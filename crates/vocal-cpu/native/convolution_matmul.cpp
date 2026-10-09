#include "internal.h"

namespace vocal::cpu {
using namespace dnnl;

// Static 1D convolution lowered to [N*T, K*C] x [K*C, O]. Activations
// remain NHWC, so interior windows are assembled with contiguous copies.
// Weight permutation and oneDNN packing are compile-time operations. Workspace
// class 4 is shared by every expansion and rebound by finalize_memory().
class ConvolutionMatmul final : public Kernel {
  Tensor source;
  int64_t input_channels, input_time, output_time, width, stride, dilation,
      padding, reduction;
  size_t rows;
  memory columns, scratch;
  primitive operation;
  std::unordered_map<int, memory> args;
  bool direct_view;

public:
  ConvolutionMatmul(Graph &g, size_t id) {
    const auto &node = g.nodes[id];
    source = g.tensors[node.inputs[0]];
    const auto &weight = g.tensors[node.inputs[1]];
    const auto &output = g.tensors[id];
    input_channels = source.shape[1];
    input_time = source.shape[3];
    output_time = output.shape[3];
    width = weight.shape[3];
    stride = node.attrs[1];
    dilation = node.attrs[3];
    padding = node.attrs[5];
    reduction = input_channels * width;
    rows = output.shape[0] * output_time;
    const auto outputs = output.shape[1];
    using dt = memory::data_type;
    using tag = memory::format_tag;
    const auto xm = memory::desc({static_cast<int64_t>(rows), reduction},
                                 dt::f32, tag::ab);
    const auto ym = memory::desc({static_cast<int64_t>(rows), outputs},
                                 dt::f32, tag::ab);
    const auto wm = memory::desc({reduction, outputs}, dt::f32, tag::any);
    const auto bm = memory::desc({1, outputs}, dt::f32, tag::ab);
    primitive_attr attr;
    attr.set_scratchpad_mode(scratchpad_mode::user);
    append_activation_post_op(attr, node);
    const bool bias = node.inputs.size() == 3;
    memory::desc packed_desc, scratch_desc, destination_desc, bias_desc;
    if (g.convolution.mode == 2) {
      const auto cw = memory::desc({outputs, reduction, 1, 1}, dt::f32, tag::any);
      const auto cb = memory::desc({outputs}, dt::f32, tag::x);
      // Binary residual descriptors use the original batch geometry.
      const auto sx = memory::desc({source.shape[0], reduction, 1, output_time}, dt::f32, tag::nhwc);
      const auto sy = memory::desc(output.shape, dt::f32, tag::nhwc);
      const auto pd = bias ? convolution_forward::primitive_desc(g.engine, prop_kind::forward_inference,
          algorithm::convolution_direct, sx, cw, cb, sy, {1, 1}, {0, 0}, {0, 0}, {0, 0}, attr)
          : convolution_forward::primitive_desc(g.engine, prop_kind::forward_inference,
          algorithm::convolution_direct, sx, cw, sy, {1, 1}, {0, 0}, {0, 0}, {0, 0}, attr);
      operation = convolution_forward(pd);
      implementation = std::string("unfold1d-pointwise:") + pd.impl_info_str();
      packed_desc = pd.weights_desc();
      scratch_desc = pd.scratchpad_desc();
      destination_desc = sy;
      bias_desc = cb;
    } else {
      // A residual in this lowering must describe [N*T,O], like the MatMul.
      Node post_node;
      post_node.activation = node.activation;
      post_node.activation_alpha = node.activation_alpha;
      if (node.residual.storage) {
        post_node.residual = node.residual;
        post_node.residual.shape = {static_cast<int64_t>(rows), outputs};
        post_node.residual.strides = {outputs, 1};
      }
      primitive_attr ma;
      ma.set_scratchpad_mode(scratchpad_mode::user);
      append_activation_post_op(ma, post_node);
      const auto pd = bias ? matmul::primitive_desc(g.engine, xm, wm, bm, ym, ma)
                           : matmul::primitive_desc(g.engine, xm, wm, ym, ma);
      operation = matmul(pd);
      implementation = std::string("unfold1d:") + pd.impl_info_str();
      packed_desc = pd.weights_desc();
      scratch_desc = pd.scratchpad_desc();
      destination_desc = ym;
      bias_desc = bm;
    }
    // Non-overlapping windows can be represented directly, provided a batch
    // transition introduces no unconsumed samples or padding.
    direct_view = source.strides[1] == 1 &&
                  source.strides[3] == input_channels && padding == 0 &&
                  dilation == 1 && stride == width &&
                  (source.shape[0] == 1 ||
                   (input_time == output_time * width &&
                    source.strides[0] == input_time * input_channels));
    if (direct_view)
      columns = memory(xm, g.engine, source.data());
    else
      columns = g.workspace(4, xm);

    // OIHW -> (K,C),O. Keep the temporary alive until synchronous packing ends.
    std::vector<float> reordered(weight.elements());
    const auto *canonical = static_cast<const float *>(weight.data());
    for (int64_t o = 0; o < outputs; ++o)
      for (int64_t c = 0; c < input_channels; ++c)
        for (int64_t k = 0; k < width; ++k)
          reordered[g.convolution.mode == 2 ? o * reduction + k * input_channels + c
                                           : (k * input_channels + c) * outputs + o] =
              canonical[(o * input_channels + c) * width + k];
    const auto raw = g.convolution.mode == 2 ? memory::desc({outputs, reduction, 1, 1}, dt::f32, tag::oihw)
                                           : memory::desc({reduction, outputs}, dt::f32, tag::ab);
    auto parameters = memory(packed_desc, g.engine);
    auto original = memory(raw, g.engine, reordered.data());
    reorder(original, parameters).execute(g.stream, original, parameters);
    g.stream.wait();
    packed_inputs = {1};
    packed_bytes = packed_desc.get_size();
    scratch = g.workspace(0, scratch_desc);
    // oneDNN checks memory descriptors at execution; reinterpret the same
    // unfolded workspace as NHWC for the pointwise primitive.
    auto prepared = g.convolution.mode == 2
        ? memory(memory::desc({source.shape[0], reduction, 1, output_time}, dt::f32, tag::nhwc), g.engine, columns.get_data_handle())
        : columns;
    if (g.convolution.mode == 2 && !direct_view) {
      prepared = g.workspace(4, prepared.get_desc());
    }
    args = {{DNNL_ARG_SRC, prepared},
            {DNNL_ARG_WEIGHTS, parameters},
            {DNNL_ARG_DST, memory(destination_desc, g.engine, output.data())},
            {DNNL_ARG_SCRATCHPAD, scratch}};
    if (bias)
      args.emplace(DNNL_ARG_BIAS,
                   memory(bias_desc, g.engine, g.tensors[node.inputs[2]].data()));
    if (node.residual.storage) {
      auto md = g.convolution.mode == 2 ? descriptor(node.residual) : ym;
      args.emplace(DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                   memory(md, g.engine, node.residual.data()));
    }
  }

  void run(Graph &g) override {
    if (!direct_view) {
      const auto *x = static_cast<const float *>(source.data());
      auto *dst = static_cast<float *>(columns.get_data_handle());
      g.pool->range(rows, 64, [&](size_t begin, size_t end) {
        for (size_t row = begin; row < end; ++row) {
          const auto batch = row / output_time;
          const auto first = static_cast<int64_t>(row % output_time) * stride -
                             padding;
          auto *patch = dst + row * reduction;
          // Unit dilation is one contiguous copy for interior NHWC patches.
          if (source.strides[1] == 1 &&
              source.strides[3] == input_channels && dilation == 1 &&
              first >= 0 && first + width <= input_time) {
            std::memcpy(patch,
                        x + batch * source.strides[0] + first * input_channels,
                        reduction * sizeof(float));
            continue;
          }
          for (int64_t k = 0; k < width; ++k) {
            const auto time = first + k * dilation;
            auto *channels = patch + k * input_channels;
            if (time < 0 || time >= input_time) {
              std::fill_n(channels, input_channels, 0.f);
            } else if (source.strides[1] == 1) {
              std::memcpy(channels,
                          x + batch * source.strides[0] +
                              time * source.strides[3],
                          input_channels * sizeof(float));
            } else {
              for (int64_t c = 0; c < input_channels; ++c)
                channels[c] = x[batch * source.strides[0] +
                                c * source.strides[1] +
                                time * source.strides[3]];
            }
          }
        }
      });
    }
    operation.execute(g.stream, args);
  }
};

std::unique_ptr<Kernel> prepare_convolution_matmul(Graph &g, size_t id) {
  const auto &node = g.nodes[id];
  const auto &x = g.tensors[node.inputs[0]];
  const auto &w = g.tensors[node.inputs[1]];
  const auto &y = g.tensors[id];
  if (!g.channels_last || node.op != 24 || x.dtype != 1 || w.dtype != 1 ||
      y.dtype != 1 || x.shape.size() != 4 || x.shape[2] != 1 ||
      y.shape[2] != 1 || w.shape[2] != 1 || node.attrs[8] != 1 ||
      node.attrs[0] != 1 || node.attrs[2] != 1 || node.attrs[4] != 0 ||
      node.attrs[6] != 0 || node.attrs[7] != 0 || !w.contiguous() ||
      !g.nodes[node.inputs[1]].folded)
    return nullptr;
  constexpr size_t max_expansion = 96 * 1024 * 1024;
  const size_t reduction = x.shape[1] * w.shape[3];
  const size_t rows = y.shape[0] * y.shape[3];
  if (reduction > max_expansion / sizeof(float) ||
      rows > max_expansion / (sizeof(float) * reduction))
    return nullptr;
  if (node.residual.storage && (node.residual.strides[1] != 1 ||
                               node.residual.strides[3] != y.shape[1]))
    return nullptr;
  return std::make_unique<ConvolutionMatmul>(g, id);
}
} // namespace vocal::cpu
