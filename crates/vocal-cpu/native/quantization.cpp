#include "internal.h"
#include <cmath>
namespace vocal::cpu {
using namespace dnnl;

void observe_input(Graph &g, size_t id) {
  auto &n = g.nodes[id];
  const auto &x = g.tensors[n.inputs[0]];
  const size_t axis = n.op == 24 || n.op == 25 ? 1 : x.shape.size() - 1;
  require(x.dtype == 1 && n.input_max.size() == x.shape[axis],
          "calibration input dtype/channels mismatch");
  const auto *data = static_cast<const float *>(x.data());
  if (x.dense()) {
    for (size_t i = 0; i < x.elements(); ++i) {
      require(std::isfinite(data[i]), "non-finite calibration input");
      const size_t channel = (i / x.strides[axis]) % x.shape[axis];
      n.input_max[channel] = std::max(n.input_max[channel], std::abs(data[i]));
    }
  } else {
    size_t tail = 1;
    for (size_t i = axis + 1; i < x.shape.size(); ++i)
      tail *= x.shape[i];
    for (size_t i = 0; i < x.elements(); ++i) {
      const float value = data[x.position(i)];
      require(std::isfinite(value), "non-finite calibration input");
      const size_t channel = (i / tail) % x.shape[axis];
      n.input_max[channel] = std::max(n.input_max[channel], std::abs(value));
    }
  }
  ++n.observations;
}

static memory::desc matrix_desc(const Tensor &t, size_t rank, bool transpose) {
  Shape dims(rank, 1), strides(rank, 1);
  const size_t offset = rank - t.shape.size();
  for (size_t i = 0; i < t.shape.size(); ++i) {
    dims[offset + i] = t.shape[i];
    strides[offset + i] = t.strides[i];
  }
  if (transpose) {
    std::swap(dims[rank - 2], dims[rank - 1]);
    std::swap(strides[rank - 2], strides[rank - 1]);
  }
  return {dims, memory::data_type::f32, strides};
}

class Int8Primitive final : public Kernel {
  primitive operation;
  std::unique_ptr<reorder> upload;
  std::unordered_map<int, memory> args, upload_args;
  std::vector<float> weight_scales, quantization_scales;
  float activation_scale = 1;

public:
  Int8Primitive(Graph &g, size_t id) {
    using dt = memory::data_type;
    using tag = memory::format_tag;
    auto &n = g.nodes[id];
    auto &x = g.tensors[n.inputs[0]];
    auto &w = g.tensors[n.inputs[1]];
    auto &y = g.tensors[id];
    const bool conv = n.op == 24 || n.op == 25;
    const bool bias = n.inputs.size() == 3;
    require((conv || n.op == 23 || n.op == 53) && x.dtype == 1 &&
                w.dtype == 1 && y.dtype == 1 && g.nodes[n.inputs[1]].folded,
            "static W8A8 requires a floating convolution or constant-weight GEMM");
    const size_t input_channels = conv ? x.shape[1] : x.shape.back();
    const size_t output_channels = conv ? y.shape[1] : y.shape.back();
    require(n.input_max.size() == input_channels, "W8A8 calibration channel mismatch");
    memory::desc original_x, raw_w, xm, wm, ym, bm;
    int groups = 1, weight_mask = 0, activation_mask = 0;
    primitive_attr attr;
    attr.set_scratchpad_mode(scratchpad_mode::user);
    attr.set_scales_mask(DNNL_ARG_SRC, 0);
    if (conv) {
      require(w.contiguous(), "W8A8 convolution weight layout must be canonical");
      auto &a = n.attrs;
      groups = a[8];
      Shape wd = n.op == 24 ? w.shape
                           : Shape{y.shape[1], x.shape[1] / groups, w.shape[2], w.shape[3]};
      if (groups != 1)
        wd = {groups, y.shape[1] / groups, x.shape[1] / groups, w.shape[2], w.shape[3]};
      raw_w = memory::desc(wd, dt::f32,
                          groups == 1 ? (n.op == 24 ? tag::oihw : tag::iohw)
                                      : (n.op == 24 ? tag::goihw : tag::giohw));
      original_x = descriptor(x);
      auto format = g.channels_last ? tag::nhwc : tag::nchw;
      xm = memory::desc(x.shape, dt::s8, format);
      wm = memory::desc(wd, dt::s8, tag::any);
      ym = memory::desc(y.shape, dt::f32, format);
      bm = memory::desc({y.shape[1]}, dt::f32, tag::x);
      weight_mask = groups == 1 ? 1 : 3;
      activation_mask = 2;
    } else {
      require(w.shape.size() == 2, "W8A8 requires static two-dimensional GEMM weights");
      size_t rank = y.shape.size();
      original_x = descriptor(x);
      raw_w = matrix_desc(w, rank, n.attrs[0] != 0);
      ym = descriptor(y);
      if (x.contiguous() && y.contiguous()) {
        rank = 2;
        original_x = memory::desc({static_cast<int64_t>(x.elements() / input_channels),
                                  static_cast<int64_t>(input_channels)}, dt::f32, tag::ab);
        raw_w = matrix_desc(w, rank, n.attrs[0] != 0);
        ym = memory::desc({static_cast<int64_t>(y.elements() / output_channels),
                          static_cast<int64_t>(output_channels)}, dt::f32, tag::ab);
      }
      const auto dims = original_x.get_dims();
      xm = memory::desc(dims, dt::s8, contiguous_strides(dims));
      wm = memory::desc(raw_w.get_dims(), dt::s8, tag::any);
      Shape bd(rank, 1);
      bd.back() = output_channels;
      bm = memory::desc(bd, dt::f32, contiguous_strides(bd));
      weight_mask = 1 << (rank - 1);
      activation_mask = 1 << (rank - 1);
    }
    attr.set_scales_mask(DNNL_ARG_WEIGHTS, weight_mask);
    append_activation_post_op(attr, n);
    memory::desc scratch;
    auto finish = [&](const auto &pd) {
      xm = pd.src_desc();
      wm = pd.weights_desc();
      ym = pd.dst_desc();
      scratch = pd.scratchpad_desc();
      implementation = pd.impl_info_str();
    };
    if (conv) {
      const auto &a = n.attrs;
      Shape stride = {a[0], a[1]}, dilation = {a[2] - 1, a[3] - 1},
            left = {a[4], a[5]}, right = {a[4] - a[6], a[5] - a[7]};
      if (n.op == 24) {
        auto pd = bias ? convolution_forward::primitive_desc(
                             g.engine, prop_kind::forward_inference, algorithm::convolution_direct,
                             xm, wm, bm, ym, stride, dilation, left, right, attr)
                       : convolution_forward::primitive_desc(
                             g.engine, prop_kind::forward_inference, algorithm::convolution_direct,
                             xm, wm, ym, stride, dilation, left, right, attr);
        finish(pd);
        operation = convolution_forward(pd);
      } else {
        auto pd = bias ? deconvolution_forward::primitive_desc(
                             g.engine, prop_kind::forward_inference, algorithm::deconvolution_direct,
                             xm, wm, bm, ym, stride, dilation, left, right, attr)
                       : deconvolution_forward::primitive_desc(
                             g.engine, prop_kind::forward_inference, algorithm::deconvolution_direct,
                             xm, wm, ym, stride, dilation, left, right, attr);
        finish(pd);
        operation = deconvolution_forward(pd);
      }
    } else {
      auto pd = bias ? matmul::primitive_desc(g.engine, xm, wm, bm, ym, attr)
                     : matmul::primitive_desc(g.engine, xm, wm, ym, attr);
      finish(pd);
      operation = matmul(pd);
    }
    // The ordinary AVX2 signed-dot fallback changes the weight grid on this
    // SDK. Refuse it instead of silently changing the calibration contract.
    require(implementation.find("vnni") != std::string::npos ||
                implementation.find("amx") != std::string::npos,
            "W8A8 requires a VNNI/AMX oneDNN implementation");
    const auto dims = raw_w.get_dims(), strides = raw_w.get_strides();
    auto coordinates = [&](size_t i, size_t &physical, size_t &ci, size_t &co) {
      size_t coords[8] = {};
      physical = 0;
      for (size_t j = dims.size(); j-- > 0;) {
        coords[j] = i % dims[j];
        i /= dims[j];
        physical += coords[j] * strides[j];
      }
      if (!conv) {
        ci = coords[dims.size() - 2];
        co = coords[dims.size() - 1];
      } else if (groups == 1) {
        co = coords[0]; ci = coords[1];
      } else {
        co = coords[0] * dims[1] + coords[1];
        ci = coords[0] * dims[2] + coords[2];
      }
    };
    const auto *weights = static_cast<const float *>(w.data());
    std::vector<float> input_weight_max(input_channels, 1e-8f), smooth(input_channels, 1);
    for (size_t i = 0; i < w.elements(); ++i) {
      size_t physical, ci, co;
      coordinates(i, physical, ci, co);
      require(std::isfinite(weights[physical]), "non-finite quantized weight");
      input_weight_max[ci] = std::max(input_weight_max[ci], std::abs(weights[physical]));
    }
    if (n.smooth_alpha >= 0) {
      require(n.smooth_alpha <= 1 && std::isfinite(n.smooth_alpha), "invalid SmoothQuant alpha");
      double log_sum = 0;
      for (size_t c = 0; c < input_channels; ++c) {
        smooth[c] = std::pow(std::max(n.input_max[c], 1e-8f), n.smooth_alpha) /
                    std::pow(input_weight_max[c], 1 - n.smooth_alpha);
        log_sum += std::log(smooth[c]);
      }
      const float mean = static_cast<float>(std::exp(log_sum / input_channels));
      for (auto &s : smooth) s /= mean;
    }
    activation_scale = 1e-8f;
    for (size_t c = 0; c < input_channels; ++c) {
      require(std::isfinite(n.input_max[c]) && n.input_max[c] >= 0 &&
                  std::isfinite(smooth[c]) && smooth[c] > 0, "invalid static calibration range");
      activation_scale = std::max(activation_scale, n.input_max[c] / smooth[c]);
    }
    activation_scale /= 127;
    const bool smooth_input = n.smooth_alpha >= 0;
    quantization_scales.resize(smooth_input ? input_channels : 1);
    if (smooth_input)
      for (size_t c = 0; c < input_channels; ++c)
        quantization_scales[c] = activation_scale * smooth[c];
    else
      quantization_scales[0] = activation_scale;
    weight_scales.assign(output_channels, 1e-8f);
    for (size_t i = 0; i < w.elements(); ++i) {
      size_t physical, ci, co;
      coordinates(i, physical, ci, co);
      weight_scales[co] = std::max(weight_scales[co], std::abs(weights[physical] * smooth[ci]));
    }
    for (auto &s : weight_scales) s /= 127;
    std::vector<int8_t> quantized(w.elements());
    for (size_t i = 0; i < w.elements(); ++i) {
      size_t physical, ci, co;
      coordinates(i, physical, ci, co);
      quantized[physical] = static_cast<int8_t>(std::clamp(
          std::nearbyint(weights[physical] * smooth[ci] / weight_scales[co]), -127.f, 127.f));
    }
    const auto quantized_md = memory::desc(dims, dt::s8, strides);
    auto raw_weight = memory(quantized_md, g.engine, quantized.data());
    auto packed_weight = memory(wm, g.engine);
    reorder(raw_weight, packed_weight).execute(g.stream, raw_weight, packed_weight);
    g.stream.wait();
    auto input = g.workspace(4, xm);
    primitive_attr qa;
    // Fold x/s and scalar quantization into one layout-aware reorder. No
    // intermediate FP32 multiply or per-call scale reduction is needed.
    qa.set_scales_mask(DNNL_ARG_DST, smooth_input ? activation_mask : 0);
    upload = std::make_unique<reorder>(reorder::primitive_desc(g.engine, original_x, g.engine, xm, qa));
    auto qscales = memory({{static_cast<int64_t>(quantization_scales.size())}, dt::f32, tag::x},
                          g.engine, quantization_scales.data());
    upload_args = {{DNNL_ARG_FROM, memory(original_x, g.engine, x.data())},
                   {DNNL_ARG_TO, input}, {DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST, qscales}};
    auto ascale = memory({{1}, dt::f32, tag::x}, g.engine, &activation_scale);
    auto wscale = memory({{static_cast<int64_t>(output_channels)}, dt::f32, tag::x},
                         g.engine, weight_scales.data());
    args = {{DNNL_ARG_SRC, input}, {DNNL_ARG_WEIGHTS, packed_weight},
             {DNNL_ARG_DST, memory(ym, g.engine, y.data())},
             {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, ascale},
             {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, wscale},
             {DNNL_ARG_SCRATCHPAD, g.workspace(0, scratch)}};
    if (bias)
      args.emplace(DNNL_ARG_BIAS, memory(bm, g.engine, g.tensors[n.inputs[2]].data()));
    packed_inputs = {1};
    packed_bytes = wm.get_size();
    auxiliary_bytes = (weight_scales.size() + quantization_scales.size() + 1) * sizeof(float);
  }
  void run(Graph &g) override {
    upload->execute(g.stream, upload_args);
    operation.execute(g.stream, args);
  }
};
std::unique_ptr<Kernel> prepare_int8(Graph &g, size_t id) {
  return std::make_unique<Int8Primitive>(g, id);
}
} // namespace vocal::cpu
