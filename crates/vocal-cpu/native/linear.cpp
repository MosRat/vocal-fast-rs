#include "internal.h"

namespace vocal::cpu {
using namespace dnnl;

class InnerProduct final : public Kernel {
  primitive operation;
  std::unordered_map<int, memory> args;

public:
  InnerProduct(Graph &g, size_t id) {
    const auto &node = g.nodes[id];
    const auto &x = g.tensors[node.inputs[0]];
    const auto &w = g.tensors[node.inputs[1]];
    const auto &y = g.tensors[id];
    const int64_t inputs = x.shape.back(), outputs = y.shape.back();
    const int64_t rows = x.elements() / inputs;
    using dt = memory::data_type;
    using tag = memory::format_tag;
    const auto xm = memory::desc({rows, inputs}, dt::f32, tag::nc);
    const auto ym = memory::desc({rows, outputs}, dt::f32, tag::nc);
    const auto wm = memory::desc({outputs, inputs}, dt::f32, tag::any);
    const auto bm = memory::desc({outputs}, dt::f32, tag::x);
    const bool transposed = node.attrs[0] != 0, bias = node.inputs.size() == 3;
    const auto raw = memory::desc({outputs, inputs}, dt::f32,
                                 transposed ? w.strides
                                            : Shape{w.strides[1], w.strides[0]});
    primitive_attr attr;
    attr.set_scratchpad_mode(scratchpad_mode::user);
    append_activation_post_op(attr, node);
    // oneDNN 3.10.2's BRGEMM InnerProduct dispatcher accepts forward_training
    // only (jit_brgemm_inner_product.hpp). For this stateless affine operator
    // its forward math is identical; no backward/training state is created.
    const auto descriptor = [&] {
      return bias ? inner_product_forward::primitive_desc(
                              g.engine, prop_kind::forward_training, xm, wm,
                              bm, ym, attr)
                         : inner_product_forward::primitive_desc(
                              g.engine, prop_kind::forward_training, xm, wm,
                              ym, attr);
    };
    auto pd = descriptor();
    // The MatMul wrapper precedes packed BRGEMM in oneDNN's dispatch list.
    // Enumerate supported implementations once, during graph compilation.
    while (std::string(pd.impl_info_str()).find("brgemm:") == std::string::npos &&
           pd.next_impl()) {}
    if (std::string(pd.impl_info_str()).find("brgemm:") == std::string::npos)
      pd = descriptor();
    operation = inner_product_forward(pd);
    implementation = std::string("inner-product:") + pd.impl_info_str();
    auto parameters = memory(pd.weights_desc(), g.engine);
    auto original = memory(raw, g.engine, w.data());
    reorder(original, parameters).execute(g.stream, original, parameters);
    g.stream.wait();
    packed_inputs = {1};
    packed_bytes = pd.weights_desc().get_size();
    args = {{DNNL_ARG_SRC, memory(xm, g.engine, x.data())},
            {DNNL_ARG_WEIGHTS, parameters},
            {DNNL_ARG_DST, memory(ym, g.engine, y.data())},
            {DNNL_ARG_SCRATCHPAD, g.workspace(0, pd.scratchpad_desc())}};
    if (bias)
      args.emplace(DNNL_ARG_BIAS,
                   memory(bm, g.engine, g.tensors[node.inputs[2]].data()));
  }
  void run(Graph &g) override { operation.execute(g.stream, args); }
};

// Token-major FP32 projections are also a channels-last 1x1 convolution.
// This alternative keeps activation memory intact and packs weights once.
class PointwiseLinear final : public Kernel {
  primitive operation;
  std::unordered_map<int, memory> args;

public:
  PointwiseLinear(Graph &g, size_t id) {
    const auto &node = g.nodes[id];
    const auto &x = g.tensors[node.inputs[0]];
    const auto &w = g.tensors[node.inputs[1]];
    const auto &y = g.tensors[id];
    const int64_t inputs = x.shape.back(), outputs = y.shape.back();
    const int64_t rows = x.elements() / inputs;
    using dt = memory::data_type;
    using tag = memory::format_tag;
    const auto xm = memory::desc({1, inputs, 1, rows}, dt::f32, tag::nhwc);
    const auto ym = memory::desc({1, outputs, 1, rows}, dt::f32, tag::nhwc);
    const auto wm = memory::desc({outputs, inputs, 1, 1}, dt::f32, tag::any);
    const auto bm = memory::desc({outputs}, dt::f32, tag::x);
    const bool transposed = node.attrs[0] != 0, bias = node.inputs.size() == 3;
    const auto raw = memory::desc(
        {outputs, inputs, 1, 1}, dt::f32,
        transposed ? Shape{w.strides[0], w.strides[1], 1, 1}
                   : Shape{w.strides[1], w.strides[0], 1, 1});
    primitive_attr attr;
    attr.set_scratchpad_mode(scratchpad_mode::user);
    append_activation_post_op(attr, node);
    const auto pd = bias ? convolution_forward::primitive_desc(
                               g.engine, prop_kind::forward_inference,
                               algorithm::convolution_direct, xm, wm, bm, ym,
                               {1, 1}, {0, 0}, {0, 0}, {0, 0}, attr)
                         : convolution_forward::primitive_desc(
                               g.engine, prop_kind::forward_inference,
                               algorithm::convolution_direct, xm, wm, ym,
                               {1, 1}, {0, 0}, {0, 0}, {0, 0}, attr);
    operation = convolution_forward(pd);
    implementation = std::string("pointwise-linear:") + pd.impl_info_str();
    auto parameters = memory(pd.weights_desc(), g.engine);
    auto original = memory(raw, g.engine, w.data());
    reorder(original, parameters).execute(g.stream, original, parameters);
    g.stream.wait();
    packed_inputs = {1};
    packed_bytes = pd.weights_desc().get_size();
    args = {{DNNL_ARG_SRC, memory(xm, g.engine, x.data())},
            {DNNL_ARG_WEIGHTS, parameters},
            {DNNL_ARG_DST, memory(ym, g.engine, y.data())},
            {DNNL_ARG_SCRATCHPAD, g.workspace(0, pd.scratchpad_desc())}};
    if (bias)
      args.emplace(DNNL_ARG_BIAS,
                   memory(bm, g.engine, g.tensors[node.inputs[2]].data()));
  }
  void run(Graph &g) override { operation.execute(g.stream, args); }
};

std::unique_ptr<Kernel> prepare_linear(Graph &g, size_t id) {
  const auto &node = g.nodes[id];
  const auto &x = g.tensors[node.inputs[0]];
  const auto &w = g.tensors[node.inputs[1]];
  const auto &y = g.tensors[id];
  if (x.dtype != 1 || w.dtype != 1 || y.dtype != 1 || w.shape.size() != 2 ||
      !x.contiguous() || !y.contiguous() || !w.dense() ||
      !g.nodes[node.inputs[1]].folded ||
      (node.inputs.size() == 3 && !g.tensors[node.inputs[2]].contiguous()))
    return nullptr;
  // Real 34/140-token captures favor MatMul for the long 3072-channel
  // reduction. Keep this heuristic opt-in: other CPUs/shapes need measurement.
  if (g.linear_mode == 3 && x.elements() / x.shape.back() >= 64 &&
      x.shape.back() >= 1536)
    return nullptr;
  if (g.linear_mode == 2 || g.linear_mode == 3)
    return std::make_unique<PointwiseLinear>(g, id);
  return std::make_unique<InnerProduct>(g, id);
}
} // namespace vocal::cpu
