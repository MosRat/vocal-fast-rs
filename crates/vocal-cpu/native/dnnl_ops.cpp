#include "internal.h"
#include <unordered_map>
namespace vocal::cpu {
using namespace dnnl;
void append_activation_post_op(primitive_attr &attr, const Node &node) {
  if (node.activation == ActivationPostOp::None && !node.residual.storage)
    return;
  post_ops post;
  if (node.residual.storage)
    post.append_binary(algorithm::binary_add, descriptor(node.residual));
  if (node.activation != ActivationPostOp::None)
    post.append_eltwise(node.activation == ActivationPostOp::Gelu
                          ? algorithm::eltwise_gelu_erf
                          : algorithm::eltwise_relu,
                      node.activation_alpha, 0);
  attr.set_post_ops(post);
}
memory::desc descriptor(const Tensor &x) {
  require(x.dtype == 1, "oneDNN compute requires FP32");
  return {x.shape, memory::data_type::f32, x.strides};
}
class Primitive final : public Kernel {
public:
  primitive operation;
  std::unordered_map<int, memory> args;
  memory scratch;
  std::unique_ptr<reorder> upload;
  memory canonical_input, prepared_input;
  void run(Graph &g) override {
    if (upload)
      upload->execute(g.stream, canonical_input, prepared_input);
    operation.execute(g.stream, args);
  }
  void workspace(Graph &g, const memory::desc &md) {
    scratch = g.workspace(0, md);
    args.emplace(DNNL_ARG_SCRATCHPAD, scratch);
  }
};
static memory packed(Graph &g, const Tensor &tensor, const memory::desc &raw,
                     const memory::desc &target) {
  auto source = memory(raw, g.engine, tensor.data());
  auto result = memory(target, g.engine);
  reorder(source, result).execute(g.stream, source, result);
  g.stream.wait();
  return result;
}
static memory::desc matrix_desc(const Tensor &t, size_t rank, bool transpose) {
  Shape dims(rank, 1), strides(rank, 1);
  size_t offset = rank - t.shape.size();
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
static std::unique_ptr<Kernel> prepare_matrix(Graph &g, size_t id) {
  auto &n = g.nodes[id];
  auto &x = g.tensors[n.inputs[0]];
  auto &w = g.tensors[n.inputs[1]];
  auto &y = g.tensors[id];
  require(x.dtype == 1 && w.dtype == 1 && y.dtype == 1,
          "matmul dtype mismatch");
  size_t rank = y.shape.size();
  auto xm = matrix_desc(x, rank, false),
       raw = matrix_desc(w, rank, n.attrs[0] != 0), ym = descriptor(y);
  bool constant = g.nodes[n.inputs[1]].folded;
  if (w.shape.size() == 2 && x.contiguous() && y.contiguous()) {
    rank = 2;
    xm = memory::desc(
        {static_cast<int64_t>(x.elements() / x.shape.back()), x.shape.back()},
        memory::data_type::f32, memory::format_tag::ab);
    ym = memory::desc(
        {static_cast<int64_t>(y.elements() / y.shape.back()), y.shape.back()},
        memory::data_type::f32, memory::format_tag::ab);
    raw = matrix_desc(w, 2, n.attrs[0] != 0);
  }
  auto wm = constant ? memory::desc(raw.get_dims(), memory::data_type::f32,
                                    memory::format_tag::any)
                     : raw;
  primitive_attr attr;
  attr.set_scratchpad_mode(scratchpad_mode::user);
  append_activation_post_op(attr, n);
  bool bias = n.inputs.size() == 3;
  memory::desc bm;
  if (bias) {
    Shape shape(rank, 1);
    shape.back() = y.shape.back();
    bm = {shape, memory::data_type::f32, contiguous_strides(shape)};
  }
  auto pd = bias ? matmul::primitive_desc(g.engine, xm, wm, bm, ym, attr)
                 : matmul::primitive_desc(g.engine, xm, wm, ym, attr);
  auto kernel = std::make_unique<Primitive>();
  kernel->operation = dnnl::matmul(pd);
  kernel->implementation = pd.impl_info_str();
  kernel->args = {{DNNL_ARG_SRC, memory(pd.src_desc(), g.engine, x.data())},
                  {DNNL_ARG_WEIGHTS, constant
                                         ? packed(g, w, raw, pd.weights_desc())
                                         : memory(raw, g.engine, w.data())},
                  {DNNL_ARG_DST, memory(pd.dst_desc(), g.engine, y.data())}};
  if (constant) {
    kernel->packed_inputs = {1};
    kernel->packed_bytes = pd.weights_desc().get_size();
  }
  if (bias)
    kernel->args.emplace(DNNL_ARG_BIAS,
                         memory(bm, g.engine, g.tensors[n.inputs[2]].data()));
  kernel->workspace(g, pd.scratchpad_desc());
  return kernel;
}
static std::unique_ptr<Kernel> convolution(Graph &g, size_t id) {
  auto &n = g.nodes[id];
  auto &x = g.tensors[n.inputs[0]];
  auto &w = g.tensors[n.inputs[1]];
  auto &y = g.tensors[id];
  auto &a = n.attrs;
  using dt = memory::data_type;
  using tag = memory::format_tag;
  require(g.nodes[n.inputs[1]].folded, "convolution requires constant weights");
  int groups = a[8];
  Shape wd;
  if (n.op == 24)
    wd = w.shape;
  else
    wd = {y.shape[1], x.shape[1] / groups, w.shape[2], w.shape[3]};
  if (groups != 1)
    wd = {groups, y.shape[1] / groups, x.shape[1] / groups, w.shape[2],
          w.shape[3]};
  auto raw = memory::desc(wd, dt::f32,
                          groups == 1 ? (n.op == 24 ? tag::oihw : tag::iohw)
                                      : (n.op == 24 ? tag::goihw : tag::giohw));
  // VFRM weights remain canonical. Only activation tensors carry NHWC strides.
  require(w.contiguous(), "convolution weights must be canonical");
  // Explicit tags also canonicalize singleton strides. A mixed NCHW/NHWC
  // descriptor otherwise selects the slow reference convolution implementation.
  auto format = g.channels_last ? tag::nhwc : tag::nchw;
  auto xm = memory::desc(x.shape, dt::f32, format),
       ym = memory::desc(y.shape, dt::f32, format),
       wm = memory::desc(wd, dt::f32, tag::any),
       bm = memory::desc({y.shape[1]}, dt::f32, tag::x);
  Shape stride = {a[0], a[1]}, dilation = {a[2] - 1, a[3] - 1},
        left = {a[4], a[5]}, right = {a[4] - a[6], a[5] - a[7]};
  primitive_attr attr;
  attr.set_scratchpad_mode(scratchpad_mode::user);
  append_activation_post_op(attr, n);
  auto kernel = std::make_unique<Primitive>();
  auto preferred = xm.get_strides();
  bool compatible = true;
  for (size_t axis = 0; axis < x.shape.size(); ++axis)
    if (x.shape[axis] > 1 && x.strides[axis] != preferred[axis])
      compatible = false;
  kernel->prepared_input =
      compatible ? memory(xm, g.engine, x.data()) : g.workspace(3, xm);
  if (!compatible) {
    kernel->canonical_input = memory(descriptor(x), g.engine, x.data());
    kernel->upload = std::make_unique<reorder>(kernel->canonical_input,
                                               kernel->prepared_input);
  }
  bool bias = n.inputs.size() == 3;
  auto finish = [&](const auto &pd) {
    kernel->implementation = pd.impl_info_str();
    if (n.residual.storage)
      kernel->implementation += ";binary_add";
    kernel->packed_inputs = {1};
    kernel->packed_bytes = pd.weights_desc().get_size();
    kernel->args = {{DNNL_ARG_SRC, kernel->prepared_input},
                    {DNNL_ARG_WEIGHTS, packed(g, w, raw, pd.weights_desc())},
                    {DNNL_ARG_DST, memory(pd.dst_desc(), g.engine, y.data())}};
    if (n.residual.storage)
      kernel->args.emplace(DNNL_ARG_ATTR_MULTIPLE_POST_OP(0) | DNNL_ARG_SRC_1,
                           memory(descriptor(n.residual), g.engine,
                                  n.residual.data()));
    if (bias)
      kernel->args.emplace(DNNL_ARG_BIAS,
                           memory(bm, g.engine, g.tensors[n.inputs[2]].data()));
    kernel->workspace(g, pd.scratchpad_desc());
  };
  if (n.op == 24) {
    auto pd = bias ? convolution_forward::primitive_desc(
                         g.engine, prop_kind::forward_inference,
                         algorithm::convolution_direct, xm, wm, bm, ym, stride,
                         dilation, left, right, attr)
                   : convolution_forward::primitive_desc(
                         g.engine, prop_kind::forward_inference,
                         algorithm::convolution_direct, xm, wm, ym, stride,
                         dilation, left, right, attr);
    finish(pd);
    kernel->operation = convolution_forward(pd);
  } else {
    auto pd = bias ? deconvolution_forward::primitive_desc(
                         g.engine, prop_kind::forward_inference,
                         algorithm::deconvolution_direct, xm, wm, bm, ym,
                         stride, dilation, left, right, attr)
                   : deconvolution_forward::primitive_desc(
                         g.engine, prop_kind::forward_inference,
                         algorithm::deconvolution_direct, xm, wm, ym, stride,
                         dilation, left, right, attr);
    finish(pd);
    kernel->operation = deconvolution_forward(pd);
  }
  return kernel;
}
std::unique_ptr<Kernel> prepare_dnnl(Graph &g, size_t id) {
  auto &n = g.nodes[id];
  if ((n.op == 23 || n.op == 53) && g.linear_mode != 0)
    if (auto kernel = prepare_linear(g, id))
      return kernel;
  if (n.op == 23 || n.op == 53)
    return prepare_matrix(g, id);
  if (n.op == 24 && g.convolution.mode != 0)
    if (auto kernel = prepare_convolution_matmul(g, id))
      return kernel;
  if (n.op == 24 || n.op == 25)
    return convolution(g, id);
  auto &x = g.tensors[n.inputs[0]];
  auto &y = g.tensors[id];
  if (x.dtype != 1 || y.dtype != 1 || !x.dense())
    return nullptr;
  primitive_attr attr;
  attr.set_scratchpad_mode(scratchpad_mode::user);
  auto kernel = std::make_unique<Primitive>();
  if (n.op == 51) {
    // Temporal GroupNorm has channel-wise affine parameters, unlike LayerNorm.
    if (n.attrs.size() != 1)
      return nullptr;
    size_t axis = n.attrs[0];
    int64_t stride = x.shape[axis];
    bool compatible = x.strides[axis] == 1 && y.strides[axis] == 1;
    for (size_t a = x.shape.size(); a-- > 0;)
      if (a != axis) {
        if (x.shape[a] > 1 &&
            (x.strides[a] != stride || y.strides[a] != stride))
          compatible = false;
        stride *= x.shape[a];
      }
    auto &scale = g.tensors[n.inputs[1]];
    auto &shift = g.tensors[n.inputs[2]];
    require(scale.elements() == static_cast<size_t>(x.shape[axis]) &&
                shift.elements() == scale.elements(),
            "layer norm affine size mismatch");
    if (!compatible || !scale.contiguous() || !shift.contiguous())
      return nullptr;
    auto md = memory::desc(
        {static_cast<int64_t>(x.elements() / x.shape[axis]), x.shape[axis]},
        memory::data_type::f32, memory::format_tag::ab);
    auto pd = layer_normalization_forward::primitive_desc(
        g.engine, prop_kind::forward_inference, md, md, n.values[0],
        normalization_flags::use_scale | normalization_flags::use_shift, attr);
    kernel->operation = layer_normalization_forward(pd);
    auto affine = memory::desc({x.shape[axis]}, memory::data_type::f32,
                               memory::format_tag::x);
    kernel->args = {{DNNL_ARG_SRC, memory(md, g.engine, x.data())},
                    {DNNL_ARG_DST, memory(md, g.engine, y.data())},
                    {DNNL_ARG_SCALE, memory(affine, g.engine, scale.data())},
                    {DNNL_ARG_SHIFT, memory(affine, g.engine, shift.data())}};
    kernel->workspace(g, pd.scratchpad_desc());
    return kernel;
  }
  if (n.op == 26) {
    auto pd = softmax_forward::primitive_desc(
        g.engine, prop_kind::forward_inference, algorithm::softmax_accurate,
        descriptor(x), descriptor(y), n.attrs[0], attr);
    kernel->operation = softmax_forward(pd);
    kernel->args = {{DNNL_ARG_SRC, memory(pd.src_desc(), g.engine, x.data())},
                    {DNNL_ARG_DST, memory(pd.dst_desc(), g.engine, y.data())}};
    kernel->workspace(g, pd.scratchpad_desc());
    return kernel;
  }
  algorithm algo;
  float alpha = 0;
  switch (n.op) {
  case 12:
    algo = algorithm::eltwise_relu;
    break;
  case 13:
    algo = algorithm::eltwise_relu;
    alpha = n.values[0];
    break;
  case 14:
    algo = algorithm::eltwise_tanh;
    break;
  case 15:
    algo = algorithm::eltwise_logistic;
    break;
  case 16:
    algo = algorithm::eltwise_gelu_erf;
    break;
  case 17:
    algo = algorithm::eltwise_exp;
    break;
  case 18:
    algo = algorithm::eltwise_log;
    break;
  case 19:
    algo = algorithm::eltwise_sqrt;
    break;
  default:
    return nullptr;
  }
  auto pd = eltwise_forward::primitive_desc(
      g.engine, prop_kind::forward_inference, algo, descriptor(x),
      descriptor(y), alpha, 0, attr);
  kernel->operation = eltwise_forward(pd);
  kernel->args = {{DNNL_ARG_SRC, memory(pd.src_desc(), g.engine, x.data())},
                  {DNNL_ARG_DST, memory(pd.dst_desc(), g.engine, y.data())}};
  kernel->workspace(g, pd.scratchpad_desc());
  return kernel;
}
} // namespace vocal::cpu
