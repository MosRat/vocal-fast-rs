#include "internal.h"
#include <unordered_map>
namespace vocal::cpu {
using namespace dnnl;
class Attention final : public Kernel {
  primitive dot, normalize, context;
  memory scores, probabilities, scratch1, scratch2, scratch3;
  std::unordered_map<int, memory> dot_args, softmax_args, context_args;

public:
  Attention(Graph &g, size_t id) {
    implementation = "onednn:dense-fp32-attention";
    auto &n = g.nodes[id];
    auto &q = g.tensors[n.inputs[0]];
    auto &k = g.tensors[n.inputs[1]];
    auto &v = g.tensors[n.inputs[2]];
    auto &y = g.tensors[id];
    Shape shape = q.shape;
    shape.back() = k.shape[k.shape.size() - 2];
    auto score_md =
        memory::desc(shape, memory::data_type::f32, contiguous_strides(shape));
    scores = g.workspace(1, score_md);
    // oneDNN inference softmax supports in-place source/destination memory.
    probabilities = scores;
    auto transposed = k;
    std::swap(transposed.shape[transposed.shape.size() - 1],
              transposed.shape[transposed.shape.size() - 2]);
    std::swap(transposed.strides[transposed.strides.size() - 1],
              transposed.strides[transposed.strides.size() - 2]);
    primitive_attr attr;
    attr.set_scratchpad_mode(scratchpad_mode::user);
    primitive_attr scaled;
    scaled.set_scratchpad_mode(scratchpad_mode::user);
    post_ops post;
    post.append_eltwise(algorithm::eltwise_linear, n.values[0], 0);
    scaled.set_post_ops(post);
    auto dp = matmul::primitive_desc(g.engine, descriptor(q),
                                     descriptor(transposed), score_md, scaled);
    dot = matmul(dp);
    scratch1 = g.workspace(0, dp.scratchpad_desc());
    dot_args = {
        {DNNL_ARG_SRC, memory(descriptor(q), g.engine, q.data())},
        {DNNL_ARG_WEIGHTS, memory(descriptor(transposed), g.engine, k.data())},
        {DNNL_ARG_DST, scores},
        {DNNL_ARG_SCRATCHPAD, scratch1}};
    auto sp = softmax_forward::primitive_desc(
        g.engine, prop_kind::forward_inference, algorithm::softmax_accurate,
        score_md, score_md, shape.size() - 1, attr);
    normalize = softmax_forward(sp);
    scratch2 = g.workspace(0, sp.scratchpad_desc());
    softmax_args = {{DNNL_ARG_SRC, scores},
                    {DNNL_ARG_DST, probabilities},
                    {DNNL_ARG_SCRATCHPAD, scratch2}};
    auto cp = matmul::primitive_desc(g.engine, score_md, descriptor(v),
                                     descriptor(y), attr);
    context = matmul(cp);
    scratch3 = g.workspace(0, cp.scratchpad_desc());
    context_args = {
        {DNNL_ARG_SRC, probabilities},
        {DNNL_ARG_WEIGHTS, memory(descriptor(v), g.engine, v.data())},
        {DNNL_ARG_DST, memory(descriptor(y), g.engine, y.data())},
        {DNNL_ARG_SCRATCHPAD, scratch3}};
  }
  void run(Graph &g) override {
    dot.execute(g.stream, dot_args);
    normalize.execute(g.stream, softmax_args);
    context.execute(g.stream, context_args);
  }
};
std::unique_ptr<Kernel> prepare_attention(Graph &g, size_t id) {
  if (g.attention.mode == 1)
    return prepare_online_attention(g, id);
  return std::make_unique<Attention>(g, id);
}
} // namespace vocal::cpu
