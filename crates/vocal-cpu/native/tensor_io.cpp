#include "internal.h"

namespace vocal::cpu {
using namespace dnnl;
class TensorReorder final : public Kernel {
  reorder operation;
  memory source, destination, scratch;
  std::unordered_map<int, memory> args;
public:
  TensorReorder(Graph &g, const Tensor &x, const Tensor &y) {
    // Reshape semantics preserve canonical element order. Reorder into x's
    // canonical shape, then expose the same storage using y's new shape.
    auto target = memory::desc(x.shape, memory::data_type::f32, contiguous_strides(x.shape));
    primitive_attr attr;
    attr.set_scratchpad_mode(scratchpad_mode::user);
    auto pd = reorder::primitive_desc(g.engine, descriptor(x), g.engine, target, attr);
    operation = reorder(pd);
    source = memory(descriptor(x), g.engine, x.data());
    destination = memory(target, g.engine, y.data());
    scratch = g.workspace(0, pd.scratchpad_desc());
    args = {{DNNL_ARG_FROM, source}, {DNNL_ARG_TO, destination}, {DNNL_ARG_SCRATCHPAD, scratch}};
    implementation = "onednn:reshape-reorder";
  }
  void run(Graph &g) override {
    operation.execute(g.stream, args);
  }
};
std::unique_ptr<Kernel> prepare_tensor_reorder(Graph &g, size_t id) {
  auto &n = g.nodes[id];
  auto &x = g.tensors[n.inputs[0]];
  auto &y = g.tensors[id];
  if (x.dtype != 1 || y.dtype != 1 || !x.dense() || !y.contiguous())
    return nullptr;
  require(x.elements() == y.elements(), "reshape reorder size mismatch");
  return std::make_unique<TensorReorder>(g, x, y);
}

// RVC relative attention gathers a tiny relative-position row into a complete
// score row. Compute each outer-row offset once instead of dividing/modding
// every element across all axes. Dynamic indices retain bounds checks.
bool gather_elements_rows(Graph &g, size_t id) {
  auto &n = g.nodes[id];
  auto &x = g.tensors[n.inputs[0]];
  auto &indices = g.tensors[n.inputs[1]];
  auto &y = g.tensors[id];
  size_t rank = x.shape.size();
  if (x.dtype != 1 || y.dtype != 1 || indices.dtype != 3 ||
      rank != y.shape.size() || n.attrs[0] != static_cast<int32_t>(rank - 1) ||
      !x.contiguous() || !indices.contiguous() || !y.contiguous() || indices.shape != y.shape)
    return false;
  for (size_t a = 0; a + 1 < rank; ++a)
    if (x.shape[a] != y.shape[a])
      return false;
  const auto *input = static_cast<const float *>(x.data());
  const auto *index = static_cast<const int32_t *>(indices.data());
  auto *output = static_cast<float *>(y.data());
  size_t width = y.shape.back(), source_width = x.shape.back();
  g.pool->range(y.elements() / width, 32, [&](size_t begin, size_t end) {
    for (size_t row = begin; row < end; ++row)
      for (size_t col = 0; col < width; ++col) {
        int64_t selected = index[row * width + col];
        if (selected < 0)
          selected += source_width;
        require(selected >= 0 && selected < static_cast<int64_t>(source_width), "gather index out of range");
        output[row * width + col] = input[row * source_width + selected];
      }
  });
  return true;
}
} // namespace vocal::cpu
