#include "internal.h"
#include <unordered_set>
namespace vocal::cpu {
void Graph::finalize_memory() {
  stream.wait();
  // All operators execute synchronously. Rebind every cached memory handle to
  // the largest allocation in its workspace class before releasing older ones.
  for (auto &[key, memory] : workspace_bindings)
    memory.set_data_handle(workspace_slots.at(key)->data);
  workspace_bindings.clear();
  compiling_workspaces.clear();

  std::unordered_set<const Storage *> live;
  auto keep = [&](size_t id) {
    if (tensors[id].storage)
      live.insert(tensors[id].storage.get());
  };
  for (auto id : outputs)
    keep(id);
  for (size_t id = 0; id < nodes.size(); ++id) {
    auto &node = nodes[id];
    if (node.op == 0)
      keep(id);
    if (node.folded || node.alias || node.op <= 1)
      continue;
    keep(id);
    if (node.residual.storage)
      live.insert(node.residual.storage.get());
    for (size_t i = 0; i < node.inputs.size(); ++i) {
      if (!node.kernel || std::find(node.kernel->packed_inputs.begin(),
                                    node.kernel->packed_inputs.end(),
                                    i) == node.kernel->packed_inputs.end())
        keep(node.inputs[i]);
    }
  }
  // Tensor views share Storage, so retain an entire allocation whenever any
  // live view or non-packed consumer still needs it, including marked
  // constants.
  for (auto &tensor : tensors)
    if (tensor.storage && !live.contains(tensor.storage.get()))
      tensor.storage.reset();
  std::erase_if(slots,
                [&](const auto &slot) { return !live.contains(slot.get()); });
}
CpuMemory Graph::memory_usage() const {
  CpuMemory result{};
  for (auto &slot : slots) {
    result.tensor_bytes += slot->bytes;
    ++result.tensor_allocations;
  }
  for (auto &[key, slot] : workspace_slots) {
    result.workspace_bytes += slot->bytes;
    ++result.workspace_allocations;
  }
  for (auto &node : nodes)
    if (node.kernel) {
      result.packed_bytes += node.kernel->packed_bytes;
      result.auxiliary_bytes += node.kernel->auxiliary_bytes;
    }
  return result;
}
} // namespace vocal::cpu
extern "C" void vocal_cpu_memory(void *ptr, CpuMemory *result) {
  *result = static_cast<vocal::cpu::Graph *>(ptr)->memory_usage();
}
