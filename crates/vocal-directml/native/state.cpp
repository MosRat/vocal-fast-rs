#include "bridge.h"

namespace vdml {
// State resources are distinct from dispatch outputs: DirectML does not need
// to support in-place aliases, and command replay always starts/ends in UAV.
void Graph::record_feedback() {
  if (options.pooled_io && !feedback.empty()) {
    transition(list.Get(), feedback_arena.Get(),
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    transition(list.Get(), state_arena.Get(),
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_DEST);
  }
  for (auto const &f : feedback) {
    auto *src = outputs[f.output].resource.Get();
    auto *dst = inputs[f.input].resource.Get();
    if (!options.pooled_io) {
      transition(list.Get(), src, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                 D3D12_RESOURCE_STATE_COPY_SOURCE);
      transition(list.Get(), dst, D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                 D3D12_RESOURCE_STATE_COPY_DEST);
    }
    list->CopyBufferRegion(dst, inputs[f.input].gpu_offset, src,
                           outputs[f.output].gpu_offset, inputs[f.input].bytes);
    if (!options.pooled_io) {
      transition(list.Get(), dst, D3D12_RESOURCE_STATE_COPY_DEST,
                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      transition(list.Get(), src, D3D12_RESOURCE_STATE_COPY_SOURCE,
                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    }
  }
  if (options.pooled_io && !feedback.empty()) {
    transition(list.Get(), state_arena.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    transition(list.Get(), feedback_arena.Get(),
               D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  }
}

void Graph::initialize_states() {
  if (feedback.empty())
    return;
  auto *d = device;
  check(d->d3d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                       IID_PPV_ARGS(&reset_allocator)),
        "create state reset allocator");
  check(d->d3d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                  reset_allocator.Get(), nullptr,
                                  IID_PPV_ARGS(&reset_list)),
        "create state reset command list");
  if (options.pooled_io)
    transition(reset_list.Get(), state_arena.Get(),
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_DEST);
  for (auto const &f : feedback) {
    auto const &s = inputs[f.input];
    std::memset(upload_ptr + s.offset, 0, static_cast<size_t>(s.bytes));
    if (!options.pooled_io)
      transition(reset_list.Get(), s.resource.Get(),
                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                 D3D12_RESOURCE_STATE_COPY_DEST);
    reset_list->CopyBufferRegion(s.resource.Get(), s.gpu_offset, upload.Get(),
                                 s.offset, s.bytes);
    if (!options.pooled_io)
      transition(reset_list.Get(), s.resource.Get(),
                 D3D12_RESOURCE_STATE_COPY_DEST,
                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  }
  if (options.pooled_io)
    transition(reset_list.Get(), state_arena.Get(),
               D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  check(reset_list->Close(), "close state reset command list");
  reset_states();
}

void Graph::reset_states() {
  if (reset_list)
    device->execute(reset_list.Get());
}
} // namespace vdml

extern "C" int vdml_reset_states(vdml::Graph *g, char **error) noexcept {
  return vdml::guarded(error, [&] {
    vdml::require(g, "null DirectML graph");
    g->reset_states();
  });
}
