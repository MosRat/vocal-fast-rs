#include "bridge.h"

namespace vdml {
namespace {
// DML tensor bindings require aligned offsets. Keeping host and GPU arenas
// identical permits one copy including padding, without repacking per frame.
uint64_t aligned(uint64_t bytes) { return (bytes + 255) & ~255ull; }
} // namespace

void Graph::allocate_io() {
  // The host-upload prefix contains only dynamic inputs. Zeroed reset data
  // follows it and lives in a separate GPU arena to preserve replay history.
  for (auto &s : inputs)
    if (!s.owned && !s.state) {
      s.offset = input_bytes;
      input_bytes += aligned(s.bytes);
    }
  upload_bytes = input_bytes;
  for (auto &s : inputs)
    if (s.state) {
      s.offset = input_bytes;
      input_bytes += aligned(s.bytes);
    }
  uint64_t feedback_bytes{};
  for (auto &s : outputs)
    if (s.readback) {
      s.offset = output_bytes;
      output_bytes += aligned(s.bytes);
    } else {
      s.gpu_offset = feedback_bytes;
      feedback_bytes += aligned(s.bytes);
    }

  auto allocate = [&](uint64_t bytes) {
    ++memory.io_resource_count;
    memory.io_buffer_bytes += bytes;
    auto resource = device->buffer(bytes);
    memory.io_allocation_bytes += allocation_bytes(*device, resource.Get());
    return resource;
  };
  if (options.pooled_io) {
    if (upload_bytes)
      input_arena = allocate(upload_bytes);
    if (input_bytes > upload_bytes)
      state_arena = allocate(input_bytes - upload_bytes);
    if (output_bytes)
      output_arena = allocate(output_bytes);
    if (feedback_bytes)
      feedback_arena = allocate(feedback_bytes);
  }
  for (auto &s : inputs) {
    if (s.owned)
      continue;
    if (options.pooled_io) {
      s.resource = s.state ? state_arena : input_arena;
      s.gpu_offset = s.state ? s.offset - upload_bytes : s.offset;
    } else {
      s.resource = allocate(s.bytes);
      s.gpu_offset = 0;
    }
  }
  for (auto &s : outputs) {
    if (options.pooled_io) {
      s.resource = s.readback ? output_arena : feedback_arena;
      if (s.readback)
        s.gpu_offset = s.offset;
    } else {
      s.resource = allocate(s.bytes);
      s.gpu_offset = 0;
    }
  }
}

void Graph::record_upload() {
  if (options.pooled_io) {
    if (!upload_bytes)
      return;
    transition(list.Get(), input_arena.Get(),
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_DEST);
    list->CopyBufferRegion(input_arena.Get(), 0, upload.Get(), 0, upload_bytes);
    transition(list.Get(), input_arena.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    memory.upload_copies = 1;
  } else {
    for (auto const &s : inputs)
      if (!s.owned && !s.state) {
        transition(list.Get(), s.resource.Get(),
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                   D3D12_RESOURCE_STATE_COPY_DEST);
        list->CopyBufferRegion(s.resource.Get(), s.gpu_offset, upload.Get(),
                               s.offset, s.bytes);
        transition(list.Get(), s.resource.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                   D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        ++memory.upload_copies;
      }
  }
}

void Graph::record_readback() {
  if (options.pooled_io) {
    if (!output_bytes)
      return;
    transition(list.Get(), output_arena.Get(),
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->CopyBufferRegion(readback.Get(), 0, output_arena.Get(), 0,
                           output_bytes);
    transition(list.Get(), output_arena.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    memory.readback_copies = 1;
  } else {
    for (auto const &s : outputs) {
      if (!s.readback)
        continue;
      transition(list.Get(), s.resource.Get(),
                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                 D3D12_RESOURCE_STATE_COPY_SOURCE);
      list->CopyBufferRegion(readback.Get(), s.offset, s.resource.Get(),
                             s.gpu_offset, s.bytes);
      transition(list.Get(), s.resource.Get(), D3D12_RESOURCE_STATE_COPY_SOURCE,
                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      ++memory.readback_copies;
    }
  }
}
} // namespace vdml
