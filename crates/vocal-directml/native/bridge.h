#pragma once
#include <DirectML.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include <windows.h>
#include <wrl/client.h>

namespace vdml {
using Microsoft::WRL::ComPtr;
inline void check(HRESULT hr, char const *operation) {
  if (FAILED(hr)) {
    char hex[24];
    sprintf_s(hex, "0x%08X", static_cast<unsigned>(hr));
    throw std::runtime_error(std::string(operation) + ": " + hex);
  }
}
inline void require(bool condition, char const *message) {
  if (!condition)
    throw std::runtime_error(message);
}
template <class F> int guarded(char **error, F &&fn) noexcept {
  try {
    fn();
    return 0;
  } catch (std::exception const &e) {
    if (error) {
      size_t n = std::strlen(e.what()) + 1;
      *error = static_cast<char *>(std::malloc(n));
      if (*error)
        std::memcpy(*error, e.what(), n);
    }
    return 1;
  } catch (...) {
    return 2;
  }
}
struct Node {
  uint32_t op, dtype, rank, shape[8];
  uint32_t const *inputs;
  size_t input_count;
  int32_t const *attrs;
  size_t attr_count;
  float values[4];
  uint8_t const *data;
  size_t data_bytes;
};
struct Binding {
  uint8_t *data;
  size_t bytes;
};
// Borrowed only during synchronous vdml_compile. Initialization completes its
// GPU fence before returning, so no graph retains a pointer into Rust weights.
struct ConstantView {
  uint8_t const *data{};
  size_t bytes{};
};
struct GraphMemory {
  uint64_t constant_bytes{}, constant_buffer_bytes{}, staging_bytes{};
  uint64_t initialization_temporary_bytes{}, execution_temporary_bytes{};
  uint64_t persistent_bytes{}, input_bytes{}, output_bytes{}, host_io_bytes{};
  uint64_t io_resource_count{}, io_buffer_bytes{}, upload_copies{},
      readback_copies{};
  uint64_t source_nodes{}, lowered_nodes{}, gemm_bias_fusions{};
  uint64_t constant_resource_count{}, constant_allocation_bytes{},
      io_allocation_bytes{};
};
struct CompileOptions {
  uint32_t pooled_io{}, contiguous_views{}, fuse_gemm_bias{};
  uint32_t pooled_weights{}, weight_staging_mib{};
};
struct GpuTimings {
  double upload_ms{}, dispatch_ms{}, readback_ms{};
};
struct Feedback {
  uint32_t input, output;
};
struct AdapterDesc {
  char const *id, *name, *reason;
  uint64_t dedicated_memory, shared_memory;
  uint32_t index, vendor_id, available, software;
};
struct Device {
  ComPtr<ID3D12Device> d3d;
  ComPtr<IDMLDevice> dml;
  ComPtr<ID3D12CommandQueue> queue;
  ComPtr<ID3D12Fence> fence;
  ComPtr<IDMLCommandRecorder> recorder;
  HANDLE event{};
  uint64_t serial{};
  bool debug{};
  std::string adapter_name;
  ~Device() {
    if (event)
      CloseHandle(event);
  }
  void submit(ID3D12GraphicsCommandList *list);
  void execute(ID3D12GraphicsCommandList *list);
  ComPtr<ID3D12Resource> buffer(uint64_t bytes,
                                D3D12_HEAP_TYPE heap = D3D12_HEAP_TYPE_DEFAULT);
};
struct Slot {
  uint64_t bytes{}, offset{};
  bool owned{};
  ComPtr<ID3D12Resource> resource;
  bool state{}, readback{true};
  uint64_t gpu_offset{};
};
struct Graph {
  Device *device{};
  ComPtr<IDMLCompiledOperator> compiled;
  ComPtr<ID3D12DescriptorHeap> descriptors;
  ComPtr<IDMLBindingTable> bindings;
  ComPtr<ID3D12CommandAllocator> allocator;
  ComPtr<ID3D12GraphicsCommandList> list;
  ComPtr<ID3D12CommandAllocator> reset_allocator;
  ComPtr<ID3D12GraphicsCommandList> reset_list;
  ComPtr<ID3D12Resource> temporary, persistent, upload, readback, timestamps;
  ComPtr<ID3D12Resource> input_arena, state_arena, output_arena, feedback_arena;
  ComPtr<ID3D12QueryHeap> queries;
  std::vector<Slot> inputs, outputs;
  std::vector<Feedback> feedback;
  uint8_t *upload_ptr{};
  uint8_t *readback_ptr{};
  uint64_t *timestamp_ptr{};
  uint64_t input_bytes{}, output_bytes{}, frequency{};
  uint64_t upload_bytes{};
  CompileOptions options;
  GraphMemory memory;
  ~Graph() {
    if (upload_ptr)
      upload->Unmap(0, nullptr);
    if (readback_ptr)
      readback->Unmap(0, nullptr);
    if (timestamp_ptr)
      timestamps->Unmap(0, nullptr);
  }
  void initialize(std::vector<ConstantView> const &constants);
  void allocate_io();
  void record_upload();
  void record_readback();
  void record();
  void initialize_states();
  void record_feedback();
  void reset_states();
  void run(Binding const *, size_t, Binding const *, size_t, GpuTimings *);
};
inline void transition(ID3D12GraphicsCommandList *list,
                       ID3D12Resource *resource, D3D12_RESOURCE_STATES before,
                       D3D12_RESOURCE_STATES after) {
  D3D12_RESOURCE_BARRIER barrier{};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Transition = {resource, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES,
                        before, after};
  list->ResourceBarrier(1, &barrier);
}
inline void uav_barrier(ID3D12GraphicsCommandList *list) {
  D3D12_RESOURCE_BARRIER b{};
  b.Type = D3D12_RESOURCE_BARRIER_TYPE_UAV;
  list->ResourceBarrier(1, &b);
}
inline uint64_t allocation_bytes(Device &device, ID3D12Resource *resource) {
  auto desc = resource->GetDesc();
  return device.d3d->GetResourceAllocationInfo(0, 1, &desc).SizeInBytes;
}
} // namespace vdml
