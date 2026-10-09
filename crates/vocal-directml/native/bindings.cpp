#include "bridge.h"
#include <d3d12sdklayers.h>
namespace vdml {
void Graph::initialize(std::vector<ConstantView> const &constants) {
  auto *d = device;
  require(constants.size() == inputs.size(), "invalid constant binding count");
  check(d->d3d->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                       IID_PPV_ARGS(&allocator)),
        "create allocator");
  check(d->d3d->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT,
                                  allocator.Get(), nullptr,
                                  IID_PPV_ARGS(&list)),
        "create command list");
  ComPtr<IDMLOperatorInitializer> init;
  IDMLCompiledOperator *operators[]{compiled.Get()};
  check(d->dml->CreateOperatorInitializer(1, operators, IID_PPV_ARGS(&init)),
        "create DirectML initializer");
  auto ip = init->GetBindingProperties(), ep = compiled->GetBindingProperties();
  D3D12_DESCRIPTOR_HEAP_DESC hd{};
  hd.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  hd.NumDescriptors = std::max<UINT>(
      1, std::max(ip.RequiredDescriptorCount, ep.RequiredDescriptorCount));
  hd.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
  check(d->d3d->CreateDescriptorHeap(&hd, IID_PPV_ARGS(&descriptors)),
        "create descriptor heap");
  DML_BINDING_TABLE_DESC bd{
      init.Get(), descriptors->GetCPUDescriptorHandleForHeapStart(),
      descriptors->GetGPUDescriptorHandleForHeapStart(), hd.NumDescriptors};
  check(d->dml->CreateBindingTable(&bd, IID_PPV_ARGS(&bindings)),
        "create DirectML binding table");
  memory.initialization_temporary_bytes = ip.TemporaryResourceSize;
  memory.execution_temporary_bytes = ep.TemporaryResourceSize;
  memory.persistent_bytes = ep.PersistentResourceSize;
  std::vector<DML_BUFFER_BINDING> initial(inputs.size());
  uint64_t constant_arena_bytes{};
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].owned) {
      require(constants[i].data && constants[i].bytes &&
                  constants[i].bytes <= inputs[i].bytes,
              "invalid constant upload extent");
      memory.constant_bytes += constants[i].bytes;
      memory.constant_buffer_bytes += inputs[i].bytes;
      inputs[i].gpu_offset = constant_arena_bytes;
      constant_arena_bytes += (inputs[i].bytes + 255) & ~255ull;
    }
  }
  // Reuse one bounded staging allocation. Submissions wait for the fence
  // before this mapped arena or the command allocator can be overwritten.
  // Large tensors span batches; scalar tensors share an allocation rather
  // than each consuming a separate 64 KiB committed upload resource.
  require(options.weight_staging_mib >= 1 && options.weight_staging_mib <= 64,
          "invalid DirectML weight staging capacity");
  auto staging_limit =
      static_cast<uint64_t>(options.weight_staging_mib) * 1024 * 1024;
  auto capacity = std::min(staging_limit, memory.constant_buffer_bytes);
  memory.staging_bytes = capacity;
  ComPtr<ID3D12Resource> constant_arena;
  if (options.pooled_weights && constant_arena_bytes) {
    constant_arena = d->buffer(constant_arena_bytes);
    memory.constant_resource_count = 1;
    memory.constant_allocation_bytes =
        allocation_bytes(*d, constant_arena.Get());
    transition(list.Get(), constant_arena.Get(),
               D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_DEST);
  }
  {
    ComPtr<ID3D12Resource> staging;
    uint8_t *mapped{};
    if (capacity) {
      staging = d->buffer(capacity, D3D12_HEAP_TYPE_UPLOAD);
      D3D12_RANGE empty{0, 0};
      check(staging->Map(0, &empty, reinterpret_cast<void **>(&mapped)),
            "map constant staging arena");
    }
    uint64_t used{};
    auto flush = [&] {
      if (!used)
        return;
      d->submit(list.Get());
      check(allocator->Reset(), "reset constant upload allocator");
      check(list->Reset(allocator.Get(), nullptr),
            "reset constant upload list");
      used = 0;
    };
    for (size_t i = 0; i < inputs.size(); ++i) {
      auto &slot = inputs[i];
      if (slot.owned) {
        if (options.pooled_weights) {
          slot.resource = constant_arena;
        } else {
          slot.gpu_offset = 0;
          slot.resource = d->buffer(slot.bytes);
          ++memory.constant_resource_count;
          memory.constant_allocation_bytes +=
              allocation_bytes(*d, slot.resource.Get());
          transition(list.Get(), slot.resource.Get(),
                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                     D3D12_RESOURCE_STATE_COPY_DEST);
        }
        uint64_t offset{};
        while (offset < slot.bytes) {
          if (used == capacity)
            flush();
          auto bytes = std::min(slot.bytes - offset, capacity - used);
          auto payload =
              offset < constants[i].bytes
                  ? std::min<uint64_t>(bytes, constants[i].bytes - offset)
                  : 0;
          if (payload)
            std::memcpy(mapped + used, constants[i].data + offset,
                        static_cast<size_t>(payload));
          if (payload < bytes)
            std::memset(mapped + used + payload, 0,
                        static_cast<size_t>(bytes - payload));
          list->CopyBufferRegion(slot.resource.Get(), slot.gpu_offset + offset,
                                 staging.Get(), used, bytes);
          offset += bytes;
          used += bytes;
        }
        if (!options.pooled_weights)
          transition(list.Get(), slot.resource.Get(),
                     D3D12_RESOURCE_STATE_COPY_DEST,
                     D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        initial[i] = {slot.resource.Get(), slot.gpu_offset, slot.bytes};
      }
    }
    if (constant_arena)
      transition(list.Get(), constant_arena.Get(),
                 D3D12_RESOURCE_STATE_COPY_DEST,
                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    flush();
    if (staging)
      staging->Unmap(0, nullptr);
  } // Release staging before allocating initializer scratch and persistent
    // data.
  uint64_t scratch =
      std::max(ip.TemporaryResourceSize, ep.TemporaryResourceSize);
  if (scratch)
    temporary = d->buffer(scratch);
  if (ep.PersistentResourceSize)
    persistent = d->buffer(ep.PersistentResourceSize);
  DML_BUFFER_BINDING temp{temporary.Get(), 0, scratch},
      persist{persistent.Get(), 0, ep.PersistentResourceSize};
  DML_BINDING_DESC tb{DML_BINDING_TYPE_BUFFER, &temp},
      pb{DML_BINDING_TYPE_BUFFER, &persist};
  if (ip.TemporaryResourceSize)
    bindings->BindTemporaryResource(&tb);
  if (persistent)
    bindings->BindOutputs(1, &pb);
  DML_BUFFER_ARRAY_BINDING array{static_cast<UINT>(initial.size()),
                                 initial.data()};
  DML_BINDING_DESC ab{DML_BINDING_TYPE_BUFFER_ARRAY, &array};
  bindings->BindInputs(1, &ab);
  ID3D12DescriptorHeap *heaps[]{descriptors.Get()};
  list->SetDescriptorHeaps(1, heaps);
  d->recorder->RecordDispatch(list.Get(), init.Get(), bindings.Get());
  uav_barrier(list.Get());
  d->submit(list.Get());
  if (d->debug) {
    ComPtr<ID3D12InfoQueue> info;
    if (SUCCEEDED(d->d3d.As(&info))) {
      for (UINT64 i = 0; i < info->GetNumStoredMessages(); ++i) {
        SIZE_T length{};
        info->GetMessage(i, nullptr, &length);
        std::vector<uint8_t> bytes(length);
        auto *message = reinterpret_cast<D3D12_MESSAGE *>(bytes.data());
        if (SUCCEEDED(info->GetMessage(i, message, &length)))
          fprintf(stderr, "D3D12: %s\n", message->pDescription);
      }
      info->ClearStoredMessages();
    }
  }
  // Owned weights have been copied into persistent operator resources.
  for (auto &slot : inputs)
    if (slot.owned)
      slot.resource.Reset();
  constant_arena.Reset();
  bd.Dispatchable = compiled.Get();
  check(bindings->Reset(&bd), "bind compiled DirectML graph");
  init.Reset();
  // Initializer-only scratch has no role in inference. Do not retain its
  // potentially larger allocation for the lifetime of the loaded model.
  if (ip.TemporaryResourceSize > ep.TemporaryResourceSize) {
    temporary.Reset();
    if (ep.TemporaryResourceSize)
      temporary = d->buffer(ep.TemporaryResourceSize);
  }
  temp = {temporary.Get(), 0, ep.TemporaryResourceSize};
  if (ep.TemporaryResourceSize)
    bindings->BindTemporaryResource(&tb);
  if (persistent)
    bindings->BindPersistentResource(&pb);
  std::vector<DML_BUFFER_BINDING> ib(inputs.size()), ob(outputs.size());
  std::vector<DML_BINDING_DESC> ids(inputs.size()), ods(outputs.size());
  allocate_io();
  for (size_t i = 0; i < inputs.size(); ++i) {
    auto &s = inputs[i];
    ib[i] = {s.resource.Get(), s.gpu_offset, s.bytes};
    ids[i] = {s.owned ? DML_BINDING_TYPE_NONE : DML_BINDING_TYPE_BUFFER,
              s.owned ? nullptr : &ib[i]};
  }
  for (size_t i = 0; i < outputs.size(); ++i) {
    auto &s = outputs[i];
    ob[i] = {s.resource.Get(), s.gpu_offset, s.bytes};
    ods[i] = {DML_BINDING_TYPE_BUFFER, &ob[i]};
  }
  bindings->BindInputs(static_cast<UINT>(ids.size()), ids.data());
  bindings->BindOutputs(static_cast<UINT>(ods.size()), ods.data());
  upload = d->buffer(input_bytes, D3D12_HEAP_TYPE_UPLOAD);
  readback = d->buffer(output_bytes, D3D12_HEAP_TYPE_READBACK);
  D3D12_RANGE empty{0, 0};
  check(upload->Map(0, &empty, reinterpret_cast<void **>(&upload_ptr)),
        "map input ring");
  // Padding is copied along with pooled inputs and must be initialized.
  std::memset(upload_ptr, 0, static_cast<size_t>(input_bytes));
  D3D12_RANGE range{0, static_cast<SIZE_T>(output_bytes)};
  check(readback->Map(0, &range, reinterpret_cast<void **>(&readback_ptr)),
        "map audio readback");
  initialize_states();
  D3D12_QUERY_HEAP_DESC qh{};
  qh.Type = D3D12_QUERY_HEAP_TYPE_TIMESTAMP;
  qh.Count = 4;
  check(d->d3d->CreateQueryHeap(&qh, IID_PPV_ARGS(&queries)),
        "create timestamp queries");
  timestamps = d->buffer(32, D3D12_HEAP_TYPE_READBACK);
  D3D12_RANGE timestamp_range{0, 32};
  check(timestamps->Map(0, &timestamp_range,
                        reinterpret_cast<void **>(&timestamp_ptr)),
        "map GPU timestamp readback");
  check(d->queue->GetTimestampFrequency(&frequency),
        "get GPU timestamp frequency");
  check(allocator->Reset(), "reset allocator after initialization");
  check(list->Reset(allocator.Get(), nullptr), "reset inference command list");
  record();
  check(list->Close(), "close reusable inference command list");
  for (auto const &slot : inputs)
    if (!slot.owned)
      memory.input_bytes += slot.bytes;
  for (auto const &slot : outputs)
    memory.output_bytes += slot.bytes;
  memory.host_io_bytes = input_bytes + output_bytes + 32;
}
} // namespace vdml

extern "C" void vdml_graph_memory(vdml::Graph const *graph,
                                  vdml::GraphMemory *out) noexcept {
  if (graph && out)
    *out = graph->memory;
}
