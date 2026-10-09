#include "bridge.h"
#include <d3d12sdklayers.h>
namespace {
std::string utf8(wchar_t const *text) {
  auto bytes =
      WideCharToMultiByte(CP_UTF8, 0, text, -1, nullptr, 0, nullptr, nullptr);
  vdml::require(bytes > 0, "encode adapter name");
  std::string result(static_cast<size_t>(bytes), '\0');
  vdml::require(WideCharToMultiByte(CP_UTF8, 0, text, -1, result.data(), bytes,
                                    nullptr, nullptr) > 0,
                "encode adapter name");
  result.pop_back();
  return result;
}
std::string adapter_id(DXGI_ADAPTER_DESC1 const &desc) {
  DISPLAYCONFIG_ADAPTER_NAME query{};
  query.header.type = DISPLAYCONFIG_DEVICE_INFO_GET_ADAPTER_NAME;
  query.header.size = sizeof(query);
  query.header.adapterId = desc.AdapterLuid;
  if (DisplayConfigGetDeviceInfo(&query.header) == ERROR_SUCCESS &&
      query.adapterDevicePath[0])
    return "dxgi:" + utf8(query.adapterDevicePath);
  // Some headless adapters do not expose a display interface. Their LUID is
  // valid for this Windows boot; a missing saved ID requires reselection.
  char id[40];
  sprintf_s(id, "dxgi-luid:%08X%08X",
            static_cast<unsigned>(desc.AdapterLuid.HighPart),
            static_cast<unsigned>(desc.AdapterLuid.LowPart));
  return id;
}
struct Adapter {
  std::string id, name, reason;
  DXGI_ADAPTER_DESC1 desc{};
  uint32_t index{};
};
using Adapters = std::vector<Adapter>;
} // namespace
namespace vdml {

ComPtr<ID3D12Resource> Device::buffer(uint64_t bytes, D3D12_HEAP_TYPE heap) {
  D3D12_HEAP_PROPERTIES hp{};
  hp.Type = heap;
  D3D12_RESOURCE_DESC desc{};
  desc.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  desc.Width = std::max<uint64_t>(bytes, 4);
  desc.Height = 1;
  desc.DepthOrArraySize = 1;
  desc.MipLevels = 1;
  desc.SampleDesc.Count = 1;
  desc.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  desc.Flags = heap == D3D12_HEAP_TYPE_DEFAULT
                   ? D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
                   : D3D12_RESOURCE_FLAG_NONE;
  auto state = heap == D3D12_HEAP_TYPE_UPLOAD
                   ? D3D12_RESOURCE_STATE_GENERIC_READ
               : heap == D3D12_HEAP_TYPE_READBACK
                   ? D3D12_RESOURCE_STATE_COPY_DEST
                   : D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
  ComPtr<ID3D12Resource> result;
  check(d3d->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &desc, state,
                                     nullptr, IID_PPV_ARGS(&result)),
        "allocate D3D12 buffer");
  return result;
}
void Device::submit(ID3D12GraphicsCommandList *list) {
  check(list->Close(), "close D3D12 command list");
  execute(list);
}
void Device::execute(ID3D12GraphicsCommandList *list) {
  ID3D12CommandList *lists[]{list};
  queue->ExecuteCommandLists(1, lists);
  auto value = ++serial;
  check(queue->Signal(fence.Get(), value), "signal D3D12 fence");
  if (fence->GetCompletedValue() < value) {
    check(fence->SetEventOnCompletion(value, event), "arm D3D12 fence");
    require(WaitForSingleObject(event, INFINITE) == WAIT_OBJECT_0,
            "wait for D3D12 fence");
  }
  check(d3d->GetDeviceRemovedReason(), "D3D12 device execution");
}
} // namespace vdml
extern "C" int vdml_device_status(vdml::Device *device, char **error) noexcept {
  return vdml::guarded(error, [&] {
    vdml::require(device, "null DirectML device");
    vdml::check(device->d3d->GetDeviceRemovedReason(),
                "DirectML device health");
  });
}
extern "C" int vdml_adapters_enumerate(Adapters **out, char **error) noexcept {
  return vdml::guarded(error, [&] {
    auto result = std::make_unique<Adapters>();
    vdml::ComPtr<IDXGIFactory6> factory;
    vdml::check(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)),
                "create DXGI factory");
    for (uint32_t index = 0;; ++index) {
      vdml::ComPtr<IDXGIAdapter1> gpu;
      auto hr = factory->EnumAdapterByGpuPreference(
          index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&gpu));
      if (hr == DXGI_ERROR_NOT_FOUND)
        break;
      vdml::check(hr, "enumerate DirectML adapters");
      Adapter entry;
      entry.index = index;
      vdml::check(gpu->GetDesc1(&entry.desc), "get adapter description");
      entry.id = adapter_id(entry.desc);
      entry.name = utf8(entry.desc.Description);
      try {
        vdml::require(!(entry.desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE),
                      "Software adapters cannot run this DirectML backend");
        vdml::ComPtr<ID3D12Device> d3d;
        vdml::ComPtr<IDMLDevice> dml;
        vdml::check(D3D12CreateDevice(gpu.Get(), D3D_FEATURE_LEVEL_11_0,
                                      IID_PPV_ARGS(&d3d)),
                    "D3D12 feature level 11.0 unavailable");
        vdml::check(DMLCreateDevice1(d3d.Get(), DML_CREATE_DEVICE_FLAG_NONE,
                                     DML_FEATURE_LEVEL_5_0, IID_PPV_ARGS(&dml)),
                    "DirectML feature level 5.0 unavailable");
      } catch (std::exception const &e) {
        entry.reason = e.what();
      }
      result->push_back(std::move(entry));
    }
    *out = result.release();
  });
}
extern "C" size_t vdml_adapters_len(Adapters const *adapters) noexcept {
  return adapters ? adapters->size() : 0;
}
extern "C" void vdml_adapters_get(Adapters const *adapters, size_t index,
                                  vdml::AdapterDesc *out) noexcept {
  if (!adapters || index >= adapters->size() || !out)
    return;
  auto const &a = (*adapters)[index];
  *out = {
      a.id.c_str(),
      a.name.c_str(),
      a.reason.c_str(),
      a.desc.DedicatedVideoMemory,
      a.desc.SharedSystemMemory,
      a.index,
      a.desc.VendorId,
      static_cast<uint32_t>(a.reason.empty()),
      static_cast<uint32_t>((a.desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0)};
}
extern "C" void vdml_adapters_free(Adapters *adapters) noexcept {
  delete adapters;
}

extern "C" int vdml_device_create(uint32_t adapter, char const *selected_id,
                                  uint32_t debug, vdml::Device **out,
                                  char **error) noexcept {
  return vdml::guarded(error, [&] {
    auto d = std::make_unique<vdml::Device>();
    vdml::ComPtr<IDXGIFactory6> factory;
    d->debug = debug != 0;
    // EnableDebugLayer is process-global and removes devices created earlier
    // (including a GUI/WebView's devices). debug selects DirectML validation;
    // D3D12 validation must be configured by the host before any D3D12 device.
    vdml::check(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)),
                "create DXGI factory");
    vdml::ComPtr<IDXGIAdapter1> gpu;
    if (selected_id && selected_id[0]) {
      for (uint32_t index = 0;; ++index) {
        auto hr = factory->EnumAdapterByGpuPreference(
            index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE, IID_PPV_ARGS(&gpu));
        if (hr == DXGI_ERROR_NOT_FOUND) {
          throw std::runtime_error(
              "Selected DirectML device is no longer available; refresh "
              "devices and select a GPU");
        }
        vdml::check(hr, "resolve selected DirectML adapter");
        DXGI_ADAPTER_DESC1 candidate{};
        vdml::check(gpu->GetDesc1(&candidate), "get adapter description");
        if (_stricmp(adapter_id(candidate).c_str(), selected_id) == 0)
          break;
        gpu.Reset();
      }
    } else {
      vdml::check(factory->EnumAdapterByGpuPreference(
                      adapter, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
                      IID_PPV_ARGS(&gpu)),
                  "select hardware adapter");
    }
    DXGI_ADAPTER_DESC1 desc{};
    vdml::check(gpu->GetDesc1(&desc), "get adapter description");
    d->adapter_name = utf8(desc.Description);
    vdml::require(!(desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE),
                  "software DirectML adapters are unsupported");
    vdml::check(D3D12CreateDevice(gpu.Get(), D3D_FEATURE_LEVEL_11_0,
                                  IID_PPV_ARGS(&d->d3d)),
                "create D3D12 device");
    vdml::check(DMLCreateDevice1(d->d3d.Get(),
                                 debug ? DML_CREATE_DEVICE_FLAG_DEBUG
                                       : DML_CREATE_DEVICE_FLAG_NONE,
                                 DML_FEATURE_LEVEL_5_0, IID_PPV_ARGS(&d->dml)),
                "create DirectML device");
    D3D12_COMMAND_QUEUE_DESC q{};
    q.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    vdml::check(d->d3d->CreateCommandQueue(&q, IID_PPV_ARGS(&d->queue)),
                "create DirectML queue");
    vdml::check(
        d->d3d->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&d->fence)),
        "create fence");
    d->event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    vdml::require(d->event, "create fence event");
    vdml::check(d->dml->CreateCommandRecorder(IID_PPV_ARGS(&d->recorder)),
                "create DirectML command recorder");
    *out = d.release();
  });
}
extern "C" void vdml_device_free(vdml::Device *d) noexcept { delete d; }
extern "C" char const *vdml_device_name(vdml::Device const *d) noexcept {
  return d ? d->adapter_name.c_str() : nullptr;
}
