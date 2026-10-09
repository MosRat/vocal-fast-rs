#include "bridge.h"
namespace vdml {
void Graph::record() {
  list->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0);
  record_upload();
  ID3D12DescriptorHeap *heaps[]{descriptors.Get()};
  list->SetDescriptorHeaps(1, heaps);
  list->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 1);
  device->recorder->RecordDispatch(list.Get(), compiled.Get(), bindings.Get());
  uav_barrier(list.Get());
  record_feedback();
  list->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 2);
  record_readback();
  list->EndQuery(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 3);
  list->ResolveQueryData(queries.Get(), D3D12_QUERY_TYPE_TIMESTAMP, 0, 4,
                         timestamps.Get(), 0);
}
void Graph::run(Binding const *in, size_t ni, Binding const *out, size_t no,
                GpuTimings *timings) {
  size_t index = 0;
  for (auto const &s : inputs)
    if (!s.owned && !s.state) {
      require(index < ni && in[index].data && in[index].bytes <= s.bytes,
              "invalid graph input binding");
      if (in[index].bytes < s.bytes)
        std::memset(upload_ptr + s.offset + in[index].bytes, 0,
                    static_cast<size_t>(s.bytes - in[index].bytes));
      std::memcpy(upload_ptr + s.offset, in[index].data, in[index].bytes);
      ++index;
    }
  auto host_outputs = std::count_if(outputs.begin(), outputs.end(),
                                    [](Slot const &s) { return s.readback; });
  require(index == ni && no == static_cast<size_t>(host_outputs),
          "graph binding count mismatch");
  index = 0;
  for (auto const &s : outputs) {
    if (!s.readback)
      continue;
    require(out[index].data && out[index].bytes <= s.bytes,
            "invalid graph output binding");
    ++index;
  }
  device->execute(list.Get());
  index = 0;
  for (auto const &s : outputs) {
    if (!s.readback)
      continue;
    std::memcpy(out[index].data, readback_ptr + s.offset, out[index].bytes);
    ++index;
  }
  auto elapsed = [&](size_t a, size_t b) {
    return static_cast<double>(timestamp_ptr[b] - timestamp_ptr[a]) * 1000. /
           frequency;
  };
  *timings = {elapsed(0, 1), elapsed(1, 2), elapsed(2, 3)};
}
} // namespace vdml
extern "C" void vdml_graph_free(vdml::Graph *g) noexcept { delete g; }
extern "C" void vdml_error_free(char *e) noexcept { std::free(e); }
extern "C" int vdml_run(vdml::Graph *g, vdml::Binding const *in, size_t ni,
                        vdml::Binding const *out, size_t no,
                        vdml::GpuTimings *timings, char **error) noexcept {
  return vdml::guarded(error, [&] {
    vdml::require(g && timings, "null DirectML graph or timings");
    g->run(in, ni, out, no, timings);
  });
}
