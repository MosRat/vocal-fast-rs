#include "internal.h"
#include <cmath>
#include <unordered_map>
namespace vocal::cpu {
extern "C" void *vocal_cpu_fft_create();
extern "C" int vocal_cpu_fft_run(void *, const float *, size_t, size_t, size_t,
                                 float *);
extern "C" void vocal_cpu_fft_destroy(void *);
struct FftContext {
  void *ptr = vocal_cpu_fft_create();
  FftContext() { require(ptr != nullptr, "FFT initialization failed"); }
  ~FftContext() { vocal_cpu_fft_destroy(ptr); }
};
class Stft final : public Kernel {
  uint32_t source, destination;
  size_t samples, frames;
  std::vector<std::unique_ptr<FftContext>> contexts;
  dnnl::memory magnitude, scratch;
  dnnl::primitive mel;
  std::unordered_map<int, dnnl::memory> args;

public:
  Stft(Graph &g, size_t id)
      : source(g.nodes[id].inputs[0]), destination(static_cast<uint32_t>(id)) {
    using namespace dnnl;
    auto &wave = g.tensors[source];
    auto &basis = g.tensors[g.nodes[id].inputs[1]];
    samples = wave.elements();
    frames = samples / 160 + 1;
    require(wave.dtype == 1 && wave.contiguous() && samples > 512,
            "invalid STFT waveform");
    require(basis.shape == Shape({128, 513}) && basis.dtype == 1,
            "invalid mel basis");
    for (int i = 0; i < g.pool->get_num_threads(); ++i)
      contexts.push_back(std::make_unique<FftContext>());
    auto md = memory::desc({513, static_cast<int64_t>(frames)},
                           memory::data_type::f32, memory::format_tag::ab);
    magnitude = memory(md, g.engine);
    auxiliary_bytes = md.get_size();
    auto output = memory::desc({128, static_cast<int64_t>(frames)},
                               memory::data_type::f32, memory::format_tag::ab);
    primitive_attr attr;
    attr.set_scratchpad_mode(scratchpad_mode::user);
    auto pd =
        matmul::primitive_desc(g.engine, descriptor(basis), md, output, attr);
    mel = matmul(pd);
    scratch = g.workspace(0, pd.scratchpad_desc());
    args = {{DNNL_ARG_SRC, memory(descriptor(basis), g.engine, basis.data())},
            {DNNL_ARG_WEIGHTS, magnitude},
            {DNNL_ARG_DST, memory(output, g.engine, g.tensors[id].data())},
            {DNNL_ARG_SCRATCHPAD, scratch}};
  }
  void run(Graph &g) override {
    int jobs = static_cast<int>(std::min(frames, contexts.size()));
    g.pool->parallel_for(jobs, [&](int i, int n) {
      require(vocal_cpu_fft_run(
                  contexts[i]->ptr,
                  static_cast<float *>(g.tensors[source].data()), samples,
                  frames * i / n, frames * (i + 1) / n,
                  static_cast<float *>(magnitude.get_data_handle())) == 0,
              "FFT execution failed");
    });
    mel.execute(g.stream, args);
    auto &y = g.tensors[destination];
    auto *data = static_cast<float *>(y.data());
    g.pool->range(y.elements(), 32768, [&](size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i)
        data[i] = std::log(std::max(data[i], 1e-5f));
    });
  }
};
std::unique_ptr<Kernel> prepare_stft(Graph &g, size_t id) {
  return std::make_unique<Stft>(g, id);
}
} // namespace vocal::cpu
