#include "internal.h"
#include <cmath>

namespace vocal::cpu {
extern "C" void vocal_cpu_channel_norm_avx2(
    const float *, size_t, float *, size_t, size_t, size_t,
    const float *, const float *, float);

// One group per channel, statistics along time. Tiling over the contiguous
// channel dimension avoids rereading a cache line for every separate channel.
// Match the decomposed graph: double reductions, FP32 centered squares,
// variance/epsilon/sqrt and division, then separate affine multiply/add.
void channel_temporal_norm(Graph &g, size_t id) {
  auto &n = g.nodes[id];
  auto &x = g.tensors[n.inputs[0]];
  auto &gamma = g.tensors[n.inputs[1]];
  auto &beta = g.tensors[n.inputs[2]];
  auto &y = g.tensors[id];
  require(x.shape.size() == 3 && x.shape[0] == 1 &&
              n.attrs.size() == 2 && n.attrs[0] == 2 && n.attrs[1] == 1 && x.dtype == 1 &&
              y.dtype == 1 && gamma.dtype == 1 && beta.dtype == 1 &&
              gamma.elements() == static_cast<size_t>(x.shape[1]) &&
              beta.elements() == gamma.elements(),
          "invalid channel temporal normalization");
  size_t channels = x.shape[1], time = x.shape[2];
  bool tiled = x.strides[1] == 1 && y.strides[1] == 1 &&
               gamma.contiguous() && beta.contiguous();
#if defined(VOCAL_CPU_AVX2)
  if (g.avx2 && tiled) {
    g.pool->range((channels + 63) / 64, 1, [&](size_t begin, size_t end) {
      for (size_t tile = begin; tile < end; ++tile) {
        size_t first = tile * 64, count = std::min(size_t(64), channels - first);
        vocal_cpu_channel_norm_avx2(
            static_cast<const float *>(x.data()) + first, x.strides[2],
            static_cast<float *>(y.data()) + first, y.strides[2], time, count,
            static_cast<const float *>(gamma.data()) + first,
            static_cast<const float *>(beta.data()) + first, n.values[0]);
      }
    });
    return;
  }
#endif
  g.pool->range(channels, 16, [&](size_t begin, size_t end) {
    for (size_t c = begin; c < end; ++c) {
      size_t ix = c * x.strides[1], iy = c * y.strides[1];
      double sum = 0, squares = 0;
      for (size_t t = 0; t < time; ++t)
        sum += x.read(ix + t * x.strides[2]);
      float mean = static_cast<float>(sum / time);
      for (size_t t = 0; t < time; ++t) {
        float d = static_cast<float>(x.read(ix + t * x.strides[2])) - mean;
        squares += d * d;
      }
      float variance = static_cast<float>(squares / time);
      float denominator = std::sqrt(variance + n.values[0]);
      float scale = static_cast<float>(gamma.read(gamma.position(c)));
      float bias = static_cast<float>(beta.read(beta.position(c)));
      for (size_t t = 0; t < time; ++t) {
        float value = (static_cast<float>(x.read(ix + t * x.strides[2])) - mean) / denominator;
        value = value * scale + bias;
        y.write(iy + t * y.strides[2], value);
      }
    }
  });
}
} // namespace vocal::cpu
