#include <cmath>
#include <cstddef>
#include <immintrin.h>

// Caller checks AVX2 and passes at most 64 contiguous channels. Scalar tails
// use the same FP64 reduction and FP32 pointwise rounding as the vector path.
extern "C" void vocal_cpu_channel_norm_avx2(
    const float *x, size_t xs, float *y, size_t ys, size_t time, size_t channels,
    const float *gamma, const float *beta, float epsilon) {
  alignas(32) double sums[64] = {}, squares[64] = {};
  alignas(32) float means[64], denominators[64];
  for (size_t t = 0; t < time; ++t) {
    size_t c = 0;
    for (; c + 8 <= channels; c += 8) {
      auto v = _mm256_loadu_ps(x + t * xs + c);
      auto lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));
      auto hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
      _mm256_store_pd(sums + c, _mm256_add_pd(_mm256_load_pd(sums + c), lo));
      _mm256_store_pd(sums + c + 4, _mm256_add_pd(_mm256_load_pd(sums + c + 4), hi));
    }
    for (; c < channels; ++c)
      sums[c] += x[t * xs + c];
  }
  for (size_t c = 0; c < channels; ++c)
    means[c] = static_cast<float>(sums[c] / time);
  for (size_t t = 0; t < time; ++t) {
    size_t c = 0;
    for (; c + 8 <= channels; c += 8) {
      auto d = _mm256_sub_ps(_mm256_loadu_ps(x + t * xs + c), _mm256_load_ps(means + c));
      auto v = _mm256_mul_ps(d, d);
      auto lo = _mm256_cvtps_pd(_mm256_castps256_ps128(v));
      auto hi = _mm256_cvtps_pd(_mm256_extractf128_ps(v, 1));
      _mm256_store_pd(squares + c, _mm256_add_pd(_mm256_load_pd(squares + c), lo));
      _mm256_store_pd(squares + c + 4, _mm256_add_pd(_mm256_load_pd(squares + c + 4), hi));
    }
    for (; c < channels; ++c) {
      float d = x[t * xs + c] - means[c];
      squares[c] += d * d;
    }
  }
  for (size_t c = 0; c < channels; ++c)
    denominators[c] = std::sqrt(static_cast<float>(squares[c] / time) + epsilon);
  for (size_t t = 0; t < time; ++t) {
    size_t c = 0;
    for (; c + 8 <= channels; c += 8) {
      auto v = _mm256_sub_ps(_mm256_loadu_ps(x + t * xs + c), _mm256_load_ps(means + c));
      v = _mm256_div_ps(v, _mm256_load_ps(denominators + c));
      v = _mm256_mul_ps(v, _mm256_loadu_ps(gamma + c));
      v = _mm256_add_ps(v, _mm256_loadu_ps(beta + c));
      _mm256_storeu_ps(y + t * ys + c, v);
    }
    for (; c < channels; ++c) {
      float v = (x[t * xs + c] - means[c]) / denominators[c];
      y[t * ys + c] = v * gamma[c] + beta[c];
    }
  }
}
