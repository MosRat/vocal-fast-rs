#include "attention_simd.h"
#include <immintrin.h>

static float sum(__m256 x) {
  __m128 a = _mm_add_ps(_mm256_castps256_ps128(x), _mm256_extractf128_ps(x, 1));
  a = _mm_hadd_ps(a, a);
  return _mm_cvtss_f32(_mm_hadd_ps(a, a));
}

// Four keys reuse each query vector. Feature dimensions and sequence tails may
// be unaligned; the caller guarantees only the final feature stride is one.
extern "C" void vocal_cpu_attention_dot_avx2(
    const float *q, const float *k, float *scores, size_t queries, size_t keys,
    size_t width, size_t qs, size_t ks, size_t ss, float scale) {
  for (size_t i = 0; i < queries; ++i) {
    const float *row = q + i * qs;
    size_t j = 0;
    for (; j + 4 <= keys; j += 4) {
      __m256 a = _mm256_setzero_ps(), b = a, c = a, d = a;
      size_t f = 0;
      for (; f + 8 <= width; f += 8) {
        auto x = _mm256_loadu_ps(row + f);
        a = _mm256_fmadd_ps(x, _mm256_loadu_ps(k + j * ks + f), a);
        b = _mm256_fmadd_ps(x, _mm256_loadu_ps(k + (j + 1) * ks + f), b);
        c = _mm256_fmadd_ps(x, _mm256_loadu_ps(k + (j + 2) * ks + f), c);
        d = _mm256_fmadd_ps(x, _mm256_loadu_ps(k + (j + 3) * ks + f), d);
      }
      float values[4] = {sum(a), sum(b), sum(c), sum(d)};
      for (; f < width; ++f)
        for (size_t t = 0; t < 4; ++t)
          values[t] += row[f] * k[(j + t) * ks + f];
      for (size_t t = 0; t < 4; ++t)
        scores[i * ss + j + t] = values[t] * scale;
    }
    for (; j < keys; ++j) {
      auto a = _mm256_setzero_ps();
      size_t f = 0;
      for (; f + 8 <= width; f += 8)
        a = _mm256_fmadd_ps(_mm256_loadu_ps(row + f),
                           _mm256_loadu_ps(k + j * ks + f), a);
      float value = sum(a);
      for (; f < width; ++f)
        value += row[f] * k[j * ks + f];
      scores[i * ss + j] = value * scale;
    }
  }
}

extern "C" void vocal_cpu_attention_update_avx2(
    const float *p, const float *v, float *numerator, size_t keys, size_t width,
    size_t vs, float decay) {
  size_t f = 0;
  const auto old_scale = _mm256_set1_ps(decay);
  for (; f + 8 <= width; f += 8) {
    auto a = _mm256_mul_ps(_mm256_loadu_ps(numerator + f), old_scale);
    for (size_t j = 0; j < keys; ++j)
      a = _mm256_fmadd_ps(_mm256_set1_ps(p[j]), _mm256_loadu_ps(v + j * vs + f), a);
    _mm256_storeu_ps(numerator + f, a);
  }
  for (; f < width; ++f) {
    float a = numerator[f] * decay;
    for (size_t j = 0; j < keys; ++j)
      a += p[j] * v[j * vs + f];
    numerator[f] = a;
  }
}
