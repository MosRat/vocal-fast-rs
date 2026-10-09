#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <immintrin.h>
// Baseline dispatcher checks AVX2 and OS vector state before calling this TU.
// One pass: residual add + leaky activation, with an unaligned scalar tail.
extern "C" void vocal_cpu_add_leaky_avx2(const float *a, const float *b,
                                         float *y, size_t n, float alpha) {
  const __m256 zero = _mm256_setzero_ps(), slope = _mm256_set1_ps(alpha);
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    const __m256 x =
        _mm256_add_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    _mm256_storeu_ps(y + i,
                     _mm256_blendv_ps(x, _mm256_mul_ps(x, slope),
                                      _mm256_cmp_ps(x, zero, _CMP_LT_OQ)));
  }
  for (; i < n; ++i) {
    float x = a[i] + b[i];
    y[i] = x < 0 ? x * alpha : x;
  }
}

extern "C" void vocal_cpu_element_avx2(uint32_t op, const float *a, size_t as,
                                       const float *b, size_t bs, float *y,
                                       size_t n, float alpha) {
  const __m256 zero = _mm256_setzero_ps(), slope = _mm256_set1_ps(alpha);
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    auto x = as ? _mm256_loadu_ps(a + i) : _mm256_set1_ps(*a);
    auto z = bs ? _mm256_loadu_ps(b + i) : _mm256_set1_ps(*b);
    switch (op) {
    case 5:
      x = _mm256_add_ps(x, z);
      break;
    case 6:
      x = _mm256_sub_ps(x, z);
      break;
    case 7:
      x = _mm256_mul_ps(x, z);
      break;
    case 8:
      x = _mm256_div_ps(x, z);
      break;
    case 10:
      x = _mm256_max_ps(x, z);
      break;
    case 11:
      x = _mm256_min_ps(x, z);
      break;
    case 12:
      x = _mm256_max_ps(x, zero);
      break;
    case 52:
      x = _mm256_add_ps(x, z);
      [[fallthrough]];
    case 13:
      x = _mm256_blendv_ps(x, _mm256_mul_ps(x, slope),
                           _mm256_cmp_ps(x, zero, _CMP_LT_OQ));
      break;
    }
    _mm256_storeu_ps(y + i, x);
  }
  for (; i < n; ++i) {
    float x = a[i * as], z = b[i * bs];
    switch (op) {
    case 5:
      x += z;
      break;
    case 6:
      x -= z;
      break;
    case 7:
      x *= z;
      break;
    case 8:
      x /= z;
      break;
    case 10:
      x = std::max(x, z);
      break;
    case 11:
      x = std::min(x, z);
      break;
    case 12:
      x = std::max(x, 0.f);
      break;
    case 52:
      x += z;
      [[fallthrough]];
    case 13:
      x = x < 0 ? x * alpha : x;
      break;
    }
    y[i] = x;
  }
}
