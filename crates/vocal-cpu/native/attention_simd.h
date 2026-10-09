#pragma once
#include <cstddef>
extern "C" void vocal_cpu_attention_dot_avx2(
    const float *q, const float *k, float *scores, size_t queries, size_t keys,
    size_t width, size_t query_stride, size_t key_stride, size_t score_stride,
    float scale);
extern "C" void vocal_cpu_attention_update_avx2(
    const float *p, const float *v, float *numerator, size_t keys, size_t width,
    size_t value_stride, float decay);
