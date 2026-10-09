#pragma once
#include <cstdint>
#include <pthreadpool.h>
void nchw_to_nhwc(const float *, float *, const int64_t *, pthreadpool_t);
void nhwc_to_nchw(const float *, float *, const int64_t *, pthreadpool_t);
