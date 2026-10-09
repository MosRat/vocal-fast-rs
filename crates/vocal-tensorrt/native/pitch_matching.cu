#include "bridge.h"
#include <cmath>

namespace {
// Calibration is rare. One block fills the contour, sorts in shared memory,
// and scales a separate output. Locked callbacks skip interpolation and sort.
__global__ void match_pitch(float const* raw, float* scaled, float const* controls,
                            float* state, float* scratch, int frames, int minimum,
                            bool streaming, int sort_size) {
    extern __shared__ float sorted[];
    __shared__ int estimate;
    __shared__ float factor;
    if (threadIdx.x == 0) {
        estimate = controls[0] > 0.0f && (!streaming || state[0] == 0.0f);
        if (estimate) {
            int first = -1, left = -1, voiced = 0;
            float* contour = sort_size ? sorted : scratch;
            for (int i = 0; i < frames; ++i) {
                if (!(raw[i] > 0.0f) || !isfinite(raw[i])) continue;
                ++voiced;
                if (first < 0) {
                    first = i;
                    for (int j = 0; j <= i; ++j) contour[j] = raw[i];
                } else {
                    float const step = (raw[i] - raw[left]) / (i - left);
                    for (int j = left; j < i; ++j) contour[j] = raw[left] + step * (j - left);
                }
                left = i;
            }
            if (voiced < minimum) {
                estimate = 0;
                if (!streaming) { state[0] = 0; state[1] = 0; }
            } else {
                for (int i = left; i < frames; ++i) contour[i] = raw[left];
            }
        }
    }
    __syncthreads();
    if (estimate && sort_size) {
        for (int i = frames + threadIdx.x; i < sort_size; i += blockDim.x) sorted[i] = INFINITY;
        __syncthreads();
        for (int k = 2; k <= sort_size; k <<= 1) {
            for (int j = k >> 1; j > 0; j >>= 1) {
                for (int i = threadIdx.x; i < sort_size; i += blockDim.x) {
                    int other = i ^ j;
                    if (other > i && ((sorted[i] > sorted[other]) == ((i & k) == 0))) {
                        float tmp = sorted[i]; sorted[i] = sorted[other]; sorted[other] = tmp;
                    }
                }
                __syncthreads();
            }
        }
    }
    if (threadIdx.x == 0) {
        if (estimate) {
            float* contour = sort_size ? sorted : scratch;
            if (!sort_size) {
                // Large custom offline profiles use an in-place selection,
                // avoiding a hard frame limit or a new device allocation.
                int lo = 0, hi = frames - 1, mid = frames / 2;
                while (lo < hi) {
                    float pivot = contour[(lo + hi) / 2];
                    int a = lo, b = hi;
                    while (a <= b) {
                        while (contour[a] < pivot) ++a;
                        while (contour[b] > pivot) --b;
                        if (a <= b) { float t = contour[a]; contour[a++] = contour[b]; contour[b--] = t; }
                    }
                    if (mid <= b) hi = b;
                    else if (mid >= a) lo = a;
                    else break;
                }
            }
            float median = contour[frames / 2];
            if (frames % 2 == 0) {
                float lower = contour[0];
                for (int i = 1; i < frames / 2; ++i) lower = fmaxf(lower, contour[i]);
                median = (median + lower) * 0.5f;
            }
            state[1] = fminf(12.0f, fmaxf(-12.0f, nearbyintf(12.0f * log2f(controls[1] / median))));
            state[0] = 1.0f;
        }
        factor = exp2f((controls[2] + (controls[0] > 0.0f ? state[1] : 0.0f)) / 12.0f);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < frames; i += blockDim.x) scaled[i] = raw[i] * factor;
}
}

extern "C" int vocal_pitch_match(float const* raw, float* scaled, float const* controls,
    float* state, float* scratch, int frames, int minimum, int streaming,
    vocal::Stream* stream, char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(raw && scaled && controls && state && scratch && stream,
                       "null pitch matching binding");
        vocal::require(frames > 0 && minimum >= 2, "invalid pitch matching geometry");
        int size = 1;
        while (size < frames && size < 8192) size <<= 1;
        if (size < frames) size = 0;
        match_pitch<<<1, 256, size * sizeof(float), stream->stream>>>(
            raw, scaled, controls, state, scratch, frames, minimum, streaming != 0, size);
        vocal::cuda_check(cudaGetLastError());
    });
}
