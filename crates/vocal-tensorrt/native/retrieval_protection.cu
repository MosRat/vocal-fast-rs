#include "bridge.h"

namespace {
__global__ void protect_features(float* features, float const* raw,
                                 float const* pitchf, float const* ratio, float const* protect,
                                 int frames, int hubert_frames, int width) {
    int const index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= frames * width || *ratio <= 0.0f || *protect >= 0.5f) return;
    int const frame = index / width;
    if (pitchf[frame] > 0.0f) return;
    int const row = min(frame / 2, hubert_frames - 1);
    int const channel = index % width;
    // RVC protects unvoiced content after 2x interpolation. Doing this before
    // interpolation would incorrectly share the mask between adjacent F0 hops.
    features[index] = features[index] * *protect + raw[row * width + channel] * (1.0f - *protect);
}
}

extern "C" int vocal_retrieval_protect(
    float* features, float const* raw, float const* pitchf, float const* ratio, float const* protect,
    int frames, int hubert_frames, int width, vocal::Stream* stream,
    char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(features && raw && pitchf && ratio && protect && stream,
                       "null retrieval protection binding");
        vocal::require(frames > 0 && hubert_frames > 0 && width > 0 &&
                       frames <= INT32_MAX / width,
                       "invalid retrieval protection dimensions");
        int const elements = frames * width;
        protect_features<<<(elements - 1) / 256 + 1, 256, 0, stream->stream>>>(
            features, raw, pitchf, ratio, protect, frames, hubert_frames, width);
        vocal::cuda_check(cudaGetLastError());
    });
}
