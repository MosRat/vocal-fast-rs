#include "bridge.h"

#include <cufft.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vocal {

namespace {

constexpr int kNfft = 1024;
constexpr int kHop = 160;
constexpr int kBins = kNfft / 2 + 1;
constexpr double kPi = 3.14159265358979323846;
__device__ __forceinline__ int reflect_index(int position, int samples) {
    // RMVPE uses a 512-sample pad and every valid window is at least 4960
    // samples.  The source can therefore cross each boundary only once;
    // keeping the two branches explicit avoids a loop and improves the
    // gather kernel's instruction path without changing PyTorch reflect
    // padding (the endpoint itself is excluded).
    if (position < 0) return -position;
    if (position >= samples) return 2 * samples - 2 - position;
    return position;
}

__global__ void gather_window(
    float const* waveform, float const* window, float* frames,
    int samples, int frame_count) {
    auto linear = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    auto total = frame_count * kNfft;
    if (linear >= total) return;
    int frame = linear / kNfft;
    int offset = linear - frame * kNfft;
    int source = frame * kHop + offset - kNfft / 2;
    source = reflect_index(source, samples);
    frames[linear] = waveform[source] * window[offset];
}

__global__ void magnitude_kernel(
    cufftComplex const* spectrum, float* magnitude, int frame_count) {
    auto linear = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    auto total = frame_count * kBins;
    if (linear >= total) return;
    // cuFFT stores one contiguous [frame, bin] row per batch.  Emit the
    // transposed [bin, frame] matrix expected by the mel GEMM directly, so
    // TensorRT does not need a separate global-memory transpose.
    int bin = linear / frame_count;
    int frame = linear - bin * frame_count;
    auto value = spectrum[frame * kBins + bin];
    magnitude[bin * frame_count + frame] = sqrtf(value.x * value.x + value.y * value.y);
}

inline void cufft_check(cufftResult status, char const* operation) {
    if (status != CUFFT_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed (cuFFT status "
                                 + std::to_string(static_cast<int>(status)) + ")");
    }
}

struct CufftFrontendImpl {
    int samples{};
    int frames{};
    cufftHandle plan{};
    float* frame_buffer{};
    cufftComplex* spectrum{};
    float* hann_window{};
    void* workspace{};
    size_t workspace_bytes{};
    cudaStream_t stream{};

    ~CufftFrontendImpl() {
        if (plan) cufftDestroy(plan);
        if (workspace) cudaFree(workspace);
        if (hann_window) cudaFree(hann_window);
        if (spectrum) cudaFree(spectrum);
        if (frame_buffer) cudaFree(frame_buffer);
    }
};

} // namespace

// Keep the ABI name in bridge.h opaque while using a concrete implementation
// in this translation unit. This avoids exposing cuFFT headers to Rust or to
// the TensorRT graph builders.
struct CufftFrontend : CufftFrontendImpl {};

extern "C" int vocal_cufft_frontend_create(
    int32_t samples, CufftFrontend** out, char** error) noexcept {
    return guarded(error, [&] {
        require(out && samples >= kNfft,
                "cuFFT frontend samples must be >= 1024");
        *out = nullptr;
        auto frontend = std::make_unique<CufftFrontend>();
        frontend->samples = samples;
        frontend->frames = samples / kHop + 1;
        int lengths[1] = {kNfft};
        // Disable cuFFT's implicit workspace allocation.  Building the plan,
        // allocating its workspace, and attaching it before the first enqueue
        // removes the first-use allocation from both profiling and graph
        // capture.
        cufft_check(cufftCreate(&frontend->plan), "cufftCreate");
        cufft_check(cufftSetAutoAllocation(frontend->plan, 0),
                    "cufftSetAutoAllocation");
        cufft_check(cufftMakePlanMany(
                        frontend->plan, 1, lengths,
                        nullptr, 1, kNfft,
                        nullptr, 1, kBins,
                        CUFFT_R2C, frontend->frames,
                        &frontend->workspace_bytes),
                    "cufftMakePlanMany");
        if (frontend->workspace_bytes != 0) {
            cuda_check(cudaMalloc(&frontend->workspace, frontend->workspace_bytes));
            cufft_check(cufftSetWorkArea(frontend->plan, frontend->workspace),
                        "cufftSetWorkArea");
        }
        cuda_check(cudaMalloc(
            reinterpret_cast<void**>(&frontend->frame_buffer),
            static_cast<size_t>(frontend->frames) * kNfft * sizeof(float)));
        cuda_check(cudaMalloc(
            reinterpret_cast<void**>(&frontend->spectrum),
            static_cast<size_t>(frontend->frames) * kBins * sizeof(cufftComplex)));
        cuda_check(cudaMalloc(
            reinterpret_cast<void**>(&frontend->hann_window),
            static_cast<size_t>(kNfft) * sizeof(float)));
        std::vector<float> hann(kNfft);
        for (int i = 0; i < kNfft; ++i) {
            hann[static_cast<size_t>(i)] =
                0.5f - 0.5f * static_cast<float>(
                    std::cos(2.0 * kPi * static_cast<double>(i) / kNfft));
        }
        cuda_check(cudaMemcpy(frontend->hann_window, hann.data(),
                              static_cast<size_t>(kNfft) * sizeof(float),
                              cudaMemcpyHostToDevice));
        *out = frontend.release();
    });
}

extern "C" int vocal_cufft_frontend_set_stream(
    CufftFrontend* frontend, Stream* stream, char** error) noexcept {
    return guarded(error, [&] {
        require(frontend && stream && stream->stream,
                "invalid cuFFT frontend stream");
        frontend->stream = stream->stream;
        cufft_check(cufftSetStream(frontend->plan, frontend->stream),
                    "cufftSetStream");
    });
}

extern "C" int vocal_cufft_frontend_enqueue(
    CufftFrontend* frontend, void const* waveform, size_t waveform_bytes,
    Memory* magnitude, Stream* stream, char** error) noexcept {
    return guarded(error, [&] {
        require(frontend && waveform && magnitude && stream && stream->stream,
                "invalid cuFFT frontend enqueue arguments");
        require(waveform_bytes >= static_cast<size_t>(frontend->samples) * sizeof(float),
                "cuFFT frontend waveform buffer is too small");
        require(magnitude->bytes >= static_cast<size_t>(frontend->frames) * kBins * sizeof(float),
                "cuFFT frontend magnitude buffer is too small");
        require(frontend->stream == stream->stream,
                "cuFFT frontend was initialized for a different CUDA stream");
        auto const* input = static_cast<float const*>(waveform);
        auto blocks = (frontend->frames * kNfft + 255) / 256;
        gather_window<<<blocks, 256, 0, frontend->stream>>>(
            input, frontend->hann_window, frontend->frame_buffer,
            frontend->samples, frontend->frames);
        cuda_check(cudaGetLastError());
        cufft_check(cufftExecR2C(frontend->plan, frontend->frame_buffer,
                                 frontend->spectrum),
                    "cufftExecR2C");
        blocks = (frontend->frames * kBins + 255) / 256;
        magnitude_kernel<<<blocks, 256, 0, frontend->stream>>>(
            frontend->spectrum, static_cast<float*>(magnitude->ptr), frontend->frames);
        cuda_check(cudaGetLastError());
    });
}

extern "C" void vocal_cufft_frontend_destroy(CufftFrontend* frontend) noexcept {
    delete frontend;
}

} // namespace vocal
