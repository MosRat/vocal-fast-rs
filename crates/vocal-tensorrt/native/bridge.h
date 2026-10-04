#pragma once
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <map>
#include <cstdint>
#include <cmath>

namespace vocal {
using LogCallback = void (*)(int severity, char const* message);
struct Logger final : nvinfer1::ILogger {
    void log(Severity severity, char const* message) noexcept override;
};
Logger& logger();
void log_message(nvinfer1::ILogger::Severity severity, std::string const& message) noexcept;
void set_log_callback(LogCallback callback) noexcept;
char* make_error(char const* message) noexcept;
extern "C" void vocal_trt_free_error(char* message);
inline void require(bool ok, std::string const& message) { if (!ok) throw std::runtime_error(message); }
inline void cuda_check(cudaError_t rc) { require(rc == cudaSuccess, cudaGetErrorString(rc)); }
template<class F> int guarded(char** error, F&& f) noexcept {
    try { f(); return 0; }
    catch (std::exception const& e) {
        if (error) *error=make_error(e.what());
        return 1;
    } catch (...) { return 2; }
}
// Keep this ABI object layout identical to the legacy shim's opaque Build
// handle.  The graph builder owns the host memory until vocal_trt_build_free.
struct Build { nvinfer1::IHostMemory* mem{}; };
struct Runtime { nvinfer1::IRuntime* runtime{}; nvinfer1::ICudaEngine* engine{}; };
struct LayerProfiler final : nvinfer1::IProfiler {
    std::map<std::string, double> times;
    void reportLayerTime(char const* name, float ms) noexcept override {
        try { times[name] += ms; } catch (...) {}
    }
};
struct Context {
    nvinfer1::IExecutionContext* context{};
    // Runtime configuration must outlive the execution context.  Keeping it
    // in the ABI handle makes the TensorRT-RTX CUDA graph strategy an
    // explicit, owned part of context lifetime rather than a process-global
    // switch.
    nvinfer1::IRuntimeConfig* runtime_config{};
    LayerProfiler profiler;
};
struct Stream { cudaStream_t stream{}; };
struct Memory { void* ptr{}; size_t bytes{}; };
struct HostMemory { void* ptr{}; size_t bytes{}; };
// A CUDA graph executable is owned by the Rust stream/pipeline wrapper.  The
// graph itself is instantiated by the end-capture call and is intentionally
// opaque across the C ABI.
struct CudaGraphExec { cudaGraphExec_t exec{}; };
// cuFFT-backed RMVPE STFT frontend. The plan and scratch buffers are owned by
// the native handle and are created before CUDA graph capture starts.
struct CufftFrontend;
// Versioned C data contract, independent of TensorRT C++ layouts. Rust validates
// the canonical package before passing these borrowed arrays for the build call.
// `data` is an F32 build-time mirror.  For VFRM F16/BF16 tensors `data16`
// points at the original little-endian bits and `dtype` carries the canonical
// TensorDType code (4 = F16, 6 = BF16, 5 = F32).  Keeping both views avoids a
// lossy expand/repack round trip when TensorRT accepts a low-precision
// constant directly, while preserving host-float graph transforms.
struct Weight {
    char const* name;
    float const* data;
    uint16_t const* data16;
    int32_t dtype;
    int64_t count;
    int32_t rank;
    int64_t dims[4];
    float scalar(int64_t index) const noexcept;
};
struct Config {
    int32_t abi_version, feature_width, uses_f0, sample_rate, hidden, latent, filter, heads;
    int32_t encoder_layers, encoder_kernel, window, relative_heads, speakers, speaker_channels;
    int32_t decoder_channels, resblock_type, upsample_count, resblock_count, flow_blocks, flow_layers, flow_kernel;
    float norm_epsilon;
    int32_t rates[8], kernels[8], res_kernels[8], res_dilations[24], flow_dilations[16];
};
struct Options {
    uint64_t workspace;
    int32_t min_frames, opt_frames, max_frames, debug_outputs, fp16;
    int32_t fused_attention;
    // 0 = FP32, 1 = FP16, 2 = BF16. Acoustic graphs keep public I/O in
    // FP32 and cast to this type at the graph boundary.
    int32_t precision;
    // Experimental explicit INT8 weight-only path. Activations remain in the
    // selected floating-point type; the graph uses per-output-channel DQ
    // weights so TensorRT can choose weight-only INT8 tactics when available.
    int32_t quantize_weights;
    // Fuse the three Q/K/V projections into one GEMM/Conv1d. This is a
    // shape-preserving optimization and can be disabled for A/B builds.
    int32_t fused_qkv;
    // Stable dispatch policy ABI. TensorRT owns the actual tactic choice;
    // values 1/2 reserve the measured 11200/44960 realtime shape policies.
    int32_t dispatch_tune;
    // Synth residual block precision: 0 keeps the accuracy-safe FP32 path;
    // 1/2 selects the graph's FP16/BF16 decoder type and removes per-stage
    // casts. This is deliberately separate from the decoder precision.
    int32_t synth_resblock_precision;
    // Bit mask selecting low-precision residual stages; bit 0 is stage 0.
    int32_t synth_resblock_stage_mask;
    // Optional per-operation residual precision. A zero mask inherits the
    // stage-wide precision fields above.
    int32_t synth_resblock_conv1_precision;
    int32_t synth_resblock_conv1_stage_mask;
    int32_t synth_resblock_conv2_precision;
    int32_t synth_resblock_conv2_stage_mask;
    // Selective Synth W8A16 Conv2 mask. Bit (resblock_index * 3 + conv2
    // suffix) enables per-output-channel INT8 weights with FP16 activation.
    // The mask is intentionally separate from the legacy global weight-only
    // switch so only validated decoder hotspots are quantized.
    uint64_t synth_w8a16_conv2_mask;
    // Experimental calibrated W8A8 masks. The activation scale is currently
    // a single per-tensor value, so the builder restricts this path to one
    // selected Conv1/Conv2 node at a time.
    uint64_t synth_w8a8_conv1_mask;
    uint64_t synth_w8a8_conv2_mask;
    float synth_w8a8_activation_scale;
    // Fold residual average into the following decoder weights.
    int32_t synth_fuse_resblock_scale;
    // Replace ConvTranspose1d with polyphase Conv1d + shuffle.
    int32_t synth_polyphase_upsample;
    // Full Synth decoder FP16 experiment. Learned decoder convolutions use
    // the decoder type; residual accumulation/source remain independently
    // selectable for accuracy.
    int32_t synth_full_fp16;
    float synth_fp16_input_scale;
    int32_t synth_fp16_residual_accum;
    int32_t synth_fp16_source;
    int32_t synth_fp16_noise;
    int32_t synth_fp16_noise_stage_mask;
    int32_t synth_fp16_post;
    // RMVPE mixed-precision regions. Public I/O remains FP32. STFT, GRU, and
    // logits are fixed to FP32 by the native graph; their fields document the
    // ABI contract and are reserved for future experiments.
    int32_t rmvpe_encoder_precision;
    int32_t rmvpe_decoder_precision;
    int32_t rmvpe_stft_precision;
    int32_t rmvpe_gru_precision;
    int32_t rmvpe_logits_precision;
    // Emit only RMVPE parity tensors (kept separate from Synth debug outputs).
    int32_t rmvpe_debug_outputs;
    // 0 = compact DFT convolution (production default), 1 = experimental
    // radix-2 graph frontend with lower cancellation error.
    int32_t rmvpe_fft_mode;
    // Keep the RVC encoder Q/K/V projections separate by default. The fused
    // HuBERT path is independent; RVC fusion changes long-sequence
    // accumulation order and is enabled only for an explicit A/B experiment.
    int32_t fused_rvc_qkv;
    // HuBERT feature extractor precision is independent from the projection
    // and Transformer encoder precision.
    int32_t hubert_feature_precision;
    // 1 = RMVPE receives [1,513,T] cuFFT magnitude from the external CUDA
    // frontend. TensorRT still owns mel projection and all learned layers.
    int32_t rmvpe_external_stft;
    // TensorRT tactic-source bitset: bit 0 edge-mask convolutions, bit 1 JIT
    // convolutions. The Rust manifest records this for reproducible builds.
    int32_t tactic_sources;
    // Optional UTF-8 timing-cache path owned by the Rust build call.
    char const* timing_cache_path;
};

inline uint16_t float_to_half_bits(float value) noexcept {
    uint32_t bits{};
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000U;
    uint32_t exponent = (bits >> 23) & 0xffU;
    uint32_t mantissa = bits & 0x7fffffU;
    if (exponent == 0xffU) {
        uint16_t payload = static_cast<uint16_t>(mantissa >> 13);
        return static_cast<uint16_t>(sign | 0x7c00U | (payload ? payload : 1U));
    }
    int32_t e = static_cast<int32_t>(exponent) - 127 + 15;
    if (e >= 31) return static_cast<uint16_t>(sign | 0x7c00U);
    if (e <= 0) {
        if (e < -10) return static_cast<uint16_t>(sign);
        mantissa |= 0x800000U;
        uint32_t shift = static_cast<uint32_t>(14 - e);
        uint32_t rounded = (mantissa + (1U << (shift - 1)) - 1U + ((mantissa >> shift) & 1U)) >> shift;
        return static_cast<uint16_t>(sign | rounded);
    }
    uint32_t rounded = mantissa + 0x1000U + ((mantissa >> 13) & 1U);
    if (rounded & 0x800000U) { rounded = 0; ++e; }
    if (e >= 31) return static_cast<uint16_t>(sign | 0x7c00U);
    return static_cast<uint16_t>(sign | (static_cast<uint32_t>(e) << 10) | (rounded >> 13));
}

inline uint16_t float_to_bf16_bits(float value) noexcept {
    uint32_t bits{};
    std::memcpy(&bits, &value, sizeof(bits));
    uint32_t rounding = 0x7fffU + ((bits >> 16) & 1U);
    return static_cast<uint16_t>((bits + rounding) >> 16);
}

// Acoustic bundles use the same checked weight ABI as the RVC graph.  The
// native HuBERT builder consumes only F32 parameter tensors; integer shape
// constants remain in the VFRM IR and are rebuilt as TensorRT shape tensors.
extern "C" int vocal_trt_build_hubert(
    Weight const* weights, size_t count, Options const* options,
    Build** out, char** error) noexcept;

// Hand-expanded RMVPE graph.  The graph owns its Fourier frontend and U-Net
// layers; the only source-dependent data crossing this ABI is the normalized
// acoustic-v1 tensor table.  The GRU recurrence is emitted through TensorRT's
// ILoop API (no ONNX parser is involved).
extern "C" int vocal_trt_build_rmvpe(
    Weight const* weights, size_t count, Options const* options,
    Build** out, char** error) noexcept;

extern "C" int vocal_cufft_frontend_create(
    int32_t samples, CufftFrontend** out, char** error) noexcept;
extern "C" int vocal_cufft_frontend_set_stream(
    CufftFrontend* frontend, Stream* stream, char** error) noexcept;
extern "C" int vocal_cufft_frontend_enqueue(
    CufftFrontend* frontend, void const* waveform, size_t waveform_bytes,
    Memory* magnitude, Stream* stream, char** error) noexcept;
extern "C" void vocal_cufft_frontend_destroy(CufftFrontend* frontend) noexcept;

// Pinned host staging is used by the realtime waveform ring. The async copy
// does not block the producer and is ordered on the pipeline CUDA stream.
extern "C" int vocal_cuda_host_alloc(size_t bytes, HostMemory** out, char** error) noexcept;
extern "C" void vocal_cuda_host_free(HostMemory* memory) noexcept;
extern "C" void* vocal_cuda_host_ptr(HostMemory* memory) noexcept;
extern "C" int vocal_cuda_upload_async(
    Memory* memory, size_t destination_offset, HostMemory* source,
    size_t source_offset, size_t bytes, Stream* stream, char** error) noexcept;
extern "C" int vocal_cuda_download_async(
    Memory* memory, size_t source_offset, HostMemory* destination,
    size_t destination_offset, size_t bytes, Stream* stream, char** error) noexcept;
}

namespace vocal {
inline float half_bits_to_float(uint16_t bits) noexcept {
    uint32_t sign = (static_cast<uint32_t>(bits & 0x8000U)) << 16;
    uint32_t exponent = (bits >> 10) & 0x1fU;
    uint32_t mantissa = bits & 0x3ffU;
    uint32_t value{};
    if (exponent == 0) {
        if (mantissa == 0) {
            value = sign;
        } else {
            // Normalize the subnormal mantissa into the single-precision
            // exponent range.  This branch is build-time only and is kept
            // deliberately independent of CUDA host headers.
            int shift = 0;
            while ((mantissa & 0x400U) == 0) {
                mantissa <<= 1;
                ++shift;
            }
            mantissa &= 0x3ffU;
            uint32_t e = static_cast<uint32_t>(127 - 15 - shift + 1);
            value = sign | (e << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1fU) {
        value = sign | 0x7f800000U | (mantissa << 13);
    } else {
        value = sign | ((exponent + 112U) << 23) | (mantissa << 13);
    }
    float result{};
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

inline float Weight::scalar(int64_t index) const noexcept {
    if (index < 0 || index >= count) return 0.0f;
    if (data != nullptr) return data[index];
    if (data16 == nullptr) return 0.0f;
    uint16_t bits = data16[index];
    if (dtype == 6) {
        uint32_t value = static_cast<uint32_t>(bits) << 16;
        float result{};
        std::memcpy(&result, &value, sizeof(result));
        return result;
    }
    return half_bits_to_float(bits);
}
} // namespace vocal
