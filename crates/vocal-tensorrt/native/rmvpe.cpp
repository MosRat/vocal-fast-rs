#include "bridge.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <map>
#include <memory>
#include <string>
#include <vector>

// RMVPE is expanded here from the acoustic-v1 contract.  The implementation
// intentionally follows Applio's E2E module: a fixed Fourier/mel frontend,
// five residual U-Net encoder stages, four intermediate stages, five decoder
// stages, and a bidirectional GRU expressed with TensorRT ILoop.  No ONNX
// parser or source graph is used at runtime.
namespace vocal {
using namespace nvinfer1;
using T = ITensor*;

inline Dims rdims(std::initializer_list<int64_t> v) {
    Dims d{};
    d.nbDims = static_cast<int>(v.size());
    std::copy(v.begin(), v.end(), d.d);
    return d;
}

struct RmvpeGraph {
    INetworkDefinition& n;
    Options const& options;
    std::map<std::string, Weight const*> weights;
    std::deque<std::vector<float>> constants;
    std::deque<std::vector<uint16_t>> low_constants;
    std::deque<std::vector<int32_t>> integers;
    int layer_id = 0;
    int active_precision = 0;

    int precision() const {
        return active_precision;
    }

    RmvpeGraph(INetworkDefinition& network, Options const& opts, Weight const* table, size_t count)
        : n(network), options(opts) {
        for (size_t i = 0; i < count; ++i) {
            require(weights.emplace(table[i].name, table + i).second, "duplicate RMVPE native weight");
        }
    }

    Weight const& weight(std::string const& name) const {
        auto it = weights.find(name);
        require(it != weights.end(), "missing RMVPE native weight: " + name);
        return *it->second;
    }
    Weights w(std::string const& name) const {
        auto const& p = weight(name);
        require(p.rank <= 4, "RMVPE weight rank exceeds ABI");
        if (precision() == 0) return {DataType::kFLOAT, p.data, p.count};
        auto* self = const_cast<RmvpeGraph*>(this);
        self->low_constants.emplace_back(static_cast<size_t>(p.count));
        auto& dst = self->low_constants.back();
        for (int64_t i = 0; i < p.count; ++i)
            dst[static_cast<size_t>(i)] = precision() == 1
                ? float_to_half_bits(p.data[i]) : float_to_bf16_bits(p.data[i]);
        return {precision() == 1 ? DataType::kHALF : DataType::kBF16, dst.data(), p.count};
    }
    DataType real_type() const {
        return precision() == 1 ? DataType::kHALF
             : precision() == 2 ? DataType::kBF16 : DataType::kFLOAT;
    }
    T output(ILayer* layer, std::string const& name = {}) {
        require(layer != nullptr, "RMVPE TensorRT layer creation failed: " + name);
        if (!name.empty()) layer->setName(name.c_str());
        auto* out = layer->getOutput(0);
        require(out != nullptr, "RMVPE layer has no output: " + name);
        return out;
    }
    T parameter(std::string const& name, Dims shape) {
        // Parameter names are ABI keys, not layer names.  The same GRU
        // parameter is referenced by the forward and reverse loop and
        // TensorRT requires layer names to be globally unique.
        return output(n.addConstant(shape, w(name)));
    }
    T constant(std::vector<float> value, Dims shape) {
        if (precision() != 0) {
            low_constants.emplace_back(value.size());
            auto& dst = low_constants.back();
            for (size_t i = 0; i < value.size(); ++i)
                dst[i] = precision() == 1 ? float_to_half_bits(value[i]) : float_to_bf16_bits(value[i]);
            return output(n.addConstant(shape, {real_type(), dst.data(), static_cast<int64_t>(dst.size())}));
        }
        constants.push_back(std::move(value));
        auto const& v = constants.back();
        return output(n.addConstant(shape, {DataType::kFLOAT, v.data(), static_cast<int64_t>(v.size())}));
    }
    T constant_fp32(std::vector<float> value, Dims shape) {
        constants.push_back(std::move(value));
        auto const& v = constants.back();
        return output(n.addConstant(shape, {DataType::kFLOAT, v.data(), static_cast<int64_t>(v.size())}));
    }
    T scalar(float value, int rank = 1) {
        Dims d{}; d.nbDims = rank; std::fill(d.d, d.d + rank, 1);
        return constant({value}, d);
    }
    T integer(int32_t value) {
        integers.push_back({value});
        auto const& v = integers.back();
        return output(n.addConstant(rdims({}), {DataType::kINT32, v.data(), 1}));
    }
    T reshape(T x, Dims shape) {
        auto* l = n.addShuffle(*x); require(l, "RMVPE reshape");
        l->setReshapeDimensions(shape); return output(l);
    }
    T transpose(T x, std::initializer_list<int> order) {
        auto* l = n.addShuffle(*x); require(l, "RMVPE transpose");
        Permutation p{}; std::copy(order.begin(), order.end(), p.order);
        l->setFirstTranspose(p); return output(l);
    }
    T ew(T a, T b, ElementWiseOperation op) {
        return output(n.addElementWise(*a, *b, op));
    }
    T add(T a, T b) { return ew(a, b, ElementWiseOperation::kSUM); }
    T mul(T a, T b) { return ew(a, b, ElementWiseOperation::kPROD); }
    T sub(T a, T b) { return ew(a, b, ElementWiseOperation::kSUB); }
    T div(T a, T b) { return ew(a, b, ElementWiseOperation::kDIV); }
    T unary(T x, UnaryOperation op) { return output(n.addUnary(*x, op)); }
    T act(T x, ActivationType op) { return output(n.addActivation(*x, op)); }
    T cast_real(T x) {
        if (real_type() == DataType::kFLOAT) return x;
        return output(n.addCast(*x, real_type()));
    }

    T conv(T x, int node, int out_channels, bool bias = true) {
        auto prefix = std::string("rmvpe.node.") + std::to_string(node);
        auto* l = n.addConvolutionNd(*x, out_channels, rdims({3, 3}),
                                     w(prefix + ".weight"),
                                     bias ? w(prefix + ".bias") : Weights{real_type(), nullptr, 0});
        require(l, "RMVPE convolution");
        l->setPaddingNd(rdims({1, 1}));
        auto name = std::string("rmvpe.conv.") + std::to_string(node) + "." + std::to_string(layer_id++);
        return output(l, name);
    }
    T zero_upsample_2d(T x, int input_width) {
        // ConvTranspose2d(k=3, stride=2, pad=1, output_padding=1) is
        // equivalent to inserting a zero between adjacent samples on both
        // spatial axes and applying a forward convolution with a spatially
        // flipped/transposed kernel.  Nearest-neighbour resize is faster to write
        // but changes the operator's phase response and was the dominant
        // source of the RMVPE logit drift.
        auto x5 = reshape(x, rdims({0,0,-1,input_width,1}));
        auto z5 = mul(x5, scalar(0.0f, 5));
        std::array<T,2> width_parts{x5, z5};
        auto* width_cat = n.addConcatenation(width_parts.data(), 2);
        require(width_cat, "RMVPE zero upsample width concat");
        width_cat->setAxis(4);
        auto width = reshape(output(width_cat), rdims({0,0,-1,2 * input_width}));
        auto width6 = reshape(width, rdims({0,0,-1,1,2 * input_width,1}));
        auto z6 = mul(width6, scalar(0.0f, 6));
        std::array<T,2> height_parts{width6, z6};
        auto* height_cat = n.addConcatenation(height_parts.data(), 2);
        require(height_cat, "RMVPE zero upsample height concat");
        height_cat->setAxis(3);
        return reshape(output(height_cat), rdims({0,0,-1,2 * input_width}));
    }

    T deconv(T x, int node, int out_channels, int input_width) {
        auto prefix = std::string("rmvpe.node.") + std::to_string(node);
        // TensorRT-RTX 1.6 may select cuDNN ConvBwdData kernels for native
        // deconvolution that are unavailable on pre-Blackwell GPUs. Express
        // the exact operator with a zero-insertion stage followed by a
        // forward convolution, transposing the ONNX [in,out,k,k] weights to
        // TensorRT's [out,in,k,k] layout.
        auto const& source = weight(prefix + ".weight");
        int in_channels = static_cast<int>(source.dims[0]);
        int kernel = static_cast<int>(source.dims[2]);
        require(source.rank == 4 && source.dims[1] == out_channels && kernel == 3,
                "RMVPE deconvolution weight shape");
        auto upsampled = zero_upsample_2d(x, input_width);
        constants.emplace_back(static_cast<size_t>(out_channels) * in_channels * kernel * kernel);
        auto& converted = constants.back();
        for (int i = 0; i < in_channels; ++i)
                for (int o = 0; o < out_channels; ++o)
                for (int h = 0; h < kernel; ++h)
                    for (int k = 0; k < kernel; ++k)
                        converted[((o * in_channels + i) * kernel + h) * kernel + k] =
                            source.data[((i * out_channels + o) * kernel + (kernel - 1 - h)) * kernel + (kernel - 1 - k)];
        Weights converted_weights{};
        if (precision() == 0) {
            converted_weights = {DataType::kFLOAT, converted.data(), static_cast<int64_t>(converted.size())};
        } else {
            low_constants.emplace_back(converted.size());
            auto& dst = low_constants.back();
            for (size_t i = 0; i < converted.size(); ++i)
                dst[i] = precision() == 1 ? float_to_half_bits(converted[i]) : float_to_bf16_bits(converted[i]);
            converted_weights = {real_type(), dst.data(), static_cast<int64_t>(dst.size())};
        }
        auto* l = n.addConvolutionNd(*upsampled, out_channels, rdims({3, 3}),
                                     converted_weights, Weights{real_type(), nullptr, 0});
        require(l, "RMVPE decoder convolution"); l->setPaddingNd(rdims({1, 1}));
        return output(l, std::string("rmvpe.deconv.") + std::to_string(node));
    }
    T batch_norm(T x, int node, int channels) {
        auto prefix = std::string("rmvpe.node.") + std::to_string(node);
        auto const& gamma = weight(prefix + ".weight");
        auto const& beta = weight(prefix + ".bias");
        auto const& mean = weight(prefix + ".mean");
        auto const& var = weight(prefix + ".var");
        require(gamma.count == channels && beta.count == channels && mean.count == channels && var.count == channels,
                "RMVPE BatchNorm channel mismatch");
        std::vector<float> scale_values(channels), shift_values(channels);
        for (int i = 0; i < channels; ++i) {
            scale_values[i] = gamma.data[i] / std::sqrt(var.data[i] + 1.0e-5f);
            shift_values[i] = beta.data[i] - mean.data[i] * scale_values[i];
        }
        constants.push_back(std::move(scale_values)); auto const& s = constants.back();
        constants.push_back(std::move(shift_values)); auto const& b = constants.back();
        Weights shift{}, scale{};
        if (precision() == 0) {
            shift = {DataType::kFLOAT, b.data(), channels};
            scale = {DataType::kFLOAT, s.data(), channels};
        } else {
            low_constants.push_back(std::vector<uint16_t>(channels));
            auto& sb = low_constants.back();
            low_constants.push_back(std::vector<uint16_t>(channels));
            auto& ss = low_constants.back();
            for (int i = 0; i < channels; ++i) {
                sb[static_cast<size_t>(i)] = precision() == 1 ? float_to_half_bits(b[i]) : float_to_bf16_bits(b[i]);
                ss[static_cast<size_t>(i)] = precision() == 1 ? float_to_half_bits(s[i]) : float_to_bf16_bits(s[i]);
            }
            shift = {real_type(), sb.data(), channels};
            scale = {real_type(), ss.data(), channels};
        }
        auto* l = n.addScale(*x, ScaleMode::kCHANNEL, shift, scale,
                             Weights{real_type(), nullptr, 0});
        require(l, "RMVPE BatchNorm");
        return output(l, prefix);
    }

    // Two Conv-BN-ReLU paths with a residual shortcut, matching ConvBlockRes.
    T residual_block(T x, int conv1, int conv2, int shortcut, int in_channels, int out_channels) {
        auto y = act(conv(x, conv1, out_channels), ActivationType::kRELU);
        y = conv(y, conv2, out_channels);
        y = act(y, ActivationType::kRELU);
        T skip = x;
        if (shortcut >= 0) skip = conv_shortcut(x, shortcut, out_channels);
        return add(y, skip);
    }
    T conv_shortcut(T x, int node, int out_channels) {
        auto prefix = std::string("rmvpe.node.") + std::to_string(node);
        auto* l = n.addConvolutionNd(*x, out_channels, rdims({1, 1}),
                                     w(prefix + ".weight"), w(prefix + ".bias"));
        require(l, "RMVPE shortcut convolution");
        return output(l, prefix);
    }
    T pool(T x) {
        auto* l = n.addPoolingNd(*x, PoolingType::kAVERAGE, rdims({2, 2}));
        require(l, "RMVPE average pool"); l->setStrideNd(rdims({2, 2}));
        l->setAverageCountExcludesPadding(false); return output(l);
    }
    T resize_like(T x, T reference) {
        auto* l = n.addResize(*x);
        require(l, "RMVPE decoder resize");
        l->setResizeMode(InterpolationMode::kNEAREST);
        auto* shape = n.addShape(*reference);
        require(shape, "RMVPE decoder reference shape");
        l->setInput(1, *shape->getOutput(0));
        return output(l);
    }
    T shape_dim(T x, int axis) {
        auto* shape = n.addShape(*x);
        require(shape, "RMVPE shape");
        auto* gathered = n.addGather(*shape->getOutput(0), *integer(axis), 0);
        require(gathered, "RMVPE shape gather");
        auto* cast = n.addCast(*gathered->getOutput(0), DataType::kINT32);
        require(cast, "RMVPE shape cast");
        return reshape(output(cast), rdims({1}));
    }
    T resize_to_shape(T x, T shape) {
        auto* l = n.addResize(*x);
        require(l, "RMVPE dynamic resize");
        l->setResizeMode(InterpolationMode::kNEAREST);
        l->setInput(1, *shape);
        return output(l);
    }
    T slice_dynamic(T x, T start, T size, int axis, int rank, int stride_value) {
        Dims starts{}; starts.nbDims = rank; std::fill(starts.d, starts.d + rank, 0);
        Dims sizes{}; sizes.nbDims = rank; std::fill(sizes.d, sizes.d + rank, 1);
        Dims strides{}; strides.nbDims = rank; std::fill(strides.d, strides.d + rank, 1);
        auto* l = n.addSlice(*x, starts, sizes, strides);
        require(l, "RMVPE dynamic slice");
        l->setInput(1, *start);
        l->setInput(2, *size);
        auto stride = reshape(integer(stride_value), rdims({1}));
        l->setInput(3, *stride);
        auto axes = reshape(integer(axis), rdims({1}));
        l->setInput(5, *axes);
        return output(l);
    }

    T pad_mel_to_multiple(T mel, T original_frames) {
        auto one = reshape(integer(1), rdims({1}));
        auto thirty_two = reshape(integer(32), rdims({1}));
        auto blocks = add(div(sub(original_frames, one), thirty_two), one);
        auto target = mul(blocks, thirty_two);
        auto pad = sub(target, original_frames);
        auto start = sub(sub(original_frames, one), one); // T - 2
        auto tail = slice_dynamic(mel, start, pad, 2, 3, -1);
        std::array<T,2> parts{mel, tail};
        auto* cat = n.addConcatenation(parts.data(), 2);
        require(cat, "RMVPE mel reflect pad");
        cat->setAxis(2);
        return output(cat, "rmvpe.frontend.mel_padded");
    }

    T fourier_mel(T waveform) {
        auto samples = shape_dim(waveform, 1);
        auto input = reshape(waveform, rdims({1, 1, 1, -1}));
        // torch.stft(center=True, pad_mode="reflect") pads 512 samples on
        // each side before the 1024-point analysis window.  Express the
        // reflection with dynamic reverse slices so the waveform profile can
        // remain shape-polymorphic.
        auto pad = reshape(integer(512), rdims({1}));
        auto left_start = pad;
        auto two = reshape(integer(2), rdims({1}));
        auto right_start = sub(samples, two);
        auto left = slice_dynamic(input, left_start, pad, 3, 4, -1);
        auto right = slice_dynamic(input, right_start, pad, 3, 4, -1);
        std::array<T,3> padded_parts{left, input, right};
        auto* padded_cat = n.addConcatenation(padded_parts.data(), 3);
        require(padded_cat, "RMVPE waveform reflect pad");
        padded_cat->setAxis(3);
        input = output(padded_cat, "rmvpe.frontend.waveform_padded");
        constexpr int kNfft = 1024;
        constexpr int kBins = 513;
        constants.emplace_back(static_cast<size_t>(2 * kBins * kNfft));
        auto& kernel = constants.back();
        for (int c = 0; c < 2 * kBins; ++c) {
            int bin = c < kBins ? c : c - kBins;
            bool imag = c >= kBins;
            for (int i = 0; i < kNfft; ++i) {
                constexpr float kPi = 3.14159265358979323846f;
                // torch.hann_window(win_length) is periodic by default;
                // using N (rather than N-1) matches its STFT frontend.
                float window = 0.5f - 0.5f * std::cos(2.0f * kPi * i / kNfft);
                float phase = 2.0f * kPi * bin * i / kNfft;
                kernel[static_cast<size_t>(c) * kNfft + i] = window * (imag ? -std::sin(phase) : std::cos(phase));
            }
        }
        auto* stft = n.addConvolutionNd(*input, 2 * kBins, rdims({1, kNfft}),
                                        [&]() {
                                            if (precision() == 0)
                                                return Weights{DataType::kFLOAT, kernel.data(), static_cast<int64_t>(kernel.size())};
                                            low_constants.emplace_back(kernel.size());
                                            auto& dst = low_constants.back();
                                            for (size_t i = 0; i < kernel.size(); ++i)
                                                dst[i] = precision() == 1 ? float_to_half_bits(kernel[i]) : float_to_bf16_bits(kernel[i]);
                                            return Weights{real_type(), dst.data(), static_cast<int64_t>(dst.size())};
                                        }(),
                                        Weights{real_type(), nullptr, 0});
        require(stft, "RMVPE Fourier convolution");
        stft->setStrideNd(rdims({1, 160}));
        stft->setPaddingNd(rdims({0, 0}));
        auto spectrum = reshape(output(stft, "rmvpe.frontend.stft"), rdims({1, 2, kBins, -1}));
        auto power = mul(spectrum, spectrum);
        auto* reduce = n.addReduce(*power, ReduceOperation::kSUM, 1U << 1, true);
        require(reduce, "RMVPE Fourier magnitude");
        auto magnitude = reshape(output(reduce), rdims({1, kBins, -1}));
        magnitude = unary(magnitude, UnaryOperation::kSQRT);
        auto mel_basis = reshape(parameter("rmvpe.mel_basis", rdims({128, kBins})), rdims({1, 128, kBins}));
        auto mel = output(n.addMatrixMultiply(*mel_basis, MatrixOperation::kNONE,
                                              *magnitude, MatrixOperation::kNONE),
                          "rmvpe.frontend.mel");
        auto* clip = n.addActivation(*mel, ActivationType::kCLIP);
        require(clip, "RMVPE mel clamp");
        clip->setAlpha(1.0e-5f);
        clip->setBeta(1.0e30f);
        mel = unary(output(clip, "rmvpe.frontend.mel_clamp"), UnaryOperation::kLOG);
        return mel;
    }

    T gru_direction(T sequence, bool reverse) {
        constexpr int H = 256;
        constexpr int I = 384;
        auto zero = constant(std::vector<float>(H, 0.0f), rdims({1, H}));
        auto shape = output(n.addShape(*sequence));
        auto len = output(n.addGather(*shape, *integer(1), 0));
        auto* loop = n.addLoop(); require(loop, "RMVPE GRU loop");
        auto* iterator = loop->addIterator(*sequence, 1, reverse); require(iterator, "RMVPE GRU iterator");
        auto* recurrence = loop->addRecurrence(*zero); require(recurrence, "RMVPE GRU recurrence");
        auto x = output(iterator, reverse ? "rmvpe.gru.reverse.x" : "rmvpe.gru.forward.x");
        auto h = output(recurrence, reverse ? "rmvpe.gru.reverse.h" : "rmvpe.gru.forward.h");
        auto prefix = std::string("rmvpe.node.356.");
        auto W = parameter(prefix + "W", rdims({2, 3 * H, I}));
        auto R = parameter(prefix + "R", rdims({2, 3 * H, H}));
        auto B = parameter(prefix + "B", rdims({2, 6 * H}));
        int direction = reverse ? 1 : 0;
        auto wdir = output(n.addGather(*W, *integer(direction), 0));
        auto rdir = output(n.addGather(*R, *integer(direction), 0));
        auto bdir = output(n.addGather(*B, *integer(direction), 0));
        auto slice = [&](T t, int start) {
            auto* l = n.addSlice(*t, rdims({0, start}), rdims({1, H}), rdims({1, 1}));
            require(l, "RMVPE GRU gate slice"); return output(l);
        };
        auto bias_slice = [&](T t, int start) {
            auto* l = n.addSlice(*t, rdims({start}), rdims({3 * H}), rdims({1}));
            require(l, "RMVPE GRU bias slice"); return reshape(output(l), rdims({1, 3 * H}));
        };
        auto b_w = bias_slice(bdir, 0);
        auto b_r = bias_slice(bdir, 3 * H);
        auto wx = output(n.addMatrixMultiply(*x, MatrixOperation::kNONE, *wdir, MatrixOperation::kTRANSPOSE));
        auto rh = output(n.addMatrixMultiply(*h, MatrixOperation::kNONE, *rdir, MatrixOperation::kTRANSPOSE));

        // ONNX GRU uses the gate order [z, r, h] (PyTorch's exported
        // parameters are reordered by the exporter) and this graph carries
        // linear_before_reset=1.  In that mode the reset gate is applied to
        // the recurrent candidate *after* adding the recurrent candidate
        // bias.  Folding all three gates into one tanh, as the old baseline
        // did, produces plausible but numerically unrelated pitch tracks.
        auto z = act(add(add(slice(wx, 0), slice(rh, 0)),
                         add(slice(b_w, 0), slice(b_r, 0))),
                     ActivationType::kSIGMOID);
        auto r = act(add(add(slice(wx, H), slice(rh, H)),
                         add(slice(b_w, H), slice(b_r, H))),
                     ActivationType::kSIGMOID);
        auto candidate_linear = add(add(slice(wx, 2 * H), slice(b_w, 2 * H)),
                                    mul(r, add(slice(rh, 2 * H), slice(b_r, 2 * H))));
        auto candidate = act(candidate_linear, ActivationType::kTANH);
        auto one = scalar(1.0f, 2);
        auto hnew = add(mul(sub(one, z), candidate), mul(z, h));
        recurrence->setInput(1, *hnew);
        auto* trip = loop->addTripLimit(*len, TripLimit::kCOUNT); require(trip, "RMVPE GRU trip limit");
        auto* out = loop->addLoopOutput(*hnew, reverse ? LoopOutput::kREVERSE : LoopOutput::kCONCATENATE, 0);
        require(out, "RMVPE GRU output"); out->setInput(1, *len);
        return reshape(output(out), rdims({1, -1, H}));
    }

    T build(T waveform, T threshold) {
        // Keep the numerically sensitive Fourier frontend in its requested
        // precision, then switch the U-Net regions independently.  Explicit
        // casts at region boundaries make the mixed-precision contract stable
        // across TensorRT tactic changes.
        T mel{};
        active_precision = 0; // STFT/mel is intentionally FP32.
        mel = fourier_mel(waveform); // [1, 128, T]
        active_precision = options.rmvpe_encoder_precision;
        auto original_frames = shape_dim(mel, 2);
        mel = pad_mel_to_multiple(mel, original_frames);
        // Python E2E.forward performs `mel.transpose(-1, -2).unsqueeze(1)`.
        // A reshape alone would reinterpret the [mel,time] row-major layout
        // and silently rotate the temporal/frequency axes, which preserves
        // aggregate mel maxima but destroys every convolutional feature.
        auto x = reshape(transpose(mel, {0, 2, 1}), rdims({1, 1, -1, 128}));
        x = cast_real(x);
        x = batch_norm(x, 27, 1);
        static int const enc[5][4][3] = {
            {{28,30,32},{34,36,-1},{39,41,-1},{44,46,-1}},
            {{50,52,54},{56,58,-1},{61,63,-1},{66,68,-1}},
            {{72,74,76},{78,80,-1},{83,85,-1},{88,90,-1}},
            {{94,96,98},{100,102,-1},{105,107,-1},{110,112,-1}},
            {{116,118,120},{122,124,-1},{127,129,-1},{132,134,-1}},
        };
        T skips[5]{};
        int in_channels = 1;
        for (int i = 0; i < 5; ++i) {
            int out_channels = 16 << i;
            for (int b = 0; b < 4; ++b) {
                x = residual_block(x, enc[i][b][0], enc[i][b][1], enc[i][b][2], in_channels, out_channels);
            }
            skips[i] = x;
            x = pool(x);
            in_channels = out_channels;
        }
        static int const mid[4][3] = {{138,140,142},{144,146,-1},{149,151,-1},{154,156,-1}};
        for (auto const& b : mid) x = residual_block(x, b[0], b[1], b[2], in_channels, 512);
        static int const dec[5][1 + 4 * 2 + 4] = {
            {219,223,225,227,229,231,234,236,239,241},
            {244,248,250,252,254,256,259,261,264,266},
            {269,273,275,277,279,281,284,286,289,291},
            {294,298,300,302,304,306,309,311,314,316},
            {319,323,325,327,329,331,334,336,339,341},
        };
        static int const dec_bn[5] = {220,245,270,295,320};
        active_precision = options.rmvpe_decoder_precision;
        for (int i = 0; i < 5; ++i) {
            int out_channels = 256 >> i;
            x = deconv(x, dec[i][0], out_channels, 4 << i);
            // Mel time is reflect-padded to a multiple of 32, so every
            // encoder pool has an even inverse and the exact deconvolution
            // dimensions already match the skip tensor. Avoid an otherwise
            // redundant dynamic resize that can perturb values on RTX 1.6.
            x = act(batch_norm(x, dec_bn[i], out_channels), ActivationType::kRELU);
            auto skip = cast_real(skips[4 - i]);
            auto* cat = n.addConcatenation(std::array<T,2>{x, skip}.data(), 2);
            require(cat, "RMVPE decoder skip concat"); cat->setAxis(1);
            x = output(cat);
            int block_in = out_channels * 2;
            for (int b = 0; b < 4; ++b) {
                int base = b == 0 ? 1 : 4 + (b - 1) * 2;
                int shortcut = b == 0 ? dec[i][3] : -1;
                x = residual_block(x, dec[i][base], dec[i][base + 1], shortcut, block_in, out_channels);
                block_in = out_channels;
            }
        }
        // Decoder output is converted to FP32 before the recurrent block. The
        // GRU and all recurrent state updates therefore remain FP32 even when
        // both U-Net halves use FP16/BF16.
        active_precision = options.rmvpe_gru_precision;
        // TensorRT-RTX requires recurrent state and input GEMMs to have the
        // same type. Keep the GRU boundary explicitly FP32 even if a caller
        // experiments with a low-precision field in the ABI.
        active_precision = 0;
        x = output(n.addCast(*x, DataType::kFLOAT), "rmvpe.gru.input_fp32");
        auto* head = n.addConvolutionNd(*x, 3, rdims({3,3}), w("rmvpe.node.344.weight"), w("rmvpe.node.344.bias"));
        require(head, "RMVPE salience head"); head->setPaddingNd(rdims({1,1}));
        // Keep the padded time context through the bidirectional GRU. The
        // reference predictor crops only after sigmoid/decode; cropping the
        // CNN output before the GRU changes every recurrent state.
        auto seq = reshape(transpose(output(head), {0,2,1,3}), rdims({1,-1,384}));
        auto fwd = gru_direction(seq, false);
        auto rev = gru_direction(seq, true);
        auto* direction_cat = n.addConcatenation(std::array<T,2>{fwd,rev}.data(), 2);
        require(direction_cat, "RMVPE bidirectional concat"); direction_cat->setAxis(2);
        auto both = output(direction_cat);
        active_precision = 0;
        both = output(n.addCast(*both, DataType::kFLOAT), "rmvpe.logits.input_fp32");
        auto logits_full = add(output(n.addMatrixMultiply(*both, MatrixOperation::kNONE,
                                                     *parameter("rmvpe.node.360.weight", rdims({1,512,360})), MatrixOperation::kNONE)),
                          reshape(parameter("rmvpe.node.361.bias", rdims({360})), rdims({1,1,360})));
        // Salience argmax and the nine-bin weighted average are extremely
        // sensitive to half-precision ties. The decode tail is always FP32;
        // the U-Net is the only region intentionally lowered by default.
        if (precision() != 0)
            logits_full = output(n.addCast(*logits_full, DataType::kFLOAT),
                                 "rmvpe.decode.fp32_logits");
        active_precision = 0;
        auto logits = slice_dynamic(logits_full, reshape(integer(0), rdims({1})), original_frames, 1, 3, 1);
        auto salience = act(logits, ActivationType::kSIGMOID);
        // RMVPE decodes a local weighted average over the winning bin and
        // its four neighbours.  TopK supplies the centre index; Gather then
        // selects the nine salience/cents values without downloading an
        // intermediate tensor to the host.
        auto* top = n.addTopK(*salience, TopKOperation::kMAX, 1, 1U << 2);
        require(top, "RMVPE salience argmax");
        auto index = top->getOutput(1);
        require(index != nullptr, "RMVPE salience indices missing");
        index->setName("rmvpe.decode.bin");
        // IPaddingLayer only accepts two trailing dimensions on RTX 1.6.
        // Make the [1,T,4] zero pads from a broadcast zero tensor instead;
        // this keeps T dynamic and has exactly the same semantics.
        auto zero_base = constant_fp32(std::vector<float>(4, 0.0f), rdims({1,1,4}));
        std::array<T,3> zero_shape_parts{reshape(integer(1), rdims({1})), shape_dim(salience, 1), reshape(integer(4), rdims({1}))};
        auto* zero_shape_layer = n.addConcatenation(zero_shape_parts.data(), 3);
        require(zero_shape_layer, "RMVPE salience padding shape");
        zero_shape_layer->setAxis(0);
        auto zero_shape = output(zero_shape_layer);
        auto left_zero = resize_to_shape(zero_base, zero_shape);
        auto right_zero = resize_to_shape(zero_base, zero_shape);
        std::array<T,3> padded_parts{left_zero, salience, right_zero};
        auto* padded_cat = n.addConcatenation(padded_parts.data(), 3);
        require(padded_cat, "RMVPE salience padding concat");
        padded_cat->setAxis(2);
        auto padded_salience = output(padded_cat, "rmvpe.decode.salience_padded");
        auto index_f = output(n.addCast(*index, DataType::kFLOAT));
        auto offsets = constant_fp32({-4.0f,-3.0f,-2.0f,-1.0f,0.0f,1.0f,2.0f,3.0f,4.0f}, rdims({1,1,9}));
        auto centres = add(add(index_f, constant_fp32({4.0f}, rdims({1,1,1}))), offsets);
        auto* indices = n.addCast(*centres, DataType::kINT32);
        require(indices, "RMVPE local-average indices");
        // GatherElements on RTX 1.6 has a shape-machine issue when the
        // indices' last extent differs from the data extent. Flatten the
        // [T,368] table and form absolute offsets, then use ordinary Gather.
        auto* frame_fill = n.addFill(rdims({1}), FillOperation::kLINSPACE, DataType::kINT32);
        require(frame_fill, "RMVPE local-average frame range");
        frame_fill->setAlpha(0.0);
        frame_fill->setBeta(1.0);
        frame_fill->setInput(0, *shape_dim(salience, 1));
        auto frame_index = reshape(output(frame_fill), rdims({1,-1,1}));
        auto base_offset = mul(frame_index, reshape(integer(368), rdims({1,1,1})));
        auto absolute = add(base_offset, output(indices));
        auto flat_indices = reshape(output(n.addCast(*absolute, DataType::kINT32)), rdims({-1}));
        auto flat_salience = reshape(padded_salience, rdims({1,-1}));
        auto* gathered = n.addGather(*flat_salience, *flat_indices, 1);
        require(gathered, "RMVPE local-average gather");
        auto local_salience = reshape(output(gathered, "rmvpe.decode.local_salience_flat"), rdims({1,-1,9}));
        // The cents table is indexed by the absolute class.  Build it once
        // as a padded [368] constant and gather the same nine absolute bins.
        std::vector<float> cents_table(368);
        for (int i = 0; i < 368; ++i) {
            int cls = i - 4;
            cents_table[static_cast<size_t>(i)] = (cls >= 0 && cls < 360)
                ? 1997.3794084376191f + 20.0f * cls
                : 0.0f;
        }
        auto cents_base = constant_fp32(std::move(cents_table), rdims({1,1,368}));
        std::array<T,3> cents_shape_parts{reshape(integer(1), rdims({1})), shape_dim(salience, 1), reshape(integer(368), rdims({1}))};
        auto* cents_shape_layer = n.addConcatenation(cents_shape_parts.data(), 3);
        require(cents_shape_layer, "RMVPE cents shape");
        cents_shape_layer->setAxis(0);
        auto cents_tensor = resize_to_shape(cents_base, output(cents_shape_layer));
        auto flat_cents = reshape(cents_tensor, rdims({1,-1}));
        auto* cents_gather = n.addGather(*flat_cents, *flat_indices, 1);
        require(cents_gather, "RMVPE local-average cents gather");
        auto local_cents = reshape(output(cents_gather, "rmvpe.decode.local_cents_flat"), rdims({1,-1,9}));
        auto weighted = mul(local_salience, local_cents);
        auto* sum = n.addReduce(*weighted, ReduceOperation::kSUM, 1U << 2, true);
        auto* denom = n.addReduce(*local_salience, ReduceOperation::kSUM, 1U << 2, true);
        require(sum && denom, "RMVPE local-average reduce");
        auto frequency_cents = div(output(sum), output(denom));
        auto frequency = mul(unary(mul(div(frequency_cents, constant_fp32({1200.0f}, rdims({1,1,1}))),
                                       constant_fp32({0.6931471805599453f}, rdims({1,1,1}))), UnaryOperation::kEXP),
                             constant_fp32({10.0f}, rdims({1,1,1})));
        frequency = reshape(frequency, rdims({1,-1}));
        auto* max_layer = n.addReduce(*salience, ReduceOperation::kMAX, 1U << 2, false);
        require(max_layer, "RMVPE threshold reduce");
        auto max_value = output(max_layer, "rmvpe.decode.salience_max");
        T threshold_decode = threshold;
        if (precision() != 0)
            threshold_decode = output(n.addCast(*threshold, DataType::kFLOAT),
                                      "rmvpe.decode.fp32_threshold");
        auto threshold_broadcast = reshape(threshold_decode, rdims({1,1}));
        auto mask = output(n.addElementWise(*max_value, *threshold_broadcast, ElementWiseOperation::kGREATER));
        mask = output(n.addCast(*mask, DataType::kFLOAT));
        // Apply the salience threshold in-graph.  The mask is broadcast over
        // the single output frequency axis; unvoiced frames are represented
        // by exactly 10 Hz and are converted to zero by the Rust caller's
        // standard RMVPE post-processing.
        auto voiced = output(n.addElementWise(*mask, *frequency, ElementWiseOperation::kPROD),
                             "rmvpe.decode.thresholded");
        return voiced;
    }
};

extern "C" int vocal_trt_build_rmvpe(
    Weight const* weights, size_t count, Options const* options,
    Build** out, char** error) noexcept {
    return guarded(error, [&] {
        require(weights && count && options && out, "invalid RMVPE build arguments");
        require(options->min_frames > 0 && options->min_frames <= options->opt_frames && options->opt_frames <= options->max_frames,
                "invalid RMVPE frame profile");
        *out = nullptr;
        std::unique_ptr<IBuilder> builder(createInferBuilder(logger())); require(bool(builder), "create RMVPE builder");
        std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(0)); require(bool(network), "create RMVPE network");
        std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig()); require(bool(config), "create RMVPE config");
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, options->workspace);
        config->clearFlag(BuilderFlag::kTF32);
        config->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);
        auto* profile = builder->createOptimizationProfile(); require(profile, "create RMVPE profile");
        auto* waveform = network->addInput("waveform", DataType::kFLOAT, Dims{2, {1, -1}}); require(waveform, "RMVPE waveform input");
        auto* threshold = network->addInput("threshold", DataType::kFLOAT, Dims{1, {1}}); require(threshold, "RMVPE threshold input");
        require(profile->setDimensions("waveform", OptProfileSelector::kMIN, rdims({1, options->min_frames * 160})), "RMVPE min profile");
        require(profile->setDimensions("waveform", OptProfileSelector::kOPT, rdims({1, options->opt_frames * 160})), "RMVPE opt profile");
        require(profile->setDimensions("waveform", OptProfileSelector::kMAX, rdims({1, options->max_frames * 160})), "RMVPE max profile");
        RmvpeGraph graph(*network, *options, weights, count);
        auto internal_waveform = graph.cast_real(waveform);
        auto internal_threshold = graph.cast_real(threshold);
        auto pitch = graph.build(internal_waveform, internal_threshold);
        if (options->precision != 0) pitch = graph.output(network->addCast(*pitch, DataType::kFLOAT), "pitchf");
        pitch->setName("pitchf");
        network->markOutput(*pitch);
        require(config->addOptimizationProfile(profile) >= 0, "RMVPE add profile");
        auto* serialized = builder->buildSerializedNetwork(*network, *config); require(serialized, "build RMVPE native graph failed");
        *out = new Build{serialized};
    });
}
}
