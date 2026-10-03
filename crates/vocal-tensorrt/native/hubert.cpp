#include "bridge.h"
#include <cmath>
#include <algorithm>
#include <deque>
#include <vector>
#include <memory>
#include <map>
#include <string>

// Hand-expanded HuBERT base inference graph.  The converter normalizes the
// ONNX initializers into stable hubert.model.* names; this file deliberately
// does not include NvOnnxParser or inspect an ONNX graph.
namespace vocal {
using namespace nvinfer1;
using T = ITensor*;
inline Dims hdims(std::initializer_list<int64_t> values) {
    Dims d{}; d.nbDims = static_cast<int>(values.size());
    std::copy(values.begin(), values.end(), d.d); return d;
}

struct HubertGraph {
    INetworkDefinition& n;
    Options const& o;
    std::map<std::string, Weight const*> weights;
    std::deque<std::vector<float>> constants;
    std::deque<std::vector<uint16_t>> low_constants;
    std::deque<std::vector<int32_t>> integers;
    int elementwise_id = 0;

    HubertGraph(INetworkDefinition& net, Options const& options, Weight const* ws, size_t count)
        : n(net), o(options) {
        for (size_t i = 0; i < count; ++i) {
            require(weights.emplace(ws[i].name, ws + i).second, "duplicate HuBERT weight");
        }
    }
    Weight const& weight(std::string const& name) const {
        auto it = weights.find(name);
        require(it != weights.end(), "missing HuBERT weight: " + name);
        return *it->second;
    }
    Weights w(std::string const& name) const {
        auto const& p = weight(name);
        require(p.rank <= 4, "HuBERT weight rank exceeds ABI");
        if (o.precision == 0) return {DataType::kFLOAT, p.data, p.count};
        auto* self = const_cast<HubertGraph*>(this);
        self->low_constants.emplace_back(static_cast<size_t>(p.count));
        auto& dst = self->low_constants.back();
        for (int64_t i = 0; i < p.count; ++i)
            dst[static_cast<size_t>(i)] = o.precision == 1
                ? float_to_half_bits(p.data[i]) : float_to_bf16_bits(p.data[i]);
        return {o.precision == 1 ? DataType::kHALF : DataType::kBF16, dst.data(), p.count};
    }
    DataType real_type() const {
        return o.precision == 1 ? DataType::kHALF
             : o.precision == 2 ? DataType::kBF16 : DataType::kFLOAT;
    }
    T output(ILayer* layer, char const* name = nullptr) {
        require(layer != nullptr, name ? std::string("HuBERT layer failed: ") + name : "HuBERT layer failed");
        if (name) layer->setName(name);
        auto* tensor = layer->getOutput(0);
        require(tensor != nullptr, "HuBERT layer has no output");
        return tensor;
    }
    T constant(std::vector<float> values, Dims dims) {
        if (o.precision != 0) {
            low_constants.emplace_back(values.size());
            auto& dst = low_constants.back();
            for (size_t i = 0; i < values.size(); ++i)
                dst[i] = o.precision == 1 ? float_to_half_bits(values[i]) : float_to_bf16_bits(values[i]);
            return output(n.addConstant(dims, {real_type(), dst.data(), static_cast<int64_t>(dst.size())}));
        }
        constants.push_back(std::move(values));
        auto& v = constants.back();
        return output(n.addConstant(dims, {DataType::kFLOAT, v.data(), static_cast<int64_t>(v.size())}));
    }
    T integer(std::vector<int32_t> values, Dims dims) {
        integers.push_back(std::move(values));
        auto& stable = integers.back();
        auto* layer = n.addConstant(dims, {DataType::kINT32, stable.data(), static_cast<int64_t>(stable.size())});
        return output(layer);
    }
    T concat(std::vector<T> values, int axis) {
        auto* layer = n.addConcatenation(values.data(), static_cast<int>(values.size()));
        require(layer, "HuBERT shape concat"); layer->setAxis(axis); return output(layer);
    }
    T parameter(std::string const& name, Dims dims) {
        return output(n.addConstant(dims, w(name)), name.c_str());
    }
    T reshape(T x, Dims dims) {
        auto* l = n.addShuffle(*x); require(l, "HuBERT reshape");
        l->setReshapeDimensions(dims); return output(l);
    }
    T transpose(T x, std::initializer_list<int> order) {
        auto* l = n.addShuffle(*x); require(l, "HuBERT transpose");
        Permutation p{}; std::copy(order.begin(), order.end(), p.order);
        l->setFirstTranspose(p); return output(l);
    }
    T ew(T a, T b, ElementWiseOperation op) {
        auto* layer = n.addElementWise(*a, *b, op);
        require(layer, "HuBERT elementwise");
        auto name = std::string("hubert.ew.") + std::to_string(elementwise_id++);
        return output(layer, name.c_str());
    }
    T add(T a, T b) { return ew(a, b, ElementWiseOperation::kSUM); }
    T mul(T a, T b) { return ew(a, b, ElementWiseOperation::kPROD); }
    T scale(T a, float v, int rank) {
        Dims d{}; d.nbDims = rank; for (int i = 0; i < rank; ++i) d.d[i] = 1;
        return mul(a, constant({v}, d));
    }
    T activation(T x, ActivationType type) { return output(n.addActivation(*x, type)); }
    T cast_real(T x) {
        if (real_type() == DataType::kFLOAT) return x;
        return output(n.addCast(*x, real_type()));
    }
    T matmul(T a, T b, bool transpose_b = false) {
        return output(n.addMatrixMultiply(*a, MatrixOperation::kNONE, *b,
                                          transpose_b ? MatrixOperation::kTRANSPOSE : MatrixOperation::kNONE));
    }
    T linear(T x, std::string const& prefix, int in, int out) {
        auto weight_name = prefix + ".weight";
        auto y = matmul(x, parameter(weight_name, hdims({1, in, out})));
        return add(y, reshape(parameter(prefix + ".bias", hdims({out})), hdims({1, 1, out})));
    }
    T fused_qkv_linear(T x, std::string const& prefix, int in, int out) {
        auto const& qw = weight(prefix + "q_proj.weight");
        auto const& kw = weight(prefix + "k_proj.weight");
        auto const& vw = weight(prefix + "v_proj.weight");
        require(qw.rank == 2 && kw.rank == 2 && vw.rank == 2,
                "HuBERT QKV weights must be rank 2");
        require(qw.dims[0] == in && kw.dims[0] == in && vw.dims[0] == in &&
                    qw.dims[1] == out && kw.dims[1] == out && vw.dims[1] == out,
                "HuBERT QKV weight layout must be [in,out]");
        require(qw.count == static_cast<int64_t>(in) * out &&
                    kw.count == qw.count && vw.count == qw.count,
                "HuBERT QKV weight size mismatch");
        constants.emplace_back(static_cast<size_t>(in) * out * 3);
        auto& fused = constants.back();
        for (int i = 0; i < in; ++i) {
            for (int j = 0; j < out; ++j) {
                auto dst = static_cast<size_t>(i) * out * 3 + j;
                fused[dst] = qw.data[static_cast<size_t>(i) * out + j];
                fused[dst + out] = kw.data[static_cast<size_t>(i) * out + j];
                fused[dst + out * 2] = vw.data[static_cast<size_t>(i) * out + j];
            }
        }
        // Convert the fused constant through the same precision path used by
        // ordinary parameters. This keeps FP32/FP16/BF16 A/B comparisons
        // numerically identical apart from the GEMM accumulation order.
        T matrix;
        if (o.precision == 0) {
            matrix = output(n.addConstant(hdims({1, in, 3 * out}),
                {DataType::kFLOAT, fused.data(), static_cast<int64_t>(fused.size())}));
        } else {
            low_constants.emplace_back(fused.size());
            auto& low = low_constants.back();
            for (size_t i = 0; i < fused.size(); ++i)
                low[i] = o.precision == 1 ? float_to_half_bits(fused[i]) : float_to_bf16_bits(fused[i]);
            matrix = output(n.addConstant(hdims({1, in, 3 * out}),
                {real_type(), low.data(), static_cast<int64_t>(low.size())}));
        }
        auto const& qb = weight(prefix + "q_proj.bias");
        auto const& kb = weight(prefix + "k_proj.bias");
        auto const& vb = weight(prefix + "v_proj.bias");
        require(qb.rank == 1 && kb.rank == 1 && vb.rank == 1 &&
                    qb.count == out && kb.count == out && vb.count == out,
                "HuBERT QKV bias layout must be [out]");
        std::vector<float> bias(static_cast<size_t>(3 * out));
        std::copy(qb.data, qb.data + out, bias.data());
        std::copy(kb.data, kb.data + out, bias.data() + out);
        std::copy(vb.data, vb.data + out, bias.data() + 2 * out);
        return add(matmul(x, matrix),
                   reshape(constant(std::move(bias), hdims({3 * out})), hdims({1, 1, 3 * out})));
    }
    T qkv_slice(T x, int start, int width) {
        std::vector<int32_t> indices(static_cast<size_t>(width));
        for (int i = 0; i < width; ++i) indices[static_cast<size_t>(i)] = start + i;
        integers.emplace_back(std::move(indices));
        auto& stable = integers.back();
        auto* layer = n.addGather(*x,
            *n.addConstant(hdims({width}), {DataType::kINT32, stable.data(), width})->getOutput(0), 2);
        require(layer, "HuBERT QKV slice");
        return reshape(output(layer), hdims({1, -1, width}));
    }
    T layer_norm(T x, std::string const& prefix, int channels) {
        auto gamma = parameter(prefix + ".weight", hdims({1, 1, channels}));
        auto beta = parameter(prefix + ".bias", hdims({1, 1, channels}));
        auto* layer = n.addNormalizationV2(*x, *gamma, *beta, 1U << 2);
        require(layer, "HuBERT LayerNorm");
        layer->setEpsilon(1.0e-5f);
        return output(layer, prefix.c_str());
    }
    T conv_feature(T x, int index, int in_channels, int out_channels, int kernel, int stride) {
        auto name = std::string("hubert.model.feature_extractor.conv_layers.") + std::to_string(index) + ".conv.weight";
        auto* layer = n.addConvolutionNd(*x, out_channels, hdims({1, kernel}), w(name), Weights{real_type(), nullptr, 0});
        require(layer, "HuBERT feature convolution");
        layer->setStrideNd(hdims({1, stride}));
        layer->setPaddingNd(hdims({0, 0}));
        return output(layer, name.c_str());
    }
    T attention(T x, int layer) {
        auto p = std::string("hubert.model.encoder.layers.") + std::to_string(layer) + ".attention.";
        T q_linear, k_linear, v_linear;
        if (o.fused_qkv) {
            auto qkv = fused_qkv_linear(x, p, 768, 768);
            q_linear = qkv_slice(qkv, 0, 768);
            k_linear = qkv_slice(qkv, 768, 768);
            v_linear = qkv_slice(qkv, 1536, 768);
        } else {
            q_linear = linear(x, p + "q_proj", 768, 768);
            k_linear = linear(x, p + "k_proj", 768, 768);
            v_linear = linear(x, p + "v_proj", 768, 768);
        }
        auto q = reshape(q_linear, hdims({1, -1, 12, 64}));
        auto k = reshape(k_linear, hdims({1, -1, 12, 64}));
        auto v = reshape(v_linear, hdims({1, -1, 12, 64}));
        if (o.fused_attention) {
            // HuBERT has ordinary self-attention without the relative-key /
            // relative-value correction used by the RVC encoder. Keep the
            // exact MatMul path below as the default, and let TensorRT-RTX
            // select its fused scaled-dot-product kernel when explicitly
            // requested.
            auto qh = transpose(q, {0, 2, 1, 3});
            auto kh = transpose(k, {0, 2, 1, 3});
            auto vh = transpose(v, {0, 2, 1, 3});
            auto* fused = n.addAttentionV2(*qh, *kh, *vh,
                AttentionNormalizationOp::kSOFTMAX, CausalMaskKind::kNONE);
            require(fused, "HuBERT fused attention creation failed");
            require(fused->setQueryForm(AttentionIOForm::kPADDED_BHND),
                    "HuBERT fused query form failed");
            require(fused->setKeyValueForm(AttentionIOForm::kPADDED_BHND),
                    "HuBERT fused key/value form failed");
            // Dynamic short chunks do not always match RTX's dedicated MHA
            // kernel constraints. Allow Myelin to decompose the operator so
            // this remains a valid A/B path instead of making the build
            // fail; Nsight decides whether a fused kernel is selected.
            require(fused->setDecomposable(true),
                    "HuBERT fused attention decomposition failed");
            fused->setName((p + "fused").c_str());
            auto context = fused->getOutput(0);
            require(context, "HuBERT fused attention output missing");
            context = transpose(context, {0, 2, 1, 3});
            context = reshape(context, hdims({1, -1, 768}));
            return linear(context, p + "out_proj", 768, 768);
        }
        q = transpose(q, {0, 2, 1, 3}); k = transpose(k, {0, 2, 1, 3}); v = transpose(v, {0, 2, 1, 3});
        auto scores = scale(matmul(q, k, true), 1.0f / std::sqrt(64.0f), 4);
        auto* softmax = n.addSoftMax(*scores); require(softmax, "HuBERT attention softmax");
        softmax->setAxes(1U << 3);
        auto context = matmul(output(softmax), v);
        context = transpose(context, {0, 2, 1, 3});
        context = reshape(context, hdims({1, -1, 768}));
        return linear(context, p + "out_proj", 768, 768);
    }
    T build(T waveform) {
        auto x = reshape(waveform, hdims({1, 1, 1, -1}));
        static int const kernels[] = {10, 3, 3, 3, 3, 2, 2};
        static int const strides[] = {5, 2, 2, 2, 2, 2, 2};
        for (int i = 0; i < 7; ++i) {
            x = conv_feature(x, i, i == 0 ? 1 : 512, 512, kernels[i], strides[i]);
            if (i == 0) {
                // HubertFeatureEncoder uses GroupNorm(512, 512), exported as
                // InstanceNormalization over the single spatial axis.
                auto scale = parameter("hubert.const.fef951e6c76ad6a0", hdims({1, 512, 1, 1}));
                auto bias = parameter("hubert.const.e5a00aa9991ac8a5", hdims({1, 512, 1, 1}));
                auto* norm = n.addNormalizationV2(*x, *scale, *bias, (1U << 2) | (1U << 3));
                require(norm, "HuBERT feature GroupNorm");
                norm->setEpsilon(1.0e-5f);
                x = output(norm, "hubert.feature_group_norm");
                // The exporter represents the learned GroupNorm affine as a
                // following Mul/Add pair. Keep that pair explicit so the
                // native graph consumes the same d8b9/685e constants.
                x = add(mul(x, parameter("hubert.const.d8b9ba9bce6ed02c", hdims({1, 512, 1, 1}))),
                        parameter("hubert.const.685ea624c050bb77", hdims({1, 512, 1, 1})));
            }
            x = activation(x, ActivationType::kGELU_ERF);
        }
        x = reshape(x, hdims({1, -1, 512}));
        x = layer_norm(x, "hubert.model.feature_projection.layer_norm", 512);
        x = linear(x, "hubert.model.feature_projection.projection", 512, 768);
        // Positional convolution uses the normalized weight exported by the
        // converter.  Keep the sequence length dynamic and crop the extra
        // right padding with a shape-driven slice in a later refinement.
        auto pos = reshape(x, hdims({1, 768, 1, -1}));
        auto* pos_layer = n.addConvolutionNd(
            *pos, 768, hdims({1, 128}),
            w("hubert.model.encoder.pos_conv_embed.conv.weight"),
            w("hubert.model.encoder.pos_conv_embed.conv.bias"));
        require(pos_layer, "HuBERT positional convolution");
        pos_layer->setNbGroups(16);
        pos_layer->setPaddingNd(hdims({0, 64}));
        auto pos4 = output(pos_layer, "hubert.encoder.pos_conv");
        auto* shape_layer = n.addShape(*x); require(shape_layer, "HuBERT sequence shape");
        auto shape32 = output(n.addCast(*output(shape_layer), DataType::kINT32));
        auto sequence = output(n.addGather(*shape32, *integer({1}, hdims({1})), 0));
        auto target = concat({integer({1, 768, 1}, hdims({3})), sequence}, 0);
        auto* crop = n.addSlice(*pos4, hdims({0, 0, 0, 0}), hdims({1, 1, 1, 1}), hdims({1, 1, 1, 1}));
        require(crop, "HuBERT positional crop"); crop->setInput(2, *target);
        pos = reshape(output(crop), hdims({1, -1, 768}));
        pos = activation(pos, ActivationType::kGELU_ERF);
        x = layer_norm(add(x, pos), "hubert.model.encoder.layer_norm", 768);
        for (int i = 0; i < 12; ++i) {
            auto prefix = std::string("hubert.model.encoder.layers.") + std::to_string(i);
            auto y = add(x, attention(x, i));
            y = layer_norm(y, prefix + ".layer_norm", 768);
            y = activation(linear(y, prefix + ".feed_forward.intermediate_dense", 768, 3072), ActivationType::kGELU_ERF);
            y = linear(y, prefix + ".feed_forward.output_dense", 3072, 768);
            x = layer_norm(add(x, y), prefix + ".final_layer_norm", 768);
        }
        return x;
    }
};

extern "C" int vocal_trt_build_hubert(
    Weight const* weights, size_t count, Options const* options,
    Build** out, char** error) noexcept {
    return guarded(error, [&] {
        require(weights && count && options && out, "invalid HuBERT build arguments");
        require(options->min_frames > 0 && options->min_frames <= options->opt_frames &&
                    options->opt_frames <= options->max_frames, "invalid HuBERT sample profile");
        *out = nullptr;
        std::unique_ptr<IBuilder> builder(createInferBuilder(logger())); require(bool(builder), "create HuBERT builder");
        std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(0)); require(bool(network), "create HuBERT network");
        std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig()); require(bool(config), "create HuBERT config");
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, options->workspace);
        config->clearFlag(BuilderFlag::kTF32);
        config->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);
        auto* profile = builder->createOptimizationProfile(); require(profile, "create HuBERT profile");
        auto* waveform = network->addInput("waveform", DataType::kFLOAT, Dims{2, {1, -1}});
        require(waveform, "create HuBERT waveform input");
        require(profile->setDimensions("waveform", OptProfileSelector::kMIN, hdims({1, options->min_frames})), "HuBERT min profile");
        require(profile->setDimensions("waveform", OptProfileSelector::kOPT, hdims({1, options->opt_frames})), "HuBERT opt profile");
        require(profile->setDimensions("waveform", OptProfileSelector::kMAX, hdims({1, options->max_frames})), "HuBERT max profile");
        HubertGraph graph(*network, *options, weights, count);
        auto* features = graph.build(graph.cast_real(waveform));
        if (options->precision != 0)
            features = graph.output(network->addCast(*features, DataType::kFLOAT), "features");
        features->setName("features");
        network->markOutput(*features);
        require(config->addOptimizationProfile(profile) >= 0, "HuBERT add profile");
        auto* serialized = builder->buildSerializedNetwork(*network, *config);
        require(serialized, "build HuBERT native graph failed");
        *out = new Build{serialized};
    });
}
}
