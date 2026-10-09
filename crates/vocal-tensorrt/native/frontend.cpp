#include "graph/layers.h"

// RVC v1/v2 alignment: repeat each HuBERT frame twice, then crop RMVPE F0
// to that length. Never stretch F0: its 10 ms time grid is already correct.
namespace vocal {
extern "C" int vocal_trt_build_rvc_frontend(
    uint64_t workspace, int feature_width, int min_frames, int opt_frames, int max_frames,
    Build** out, char** error) noexcept {
    return guarded(error, [&] {
        require(out && min_frames > 0 && min_frames <= opt_frames &&
                opt_frames <= max_frames, "invalid frontend build arguments");
        require(feature_width == 256 || feature_width == 768, "unsupported RVC frontend feature width");
        *out = nullptr;
        auto b = std::unique_ptr<IBuilder>(createInferBuilder(logger()));
        require(bool(b), "create frontend builder");
        auto n = std::unique_ptr<INetworkDefinition>(b->createNetworkV2(0));
        auto bc = std::unique_ptr<IBuilderConfig>(b->createBuilderConfig());
        require(n && bc, "create frontend network/config");
        bc->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, workspace);
        bc->clearFlag(BuilderFlag::kTF32);
        bc->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);
        auto profile = b->createOptimizationProfile();
        require(profile, "create frontend profile");
        Config dummy{}; Options options{};
        Layers g(*n, dummy, options, nullptr, 0);
        auto input = [&](char const* name, Dims d) {
            auto t = n->addInput(name, DataType::kFLOAT, d);
            require(t, "create frontend input");
            auto a=d, o=d, z=d;
            a.d[1]=min_frames; o.d[1]=opt_frames; z.d[1]=max_frames;
            require(profile->setDimensions(name, OptProfileSelector::kMIN,a), "frontend min");
            require(profile->setDimensions(name, OptProfileSelector::kOPT,o), "frontend opt");
            require(profile->setDimensions(name, OptProfileSelector::kMAX,z), "frontend max");
            return t;
        };
        auto h = input("hubert_features", dims({1,-1,feature_width}));
        auto f0 = input("pitchf_in", dims({1,-1}));
        // fairseq HuBERT realtime inference appends the final feature frame
        // before the 2x RVC interpolation. Recreate that boundary in the
        // native graph so p_len and F0 cache lengths match WebUI/Applio.
        auto h_len = g.dimension(h,1);
        auto h_start = g.concat({g.iv({0}), g.sub(h_len,g.iv({1})), g.iv({0})},0);
        auto h_last_layer = n->addSlice(*h,dims({0,0,0}),dims({1,1,feature_width}),dims({1,1,1}));
        require(h_last_layer,"HuBERT final frame slice");
        h_last_layer->setInput(1,*h_start);
        auto h_last = g.output(h_last_layer,"hubert.final_frame");
        auto h_aug = g.concat({h,h_last},1);
        auto frames = g.mul(g.dimension(h_aug,1),g.iv({2}));
        // The caller binds only the RVC-valid prefix of RMVPE F0 to `f0`.
        // Taking the shorter interpolated HuBERT/F0 length therefore crops
        // the extra RMVPE boundary frame (282 -> 281 at 44960) without a
        // shape-control input, which keeps this graph safe on RTX runtimes
        // that have unstable dynamic shape-input execution paths.
        auto target_frames = g.ew(frames,g.dimension(f0,1),ElementWiseOperation::kMIN);
        auto pitch_shape = g.concat({g.iv({1}),target_frames},0);
        auto crop = n->addSlice(*f0,dims({0,0}),dims({1,1}),dims({1,1}));
        require(crop,"crop F0");
        crop->setInput(2,*pitch_shape);
        auto pitchf = g.output(crop,"crop_f0");
        auto resized = g.transpose(g.resize(g.transpose(h_aug,{0,2,1}),2),{0,2,1});
        auto feat_shape = g.concat({g.iv({1}),target_frames,g.iv({feature_width})},0);
        auto feat_crop_layer = n->addSlice(*resized,dims({0,0,0}),dims({1,1,1}),dims({1,1,1}));
        require(feat_crop_layer,"crop HuBERT");
        feat_crop_layer->setInput(2,*feat_shape);
        auto feats = g.output(feat_crop_layer,"crop_hubert");
        float const mel_min=1127.0f*std::log1p(50.0f/700.0f);
        float const mel_max=1127.0f*std::log1p(1100.0f/700.0f);
        auto mel = g.scale(g.unary(g.add(g.scale(pitchf,1.0f/700.0f),
            g.scalar(1.0f,2)),UnaryOperation::kLOG),1127.0f);
        auto coarse = g.add(g.scale(g.sub(mel,g.scalar(mel_min,2)),
            254.0f/(mel_max-mel_min)),g.scalar(1.0f,2));
        coarse = g.ew(g.ew(coarse,g.scalar(1.0f,2),ElementWiseOperation::kMAX),
            g.scalar(255.0f,2),ElementWiseOperation::kMIN);
        auto pitch = g.cast(g.unary(coarse,UnaryOperation::kROUND),DataType::kINT32);
        g.mark(feats,"feats"); g.mark(pitchf,"pitchf"); g.mark(pitch,"pitch");
        require(bc->addOptimizationProfile(profile)>=0,"add frontend profile");
        auto mem = b->buildSerializedNetwork(*n,*bc);
        require(mem,"build frontend failed");
        *out=new Build{mem};
    });
}
}
