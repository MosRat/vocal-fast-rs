#include "layers.h"
namespace vocal {
extern "C" int vocal_trt_build_rvc(Config const* c,Weight const* weights,size_t count,Options const* options,Build** out,char** error) noexcept {
    return guarded(error,[&]{
        require(c&&weights&&options&&out&&c->abi_version==2,"invalid RVC build ABI");
        require(options->min_frames>0&&options->min_frames<=options->opt_frames&&options->opt_frames<=options->max_frames,"invalid frame profile");
        require(options->synth_resblock_stage_mask>=0,"invalid Synth residual stage mask");
        uint32_t valid_stage_mask=c->upsample_count>=31?0xffffffffu:((1u<<c->upsample_count)-1u);
        require((static_cast<uint32_t>(options->synth_resblock_stage_mask)&~valid_stage_mask)==0,
                "Synth residual stage mask exceeds upsample stages");
        auto b=std::unique_ptr<IBuilder>(createInferBuilder(logger()));require(bool(b),"createInferBuilder");
        auto n=std::unique_ptr<INetworkDefinition>(b->createNetworkV2(0));require(bool(n),"createNetwork");
        auto bc=std::unique_ptr<IBuilderConfig>(b->createBuilderConfig());require(bool(bc),"builder config");
        bc->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE,options->workspace);
        bc->clearFlag(BuilderFlag::kTF32); // keep the baseline deterministic
        bc->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);
        // Synth precision is represented explicitly in the native graph by
        // Layers::real_type(). The public bindings remain FP32 and are cast at
        // the graph boundary, so no Rust buffer ABI changes are required.
        auto profile=b->createOptimizationProfile();require(profile,"profile");
        Layers g(*n,*c,*options,weights,count);
        auto input=[&](char const* name,DataType type,Dims d,int scale=1){
            auto t=n->addInput(name,type,d);require(t,std::string("input ")+name);
            Dims a=d,o=d,z=d;bool dynamic=false;
            for(int i=0;i<d.nbDims;i++)if(d.d[i]==-1){dynamic=true;a.d[i]=options->min_frames*scale;o.d[i]=options->opt_frames*scale;z.d[i]=options->max_frames*scale;}
            if(dynamic){require(profile->setDimensions(name,OptProfileSelector::kMIN,a),"min profile");require(profile->setDimensions(name,OptProfileSelector::kOPT,o),"opt profile");require(profile->setDimensions(name,OptProfileSelector::kMAX,z),"max profile");}
            return t;
        };
        auto feats=input("feats",DataType::kFLOAT,dims({1,-1,c->feature_width}));
        // TensorRT shape tensors and gather indices are INT32 throughout the
        // native graph.  The public runtime ABI may still be fed i32 values;
        // keeping one integer type avoids implicit INT64/INT32 graph errors.
        auto length=input("p_len",DataType::kINT32,dims({1})),sid=input("sid",DataType::kINT32,dims({1}));
        auto rnd=input("rnd",DataType::kFLOAT,dims({1,c->latent,-1}));
        T pitch=nullptr,pitchf=nullptr,noise=nullptr,mask=nullptr;
        if(c->uses_f0){int up=1;for(int i=0;i<c->upsample_count;i++)up*=c->rates[i];pitch=input("pitch",DataType::kINT32,dims({1,-1}));pitchf=input("pitchf",DataType::kFLOAT,dims({1,-1}));noise=input("nsf_noise",DataType::kFLOAT,dims({1,1,-1}),up);}
        // Keep encoder/flow in FP32: their normalization, affine coupling,
        // and exponential/log path are accuracy-sensitive. The selected
        // low-precision type is enabled only for the NSF/HiFi-GAN decoder
        // below, where convolution arithmetic dominates runtime.
        auto feats_internal=g.cast_real(feats);
        auto rnd_internal=g.cast_real(rnd);
        auto z=g.encoder(feats_internal,pitch,length,rnd_internal,mask);
        auto speaker=g.reshape(g.gather(g.parameter("emb_g.weight",dims({c->speakers,c->speaker_channels})),sid,0),dims({1,c->speaker_channels,1}));
        z=g.flow(z,mask,speaker);
        if(options->precision!=0) {
            g.use_low_precision();
            z=g.cast_real(z);
            speaker=g.cast_real(speaker);
            // Keep F0 and deterministic NSF noise in FP32. The decoder owns
            // any explicit cast needed for its low-precision convolutions;
            // quantizing pitch before source synthesis amplifies phase error.
        }
        auto audio=g.decoder(z,speaker,pitchf,noise);
        if(options->precision!=0) audio=g.cast(audio,DataType::kFLOAT);
        g.mark(audio,"audio");
        require(g.used.size()==count,"not every canonical weight was consumed");
        require(bc->addOptimizationProfile(profile)>=0,"addOptimizationProfile");
        std::fprintf(stderr,"[RVC] %d native layers, %zu weights, F0=%d, fused_hubert_qkv=%d, fused_rvc_qkv=%d, dispatch=%d\n",
                     n->getNbLayers(),g.used.size(),c->uses_f0,options->fused_qkv,
                     options->fused_rvc_qkv,options->dispatch_tune);
        auto result=new Build();
        result->mem=b->buildSerializedNetwork(*n,*bc);
        require(result->mem,"buildSerializedNetwork RVC failed");
        *out=result;
    });
}
}
