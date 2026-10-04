#include "layers.h"
#include <cstdio>
#include <fstream>
#include <sstream>
namespace vocal {
extern "C" int vocal_trt_build_rvc(Config const* c,Weight const* weights,size_t count,Options const* options,Build** out,char** error) noexcept {
    return guarded(error,[&]{
        require(c&&weights&&options&&out&&c->abi_version==2,"invalid RVC build ABI");
        require(options->min_frames>0&&options->min_frames<=options->opt_frames&&options->opt_frames<=options->max_frames,"invalid frame profile");
        require(options->synth_resblock_stage_mask>=0,"invalid Synth residual stage mask");
        require(options->tactic_sources>=0 && options->tactic_sources<=3,
                "invalid TensorRT tactic source mask");
        require(options->synth_resblock_conv1_precision>=0 && options->synth_resblock_conv1_precision<=2,
                "invalid Synth conv1 precision");
        require(options->synth_resblock_conv2_precision>=0 && options->synth_resblock_conv2_precision<=2,
                "invalid Synth conv2 precision");
        require(options->synth_full_fp16==0 || options->synth_full_fp16==1,
                "invalid Synth full FP16 option");
        require(options->synth_fp16_residual_accum==0 || options->synth_fp16_residual_accum==1,
                "invalid Synth FP16 residual accumulation option");
        require(options->synth_fp16_source==0 || options->synth_fp16_source==1,
                "invalid Synth FP16 source option");
        require(options->synth_fp16_noise==0 || options->synth_fp16_noise==1,
                "invalid Synth FP16 noise option");
        require(options->synth_fp16_noise_stage_mask>=0,
                "invalid Synth FP16 noise stage mask");
        require(options->synth_fp16_post==0 || options->synth_fp16_post==1,
                "invalid Synth FP16 post option");
        require(std::isfinite(options->synth_fp16_input_scale) &&
                    options->synth_fp16_input_scale>0.0f,
                "invalid Synth FP16 input scale");
        if(options->synth_full_fp16 || options->synth_fp16_residual_accum ||
           options->synth_fp16_source || options->synth_fp16_noise ||
           options->synth_fp16_post) {
            require(options->precision==1,
                    "Synth FP16 decoder options require FP16 Synth precision");
        }
        if(options->synth_fp16_noise_stage_mask!=0 && options->precision!=1)
            require(false,"Synth FP16 noise stage mask requires FP16 Synth precision");
        require(options->synth_fp16_residual_accum==0 || options->synth_full_fp16!=0,
                "Synth FP16 residual accumulation requires full FP16 decoder");
        if(options->synth_w8a16_conv2_mask!=0) {
            require(options->precision==1,
                    "Synth W8A16 requires Synth FP16 activation precision");
            require(options->quantize_weights==0,
                    "Synth W8A16 mask cannot be combined with global weight-only INT8");
        }
        auto w8a8_mask=options->synth_w8a8_conv1_mask|options->synth_w8a8_conv2_mask;
        if(w8a8_mask!=0) {
            require(options->precision==1,
                    "Synth W8A8 requires Synth FP16 activation precision");
            require(options->quantize_weights==0,
                    "Synth W8A8 mask cannot be combined with global weight-only INT8");
            int selected=0; for(auto bits=w8a8_mask;bits;bits>>=1) selected+=static_cast<int>(bits&1u);
            require(selected==1,
                    "Synth W8A8 currently supports exactly one selected Conv1/Conv2 node");
            require(options->synth_w8a8_activation_scale>0.0f &&
                        std::isfinite(options->synth_w8a8_activation_scale),
                    "invalid Synth W8A8 activation scale");
            require((options->synth_w8a16_conv2_mask&options->synth_w8a8_conv2_mask)==0,
                    "Synth W8A8 and W8A16 masks select the same Conv2 node");
        }
        require(options->synth_resblock_conv1_stage_mask>=0 && options->synth_resblock_conv2_stage_mask>=0,
                "invalid Synth residual operation mask");
        uint32_t valid_stage_mask=c->upsample_count>=31?0xffffffffu:((1u<<c->upsample_count)-1u);
        require((static_cast<uint32_t>(options->synth_resblock_stage_mask)&~valid_stage_mask)==0,
                "Synth residual stage mask exceeds upsample stages");
        require((static_cast<uint32_t>(options->synth_fp16_noise_stage_mask)&~valid_stage_mask)==0,
                "Synth FP16 noise stage mask exceeds upsample stages");
        require((static_cast<uint32_t>(options->synth_resblock_conv1_stage_mask)&~valid_stage_mask)==0 &&
                    (static_cast<uint32_t>(options->synth_resblock_conv2_stage_mask)&~valid_stage_mask)==0,
                "Synth residual operation mask exceeds upsample stages");
        if(options->synth_polyphase_upsample) {
            for(int i=0;i<c->upsample_count;++i) {
                require(c->rates[i]>0 && c->kernels[i]>=c->rates[i] &&
                            (c->kernels[i]-c->rates[i])%2==0,
                        "Synth polyphase upsample requires symmetric output-length kernels");
            }
            require(options->quantize_weights==0,
                    "Synth polyphase upsample does not support weight-only INT8");
        }
        auto b=std::unique_ptr<IBuilder>(createInferBuilder(logger()));require(bool(b),"createInferBuilder");
        auto n=std::unique_ptr<INetworkDefinition>(b->createNetworkV2(0));require(bool(n),"createNetwork");
        auto bc=std::unique_ptr<IBuilderConfig>(b->createBuilderConfig());require(bool(bc),"builder config");
        bc->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE,options->workspace);
        bc->clearFlag(BuilderFlag::kTF32); // keep the baseline deterministic
        require(bc->setTacticSources(static_cast<TacticSources>(options->tactic_sources)),
                "set TensorRT tactic sources failed");
        bc->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);
        // TensorRT-RTX 1.6 exposes the standard timing-cache ABI but marks it
        // deprecated/no-op. Keep the file contract anyway: on SDKs that
        // implement it this persists fixed-profile tactics, while RTX 1.6
        // safely reports an empty cache and continues with normal selection.
        std::unique_ptr<ITimingCache> timing_cache;
        std::string timing_cache_path;
        if(auto const* path=options->timing_cache_path;path && path[0]!='\0') {
            timing_cache_path=path;
            std::vector<unsigned char> blob;
            std::ifstream input(timing_cache_path,std::ios::binary);
            if(input) {
                input.seekg(0,std::ios::end);
                auto size=input.tellg();
                require(size>=0,"timing cache seek failed");
                blob.resize(static_cast<size_t>(size));
                input.seekg(0,std::ios::beg);
                if(!blob.empty()) input.read(reinterpret_cast<char*>(blob.data()),static_cast<std::streamsize>(blob.size()));
                require(input.good() || input.eof(),"timing cache read failed");
            }
            timing_cache.reset(bc->createTimingCache(blob.empty()?nullptr:blob.data(),blob.size()));
            require(bool(timing_cache),"create timing cache failed");
            require(bc->setTimingCache(*timing_cache,true),"attach timing cache failed");
        }
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
        // Keep F0 and deterministic NSF noise in FP32 by default. The
        // decoder owns any explicit cast needed for its low-precision
        // convolutions; source FP16 is a separate diagnostic option because
        // phase accumulation is more sensitive than Conv arithmetic.
        }
        auto audio=g.decoder(z,speaker,pitchf,noise);
        if(options->precision!=0) audio=g.cast(audio,DataType::kFLOAT);
        g.mark(audio,"audio");
        require(g.used.size()==count,"not every canonical weight was consumed");
        require(bc->addOptimizationProfile(profile)>=0,"addOptimizationProfile");
        log_message(nvinfer1::ILogger::Severity::kINFO,
                    std::string("[RVC] ") + std::to_string(n->getNbLayers()) +
                    " native layers, " + std::to_string(g.used.size()) +
                    " weights, F0=" + std::to_string(c->uses_f0) +
                    ", synth_full_fp16=" + std::to_string(options->synth_full_fp16) +
                    ", synth_fp16_input_scale=" + std::to_string(options->synth_fp16_input_scale) +
                    ", synth_fp16_residual_accum=" + std::to_string(options->synth_fp16_residual_accum) +
                    ", synth_fp16_source=" + std::to_string(options->synth_fp16_source) +
                    ", synth_fp16_noise=" + std::to_string(options->synth_fp16_noise) +
                    ", synth_fp16_noise_stage_mask=" + std::to_string(options->synth_fp16_noise_stage_mask) +
                    ", synth_fp16_post=" + std::to_string(options->synth_fp16_post) +
                    ", fused_hubert_qkv=" + std::to_string(options->fused_qkv) +
                    ", fused_rvc_qkv=" + std::to_string(options->fused_rvc_qkv) +
                    ", synth_w8a16_conv2_mask=0x" + [&]{
                        std::ostringstream stream; stream << std::hex << options->synth_w8a16_conv2_mask; return stream.str();
                    }() +
                    ", synth_w8a8_conv1_mask=0x" + [&]{
                        std::ostringstream stream; stream << std::hex << options->synth_w8a8_conv1_mask; return stream.str();
                    }() +
                    ", synth_w8a8_conv2_mask=0x" + [&]{
                        std::ostringstream stream; stream << std::hex << options->synth_w8a8_conv2_mask; return stream.str();
                    }() +
                    ", dispatch=" + std::to_string(options->dispatch_tune));
        auto result=new Build();
        result->mem=b->buildSerializedNetwork(*n,*bc);
        require(result->mem,"buildSerializedNetwork RVC failed");
        if(timing_cache) {
            std::unique_ptr<IHostMemory> serialized(timing_cache->serialize());
            if(serialized && serialized->size()>0 && !timing_cache_path.empty()) {
                std::string temporary=timing_cache_path+".tmp";
                std::ofstream output(temporary,std::ios::binary|std::ios::trunc);
                require(bool(output),"open timing cache output failed");
                output.write(static_cast<char const*>(serialized->data()),static_cast<std::streamsize>(serialized->size()));
                require(output.good(),"write timing cache failed");
                output.close();
                std::remove(timing_cache_path.c_str());
                require(std::rename(temporary.c_str(),timing_cache_path.c_str())==0,"replace timing cache failed");
                log_message(nvinfer1::ILogger::Severity::kINFO,
                            std::string("[RVC] timing cache updated path=") +
                            timing_cache_path + " bytes=" + std::to_string(serialized->size()));
            } else if(!timing_cache_path.empty()) {
                log_message(nvinfer1::ILogger::Severity::kWARNING,
                            std::string("[RVC] timing cache API returned no serialized data path=") +
                            timing_cache_path);
            }
        }
        *out=result;
    });
}
}
