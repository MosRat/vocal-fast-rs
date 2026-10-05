#include "layers.h"
namespace vocal {
T Layers::source(T pitchf,T noise) {
    int up=1;for(int i=0;i<c.upsample_count;i++)up*=c.rates[i];
    // RVC _f02sine: accumulate centered frame phase, then expand intra-frame
    // sample positions. Fundamental has zero random initial phase.
    auto f=reshape(pitchf,dims({1,-1,1}));
    auto cycle=scale(f,static_cast<float>(up)/c.sample_rate);
    auto centered=sub(cycle,unary(add(cycle,scalar(0.5f,3)),UnaryOperation::kFLOOR));
    auto phase=cumulative(centered,1,true);
    phase=sub(phase,unary(phase,UnaryOperation::kFLOOR));
    std::vector<float> offsets(up);for(int i=0;i<up;i++)offsets[i]=static_cast<float>(i+1)/c.sample_rate;
    auto rad=add(mul(f,constant(offsets,dims({1,1,up}))),phase);
    auto sine=scale(unary(scale(reshape(rad,dims({1,1,-1})),6.283185307179586f),UnaryOperation::kSIN),0.1f);
    auto uv=resize(cast(ew(reshape(pitchf,dims({1,1,-1})),scalar(0,3),ElementWiseOperation::kGREATER),real_type()),up);
    auto amplitude=add(scale(uv,0.003f),scale(sub(scalar(1,3),uv),0.1f/3));
    auto source=add(mul(sine,uv),mul(noise,amplitude));
    source=act(add(mul(source,parameter("dec.m_source.l_linear.weight",dims({1,1,1}))),parameter("dec.m_source.l_linear.bias",dims({1,1,1}))),ActivationType::kTANH);
    if(o.debug_outputs)mark(source,"nsf_source");return source;
}
T Layers::decoder(T z,T speaker,T pitchf,T noise) {
    // Keep the input projection and speaker conditioning in FP32, then cast
    // once at the decoder boundary. The large upsample path remains low
    // precision while the boundary avoids an unnecessary half round-trip.
    DataType decoder_type=real_type();
    bool full_fp16=decoder_type==DataType::kHALF && o.synth_full_fp16!=0;
    auto fp32_stage=[&](int stage) {
        return decoder_type==DataType::kHALF &&
            ((static_cast<uint32_t>(o.synth_fp16_fp32_stage_mask) >> stage) & 1u) != 0;
    };
    auto coherent_stage=[&](int stage) {
        return decoder_type==DataType::kHALF && !fp32_stage(stage) &&
            ((static_cast<uint32_t>(o.synth_fp16_coherent_stage_mask) >> stage) & 1u) != 0;
    };
    // Keep the large pre-projection activation in a compact FP16 range.  The
    // inverse factor is folded into the first linear upsample kernel below,
    // so the transform is exact before FP16 rounding (bias is not scaled).
    float decoder_input_scale=1.0f;
    T x=nullptr;
    if(decoder_type!=DataType::kFLOAT) {
        if(full_fp16) {
            active_type=decoder_type;
            x=add(conv(z,"dec.conv_pre"),conv(speaker,"dec.cond"));
        } else {
            auto z_fp=cast(z,DataType::kFLOAT);
            auto speaker_fp=cast(speaker,DataType::kFLOAT);
            active_type=DataType::kFLOAT;
            x=add(conv(z_fp,"dec.conv_pre"),conv(speaker_fp,"dec.cond"));
        }
        active_type=decoder_type;
        if (decoder_type==DataType::kHALF && !full_fp16 &&
            o.synth_fp16_input_scale!=1.0f) {
            decoder_input_scale=o.synth_fp16_input_scale;
            active_type=DataType::kFLOAT;
            x=scale(x,decoder_input_scale);
            active_type=decoder_type;
        }
        if(x->getType()!=decoder_type) x=cast(x,decoder_type);
    } else {
        x=add(conv(z,"dec.conv_pre"),conv(speaker,"dec.cond"));
    }
    if(o.debug_outputs)mark(x,"decoder_pre");
    T harmonic=nullptr;
    T harmonic_low=nullptr;
    bool any_noise_low=decoder_type==DataType::kHALF &&
        (full_fp16 || o.synth_fp16_noise!=0 || o.synth_fp16_noise_stage_mask!=0 ||
         o.synth_fp16_coherent_stage_mask!=0);
    if(c.uses_f0) {
        if(decoder_type!=DataType::kFLOAT && o.synth_fp16_source!=0) {
            active_type=decoder_type;
            harmonic=source(cast(pitchf,decoder_type),cast(noise,decoder_type));
        } else if(decoder_type!=DataType::kFLOAT) {
            auto pitchf_fp=cast(pitchf,DataType::kFLOAT);
            auto noise_fp=cast(noise,DataType::kFLOAT);
            active_type=DataType::kFLOAT;
            harmonic=source(pitchf_fp,noise_fp);
        } else {
            harmonic=source(pitchf,noise);
        }
        if(decoder_type!=DataType::kFLOAT) active_type=decoder_type;
        // Keep phase/source generation in FP32 by default, but allow the
        // learned noise convolutions to consume a single decoder-domain copy.
        // The default FP32 noise path keeps the source in FP32 directly; this
        // removes the previous source FP32 -> FP16 -> FP32 round trip as well
        // as the per-stage conversion of the same tensor.
        if(any_noise_low && harmonic->getType()!=decoder_type) {
            active_type=decoder_type;
            harmonic_low=cast(harmonic,decoder_type);
        } else if(any_noise_low) {
            harmonic_low=harmonic;
        }
    }
    auto precision_type=[&](int code)->DataType {
        return code==1?DataType::kHALF:code==2?DataType::kBF16:DataType::kFLOAT;
    };
    bool fuse_residual_scale=o.synth_fuse_resblock_scale!=0;
    float residual_scale=1.0f/static_cast<float>(c.resblock_count);
    for(int i=0;i<c.upsample_count;i++) {
        bool keep_fp32=fp32_stage(i);
        if(keep_fp32) {
            active_type=DataType::kFLOAT;
            if(x->getType()!=DataType::kFLOAT) x=cast(x,DataType::kFLOAT);
        } else if(decoder_type!=DataType::kFLOAT && x->getType()!=decoder_type) {
            active_type=decoder_type;
            x=cast(x,decoder_type);
        }
        float input_scale=fuse_residual_scale && i>0?residual_scale:1.0f;
        if (i==0) input_scale/=decoder_input_scale;
        auto up_prefix="dec.ups."+std::to_string(i);
        auto up_input=act(x,ActivationType::kLEAKY_RELU,0.1f);
        if(o.synth_polyphase_upsample)
            x=polyphase_deconv(up_input,up_prefix,c.rates[i],(c.kernels[i]-c.rates[i])/2,input_scale);
        else
            x=deconv(up_input,up_prefix,c.rates[i],(c.kernels[i]-c.rates[i])/2,input_scale);
        if(o.debug_outputs)mark(x,("decoder_up_"+std::to_string(i)).c_str());
        if(harmonic) {
            int tail=1;for(int j=i+1;j<c.upsample_count;j++)tail*=c.rates[j];
            bool noise_low=decoder_type==DataType::kHALF && !keep_fp32 &&
                (full_fp16 || o.synth_fp16_noise!=0 ||
                 coherent_stage(i) ||
                 ((o.synth_fp16_noise_stage_mask >> i) & 1) != 0);
            if(noise_low) {
                active_type=decoder_type;
                x=add(x,conv(harmonic_low,"dec.noise_convs."+std::to_string(i),tail,1,i+1==c.upsample_count?0:tail/2));
            } else if(decoder_type!=DataType::kFLOAT) {
                active_type=DataType::kFLOAT;
                x=add(cast(x,DataType::kFLOAT),conv(harmonic,"dec.noise_convs."+std::to_string(i),tail,1,i+1==c.upsample_count?0:tail/2));
                if(!keep_fp32) {
                    active_type=decoder_type;
                    x=cast(x,decoder_type);
                }
            } else {
                x=add(x,conv(harmonic,"dec.noise_convs."+std::to_string(i),tail,1,i+1==c.upsample_count?0:tail/2));
            }
            if(o.debug_outputs)mark(x,("decoder_noise_"+std::to_string(i)).c_str());
        }
        // Residual blocks are the largest repeated decoder workload.  The
        // stable path evaluates them in FP32, but an explicit low-precision
        // mode keeps x in the decoder type and avoids two cast kernels per
        // stage. Noise convolution can independently use the decoder type via
        // the stage mask while source phase arithmetic remains FP32.
        bool residual_low=!keep_fp32 && decoder_type!=DataType::kFLOAT && o.synth_resblock_precision!=0 &&
            ((o.synth_resblock_stage_mask >> i) & 1) != 0;
        bool coherent_low=coherent_stage(i);
        bool residual_compute_low=!keep_fp32 && (full_fp16 || residual_low || coherent_low);
        bool residual_accum_low=!keep_fp32 &&
            ((full_fp16 && o.synth_fp16_residual_accum!=0) ||
             residual_low || coherent_low);
        DataType accum_type=residual_accum_low?decoder_type:DataType::kFLOAT;
        if(accum_type==DataType::kFLOAT) {
            if(decoder_type!=DataType::kFLOAT) {
                active_type=DataType::kFLOAT;
                x=cast(x,DataType::kFLOAT);
            }
        } else {
            active_type=decoder_type;
        }
        auto operation_type=[&](int precision,int mask)->DataType {
            if(decoder_type==DataType::kFLOAT) return DataType::kFLOAT;
            if(residual_compute_low) return decoder_type;
            if(precision!=0 && ((mask>>i)&1)!=0) return precision_type(precision);
            return accum_type;
        };
        DataType conv1_type=operation_type(o.synth_resblock_conv1_precision,
                                           o.synth_resblock_conv1_stage_mask);
        DataType conv2_type=operation_type(o.synth_resblock_conv2_precision,
                                           o.synth_resblock_conv2_stage_mask);
        T sum=nullptr;
        for(int j=0;j<c.resblock_count;j++) {
            auto p="dec.resblocks."+std::to_string(i*c.resblock_count+j);auto y=x;
            int count=c.resblock_type==1?3:2;
            for(int k=0;k<count;k++) {
                auto suffix=std::to_string(k);
                int const dilation_index=j*(c.resblock_type==1?3:2)+k;
                auto a=y;
                auto conv1_name=p+(c.resblock_type==1?".convs1.":".convs.")+suffix;
                // A selected W8A8 node must live in the decoder's low
                // precision domain.  Previously the default FP32 residual
                // accumulator won here, so the graph became
                //   FP32 -> Q/DQ -> FP32 convolution
                // and TensorRT had no typed low-precision convolution to
                // which it could apply an INT8 tactic.  Keep the residual
                // sum in FP32, but make the selected convolution itself
                // explicitly FP16 (the W8A8 path is currently FP16-only).
                auto current_conv1_type=w8a8_selected(conv1_name)?decoder_type:conv1_type;
                if(a->getType()!=current_conv1_type) a=cast(a,current_conv1_type);
                active_type=current_conv1_type;
                a=act(a,ActivationType::kLEAKY_RELU,0.1f);
                a=conv(a,conv1_name,1,c.res_dilations[dilation_index]);
                if(c.resblock_type==1) {
                    // The validated W8A16 mask is applied only to selected
                    // Conv2 nodes.  Keep the surrounding residual sum in
                    // FP32 unless the existing stage precision explicitly
                    // requests low-precision accumulation.
                    auto conv2_name=p+".convs2."+suffix;
                    auto current_conv2_type=w8a8_selected(conv2_name)?decoder_type:
                        (w8a16_selected(conv2_name)?DataType::kHALF:conv2_type);
                    if(a->getType()!=current_conv2_type) a=cast(a,current_conv2_type);
                    active_type=current_conv2_type;
                    a=conv(act(a,ActivationType::kLEAKY_RELU,0.1f),p+".convs2."+suffix);
                }
                if(a->getType()!=accum_type) a=cast(a,accum_type);
                active_type=accum_type;
                y=add(y,a);
            }
            sum=sum?add(sum,y):y;
        }
        if(fuse_residual_scale) {
            x=sum;
        } else {
            active_type=accum_type;
            x=scale(sum,residual_scale);
        }
        if(accum_type==DataType::kFLOAT &&
           decoder_type!=DataType::kFLOAT &&
           !keep_fp32) {
            active_type=decoder_type;
            x=cast(x,decoder_type);
        }
        if(o.debug_outputs)mark(x,("decoder_res_"+std::to_string(i)).c_str());
    }
    // Applio's generator calls F.leaky_relu(x) here, whose PyTorch default
    // negative slope is 0.01. The loop above deliberately uses 0.1.
    auto post_input=act(x,ActivationType::kLEAKY_RELU,0.01f);
    if(o.debug_outputs) mark(post_input,"decoder_post_input");
    T pre=nullptr;
    if(decoder_type!=DataType::kFLOAT) {
        if(full_fp16) {
            active_type=decoder_type;
            pre=conv(post_input,"dec.conv_post",1,1,-1,false,
                     fuse_residual_scale?residual_scale:1.0f);
        } else if(o.synth_fp16_post!=0) {
            active_type=decoder_type;
            pre=conv(post_input,"dec.conv_post",1,1,-1,false,
                     fuse_residual_scale?residual_scale:1.0f);
            // Keep the nonlinear output boundary in FP32 even when the
            // small final projection is selected for FP16 execution.
            active_type=DataType::kFLOAT;
            pre=cast(pre,DataType::kFLOAT);
        } else {
            active_type=DataType::kFLOAT;
            pre=conv(cast(post_input,DataType::kFLOAT),"dec.conv_post",1,1,-1,false,
                     fuse_residual_scale?residual_scale:1.0f);
            active_type=decoder_type;
        }
    } else {
        pre=conv(post_input,"dec.conv_post",1,1,-1,false,
                 fuse_residual_scale?residual_scale:1.0f);
    }
    if(o.debug_outputs)mark(pre,"audio_pre_tanh");
    if(decoder_type!=DataType::kFLOAT) {
        if(full_fp16) {
            active_type=decoder_type;
            return act(pre,ActivationType::kTANH);
        } else {
            active_type=DataType::kFLOAT;
            auto audio=act(pre,ActivationType::kTANH);
            active_type=decoder_type;
            return audio;
        }
    }
    return act(pre,ActivationType::kTANH);
}
}
