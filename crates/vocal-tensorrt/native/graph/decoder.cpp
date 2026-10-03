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
    T x=nullptr;
    if(decoder_type!=DataType::kFLOAT) {
        auto z_fp=cast(z,DataType::kFLOAT);
        auto speaker_fp=cast(speaker,DataType::kFLOAT);
        active_type=DataType::kFLOAT;
        x=add(conv(z_fp,"dec.conv_pre"),conv(speaker_fp,"dec.cond"));
        active_type=decoder_type;
        x=cast(x,decoder_type);
    } else {
        x=add(conv(z,"dec.conv_pre"),conv(speaker,"dec.cond"));
    }
    if(o.debug_outputs)mark(x,"decoder_pre");
    T harmonic=nullptr;
    if(c.uses_f0) {
        if(decoder_type!=DataType::kFLOAT) {
            auto pitchf_fp=cast(pitchf,DataType::kFLOAT);
            auto noise_fp=cast(noise,DataType::kFLOAT);
            active_type=DataType::kFLOAT;
            harmonic=source(pitchf_fp,noise_fp);
            active_type=decoder_type;
            harmonic=cast(harmonic,decoder_type);
        } else {
            harmonic=source(pitchf,noise);
        }
    }
    for(int i=0;i<c.upsample_count;i++) {
        x=deconv(act(x,ActivationType::kLEAKY_RELU,0.1f),"dec.ups."+std::to_string(i),c.rates[i],(c.kernels[i]-c.rates[i])/2);
        if(o.debug_outputs)mark(x,("decoder_up_"+std::to_string(i)).c_str());
        if(harmonic) {
            int tail=1;for(int j=i+1;j<c.upsample_count;j++)tail*=c.rates[j];
            if(decoder_type!=DataType::kFLOAT) {
                active_type=DataType::kFLOAT;
                x=add(cast(x,DataType::kFLOAT),conv(cast(harmonic,DataType::kFLOAT),"dec.noise_convs."+std::to_string(i),tail,1,i+1==c.upsample_count?0:tail/2));
                active_type=decoder_type;
                x=cast(x,decoder_type);
            } else {
                x=add(x,conv(harmonic,"dec.noise_convs."+std::to_string(i),tail,1,i+1==c.upsample_count?0:tail/2));
            }
            if(o.debug_outputs)mark(x,("decoder_noise_"+std::to_string(i)).c_str());
        }
        // Residual blocks are the largest repeated decoder workload.  The
        // stable path evaluates them in FP32, but an explicit low-precision
        // mode keeps x in the decoder type and avoids two cast kernels per
        // stage.  Noise convolution remains FP32 in both modes because its
        // source phase is the most sensitive NSF path.
        bool residual_low=decoder_type!=DataType::kFLOAT && o.synth_resblock_precision!=0 &&
            ((o.synth_resblock_stage_mask >> i) & 1) != 0;
        if(!residual_low) {
            if(decoder_type!=DataType::kFLOAT) {
                active_type=DataType::kFLOAT;
                x=cast(x,DataType::kFLOAT);
            }
        } else {
            active_type=decoder_type;
        }
        T sum=nullptr;
        for(int j=0;j<c.resblock_count;j++) {
            auto p="dec.resblocks."+std::to_string(i*c.resblock_count+j);auto y=x;
            int count=c.resblock_type==1?3:2;
            for(int k=0;k<count;k++) {
                auto a=act(y,ActivationType::kLEAKY_RELU,0.1f);auto suffix=std::to_string(k);
                int const dilation_index=j*(c.resblock_type==1?3:2)+k;
                a=conv(a,p+(c.resblock_type==1?".convs1.":".convs.")+suffix,1,c.res_dilations[dilation_index]);
                if(c.resblock_type==1) a=conv(act(a,ActivationType::kLEAKY_RELU,0.1f),p+".convs2."+suffix);
                y=add(y,a);
            }
            sum=sum?add(sum,y):y;
        }
        x=scale(sum,1.0f/c.resblock_count);
        if(!residual_low && decoder_type!=DataType::kFLOAT) {
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
        active_type=DataType::kFLOAT;
        pre=conv(cast(post_input,DataType::kFLOAT),"dec.conv_post",1,1,-1,false);
        active_type=decoder_type;
    } else {
        pre=conv(post_input,"dec.conv_post",1,1,-1,false);
    }
    if(o.debug_outputs)mark(pre,"audio_pre_tanh");
    if(decoder_type!=DataType::kFLOAT) {
        active_type=DataType::kFLOAT;
        auto audio=act(pre,ActivationType::kTANH);
        active_type=decoder_type;
        return audio;
    }
    return act(pre,ActivationType::kTANH);
}
}
