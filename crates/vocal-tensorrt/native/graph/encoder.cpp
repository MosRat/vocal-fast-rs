#include "layers.h"
namespace vocal {
T Layers::attention(T x,T mask,int layer) {
    auto p="enc_p.encoder.attn_layers."+std::to_string(layer);
    int h=c.heads,d=c.hidden/h,r=2*c.window+1;
    auto heads=[&](char const* k){return transpose(reshape(conv(x,p+".conv_"+k),dims({h,d,-1})),{0,2,1});};
    auto fused_heads=[&](T qkv,int offset){
        return transpose(reshape(slice(qkv,1,offset,h*d),dims({h,d,-1})),{0,2,1});
    };

    // TensorRT's IAttention gives the builder a chance to select its fused
    // scaled-dot-product kernel.  It has no representation for RVC's
    // relative-key/relative-value correction, so only a relative-free graph
    // may opt into this path.  Relative RVC checkpoints always use the exact
    // native implementation below and therefore retain parity with Python.
    if (o.fused_attention && c.window == 0 && c.relative_heads == 0) {
        auto q=reshape(heads("q"),dims({1,h,-1,d}));
        auto k=reshape(heads("k"),dims({1,h,-1,d}));
        auto v=reshape(heads("v"),dims({1,h,-1,d}));
        auto fused=n.addAttentionV2(*q,*k,*v,AttentionNormalizationOp::kSOFTMAX,
                                    CausalMaskKind::kNONE);
        require(fused,"TensorRT IAttention creation failed");
        require(fused->setQueryForm(AttentionIOForm::kPADDED_BHND),
                "set IAttention query form failed");
        require(fused->setKeyValueForm(AttentionIOForm::kPADDED_BHND),
                "set IAttention key/value form failed");
        require(fused->setDecomposable(false),
                "set IAttention fused mode failed");
        fused->setName((p+".fused").c_str());
        auto y4=fused->getOutput(0);
        require(y4,"TensorRT IAttention output missing");
        auto y=transpose(reshape(y4,dims({h,-1,d})),{0,2,1});
        return conv(reshape(y,dims({1,c.hidden,-1})),p+".conv_o");
    }

    T q,k,v;
    if (o.fused_qkv && !o.quantize_weights) {
        auto qkv=fused_qkv_conv(x,p+".conv_",c.hidden);
        q=fused_heads(qkv,0);
        k=fused_heads(qkv,c.hidden);
        v=fused_heads(qkv,2*c.hidden);
    } else {
        q=heads("q");
        k=heads("k");
        v=heads("v");
    }
    q=scale(q,1/std::sqrt(static_cast<float>(d)));
    auto length=dimension(x,2),positions=range(length,o.max_frames);
    auto row=reshape(positions,dims({1,-1,1})),col=reshape(positions,dims({1,1,-1}));
    auto delta=sub(col,row);
    auto zero=integer({0},dims({1,1,1}));
    auto wi=integer({c.window},dims({1,1,1}));
    auto index=add(delta,wi);
    auto valid=ew(ew(index,integer({-1},dims({1,1,1})),ElementWiseOperation::kGREATER),ew(index,integer({r},dims({1,1,1})),ElementWiseOperation::kLESS),ElementWiseOperation::kAND);
    index=ew(ew(index,zero,ElementWiseOperation::kMAX),integer({r-1},dims({1,1,1})),ElementWiseOperation::kMIN);
    index=broadcast(index,concat({iv({h}),length,length},0));
    auto relk=parameter(p+".emb_rel_k",dims({c.relative_heads,r,d}));
    auto local=gather_elements(matmul(q,relk,true),index,2);
    auto scores=add(matmul(q,k,true),select(valid,local,scalar(0,3)));
    auto allowed=mul(transpose(mask,{0,2,1}),mask);
    scores=select(ew(allowed,scalar(0,3),ElementWiseOperation::kGREATER),scores,scalar(-1e4f,3));
    auto sm=n.addSoftMax(*scores);require(sm,"attention softmax");sm->setAxes(1U<<2);auto prob=output(sm,p+".softmax");

    std::vector<int32_t> offsets(r);for(int i=0;i<r;i++)offsets[i]=i-c.window;
    auto columns=add(row,integer(offsets,dims({1,1,r})));
    auto end=reshape(length,dims({1,1,1}));
    auto valid_columns=ew(ew(columns,integer({-1},dims({1,1,1})),ElementWiseOperation::kGREATER),ew(columns,end,ElementWiseOperation::kLESS),ElementWiseOperation::kAND);
    columns=ew(ew(columns,zero,ElementWiseOperation::kMAX),sub(end,integer({1},dims({1,1,1}))),ElementWiseOperation::kMIN);
    columns=broadcast(columns,concat({iv({h}),length,iv({r})},0));
    auto relative_prob=select(valid_columns,gather_elements(prob,columns,2),scalar(0,3));
    auto relv=parameter(p+".emb_rel_v",dims({c.relative_heads,r,d}));
    auto y=add(matmul(prob,v),matmul(relative_prob,relv));
    return conv(reshape(transpose(y,{0,2,1}),dims({1,c.hidden,-1})),p+".conv_o");
}
T Layers::encoder(T feats,T pitch,T lengths,T rnd,T& mask) {
    auto x=add(matmul(feats,parameter("enc_p.emb_phone.weight",dims({1,c.feature_width,c.hidden}))),parameter("enc_p.emb_phone.bias",dims({1,1,c.hidden})));
    if(c.uses_f0) x=add(x,gather(parameter("enc_p.emb_pitch.weight",dims({256,c.hidden})),pitch,0));
    x=transpose(act(scale(x,std::sqrt(static_cast<float>(c.hidden))),ActivationType::kLEAKY_RELU,0.1f),{0,2,1});
    auto positions=reshape(range(dimension(x,2),o.max_frames),dims({1,1,-1}));
    auto length=reshape(lengths,dims({1,1,1}));
    mask=cast(ew(positions,length,ElementWiseOperation::kLESS),real_type());
    x=mul(x,mask);
    for(int i=0;i<c.encoder_layers;i++) {
        auto suffix=std::to_string(i);
        x=norm(add(x,attention(x,mask,i)),"enc_p.encoder.norm_layers_1."+suffix);
        auto p="enc_p.encoder.ffn_layers."+suffix;
        auto y=act(conv(mul(x,mask),p+".conv_1"),ActivationType::kRELU);
        y=mul(conv(mul(y,mask),p+".conv_2"),mask);
        x=norm(add(x,y),"enc_p.encoder.norm_layers_2."+suffix);
    }
    auto stats=mul(conv(mul(x,mask),"enc_p.proj"),mask);
    auto mean=slice(stats,1,0,c.latent),logs=slice(stats,1,c.latent,c.latent);
    auto z=mul(add(mean,scale(mul(unary(logs,UnaryOperation::kEXP),rnd),0.66666f)),mask);
    if(o.debug_outputs) mark(z,"encoder_z");
    return z;
}
}
