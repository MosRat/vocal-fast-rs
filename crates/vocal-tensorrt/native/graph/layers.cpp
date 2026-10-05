#include "layers.h"
#include <cstdio>
namespace vocal {
Layers::Layers(INetworkDefinition& net,Config const& config,Options const& options,Weight const* w,size_t count):n(net),c(config),o(options) {
    for(size_t i=0;i<count;++i) require(weights.emplace(w[i].name,w+i).second,"duplicate native weight");
}
T Layers::output(ILayer* layer,std::string const& name) {
    require(layer!=nullptr,"TensorRT layer creation failed: "+name);
    if(!name.empty()) layer->setName(name.c_str());
    auto t=layer->getOutput(0); require(t!=nullptr,"TensorRT missing output: "+name); return t;
}
Weight const& Layers::weight(std::string const& name) {
    auto it=weights.find(name); require(it!=weights.end(),"missing canonical weight: "+name); used.insert(name); return *it->second;
}
DataType Layers::real_type() const {
    return active_type;
}
T Layers::cast_real(T x) {
    return real_type()==DataType::kFLOAT ? x : cast(x,real_type());
}
void Layers::use_low_precision() {
    active_type=o.precision==1 ? DataType::kHALF : o.precision==2 ? DataType::kBF16 : DataType::kFLOAT;
}
void Layers::use_synth_precision() {
    active_type=o.synth_precision==1 ? DataType::kHALF :
        o.synth_precision==2 ? DataType::kBF16 : DataType::kFLOAT;
}
bool Layers::w8a16_selected(std::string const& name) const {
    if(o.synth_w8a16_conv2_mask==0 || o.synth_precision!=1) return false;
    int block=-1,suffix=-1;
    auto const* text=name.c_str();
    if(std::sscanf(text,"dec.resblocks.%d.convs2.%d",&block,&suffix)!=2) return false;
    if(block<0 || block>=21 || suffix<0 || suffix>=3) return false;
    uint64_t bit=uint64_t{1} << static_cast<unsigned>(block*3+suffix);
    return (o.synth_w8a16_conv2_mask&bit)!=0;
}
bool Layers::w8a8_selected(std::string const& name) const {
    if(o.synth_w8a8_conv1_mask==0 && o.synth_w8a8_conv2_mask==0) return false;
    int block=-1,suffix=-1;
    auto const* text=name.c_str();
    if(std::sscanf(text,"dec.resblocks.%d.convs1.%d",&block,&suffix)==2) {
        if(block<0 || block>=21 || suffix<0 || suffix>=3) return false;
        uint64_t bit=uint64_t{1} << static_cast<unsigned>(block*3+suffix);
        return (o.synth_w8a8_conv1_mask&bit)!=0;
    }
    if(std::sscanf(text,"dec.resblocks.%d.convs2.%d",&block,&suffix)==2) {
        if(block<0 || block>=21 || suffix<0 || suffix>=3) return false;
        uint64_t bit=uint64_t{1} << static_cast<unsigned>(block*3+suffix);
        return (o.synth_w8a8_conv2_mask&bit)!=0;
    }
    return false;
}
T Layers::quantized_activation(T x,std::string const& name) {
    require(o.synth_w8a8_activation_scale>0.0f && std::isfinite(o.synth_w8a8_activation_scale),
            "invalid W8A8 activation scale");
    floats.emplace_back(1, o.synth_w8a8_activation_scale);
    auto& storage=floats.back();
    auto* scale=n.addConstant(dims({1}),{DataType::kFLOAT,storage.data(),1});
    require(scale,"W8A8 activation scale");
    auto* scale_tensor=scale->getOutput(0);
    require(scale_tensor,"W8A8 activation scale output");
    auto* q=n.addQuantize(*x,*scale_tensor,DataType::kINT8);
    require(q,"W8A8 activation quantize");
    auto* q_tensor=output(q,name+".w8a8_q");
    auto* dq=n.addDequantize(*q_tensor,*scale_tensor,real_type());
    require(dq,"W8A8 activation dequantize");
    return output(dq,name+".w8a8_dq");
}
Weights Layers::w(std::string const& name) {
    auto& p=weight(name);
    if(real_type()==DataType::kFLOAT) return {DataType::kFLOAT,p.data,p.count};
    // Preserve an already low-precision VFRM constant all the way into the
    // TensorRT network.  This avoids the previous F16 -> F32 -> F16 host
    // round trip and lets the builder account for the exact stored bits.
    if (p.data16 != nullptr &&
        ((real_type()==DataType::kHALF && p.dtype==4) ||
         (real_type()==DataType::kBF16 && p.dtype==6))) {
        return {real_type(), p.data16, p.count};
    }
    low_constants.emplace_back(static_cast<size_t>(p.count));
    auto& dst=low_constants.back();
    for(int64_t i=0;i<p.count;++i)
        dst[static_cast<size_t>(i)]=real_type()==DataType::kHALF?float_to_half_bits(p.scalar(i)):float_to_bf16_bits(p.scalar(i));
    return {real_type(),dst.data(),p.count};
}
Weights Layers::w_scaled(std::string const& name,float scale) {
    if(scale==1.0f) return w(name);
    auto& p=weight(name);
    require(!o.quantize_weights,"scaled Synth weights are incompatible with weight-only INT8");
    floats.emplace_back(static_cast<size_t>(p.count));
    auto& values=floats.back();
    for(int64_t i=0;i<p.count;++i) values[static_cast<size_t>(i)]=p.scalar(i)*scale;
    if(real_type()==DataType::kFLOAT) return {DataType::kFLOAT,values.data(),p.count};
    low_constants.emplace_back(static_cast<size_t>(p.count));
    auto& dst=low_constants.back();
    for(int64_t i=0;i<p.count;++i)
        dst[static_cast<size_t>(i)]=real_type()==DataType::kHALF?float_to_half_bits(values[static_cast<size_t>(i)]):float_to_bf16_bits(values[static_cast<size_t>(i)]);
    return {real_type(),dst.data(),p.count};
}
T Layers::quantized_parameter(std::string const& name,Dims d,int axis) {
    auto& p=weight(name);
    // Keep biases, normalization affine terms and scalar lookup tables in
    // floating point. They are bandwidth-light and quantizing them makes
    // residual/normalization error accumulate without an INT8 MAC benefit.
    bool eligible=(o.quantize_weights || w8a16_selected(name) || w8a8_selected(name)) && p.rank>=2 &&
        name.find(".bias")==std::string::npos &&
        name.find(".gamma")==std::string::npos &&
        name.find(".beta")==std::string::npos;
    if(!eligible) return output(n.addConstant(d,w(name)),name);
    axis=axis<0?0:axis;
    require(axis<d.nbDims,"INT8 weight axis");
    int channels=static_cast<int>(d.d[axis]);
    require(channels>0,"INT8 weight channel dimension");
    std::vector<float> scales(static_cast<size_t>(channels),0.0f);
    int64_t inner=1;
    for(int i=axis+1;i<d.nbDims;++i) inner*=d.d[i];
    for(int64_t i=0;i<p.count;++i) {
        int ch=static_cast<int>((i/inner)%channels);
        scales[static_cast<size_t>(ch)]=std::max(scales[static_cast<size_t>(ch)],std::fabs(p.data[i]));
    }
    for(float& s:scales) s=std::max(s/127.0f,1.0e-8f);
    quantized.emplace_back(static_cast<size_t>(p.count));
    auto& q=quantized.back();
    for(int64_t i=0;i<p.count;++i) {
        int ch=static_cast<int>((i/inner)%channels);
        float v=std::nearbyint(p.data[i]/scales[static_cast<size_t>(ch)]);
        v=std::max(-127.0f,std::min(127.0f,v));
        q[static_cast<size_t>(i)]=static_cast<int8_t>(v);
    }
    auto* raw_layer=n.addConstant(d,{DataType::kINT8,q.data(),p.count});
    require(raw_layer,"INT8 weight constant");
    floats.push_back(std::move(scales));
    auto& scale_storage=floats.back();
    auto* scale_layer=n.addConstant(dims({channels}),{DataType::kFLOAT,scale_storage.data(),channels});
    require(scale_layer,"INT8 weight scale");
    auto* raw=output(raw_layer,name+".int8_q");
    auto* scale_tensor=output(scale_layer,name+".int8_scale");
    auto* dq=n.addDequantize(*raw,*scale_tensor,real_type());
    require(dq,"INT8 weight dequantize");
    dq->setAxis(axis);
    return output(dq,name+".int8_dq");
}
T Layers::constant(std::vector<float> data,Dims d) {
    if(real_type()==DataType::kFLOAT) {
        floats.push_back(std::move(data)); auto& v=floats.back();
        return output(n.addConstant(d,{DataType::kFLOAT,v.data(),static_cast<int64_t>(v.size())}));
    }
    low_constants.emplace_back(data.size()); auto& v=low_constants.back();
    for(size_t i=0;i<data.size();++i)
        v[i]=real_type()==DataType::kHALF?float_to_half_bits(data[i]):float_to_bf16_bits(data[i]);
    return output(n.addConstant(d,{real_type(),v.data(),static_cast<int64_t>(v.size())}));
}
T Layers::integer(std::vector<int32_t> data,Dims d) { ints.push_back(std::move(data)); auto& v=ints.back(); return output(n.addConstant(d,{DataType::kINT32,v.data(),static_cast<int64_t>(v.size())})); }
T Layers::iv(std::initializer_list<int32_t> v) { return integer(v,dims({static_cast<int64_t>(v.size())})); }
T Layers::scalar(float v,int rank) { Dims d{};d.nbDims=rank;std::fill(d.d,d.d+rank,1);return constant({v},d); }
T Layers::parameter(std::string const& name,Dims d) {
    int axis=(d.nbDims>=3 && d.d[0]==1)?d.nbDims-1:0;
    return quantized_parameter(name,d,axis);
}
T Layers::reshape(T x,Dims d) { auto l=n.addShuffle(*x);require(l,"shuffle");l->setReshapeDimensions(d);return output(l); }
T Layers::transpose(T x,std::initializer_list<int> perm) { auto l=n.addShuffle(*x);require(l,"transpose");Permutation p{};std::copy(perm.begin(),perm.end(),p.order);l->setFirstTranspose(p);return output(l); }
T Layers::ew(T a,T b,ElementWiseOperation op) { return output(n.addElementWise(*a,*b,op)); }
T Layers::act(T x,ActivationType type,float alpha) { auto l=n.addActivation(*x,type);require(l,"activation");l->setAlpha(alpha);return output(l); }
T Layers::gather(T x,T indices,int axis) { return output(n.addGather(*x,*indices,axis)); }
T Layers::gather_elements(T x,T indices,int axis) { auto l=n.addGatherV2(*x,*indices,GatherMode::kELEMENT);require(l,"gather elements");l->setGatherAxis(axis);return output(l); }
T Layers::dimension(T x,int axis) {
    // TensorRT may expose shape tensors as INT64 even when the source input
    // uses INT32.  Normalize all graph index arithmetic to INT32.
    return cast(gather(shape(x),iv({axis}),0),DataType::kINT32);
}
T Layers::concat(std::vector<T> x,int axis) { auto l=n.addConcatenation(x.data(),static_cast<int>(x.size()));require(l,"concat");l->setAxis(axis);return output(l); }
T Layers::slice(T x,int axis,int start,int count,int stride) {
    std::vector<int32_t> indices(count); for(int i=0;i<count;++i) indices[i]=start+i*stride;
    return gather(x,integer(indices,dims({count})),axis);
}
T Layers::range(T length,int maximum) {
    std::vector<int32_t> v(maximum);for(int i=0;i<maximum;++i)v[i]=i;
    auto l=n.addSlice(*integer(v,dims({maximum})),dims({0}),dims({1}),dims({1}));require(l,"range slice");l->setInput(2,*length);return output(l);
}
T Layers::matmul(T a,T b,bool tb) { return output(n.addMatrixMultiply(*a,MatrixOperation::kNONE,*b,tb?MatrixOperation::kTRANSPOSE:MatrixOperation::kNONE)); }
T Layers::conv(T x,std::string const& p,int stride,int dilation,int padding,bool bias,float weight_scale) {
    auto& k=weight(p+".weight");require(k.rank==3,"conv rank "+p);int channels=static_cast<int>(k.dims[0]),kernel=static_cast<int>(k.dims[2]);
    // TensorRT accepts an empty Weights object for a bias-less convolution,
    // but TensorRT-RTX has produced unstable results for that form in some
    // tactics (notably the final 1x7 Synth projection).  Materialize an
    // explicit zero bias so every tactic has the same well-defined affine
    // operation as PyTorch Conv1d(bias=False).
    Weights bias_weights{};
    if (bias) {
        bias_weights=w(p+".bias");
    } else {
        floats.emplace_back(static_cast<size_t>(channels),0.0f);
        auto& zeros=floats.back();
        if (real_type()==DataType::kFLOAT) {
            bias_weights={DataType::kFLOAT,zeros.data(),channels};
        } else {
            low_constants.emplace_back(static_cast<size_t>(channels));
            auto& low=low_constants.back();
            for (int i=0;i<channels;++i)
                low[static_cast<size_t>(i)]=real_type()==DataType::kHALF?float_to_half_bits(0.0f):float_to_bf16_bits(0.0f);
            bias_weights={real_type(),low.data(),channels};
        }
    }
    bool const w8a8=w8a8_selected(p);
    if(w8a8) {
        require(real_type()==DataType::kHALF,
                "W8A8 selected convolution requires FP16 decoder precision: "+p);
        require(x->getType()==DataType::kHALF,
                "W8A8 selected convolution entered with non-FP16 activation: "+p);
        x=quantized_activation(x,p);
    }
    auto x4=reshape(x,dims({1,x->getDimensions().d[1],1,-1}));
    IConvolutionLayer* l=nullptr;
    if((o.quantize_weights || w8a16_selected(p) || w8a8_selected(p)) && weight_scale==1.0f) {
        auto kernel_tensor=quantized_parameter(p+".weight",dims({channels,k.dims[1],1,kernel}),0);
        l=n.addConvolutionNd(*x4,channels,dims({1,kernel}),Weights{real_type(),nullptr,0},bias_weights);
        require(l,"conv "+p); l->setInput(1,*kernel_tensor);
    } else {
        // Folding a preceding residual scale into a convolution scales only
        // the kernel.  The affine bias is applied after the convolution and
        // must remain unchanged: W*(s*x)+b == (s*W)*x+b.
        Weights kernel_weights=w_scaled(p+".weight",weight_scale);
        l=n.addConvolutionNd(*x4,channels,dims({1,kernel}),kernel_weights,bias_weights);
        require(l,"conv "+p);
    }
    l->setStrideNd(dims({1,stride}));l->setDilationNd(dims({1,dilation}));l->setPaddingNd(dims({0,padding<0?(kernel-1)*dilation/2:padding}));
    if(w8a8) {
        require(l->getOutput(0)->getType()==DataType::kHALF,
                "W8A8 selected convolution did not infer FP16 output: "+p);
    }
    return reshape(output(l,p),dims({1,channels,-1}));
}
T Layers::fused_qkv_conv(T x,std::string const& p,int channels) {
    auto const& q=weight(p+"q.weight");
    auto const& k=weight(p+"k.weight");
    auto const& v=weight(p+"v.weight");
    require(q.rank==3 && k.rank==3 && v.rank==3,"QKV Conv1d rank");
    require(q.dims[0]==channels && k.dims[0]==channels && v.dims[0]==channels,
            "QKV Conv1d output channel layout: " + std::to_string(q.dims[0]) + "," +
                std::to_string(k.dims[0]) + "," + std::to_string(v.dims[0]) +
                " expected " + std::to_string(channels));
    require(q.dims[1]==k.dims[1] && q.dims[1]==v.dims[1],
            "QKV Conv1d input channel layout");
    require(q.dims[2]==k.dims[2] && q.dims[2]==v.dims[2],
            "QKV Conv1d kernel layout");
    int in=static_cast<int>(q.dims[1]), kernel=static_cast<int>(q.dims[2]);
    auto plane=static_cast<size_t>(in)*kernel;
    std::vector<float> kernel_values(static_cast<size_t>(3*channels)*plane);
    std::copy(q.data,q.data+q.count,kernel_values.data());
    std::copy(k.data,k.data+k.count,kernel_values.data()+q.count);
    std::copy(v.data,v.data+v.count,kernel_values.data()+q.count+k.count);
    auto const& qb=weight(p+"q.bias");
    auto const& kb=weight(p+"k.bias");
    auto const& vb=weight(p+"v.bias");
    require(qb.rank==1 && kb.rank==1 && vb.rank==1 && qb.count==channels &&
            kb.count==channels && vb.count==channels,"QKV Conv1d bias layout");
    std::vector<float> bias(static_cast<size_t>(3*channels));
    std::copy(qb.data,qb.data+channels,bias.data());
    std::copy(kb.data,kb.data+channels,bias.data()+channels);
    std::copy(vb.data,vb.data+channels,bias.data()+2*channels);
    floats.push_back(std::move(kernel_values));
    auto& stable_kernel=floats.back();
    floats.push_back(std::move(bias));
    auto& stable_bias=floats.back();
    Weights fused_weights;
    Weights fused_bias;
    if(real_type()==DataType::kFLOAT) {
        fused_weights={DataType::kFLOAT,stable_kernel.data(),static_cast<int64_t>(stable_kernel.size())};
        fused_bias={DataType::kFLOAT,stable_bias.data(),static_cast<int64_t>(stable_bias.size())};
    } else {
        low_constants.emplace_back(stable_kernel.size()); auto& low_kernel=low_constants.back();
        for(size_t i=0;i<stable_kernel.size();++i)
            low_kernel[i]=real_type()==DataType::kHALF?float_to_half_bits(stable_kernel[i]):float_to_bf16_bits(stable_kernel[i]);
        low_constants.emplace_back(stable_bias.size()); auto& low_bias=low_constants.back();
        for(size_t i=0;i<stable_bias.size();++i)
            low_bias[i]=real_type()==DataType::kHALF?float_to_half_bits(stable_bias[i]):float_to_bf16_bits(stable_bias[i]);
        fused_weights={real_type(),low_kernel.data(),static_cast<int64_t>(low_kernel.size())};
        fused_bias={real_type(),low_bias.data(),static_cast<int64_t>(low_bias.size())};
    }
    auto* layer=n.addConvolutionNd(*reshape(x,dims({1,x->getDimensions().d[1],1,-1})),
        3*channels,dims({1,kernel}),
        fused_weights,fused_bias);
    require(layer,"fused QKV Conv1d");
    layer->setPaddingNd(dims({0,(kernel-1)/2}));
    return reshape(output(layer,p+"qkv"),dims({1,3*channels,-1}));
}
T Layers::deconv(T x,std::string const& p,int stride,int padding,float weight_scale) {
    auto& k=weight(p+".weight");int channels=static_cast<int>(k.dims[1]);
    IDeconvolutionLayer* l=nullptr;
    if(o.quantize_weights && weight_scale==1.0f) {
        auto kernel_tensor=quantized_parameter(p+".weight",dims({k.dims[0],k.dims[1],1,k.dims[2]}),1);
        l=n.addDeconvolutionNd(*reshape(x,dims({1,k.dims[0],1,-1})),channels,dims({1,k.dims[2]}),Weights{real_type(),nullptr,0},w(p+".bias"));
        require(l,"deconv "+p); l->setInput(1,*kernel_tensor);
    } else {
        // See conv(): residual-scale folding never scales the bias term.
        l=n.addDeconvolutionNd(*reshape(x,dims({1,k.dims[0],1,-1})),channels,dims({1,k.dims[2]}),w_scaled(p+".weight",weight_scale),w(p+".bias"));
        require(l,"deconv "+p);
    }
    l->setStrideNd(dims({1,stride}));l->setPaddingNd(dims({0,padding}));return reshape(output(l,p),dims({1,channels,-1}));
}
T Layers::polyphase_deconv(T x,std::string const& p,int stride,int padding,float weight_scale) {
    auto& k=weight(p+".weight");
    auto& b=weight(p+".bias");
    require(!o.quantize_weights,"polyphase upsample does not support weight-only INT8");
    require(k.rank==3 && b.rank==1,"polyphase deconv rank "+p);
    require(stride>0 && k.dims[2]>0,
            "polyphase kernel/stride must be positive: "+p);
    require(padding==(k.dims[2]-stride)/2,
            "polyphase padding is not symmetric: "+p);
    int input_channels=static_cast<int>(k.dims[0]);
    int output_channels=static_cast<int>(k.dims[1]);
    int kernel=static_cast<int>(k.dims[2]);
    require(b.count==output_channels,"polyphase bias shape "+p);

    // Split every phase when padding is non-zero.  Even divisible kernels
    // (K=4/stride=2) have wrapped phases, so one symmetric Conv1d cannot
    // represent all phase boundaries with one symmetric padding value.  The
    // phase path below uses asymmetric pre/post padding to keep exactly one
    // output sample per input sample for every phase.
    std::vector<T> phases;
    phases.reserve(static_cast<size_t>(stride));
    for(int phase=0;phase<stride;++phase) {
        int first=(phase+padding)%stride;
        int phase_taps=(kernel-first+stride-1)/stride;
        require(phase_taps>0,"polyphase phase has no taps");
        // A wrapped phase corresponds to q=n+t-1 rather than q=n+t.
        // Asymmetric pre/post padding expresses that shift while keeping
        // exactly one output sample per input sample.
        bool wrapped=first<padding;
        require(!wrapped || phase_taps>=2,
                "wrapped polyphase phase needs at least two taps");
        std::vector<float> phase_weights(static_cast<size_t>(output_channels)*input_channels*phase_taps,0.0f);
        for(int co=0;co<output_channels;++co)
            for(int ci=0;ci<input_channels;++ci)
                for(int t=0;t<phase_taps;++t) {
                    int source_kernel=first+stride*t;
                    int conv_t=phase_taps-1-t;
                    size_t dst=((static_cast<size_t>(co)*input_channels+ci)*phase_taps)+conv_t;
                    size_t src=(static_cast<size_t>(ci)*output_channels+co)*kernel+source_kernel;
                    phase_weights[dst]=k.data[src]*weight_scale;
                }
        std::vector<float> phase_bias(static_cast<size_t>(output_channels));
        std::copy(b.data,b.data+b.count,phase_bias.data());
        Weights phase_kernel;
        Weights phase_bias_weights;
        if(real_type()==DataType::kFLOAT) {
            floats.push_back(std::move(phase_weights));
            floats.push_back(std::move(phase_bias));
            phase_kernel={DataType::kFLOAT,floats[floats.size()-2].data(),
                          static_cast<int64_t>(floats[floats.size()-2].size())};
            phase_bias_weights={DataType::kFLOAT,floats.back().data(),
                                static_cast<int64_t>(floats.back().size())};
        } else {
            low_constants.emplace_back(phase_weights.size());
            auto& low_kernel=low_constants.back();
            for(size_t i=0;i<phase_weights.size();++i)
                low_kernel[i]=real_type()==DataType::kHALF?float_to_half_bits(phase_weights[i]):float_to_bf16_bits(phase_weights[i]);
            low_constants.emplace_back(phase_bias.size());
            auto& low_bias=low_constants.back();
            for(size_t i=0;i<phase_bias.size();++i)
                low_bias[i]=real_type()==DataType::kHALF?float_to_half_bits(phase_bias[i]):float_to_bf16_bits(phase_bias[i]);
            phase_kernel={real_type(),low_kernel.data(),static_cast<int64_t>(low_kernel.size())};
            phase_bias_weights={real_type(),low_bias.data(),static_cast<int64_t>(low_bias.size())};
        }
        auto* layer=n.addConvolutionNd(*reshape(x,dims({1,input_channels,1,-1})),
                                       output_channels,dims({1,phase_taps}),phase_kernel,phase_bias_weights);
        require(layer,"polyphase phase convolution "+p);
        // Non-wrapped phases need x[q-t].  Wrapped phases need x[q+1-t].
        // Asymmetric padding expresses that shift while retaining length L.
        layer->setPrePadding(dims({0,wrapped?phase_taps-2:phase_taps-1}));
        layer->setPostPadding(dims({0,wrapped?1:0}));
        T phase_output=reshape(output(layer,p+".phase_"+std::to_string(phase)),dims({1,output_channels,-1}));
        phases.push_back(phase_output);
    }
    auto packed=concat(phases,1);
    auto packed_shape=reshape(packed,dims({1,stride,output_channels,-1}));
    return reshape(transpose(packed_shape,{0,2,3,1}),dims({1,output_channels,-1}));
}
T Layers::norm(T x,std::string const& p) {
    auto mean=output(n.addReduce(*x,ReduceOperation::kAVG,1U<<1,true));auto centered=sub(x,mean);
    auto variance=output(n.addReduce(*mul(centered,centered),ReduceOperation::kAVG,1U<<1,true));
    auto normalized=ew(centered,unary(add(variance,scalar(c.norm_epsilon,3)),UnaryOperation::kSQRT),ElementWiseOperation::kDIV);
    return add(mul(normalized,parameter(p+".gamma",dims({1,c.hidden,1}))),parameter(p+".beta",dims({1,c.hidden,1})));
}
T Layers::resize(T x,int factor) { auto l=n.addResize(*x);require(l,"resize");float scales[]{1,1,static_cast<float>(factor)};l->setScales(scales,3);l->setResizeMode(InterpolationMode::kNEAREST);l->setCoordinateTransformation(ResizeCoordinateTransformation::kASYMMETRIC);return output(l); }
T Layers::cumulative(T x,int axis,bool exclusive) { return output(n.addCumulative(*x,*integer({axis},dims({})),CumulativeOperation::kSUM,exclusive,false)); }
T Layers::broadcast(T x,T target) { auto d=x->getDimensions();Dims start{},stride{};start.nbDims=stride.nbDims=d.nbDims; for(int i=0;i<d.nbDims;i++)stride.d[i]=d.d[i]==1?0:1;auto l=n.addSlice(*x,start,d,stride);require(l,"broadcast");l->setInput(2,*target);return output(l); }
}
