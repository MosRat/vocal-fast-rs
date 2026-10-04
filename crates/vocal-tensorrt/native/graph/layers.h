#pragma once
#include "../bridge.h"
#include <deque>
#include <map>
#include <set>
#include <vector>
#include <algorithm>
#include <cmath>

namespace vocal {
using namespace nvinfer1;
using T = ITensor*;
inline Dims dims(std::initializer_list<int64_t> values) { Dims d{}; d.nbDims=static_cast<int>(values.size()); std::copy(values.begin(),values.end(),d.d); return d; }
struct Layers {
    INetworkDefinition& n;
    Config const& c;
    Options const& o;
    std::map<std::string,Weight const*> weights;
    std::set<std::string> used;
    std::deque<std::vector<float>> floats;
    std::deque<std::vector<uint16_t>> low_constants;
    std::deque<std::vector<int8_t>> quantized;
    std::deque<std::vector<int32_t>> ints;
    DataType active_type{DataType::kFLOAT};
    Layers(INetworkDefinition& net, Config const& config, Options const& options, Weight const* w, size_t count);
    T output(ILayer* layer, std::string const& name="");
    Weight const& weight(std::string const& name);
    Weights w(std::string const& name);
    Weights w_scaled(std::string const& name, float scale);
    DataType real_type() const;
    T cast_real(T x);
    void use_low_precision();
    // Create a build-time INT8 per-output-channel weight followed by an
    // explicit dequantize node. This keeps the experiment isolated from the
    // public floating-point graph ABI; the builder may or may not fold the
    // pair into a native INT8 tactic on a given TensorRT-RTX release.
    T quantized_parameter(std::string const& name, Dims shape, int axis=0);
    T constant(std::vector<float> data, Dims shape);
    T integer(std::vector<int32_t> data, Dims shape);
    T iv(std::initializer_list<int32_t> values);
    T scalar(float value, int rank);
    T parameter(std::string const& name, Dims shape);
    T reshape(T x, Dims d);
    T transpose(T x, std::initializer_list<int> permutation);
    T ew(T a,T b,ElementWiseOperation op);
    T add(T a,T b) { return ew(a,b,ElementWiseOperation::kSUM); }
    T mul(T a,T b) { return ew(a,b,ElementWiseOperation::kPROD); }
    T sub(T a,T b) { return ew(a,b,ElementWiseOperation::kSUB); }
    T scale(T a,float v) { return mul(a,scalar(v,a->getDimensions().nbDims)); }
    T unary(T x,UnaryOperation op) { return output(n.addUnary(*x,op)); }
    T cast(T x,DataType type) { return output(n.addCast(*x,type)); }
    T act(T x,ActivationType type,float alpha=0.1f);
    T shape(T x) { return output(n.addShape(*x)); }
    T dimension(T x,int axis);
    T concat(std::vector<T> tensors,int axis);
    T gather(T x,T indices,int axis);
    T gather_elements(T x,T indices,int axis);
    T slice(T x,int axis,int start,int count,int stride=1);
    T range(T length,int maximum);
    T matmul(T a,T b,bool transpose_b=false);
    T conv(T x,std::string const& prefix,int stride=1,int dilation=1,int padding=-1,bool bias=true,float weight_scale=1.0f);
    T fused_qkv_conv(T x,std::string const& prefix,int channels);
    T deconv(T x,std::string const& prefix,int stride,int padding,float weight_scale=1.0f);
    T polyphase_deconv(T x,std::string const& prefix,int stride,int padding,float weight_scale=1.0f);
    T norm(T x,std::string const& prefix);
    T select(T condition,T yes,T no) { return output(n.addSelect(*condition,*yes,*no)); }
    T resize(T x,int factor);
    T cumulative(T x,int axis,bool exclusive);
    T broadcast(T x,T target_shape);
    void mark(T x,char const* name) {
        // Debug buffers use the stable public F32 ABI. Low-precision Synth
        // tensors are explicitly cast before being exposed so the CLI does
        // not interpret packed FP16 bytes as F32 values.
        if (x->getType()==DataType::kHALF || x->getType()==DataType::kBF16)
            x=cast(x,DataType::kFLOAT);
        x->setName(name); n.markOutput(*x);
    }
    T encoder(T feats,T pitch,T lengths,T rnd,T& mask);
    T attention(T x,T mask,int layer);
    T flow(T z,T mask,T speaker);
    T decoder(T z,T speaker,T pitchf,T noise);
    T source(T pitchf,T noise);
};
}
