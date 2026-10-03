// Plugin seam reserved for composite RVC stages that are not expressible as
// one TensorRT primitive. The current VFRM graph expands these stages into
// native layers; keeping this implementation isolated makes a future plugin
// registry independent from runtime and graph construction.
#include "bridge.h"
#include <NvInferRuntimePlugin.h>

#include <cstring>
#include <string>

namespace vocal::plugin {
using namespace nvinfer1;

class Stage final : public IPluginV2DynamicExt {
public:
    explicit Stage(char const* name = "rvc_stage") : name_(name ? name : "rvc_stage") {}
    Stage(Stage const&) = default;

    IPluginV2DynamicExt* clone() const noexcept override { return new Stage(*this); }
    char const* getPluginType() const noexcept override { return "vocal.rvc.stage"; }
    char const* getPluginVersion() const noexcept override { return "1"; }
    int32_t getNbOutputs() const noexcept override { return 1; }
    DimsExprs getOutputDimensions(
        int32_t index, DimsExprs const* inputs, int32_t count, IExprBuilder&) noexcept override {
        return index == 0 && count ? inputs[0] : DimsExprs{-1, {}};
    }
    bool supportsFormatCombination(
        int32_t position, PluginTensorDesc const* io, int32_t inputs, int32_t outputs) noexcept override {
        return position < inputs + outputs && io[position].format == TensorFormat::kLINEAR
            && io[position].type == DataType::kFLOAT;
    }
    void configurePlugin(
        DynamicPluginTensorDesc const*, int32_t, DynamicPluginTensorDesc const*, int32_t) noexcept override {}
    size_t getWorkspaceSize(
        PluginTensorDesc const*, int32_t, PluginTensorDesc const*, int32_t) const noexcept override { return 0; }
    int32_t enqueue(
        PluginTensorDesc const*, PluginTensorDesc const* output, void const* const* input,
        void* const* outputs, void*, cudaStream_t stream) noexcept override {
        if (!output || !input || !outputs || !input[0] || !outputs[0]) return 1;
        size_t elements = 1;
        for (int axis = 0; axis < output[0].dims.nbDims; ++axis) {
            if (output[0].dims.d[axis] < 0) return 1;
            elements *= static_cast<size_t>(output[0].dims.d[axis]);
        }
        return cudaMemcpyAsync(
            outputs[0], input[0], elements * sizeof(float), cudaMemcpyDeviceToDevice, stream)
            == cudaSuccess ? 0 : 1;
    }
    DataType getOutputDataType(int32_t, DataType const* input, int32_t) const noexcept override {
        return input[0];
    }
    size_t getSerializationSize() const noexcept override { return name_.size() + 1; }
    void serialize(void* data) const noexcept override { std::memcpy(data, name_.c_str(), name_.size() + 1); }
    int32_t initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    void destroy() noexcept override { delete this; }
    void setPluginNamespace(AsciiChar const* value) noexcept override { namespace_ = value ? value : ""; }
    AsciiChar const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

private:
    std::string name_;
    std::string namespace_;
};
}
