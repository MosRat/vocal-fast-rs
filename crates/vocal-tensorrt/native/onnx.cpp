#include "bridge.h"
#include <NvOnnxParser.h>
#include <memory>

using namespace nvinfer1;

static int build_onnx(
    char const* file, uint64_t workspace, int /*fp16*/, int min_frames,
    int opt_frames, int max_frames, int min_samples, int opt_samples,
    int max_samples, int profile_kind, vocal::Build** out, char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(file && out, "invalid ONNX build arguments");
        vocal::require(min_frames > 0 && min_frames <= opt_frames && opt_frames <= max_frames,
                       "invalid ONNX frame profile");
        vocal::require(min_samples > 0 && min_samples <= opt_samples && opt_samples <= max_samples,
                       "invalid ONNX sample profile");
        *out = nullptr;

        std::unique_ptr<IBuilder> builder(createInferBuilder(vocal::logger()));
        vocal::require(builder != nullptr, "createInferBuilder failed");
        std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(0));
        vocal::require(network != nullptr, "createNetwork failed");
        std::unique_ptr<nvonnxparser::IParser> parser(
            nvonnxparser::createParser(*network, vocal::logger()));
        vocal::require(parser != nullptr, "create ONNX parser failed");
        vocal::require(parser->parseFromFile(file, 2), "ONNX parse failed");

        std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
        vocal::require(config != nullptr, "createBuilderConfig failed");
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, workspace);
        config->clearFlag(BuilderFlag::kTF32);
        config->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);
        // TensorRT owns optimization profiles after addOptimizationProfile;
        // IOptimizationProfile has a protected destructor and cannot be put
        // in std::unique_ptr with the default deleter.
        auto* profile = builder->createOptimizationProfile();
        vocal::require(profile != nullptr, "createOptimizationProfile failed");

        for (int i = 0; i < network->getNbInputs(); ++i) {
            auto* tensor = network->getInput(i);
            auto dimensions = tensor->getDimensions();
            auto minimum = dimensions;
            auto optimum = dimensions;
            auto maximum = dimensions;
            bool dynamic = false;
            for (int axis = 0; axis < dimensions.nbDims; ++axis) {
                if (dimensions.d[axis] < 0) {
                    dynamic = true;
                    // Waveform inputs use a sample profile.  Every other
                    // dynamic input is treated as a frame profile; outputs do
                    // not need profile dimensions.
                    auto const* name = tensor->getName();
                    bool samples = profile_kind != 0 &&
                        (std::strstr(name, "waveform") || std::strstr(name, "audio"));
                    minimum.d[axis] = samples ? min_samples : min_frames;
                    optimum.d[axis] = samples ? opt_samples : opt_frames;
                    maximum.d[axis] = samples ? max_samples : max_frames;
                }
            }
            if (dynamic) {
                vocal::require(profile->setDimensions(tensor->getName(), OptProfileSelector::kMIN, minimum),
                               "set ONNX minimum profile failed");
                vocal::require(profile->setDimensions(tensor->getName(), OptProfileSelector::kOPT, optimum),
                               "set ONNX optimum profile failed");
                vocal::require(profile->setDimensions(tensor->getName(), OptProfileSelector::kMAX, maximum),
                               "set ONNX maximum profile failed");
            }
        }
        vocal::require(config->addOptimizationProfile(profile) >= 0,
                       "add ONNX optimization profile failed");
        auto* serialized = builder->buildSerializedNetwork(*network, *config);
        vocal::require(serialized != nullptr, "buildSerializedNetwork failed");
        *out = new vocal::Build{serialized};
    });
}

extern "C" int vocal_trt_build_onnx_profile(
    char const* file, uint64_t workspace, int fp16, int min_frames,
    int opt_frames, int max_frames, int min_samples, int opt_samples,
    int max_samples, int profile_kind, vocal::Build** out, char** error) noexcept {
    return build_onnx(file, workspace, fp16, min_frames, opt_frames, max_frames,
                      min_samples, opt_samples, max_samples, profile_kind, out, error);
}

extern "C" int vocal_trt_build_onnx(
    char const* file, uint64_t workspace, int fp16, int min_frames,
    int opt_frames, int max_frames, vocal::Build** out, char** error) noexcept {
    return build_onnx(file, workspace, fp16, min_frames, opt_frames, max_frames,
                      min_frames, opt_frames, max_frames, 0, out, error);
}
