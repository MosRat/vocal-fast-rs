#include "bridge.h"
#include "nvtx.h"

#include <memory>

using namespace nvinfer1;

extern "C" int vocal_trt_runtime_load(
    unsigned char const* data, size_t bytes, vocal::Runtime** out, char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(data && bytes && out, "invalid runtime load arguments");
        *out = nullptr;
        std::unique_ptr<IRuntime> runtime(createInferRuntime(vocal::logger()));
        vocal::require(runtime != nullptr, "createInferRuntime failed");
        std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(data, bytes));
        vocal::require(engine != nullptr, "deserializeCudaEngine failed");
        auto* result = new vocal::Runtime{runtime.release(), engine.release()};
        *out = result;
    });
}

extern "C" void vocal_trt_runtime_free(vocal::Runtime* runtime) noexcept {
    if (!runtime) return;
    delete runtime->engine;
    delete runtime->runtime;
    delete runtime;
}

static int create_context(
    vocal::Runtime* runtime, int32_t cuda_graph_strategy,
    unsigned char const* cached, size_t cached_bytes, int32_t* cache_accepted,
    vocal::Context** out, char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(runtime && runtime->engine && out, "invalid runtime handle");
        *out = nullptr;
        std::unique_ptr<nvinfer1::IRuntimeConfig> runtime_config(
            runtime->engine->createRuntimeConfig());
        vocal::require(runtime_config != nullptr, "createRuntimeConfig failed");
        std::unique_ptr<nvinfer1::IRuntimeCache> cache(runtime_config->createRuntimeCache());
        if (cache_accepted) *cache_accepted = 0;
        if (cache) {
            if (cached && cached_bytes && cache->deserialize(cached, cached_bytes)) {
                if (cache_accepted) *cache_accepted = 1;
            }
            if (!runtime_config->setRuntimeCache(*cache)) {
                cache.reset();
                if (cache_accepted) *cache_accepted = 0;
            }
        }
        if (cuda_graph_strategy != 0) {
            vocal::require(
                cuda_graph_strategy == 1,
                "unsupported TensorRT CUDA graph strategy");
            vocal::require(
                runtime_config->setCudaGraphStrategy(
                    nvinfer1::CudaGraphStrategy::kWHOLE_GRAPH_CAPTURE),
                "set TensorRT CUDA graph strategy failed");
        } else {
            vocal::require(
                runtime_config->setCudaGraphStrategy(
                    nvinfer1::CudaGraphStrategy::kDISABLED),
                "disable TensorRT CUDA graph strategy failed");
        }
        std::unique_ptr<IExecutionContext> context(
            runtime->engine->createExecutionContext(runtime_config.get()));
        vocal::require(context != nullptr, "createExecutionContext failed");
        auto result = std::make_unique<vocal::Context>();
        result->context = context.release();
        result->engine = runtime->engine;
        result->runtime_config = runtime_config.release();
        result->runtime_cache = cache.release();
        *out = result.release();
    });
}

extern "C" int vocal_trt_context_create(
    vocal::Runtime* runtime, vocal::Context** out, char** error) noexcept {
    return create_context(runtime, 0, nullptr, 0, nullptr, out, error);
}

extern "C" int vocal_trt_context_create_with_options(
    vocal::Runtime* runtime, int32_t cuda_graph_strategy,
    vocal::Context** out, char** error) noexcept {
    return create_context(runtime, cuda_graph_strategy, nullptr, 0, nullptr, out, error);
}

extern "C" int vocal_trt_context_create_cached(
    vocal::Runtime* runtime, int32_t strategy, unsigned char const* cached,
    size_t cached_bytes, int32_t* accepted, vocal::Context** out, char** error) noexcept {
    return create_context(runtime, strategy, cached, cached_bytes, accepted, out, error);
}

extern "C" int vocal_trt_context_cache(vocal::Context* context, vocal::Build** out,
                                       char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(context && out, "invalid context cache arguments");
        *out = nullptr;
        if (!context->runtime_cache) return;
        auto result = std::make_unique<vocal::Build>();
        result->mem = context->runtime_cache->serialize();
        if (result->mem) *out = result.release();
    });
}

extern "C" void vocal_trt_context_free(vocal::Context* context) noexcept {
    if (context) {
        delete context->context;
        delete context->runtime_config;
        delete context->runtime_cache;
        delete context;
    }
}

extern "C" int vocal_trt_context_is_stream_capturable(
    vocal::Context* context, void* stream, int32_t* capturable,
    char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(context && context->context && stream && capturable,
                       "invalid stream capturability arguments");
        *capturable = context->context->isStreamCapturable(
            static_cast<cudaStream_t>(stream)) ? 1 : 0;
    });
}

extern "C" int vocal_trt_set_input_shape(
    vocal::Context* context, char const* name, int64_t const* dimensions,
    int32_t rank, char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(context && context->context && name && dimensions, "invalid shape arguments");
        vocal::require(rank >= 0 && rank <= Dims::MAX_DIMS, "invalid input rank");
        Dims shape{};
        shape.nbDims = rank;
        for (int32_t i = 0; i < rank; ++i) shape.d[i] = dimensions[i];
        vocal::require(context->context->setInputShape(name, shape), "setInputShape failed");
    });
}

extern "C" int vocal_trt_set_tensor_address(
    vocal::Context* context, char const* name, void* address, char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(context && context->context && name && address, "invalid tensor address arguments");
        vocal::require(context->context->setTensorAddress(name, address), "setTensorAddress failed");
    });
}

extern "C" int vocal_trt_get_tensor_shape(
    vocal::Context* context, char const* name, int64_t* dimensions,
    int32_t capacity, int32_t* rank, char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(context && context->context && name && dimensions && rank,
                       "invalid tensor shape arguments");
        vocal::require(capacity >= 0 && capacity <= Dims::MAX_DIMS,
                       "invalid tensor shape capacity");
        Dims shape = context->context->getTensorShape(name);
        vocal::require(shape.nbDims >= 0 && shape.nbDims <= capacity,
                       "tensor shape buffer is too small");
        *rank = shape.nbDims;
        for (int32_t i = 0; i < shape.nbDims; ++i) dimensions[i] = shape.d[i];
    });
}

extern "C" int vocal_trt_context_has_tensor(
    vocal::Context* context, char const* name, int32_t* found,
    char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(context && context->context && context->engine && name && found,
                       "invalid tensor existence arguments");
        *found = 0;
        int32_t count = context->engine->getNbIOTensors();
        for (int32_t index = 0; index < count; ++index) {
            auto const* tensor_name = context->engine->getIOTensorName(index);
            if (tensor_name && std::strcmp(tensor_name, name) == 0) {
                *found = 1;
                break;
            }
        }
    });
}

extern "C" int vocal_trt_enqueue(
    vocal::Context* context, void* stream, char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(context && context->context && stream, "invalid enqueue arguments");
        vocal::require(context->context->enqueueV3(static_cast<cudaStream_t>(stream)), "enqueueV3 failed");
    });
}

extern "C" int vocal_trt_enqueue_named(
    vocal::Context* context, void* stream, char const* range_name,
    char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(context && context->context && stream,
                       "invalid enqueue arguments");
        if (range_name && range_name[0] != '\0') {
            nvtx3::scoped_range range{range_name};
            vocal::require(context->context->enqueueV3(
                               static_cast<cudaStream_t>(stream)),
                           "enqueueV3 failed");
        } else {
            vocal::require(context->context->enqueueV3(
                               static_cast<cudaStream_t>(stream)),
                           "enqueueV3 failed");
        }
    });
}

extern "C" void vocal_trt_profile_enable(vocal::Context* c, int enabled) noexcept {
    c->profiler.times.clear();
    c->context->setProfiler(enabled ? &c->profiler : nullptr);
}
extern "C" void vocal_trt_profile_read(vocal::Context* c, void* user,
    void (*callback)(void*, char const*, double)) noexcept {
    for (auto const& item : c->profiler.times) callback(user, item.first.c_str(), item.second);
}
