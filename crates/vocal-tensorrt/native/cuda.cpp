#include "bridge.h"
#include "nvtx.h"

extern "C" int vocal_cuda_event_create(void** out,char** error) noexcept {
    return vocal::guarded(error,[&]{
        cudaEvent_t e{}; vocal::cuda_check(cudaEventCreate(&e)); *out=e;
    });
}
extern "C" void vocal_cuda_event_free(void* e) noexcept {
    if(e) cudaEventDestroy(static_cast<cudaEvent_t>(e));
}
extern "C" int vocal_cuda_event_record(void* e,void* stream,char** error) noexcept {
    return vocal::guarded(error,[&]{
        vocal::cuda_check(cudaEventRecord(static_cast<cudaEvent_t>(e),static_cast<cudaStream_t>(stream)));
    });
}
extern "C" int vocal_cuda_event_elapsed(void* start,void* end,float* ms,char** error) noexcept {
    return vocal::guarded(error,[&]{
        vocal::cuda_check(cudaEventElapsedTime(ms,static_cast<cudaEvent_t>(start),static_cast<cudaEvent_t>(end)));
    });
}

extern "C" int vocal_cuda_stream_create(
    vocal::Stream** out, char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(out, "null CUDA stream output");
        *out = nullptr;
        auto* stream = new vocal::Stream{};
        auto status = cudaStreamCreate(&stream->stream);
        if (status != cudaSuccess) {
            delete stream;
            throw std::runtime_error(cudaGetErrorString(status));
        }
        *out = stream;
    });
}

extern "C" void vocal_cuda_stream_destroy(vocal::Stream* stream) noexcept {
    if (!stream) return;
    if (stream->stream) cudaStreamDestroy(stream->stream);
    delete stream;
}

extern "C" int vocal_cuda_stream_sync(
    vocal::Stream* stream, char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(stream && stream->stream, "invalid CUDA stream");
        auto status = cudaStreamSynchronize(stream->stream);
        vocal::require(status == cudaSuccess, cudaGetErrorString(status));
    });
}

extern "C" void* vocal_cuda_stream_ptr(vocal::Stream* stream) noexcept {
    return stream ? static_cast<void*>(stream->stream) : nullptr;
}

extern "C" int vocal_cuda_graph_begin(
    vocal::Stream* stream, char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(stream && stream->stream, "invalid CUDA graph stream");
        vocal::cuda_check(cudaStreamBeginCapture(stream->stream, cudaStreamCaptureModeGlobal));
    });
}

extern "C" int vocal_cuda_graph_end(
    vocal::Stream* stream, vocal::CudaGraphExec** out, char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(stream && stream->stream && out, "invalid CUDA graph end arguments");
        *out = nullptr;
        cudaGraph_t graph{};
        vocal::cuda_check(cudaStreamEndCapture(stream->stream, &graph));
        vocal::require(graph != nullptr, "CUDA graph capture returned null graph");
        auto* exec = new vocal::CudaGraphExec{};
        auto status = cudaGraphInstantiate(&exec->exec, graph, 0);
        cudaGraphDestroy(graph);
        if (status != cudaSuccess) {
            delete exec;
            throw std::runtime_error(cudaGetErrorString(status));
        }
        *out = exec;
    });
}

extern "C" int vocal_cuda_graph_abort(
    vocal::Stream* stream, char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(stream && stream->stream, "invalid CUDA graph abort stream");
        cudaGraph_t graph{};
        auto status = cudaStreamEndCapture(stream->stream, &graph);
        if (graph) cudaGraphDestroy(graph);
        // cudaErrorStreamCaptureInvalidated is expected when a captured
        // TensorRT enqueue failed.  The stream is usable after the abort.
        if (status != cudaSuccess && status != cudaErrorStreamCaptureInvalidated)
            throw std::runtime_error(cudaGetErrorString(status));
    });
}

extern "C" int vocal_cuda_graph_launch(
    vocal::CudaGraphExec* graph, vocal::Stream* stream, char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(graph && graph->exec && stream && stream->stream,
                       "invalid CUDA graph launch arguments");
        vocal::cuda_check(cudaGraphLaunch(graph->exec, stream->stream));
    });
}

extern "C" void vocal_cuda_graph_exec_destroy(vocal::CudaGraphExec* graph) noexcept {
    if (!graph) return;
    if (graph->exec) cudaGraphExecDestroy(graph->exec);
    delete graph;
}

extern "C" int vocal_cuda_malloc(
    size_t bytes, vocal::Memory** out, char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(bytes > 0 && out, "invalid CUDA allocation size");
        *out = nullptr;
        auto* memory = new vocal::Memory{};
        auto status = cudaMalloc(&memory->ptr, bytes);
        if (status != cudaSuccess) {
            delete memory;
            throw std::runtime_error(cudaGetErrorString(status));
        }
        memory->bytes = bytes;
        *out = memory;
    });
}

extern "C" void vocal_cuda_free(vocal::Memory* memory) noexcept {
    if (!memory) return;
    if (memory->ptr) cudaFree(memory->ptr);
    delete memory;
}

extern "C" void* vocal_cuda_ptr(vocal::Memory* memory) noexcept {
    return memory ? memory->ptr : nullptr;
}

extern "C" int vocal_cuda_host_alloc(
    size_t bytes, vocal::HostMemory** out, char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(bytes > 0 && out, "invalid pinned host allocation size");
        *out = nullptr;
        auto* memory = new vocal::HostMemory{};
        auto status = cudaHostAlloc(&memory->ptr, bytes, cudaHostAllocPortable);
        if (status != cudaSuccess) {
            delete memory;
            throw std::runtime_error(cudaGetErrorString(status));
        }
        memory->bytes = bytes;
        *out = memory;
    });
}

extern "C" void vocal_cuda_host_free(vocal::HostMemory* memory) noexcept {
    if (!memory) return;
    if (memory->ptr) cudaFreeHost(memory->ptr);
    delete memory;
}

extern "C" void* vocal_cuda_host_ptr(vocal::HostMemory* memory) noexcept {
    return memory ? memory->ptr : nullptr;
}

extern "C" int vocal_cuda_upload_async(
    vocal::Memory* memory, size_t destination_offset,
    vocal::HostMemory* source, size_t source_offset, size_t bytes,
    vocal::Stream* stream, char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(memory && source && stream && stream->stream,
                       "invalid async CUDA upload arguments");
        vocal::require(destination_offset <= memory->bytes &&
                           bytes <= memory->bytes - destination_offset,
                       "CUDA async destination range exceeds allocation");
        vocal::require(source_offset <= source->bytes &&
                           bytes <= source->bytes - source_offset,
                       "CUDA async host range exceeds allocation");
        auto* dst = static_cast<unsigned char*>(memory->ptr) + destination_offset;
        auto* src = static_cast<unsigned char*>(source->ptr) + source_offset;
        nvtx3::scoped_range range{"rvc.h2d.async"};
        vocal::cuda_check(cudaMemcpyAsync(dst, src, bytes,
                                          cudaMemcpyHostToDevice, stream->stream));
    });
}

extern "C" int vocal_cuda_download_async(
    vocal::Memory* memory, size_t source_offset,
    vocal::HostMemory* destination, size_t destination_offset, size_t bytes,
    vocal::Stream* stream, char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(memory && destination && stream && stream->stream,
                       "invalid async CUDA download arguments");
        vocal::require(source_offset <= memory->bytes &&
                           bytes <= memory->bytes - source_offset,
                       "CUDA async source range exceeds allocation");
        vocal::require(destination_offset <= destination->bytes &&
                           bytes <= destination->bytes - destination_offset,
                       "CUDA async host destination range exceeds allocation");
        auto* src = static_cast<unsigned char*>(memory->ptr) + source_offset;
        auto* dst = static_cast<unsigned char*>(destination->ptr) + destination_offset;
        nvtx3::scoped_range range{"rvc.d2h.async"};
        vocal::cuda_check(cudaMemcpyAsync(dst, src, bytes,
                                          cudaMemcpyDeviceToHost, stream->stream));
    });
}

extern "C" int vocal_cuda_memset(
    vocal::Memory* memory, int value, size_t bytes, char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(memory && bytes <= memory->bytes, "invalid CUDA memset range");
        auto status = cudaMemset(memory->ptr, value, bytes);
        vocal::require(status == cudaSuccess, cudaGetErrorString(status));
    });
}

extern "C" int vocal_cuda_upload(
    vocal::Memory* memory, void const* source, size_t bytes, char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(memory && (bytes == 0 || source) && bytes <= memory->bytes, "invalid CUDA upload range");
        nvtx3::scoped_range range{"rvc.h2d"};
        auto status = cudaMemcpy(memory->ptr, source, bytes, cudaMemcpyHostToDevice);
        vocal::require(status == cudaSuccess, cudaGetErrorString(status));
    });
}

extern "C" int vocal_cuda_download(
    vocal::Memory* memory, void* destination, size_t bytes, char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(memory && (bytes == 0 || destination) && bytes <= memory->bytes, "invalid CUDA download range");
        nvtx3::scoped_range range{"rvc.d2h"};
        auto status = cudaMemcpy(destination, memory->ptr, bytes, cudaMemcpyDeviceToHost);
        vocal::require(status == cudaSuccess, cudaGetErrorString(status));
    });
}

extern "C" int vocal_cuda_device_copy(
    vocal::Memory* destination, size_t destination_offset,
    vocal::Memory* source, size_t source_offset, size_t bytes,
    vocal::Stream* stream, char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(destination && source && stream && stream->stream,
                       "invalid CUDA device copy arguments");
        vocal::require(destination_offset <= destination->bytes &&
                           bytes <= destination->bytes - destination_offset,
                       "CUDA destination copy range exceeds allocation");
        vocal::require(source_offset <= source->bytes && bytes <= source->bytes - source_offset,
                       "CUDA source copy range exceeds allocation");
        auto* dst = static_cast<unsigned char*>(destination->ptr) + destination_offset;
        auto* src = static_cast<unsigned char*>(source->ptr) + source_offset;
        vocal::cuda_check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice,
                                          stream->stream));
    });
}
