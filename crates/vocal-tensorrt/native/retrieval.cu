#include "bridge.h"
#include "retrieval_kernels.cuh"

#include <algorithm>
#include <limits>

namespace vocal {

struct RetrievalGpuIndex {
    int32_t dimensions{};
    int32_t nlist{};
    int32_t nprobe{};
    size_t vector_count{};
    float* centroids{};
    int32_t* offsets{};
    float* vectors{};
    int32_t* ids{};
};

namespace {

void free_index(RetrievalGpuIndex* index) noexcept {
    if (!index) return;
    if (index->centroids) cudaFree(index->centroids);
    if (index->offsets) cudaFree(index->offsets);
    if (index->vectors) cudaFree(index->vectors);
    if (index->ids) cudaFree(index->ids);
    delete index;
}

} // namespace
} // namespace vocal

extern "C" int vocal_faiss_gpu_create(
    int32_t dimensions, int32_t nlist, int32_t nprobe, size_t vector_count,
    float const* centroids, size_t centroid_count,
    int32_t const* offsets, size_t offset_count,
    float const* vectors, size_t vector_value_count,
    int32_t const* ids, size_t id_count,
    vocal::RetrievalGpuIndex** out, char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(out, "null FAISS GPU index output");
        vocal::require(dimensions > 0 && dimensions <= 4096,
                       "invalid FAISS GPU index dimension");
        vocal::require(nlist > 0 && nprobe > 0 && nprobe <= nlist && nprobe <= 32,
                       "invalid FAISS GPU IVF configuration");
        vocal::require(vector_count > 0 && vector_count <= static_cast<size_t>(INT32_MAX),
                       "invalid FAISS GPU vector count");
        vocal::require(centroids && centroid_count == static_cast<size_t>(nlist) * dimensions,
                       "invalid FAISS centroid payload");
        vocal::require(offsets && offset_count == static_cast<size_t>(nlist) + 1,
                       "invalid FAISS list offsets");
        vocal::require(vectors && vector_value_count == vector_count * static_cast<size_t>(dimensions),
                       "invalid FAISS vector payload");
        vocal::require(ids && id_count == vector_count, "invalid FAISS id payload");
        vocal::require(offsets[0] == 0 && offsets[nlist] == static_cast<int32_t>(vector_count),
                       "FAISS list offsets do not cover vectors");
        for (int32_t i = 0; i < nlist; ++i)
            vocal::require(offsets[i] <= offsets[i + 1], "FAISS list offsets are not monotonic");
        *out = nullptr;
        auto* index = new vocal::RetrievalGpuIndex{};
        index->dimensions = dimensions;
        index->nlist = nlist;
        index->nprobe = nprobe;
        index->vector_count = vector_count;
        auto alloc_copy = [](void** destination, void const* source, size_t bytes) {
            vocal::cuda_check(cudaMalloc(destination, bytes));
            auto status = cudaMemcpy(*destination, source, bytes, cudaMemcpyHostToDevice);
            if (status != cudaSuccess) {
                cudaFree(*destination);
                *destination = nullptr;
                vocal::cuda_check(status);
            }
        };
        try {
            alloc_copy(reinterpret_cast<void**>(&index->centroids), centroids,
                       centroid_count * sizeof(float));
            alloc_copy(reinterpret_cast<void**>(&index->offsets), offsets,
                       offset_count * sizeof(int32_t));
            alloc_copy(reinterpret_cast<void**>(&index->vectors), vectors,
                       vector_value_count * sizeof(float));
            alloc_copy(reinterpret_cast<void**>(&index->ids), ids,
                       id_count * sizeof(int32_t));
        } catch (...) {
            vocal::free_index(index);
            throw;
        }
        *out = index;
    });
}

extern "C" void vocal_faiss_gpu_destroy(vocal::RetrievalGpuIndex* index) noexcept {
    vocal::free_index(index);
}

extern "C" int vocal_faiss_gpu_blend(
    vocal::RetrievalGpuIndex* index, float* features, size_t frames, int32_t width,
    float const* index_rate, vocal::Stream* stream, char** error) noexcept {
    return vocal::guarded(error, [&] {
        vocal::require(index && features && frames > 0 && index_rate,
                       "invalid FAISS GPU blend arguments");
        vocal::require(frames <= static_cast<size_t>(INT32_MAX),
                       "FAISS frame count exceeds CUDA grid limit");
        vocal::require(width == index->dimensions, "FAISS feature width mismatch");
        vocal::require(stream && stream->stream, "invalid FAISS CUDA stream");
        if (index->nprobe == 1) {
            // Measured on 34/140-frame RVC windows: small query batches need
            // more warps per query; larger batches benefit from more blocks
            // resident per SM. Keep this dispatch out of the kernel/ABI.
            auto launch = frames <= 64 && index->nlist >= 128
                ? vocal::retrieval::launch_block_retrieval<16> : vocal::retrieval::launch_block_retrieval<8>;
            launch(index->dimensions, index->nlist, index->centroids, index->offsets,
                   index->vectors, features, frames, index_rate, stream->stream);
            vocal::cuda_check(cudaGetLastError());
            return;
        }
        // One warp owns one frame.  Four warps per block gives enough active
        // warps for the centroid scan without making the dynamic register
        // footprint of top-k state unnecessarily large.
        constexpr int threads = 128;
        constexpr int warps_per_block = threads / 32;
        int blocks = static_cast<int>((frames + warps_per_block - 1) / warps_per_block);
        vocal::retrieval::blend_ivf_kernel<<<blocks, threads, 0, stream->stream>>>(
            index->dimensions, index->nlist, index->nprobe,
            index->centroids, index->offsets, index->vectors, features, frames, index_rate);
        vocal::cuda_check(cudaGetLastError());
    });
}
