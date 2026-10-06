#include "bridge.h"

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

constexpr int kMaxProbe = 32;
constexpr int kTopK = 8;

__device__ __forceinline__ float squared_distance(
    float const* lhs, float const* rhs, int32_t dimensions) {
    float distance = 0.0f;
    for (int32_t i = 0; i < dimensions; ++i) {
        float delta = lhs[i] - rhs[i];
        distance = fmaf(delta, delta, distance);
    }
    return distance;
}

__device__ __forceinline__ void insert_best(
    float distance, int32_t candidate, float* distances, int32_t* candidates) {
    if (distance >= distances[kTopK - 1]) return;
    int slot = kTopK - 1;
    while (slot > 0 && distance < distances[slot - 1]) {
        distances[slot] = distances[slot - 1];
        candidates[slot] = candidates[slot - 1];
        --slot;
    }
    distances[slot] = distance;
    candidates[slot] = candidate;
}

__global__ void blend_ivf_kernel(
    RetrievalGpuIndex const* /*unused*/, int32_t dimensions, int32_t nlist,
    int32_t nprobe, size_t vector_count, float const* centroids,
    int32_t const* offsets, float const* vectors, float* features,
    size_t frames, float const* index_rate) {
    size_t frame = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (frame >= frames) return;
    float ratio = fminf(fmaxf(*index_rate, 0.0f), 1.0f);
    if (ratio <= 0.0f) return;

    float* query = features + frame * static_cast<size_t>(dimensions);
    int32_t probe_lists[kMaxProbe];
    float probe_distances[kMaxProbe];
    int probe_count = min(nprobe, kMaxProbe);
    for (int i = 0; i < probe_count; ++i) {
        probe_lists[i] = -1;
        probe_distances[i] = 1.0e30f;
    }
    // The shipped RVC indexes use nprobe=1. Keeping this scan in one thread
    // makes that hot path compact while still accepting larger nprobe values
    // for compatible IVF-Flat indexes.
    for (int32_t list = 0; list < nlist; ++list) {
        float distance = squared_distance(
            query, centroids + static_cast<size_t>(list) * dimensions, dimensions);
        int slot = probe_count - 1;
        if (distance >= probe_distances[slot]) continue;
        while (slot > 0 && distance < probe_distances[slot - 1]) {
            probe_distances[slot] = probe_distances[slot - 1];
            probe_lists[slot] = probe_lists[slot - 1];
            --slot;
        }
        probe_distances[slot] = distance;
        probe_lists[slot] = list;
    }

    float best_distances[kTopK];
    int32_t best_vectors[kTopK];
    for (int i = 0; i < kTopK; ++i) {
        best_distances[i] = 1.0e30f;
        best_vectors[i] = -1;
    }
    for (int probe = 0; probe < probe_count; ++probe) {
        int32_t list = probe_lists[probe];
        if (list < 0) continue;
        int32_t begin = offsets[list];
        int32_t end = offsets[list + 1];
        for (int32_t row = begin; row < end; ++row) {
            float distance = squared_distance(
                query, vectors + static_cast<size_t>(row) * dimensions, dimensions);
            insert_best(distance, row, best_distances, best_vectors);
        }
    }

    // A zero-distance neighbor is an exact match. It avoids the singular
    // inverse-square weighting used by the reference implementations and is
    // equivalent to a one-hot retrieved feature.
    int32_t exact = -1;
    for (int i = 0; i < kTopK; ++i) {
        if (best_vectors[i] >= 0 && best_distances[i] <= 1.0e-12f) {
            exact = best_vectors[i];
            break;
        }
    }
    if (exact >= 0) {
        float const* value = vectors + static_cast<size_t>(exact) * dimensions;
        for (int32_t i = 0; i < dimensions; ++i)
            query[i] = fmaf(ratio, value[i], (1.0f - ratio) * query[i]);
        return;
    }

    float weight_sum = 0.0f;
    float weights[kTopK];
    for (int i = 0; i < kTopK; ++i) {
        if (best_vectors[i] < 0) {
            weights[i] = 0.0f;
            continue;
        }
        float score = fmaxf(best_distances[i], 1.0e-12f);
        // FAISS IndexFlatL2 reports squared L2 distances. Applio/RVC applies
        // (1 / score)^2, so preserve that convention here.
        float weight = 1.0f / (score * score);
        weights[i] = weight;
        weight_sum += weight;
    }
    if (!(weight_sum > 0.0f) || !isfinite(weight_sum)) return;
    float inverse = 1.0f / weight_sum;
    for (int32_t i = 0; i < dimensions; ++i) {
        float retrieved = 0.0f;
        for (int k = 0; k < kTopK; ++k) {
            if (best_vectors[k] >= 0) {
                retrieved = fmaf(
                    weights[k] * inverse,
                    vectors[static_cast<size_t>(best_vectors[k]) * dimensions + i],
                    retrieved);
            }
        }
        query[i] = fmaf(ratio, retrieved, (1.0f - ratio) * query[i]);
    }
}

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
        vocal::require(width == index->dimensions, "FAISS feature width mismatch");
        vocal::require(stream && stream->stream, "invalid FAISS CUDA stream");
        constexpr int threads = 64;
        int blocks = static_cast<int>((frames + threads - 1) / threads);
        vocal::blend_ivf_kernel<<<blocks, threads, 0, stream->stream>>>(
            index, index->dimensions, index->nlist, index->nprobe, index->vector_count,
            index->centroids, index->offsets, index->vectors, features, frames, index_rate);
        vocal::cuda_check(cudaGetLastError());
    });
}
