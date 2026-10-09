// Included inside retrieval_kernels.cuh's namespace; shares the FP32 distance
// and top-k primitives with the general multi-probe implementation.

template<int Width>
__device__ __forceinline__ float cached_distance(
    float const* cached, float const* query, float const* row, int dimensions, int lane) {
    if constexpr (Width == 0) {
        return squared_distance_warp(query, row, dimensions, lane);
    } else {
        float value = 0.0f;
        #pragma unroll
        for (int i = 0; i < Width / 32; ++i) {
            float delta = cached[i] - row[i * 32 + lane];
            value = fmaf(delta, delta, value);
        }
        return warp_sum(value);
    }
}

// RVC normally probes one IVF list. A whole block cooperates on one query:
// independent warps scan disjoint centroid/candidate rows, then merge top-8.
// No global scratch, allocations, extra launches or approximate distances.
template<int Warps, int Width = 0>
__global__ void blend_ivf_block_kernel(
    int32_t dimensions, int32_t nlist, float const* centroids,
    int32_t const* offsets, float const* vectors, float* features,
    float const* index_rate) {
    int const lane = threadIdx.x & 31;
    int const warp = threadIdx.x / 32;
    float ratio = fminf(fmaxf(*index_rate, 0.0f), 1.0f);
    if (ratio <= 0.0f) return;
    float* query = features + static_cast<size_t>(blockIdx.x) * dimensions;
    extern __shared__ float query_cache[];
    __shared__ float partial_distances[Warps][kTopK];
    __shared__ int32_t partial_rows[Warps][kTopK];
    __shared__ int32_t selected_list;
    __shared__ int32_t selected_rows[kTopK];
    __shared__ float weights[kTopK];
    __shared__ float weight_sum;
    for (int i = threadIdx.x; i < dimensions; i += blockDim.x)
        query_cache[i] = query[i];
    __syncthreads();
    // Fixed RVC v1/v2 widths let nvcc keep each lane's query in registers
    // across every centroid/vector, eliminating repeated shared loads.
    float cached[Width == 0 ? 1 : Width / 32];
    if constexpr (Width != 0) {
        #pragma unroll
        for (int i = 0; i < Width / 32; ++i)
            cached[i] = query_cache[i * 32 + lane];
    }

    // For Flat-L2 there is no coarse search: the only list contains all rows.
    if (nlist == 1) {
        if (threadIdx.x == 0) selected_list = 0;
    } else {
        float best = 1.0e30f;
        int32_t best_list = -1;
        for (int32_t list = warp; list < nlist; list += Warps) {
            float distance = cached_distance<Width>(
                cached, query_cache, centroids + static_cast<size_t>(list) * dimensions,
                dimensions, lane);
            if (lane == 0 && distance < best) {
                best = distance;
                best_list = list;
            }
        }
        if (lane == 0) {
            partial_distances[warp][0] = best;
            partial_rows[warp][0] = best_list;
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            int32_t list = -1;
            float distance = 1.0e30f;
            for (int w = 0; w < Warps; ++w) {
                float value = partial_distances[w][0];
                int32_t row = partial_rows[w][0];
                if (row >= 0 && (value < distance || (value == distance && row < list))) {
                    distance = value;
                    list = row;
                }
            }
            selected_list = list;
        }
    }
    __syncthreads();
    if (selected_list < 0) return;

    float best_distances[kTopK];
    int32_t best_rows[kTopK];
    for (int k = 0; k < kTopK; ++k) {
        best_distances[k] = 1.0e30f;
        best_rows[k] = -1;
    }
    int32_t begin = offsets[selected_list];
    int32_t end = offsets[selected_list + 1];
    for (int32_t row = begin + warp; row < end; row += Warps) {
        float distance = cached_distance<Width>(
            cached, query_cache, vectors + static_cast<size_t>(row) * dimensions, dimensions, lane);
        if (lane == 0) insert_best(distance, row, best_distances, best_rows);
    }
    if (lane == 0) {
        for (int k = 0; k < kTopK; ++k) {
            partial_distances[warp][k] = best_distances[k];
            partial_rows[warp][k] = best_rows[k];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        // Merge the sorted warp lists. Ties use original inverted-list order,
        // preserving the serial scan's exact-match and top-k behavior.
        int heads[Warps]{};
        weight_sum = 0.0f;
        bool exact = false;
        for (int k = 0; k < kTopK; ++k) {
            int selected_warp = -1;
            int32_t row = -1;
            float distance = 1.0e30f;
            for (int w = 0; w < Warps; ++w) {
                int h = heads[w];
                if (h >= kTopK) continue;
                int32_t candidate = partial_rows[w][h];
                float value = partial_distances[w][h];
                if (candidate >= 0 &&
                    (value < distance || (value == distance && candidate < row))) {
                    distance = value;
                    row = candidate;
                    selected_warp = w;
                }
            }
            if (selected_warp >= 0) ++heads[selected_warp];
            if (k == 0) exact = row >= 0 && distance <= 1.0e-12f;
            selected_rows[k] = row;
            float score = fmaxf(distance, 1.0e-12f);
            float weight = exact ? (k == 0 ? 1.0f : 0.0f)
                                 : (row >= 0 ? 1.0f / (score * score) : 0.0f);
            weights[k] = weight;
            weight_sum += weight;
        }
    }
    __syncthreads();
    if (!(weight_sum > 0.0f) || !isfinite(weight_sum)) return;
    float inverse = 1.0f / weight_sum;
    for (int i = threadIdx.x; i < dimensions; i += blockDim.x) {
        float retrieved = 0.0f;
        for (int k = 0; k < kTopK; ++k) {
            if (selected_rows[k] >= 0)
                retrieved = fmaf(weights[k] * inverse,
                    vectors[static_cast<size_t>(selected_rows[k]) * dimensions + i], retrieved);
        }
        query[i] = fmaf(ratio, retrieved, (1.0f - ratio) * query_cache[i]);
    }
}

template<int Warps>
void launch_block_retrieval(
    int dimensions, int nlist, float const* centroids, int32_t const* offsets,
    float const* vectors, float* features, size_t frames, float const* rate, cudaStream_t stream) {
    auto kernel = blend_ivf_block_kernel<Warps>;
    if (dimensions == 768) kernel = blend_ivf_block_kernel<Warps, 768>;
    if (dimensions == 256) kernel = blend_ivf_block_kernel<Warps, 256>;
    kernel<<<static_cast<unsigned>(frames), Warps * 32, dimensions * sizeof(float), stream>>>(
        dimensions, nlist, centroids, offsets, vectors, features, rate);
}
