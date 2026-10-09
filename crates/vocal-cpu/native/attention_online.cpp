#include "internal.h"
#include "attention_simd.h"
#include <cmath>
#include <limits>

namespace vocal::cpu {
class OnlineAttention final : public Kernel {
  Tensor q, k, v, y;
  dnnl::memory workspace;
  size_t qt, kt, query_blocks, jobs, worker_floats;
  float scale;
  bool simd;

public:
  OnlineAttention(Graph &g, size_t id) {
    const auto &n = g.nodes[id];
    q = g.tensors[n.inputs[0]];
    k = g.tensors[n.inputs[1]];
    v = g.tensors[n.inputs[2]];
    y = g.tensors[id];
    require(q.shape.size() == 4 && k.shape.size() == 4 && v.shape.size() == 4 &&
                y.shape.size() == 4 && q.dtype == 1 && k.dtype == 1 &&
                v.dtype == 1 && y.dtype == 1, "online attention requires rank-four FP32");
    require(q.shape[0] == k.shape[0] && q.shape[1] == k.shape[1] &&
                k.shape[0] == v.shape[0] && k.shape[1] == v.shape[1] &&
                q.shape[3] == k.shape[3] && k.shape[2] == v.shape[2] &&
                y.shape == Shape({q.shape[0], q.shape[1], q.shape[2], v.shape[3]}),
            "online attention Q/K/V shape mismatch");
    scale = n.values[0];
    require(std::isfinite(scale), "non-finite attention scale");
    qt = g.attention.query_tile;
    kt = g.attention.key_tile;
    query_blocks = (q.shape[2] + qt - 1) / qt;
    jobs = q.shape[0] * q.shape[1] * query_blocks;
    // Each logical worker exclusively owns its scratch. Cache-line rounding
    // avoids false sharing; the allocation is reused by all attention layers.
    worker_floats = ((qt * (kt + v.shape[3] + 2) + 15) / 16) * 16;
    auto md = dnnl::memory::desc(
        {static_cast<int64_t>(worker_floats * g.pool->get_num_threads())},
        dnnl::memory::data_type::f32, dnnl::memory::format_tag::x);
    workspace = g.workspace(2, md);
    simd = g.avx2 && q.strides[3] == 1 && k.strides[3] == 1 && v.strides[3] == 1;
    implementation = simd ? "avx2:online-fp32-attention" : "scalar:online-fp32-attention";
  }

  void run(Graph &g) override {
    auto *scratch = static_cast<float *>(workspace.get_data_handle());
    const size_t workers = std::min<size_t>(jobs, g.pool->get_num_threads());
    g.pool->parallel_for(static_cast<int>(workers), [&](int worker, int count) {
      float *scores = scratch + worker * worker_floats;
      float *numerator = scores + qt * kt;
      float *maximum = numerator + qt * v.shape[3];
      float *denominator = maximum + qt;
      for (size_t job = jobs * worker / count; job < jobs * (worker + 1) / count; ++job) {
        size_t head = job / query_blocks, block = job % query_blocks;
        size_t batch = head / q.shape[1];
        head %= q.shape[1];
        size_t first = block * qt, rows = std::min<size_t>(qt, q.shape[2] - first);
        auto *query = static_cast<const float *>(q.data()) + batch * q.strides[0] + head * q.strides[1] + first * q.strides[2];
        auto *key = static_cast<const float *>(k.data()) + batch * k.strides[0] + head * k.strides[1];
        auto *value = static_cast<const float *>(v.data()) + batch * v.strides[0] + head * v.strides[1];
        std::fill(numerator, numerator + rows * v.shape[3], 0.f);
        std::fill(maximum, maximum + rows, -std::numeric_limits<float>::infinity());
        std::fill(denominator, denominator + rows, 0.f);
        for (size_t start = 0; start < static_cast<size_t>(k.shape[2]); start += kt) {
          size_t columns = std::min<size_t>(kt, k.shape[2] - start);
          const float *keys = key + start * k.strides[2];
          const float *values = value + start * v.strides[2];
#ifdef VOCAL_CPU_AVX2
          if (simd)
            vocal_cpu_attention_dot_avx2(query, keys, scores, rows, columns,
                                        q.shape[3], q.strides[2], k.strides[2], kt, scale);
          else
#endif
            for (size_t i = 0; i < rows; ++i)
              for (size_t j = 0; j < columns; ++j) {
                float dot = 0;
                for (size_t f = 0; f < static_cast<size_t>(q.shape[3]); ++f)
                  dot += query[i * q.strides[2] + f * q.strides[3]] * keys[j * k.strides[2] + f * k.strides[3]];
                scores[i * kt + j] = dot * scale;
              }
          for (size_t i = 0; i < rows; ++i) {
            float *p = scores + i * kt;
            float next_max = std::max(maximum[i], *std::max_element(p, p + columns));
            float decay = std::exp(maximum[i] - next_max), sum = 0;
            for (size_t j = 0; j < columns; ++j) {
              p[j] = std::exp(p[j] - next_max);
              sum += p[j];
            }
            float *acc = numerator + i * v.shape[3];
#ifdef VOCAL_CPU_AVX2
            if (simd)
              vocal_cpu_attention_update_avx2(p, values, acc, columns, v.shape[3], v.strides[2], decay);
            else
#endif
              for (size_t f = 0; f < static_cast<size_t>(v.shape[3]); ++f) {
                float dot = acc[f] * decay;
                for (size_t j = 0; j < columns; ++j)
                  dot += p[j] * values[j * v.strides[2] + f * v.strides[3]];
                acc[f] = dot;
              }
            denominator[i] = denominator[i] * decay + sum;
            maximum[i] = next_max;
          }
        }
        auto *output = static_cast<float *>(y.data()) + batch * y.strides[0] + head * y.strides[1] + first * y.strides[2];
        for (size_t i = 0; i < rows; ++i)
          for (size_t f = 0; f < static_cast<size_t>(v.shape[3]); ++f)
            output[i * y.strides[2] + f * y.strides[3]] = numerator[i * v.shape[3] + f] / denominator[i];
      }
    });
  }
};

std::unique_ptr<Kernel> prepare_online_attention(Graph &g, size_t id) {
  return std::make_unique<OnlineAttention>(g, id);
}
} // namespace vocal::cpu
