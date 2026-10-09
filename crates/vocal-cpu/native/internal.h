#pragma once
#include "bridge.h"
#include "tensor.h"
#include "threadpool.h"
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_threadpool.hpp>
#include <unordered_map>
namespace vocal::cpu {
static_assert(DNNL_CPU_RUNTIME == DNNL_RUNTIME_THREADPOOL,
              "vocal-cpu requires a standalone oneDNN THREADPOOL SDK");
// Primitive construction and execution can move between Rust worker threads.
// Scope the thread-local oneDNN planning policy to this graph's actual pool.
class Concurrency final {
  int previous = 0;

public:
  explicit Concurrency(int threads) {
    dnnl_threadpool_interop_get_max_concurrency(&previous);
    require(dnnl_threadpool_interop_set_max_concurrency(threads) ==
                dnnl_success,
            "oneDNN thread configuration failed");
  }
  ~Concurrency() { dnnl_threadpool_interop_set_max_concurrency(previous); }
};
struct Graph;
struct Kernel {
  // Inputs copied into owned, immutable oneDNN parameter memory at compile
  // time.
  std::vector<size_t> packed_inputs;
  size_t packed_bytes = 0, auxiliary_bytes = 0;
  std::string implementation;
  virtual ~Kernel() = default;
  virtual void run(Graph &) = 0;
};
enum class ActivationPostOp { None, Relu, Gelu };
struct Node {
  uint32_t op = 0;
  std::vector<uint32_t> inputs;
  std::vector<int32_t> attrs;
  float values[4] = {};
  bool folded = false, alias = false;
  ActivationPostOp activation = ActivationPostOp::None;
  float activation_alpha = 0;
  // Logical residual view; its storage is kept through the original Add's
  // lifetime. A non-null storage indicates a fused binary addition.
  Tensor residual;
  uint32_t quantization = 0;
  float smooth_alpha = -1;
  std::vector<float> input_max;
  size_t observations = 0;
  std::unique_ptr<Kernel> kernel;
};
struct Graph {
  std::shared_ptr<Threadpool> pool;
  dnnl::engine engine{dnnl::engine::kind::cpu, 0};
  dnnl::stream stream;
  std::vector<Node> nodes;
  std::vector<Tensor> tensors;
  std::vector<uint32_t> outputs;
  std::vector<double> timings;
  std::vector<std::shared_ptr<Storage>> slots;
  bool avx2, profile, channels_last, fast_tensor_io, ready = false;
  CpuAttention attention{};
  CpuConvolution convolution{};
  uint32_t linear_mode = 0;
  std::unordered_map<size_t, std::shared_ptr<Storage>> workspace_slots;
  std::vector<std::pair<size_t, dnnl::memory>> workspace_bindings;
  std::vector<std::shared_ptr<Storage>> compiling_workspaces;
  dnnl::memory workspace(size_t key, const dnnl::memory::desc &md) {
    auto &slot = workspace_slots[key];
    if (!slot || slot->bytes < md.get_size()) {
      slot = std::make_shared<Storage>(md.get_size());
      compiling_workspaces.push_back(slot);
    }
    auto memory = dnnl::memory(md, engine, slot->data);
    workspace_bindings.emplace_back(key, memory);
    return memory;
  }
  void finalize_memory();
  CpuMemory memory_usage() const;
  explicit Graph(int n, uint32_t spin, bool pin)
      : pool(shared_pool(n, spin, pin)),
        stream(dnnl::threadpool_interop::make_stream(engine, pool.get())) {}
};
std::unique_ptr<Kernel> prepare_dnnl(Graph &, size_t);
std::unique_ptr<Kernel> prepare_convolution_matmul(Graph &, size_t);
std::unique_ptr<Kernel> prepare_linear(Graph &, size_t);
void append_activation_post_op(dnnl::primitive_attr &, const Node &);
std::unique_ptr<Kernel> prepare_int8(Graph &, size_t);
void observe_input(Graph &, size_t);
std::unique_ptr<Kernel> prepare_attention(Graph &, size_t);
std::unique_ptr<Kernel> prepare_online_attention(Graph &, size_t);
std::unique_ptr<Kernel> prepare_gru(Graph &, size_t);
std::unique_ptr<Kernel> prepare_stft(Graph &, size_t);
void execute_elements(Graph &, size_t);
void channel_temporal_norm(Graph &, size_t);
std::unique_ptr<Kernel> prepare_tensor_reorder(Graph &, size_t);
bool gather_elements_rows(Graph &, size_t);
dnnl::memory::desc descriptor(const Tensor &);
extern "C" void vocal_cpu_add_leaky_avx2(const float *, const float *, float *,
                                         size_t, float);
} // namespace vocal::cpu
