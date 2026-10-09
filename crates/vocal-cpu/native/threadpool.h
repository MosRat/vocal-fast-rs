#pragma once
#include "affinity.h"
#include <atomic>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <oneapi/dnnl/dnnl_threadpool_iface.hpp>
#include <thread>
#include <vector>
namespace vocal::cpu {
class Threadpool final : public dnnl::threadpool_interop::threadpool_iface {
  int count;
  uint32_t spin_us;
  std::vector<uintptr_t> masks;
  std::vector<std::thread> workers;
  std::mutex mutex, job_mutex;
  std::condition_variable work, done;
  std::function<void(int, int)> task;
  std::atomic<int> next{0};
  int total = 0, pending = 0;
  std::atomic<uint64_t> generation{0};
  bool stopping = false;
  std::exception_ptr failure;
  static inline thread_local bool active = false;
  void consume();
  void shutdown();

public:
  explicit Threadpool(int, uint32_t, bool);
  ~Threadpool() override;
  int get_num_threads() const override { return count; }
  uintptr_t caller_affinity() const { return masks.empty() ? 0 : masks[0]; }
  bool get_in_parallel() const override { return active; }
  uint64_t get_flags() const override { return 0; }
  void parallel_for(int, const std::function<void(int, int)> &) override;
  void range(size_t, size_t, const std::function<void(size_t, size_t)> &);
};
std::shared_ptr<Threadpool> shared_pool(int, uint32_t, bool);
} // namespace vocal::cpu
