#pragma once
#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <oneapi/dnnl/dnnl_threadpool_iface.hpp>
#include <thread>
#include <vector>

// A synchronous persistent pool: the calling thread participates in each job.
class Threadpool final : public dnnl::threadpool_interop::threadpool_iface {
  int count;
  std::vector<std::thread> workers;
  std::mutex mutex;
  std::condition_variable work, done;
  std::function<void(int, int)> task;
  std::atomic<int> next{0};
  int total = 0, pending = 0;
  uint64_t generation = 0;
  bool stopping = false;
  static inline thread_local bool active = false;
  void consume() {
    active = true;
    for (int i; (i = next.fetch_add(1, std::memory_order_relaxed)) < total;)
      task(i, total);
    active = false;
  }

public:
  explicit Threadpool(int n) : count(n) {
    for (int i = 1; i < n; ++i)
      workers.emplace_back([this] {
        uint64_t seen = 0;
        for (;;) {
          std::unique_lock lock(mutex);
          work.wait(lock, [&] { return stopping || generation != seen; });
          if (stopping)
            return;
          seen = generation;
          lock.unlock();
          consume();
          lock.lock();
          if (--pending == 0)
            done.notify_one();
        }
      });
  }
  ~Threadpool() override {
    {
      std::lock_guard lock(mutex);
      stopping = true;
    }
    work.notify_all();
    for (auto &t : workers)
      t.join();
  }
  int get_num_threads() const override { return count; }
  bool get_in_parallel() const override { return active; }
  uint64_t get_flags() const override { return 0; }
  void parallel_for(int n, const std::function<void(int, int)> &fn) override {
    if (active || count == 1 || n == 1) {
      for (int i = 0; i < n; ++i)
        fn(i, n);
      return;
    }
    {
      std::lock_guard lock(mutex);
      task = fn;
      total = n;
      next = 0;
      pending = count - 1;
      ++generation;
    }
    work.notify_all();
    consume();
    std::unique_lock lock(mutex);
    done.wait(lock, [&] { return pending == 0; });
    task = {};
  }
};
