#include "threadpool.h"
#include <algorithm>
#include <chrono>
#if defined(_M_X64) || defined(__x86_64__)
#include <immintrin.h>
#endif
#include <unordered_map>
namespace vocal::cpu {
void Threadpool::consume() {
  active = true;
  try {
    for (int i; (i = next.fetch_add(1, std::memory_order_relaxed)) < total;)
      task(i, total);
  } catch (...) {
    std::lock_guard lock(mutex);
    if (!failure)
      failure = std::current_exception();
    next = total;
  }
  active = false;
}
Threadpool::Threadpool(int n, uint32_t spin, bool pin)
    : count(n), spin_us(spin), masks(performance_core_masks(pin)) {
  try {
    for (int i = 1; i < n; ++i)
      workers.emplace_back([this, i] {
        ThreadAffinity affinity(masks.empty() ? 0 : masks[i % masks.size()]);
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
          lock.unlock();
          if (spin_us) {
            auto deadline = std::chrono::steady_clock::now() +
                            std::chrono::microseconds(spin_us);
            // Bounded spin bridges adjacent operators. The mutex still
            // publishes task state; workers sleep when waiting for the next
            // audio callback.
            while (generation.load(std::memory_order_acquire) == seen &&
                   std::chrono::steady_clock::now() < deadline) {
#if defined(_M_X64) || defined(__x86_64__)
              _mm_pause();
#else
              std::this_thread::yield();
#endif
            }
          }
        }
      });
  } catch (...) {
    // Join already-created workers if a later std::thread allocation fails.
    shutdown();
    throw;
  }
}
void Threadpool::shutdown() {
  {
    std::lock_guard lock(mutex);
    stopping = true;
  }
  work.notify_all();
  for (auto &t : workers)
    t.join();
}
Threadpool::~Threadpool() { shutdown(); }
void Threadpool::parallel_for(int n, const std::function<void(int, int)> &fn) {
  if (active || count == 1 || n <= 1) {
    for (int i = 0; i < n; ++i)
      fn(i, n);
    return;
  }
  std::unique_lock job(job_mutex);
  {
    std::lock_guard lock(mutex);
    failure = nullptr;
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
  if (failure)
    std::rethrow_exception(failure);
}
void Threadpool::range(size_t n, size_t grain,
                       const std::function<void(size_t, size_t)> &fn) {
  int jobs = static_cast<int>(std::min<size_t>(count, (n + grain - 1) / grain));
  if (jobs <= 1) {
    fn(0, n);
    return;
  }
  parallel_for(jobs,
               [&](int i, int jobs) { fn(n * i / jobs, n * (i + 1) / jobs); });
}
std::shared_ptr<Threadpool> shared_pool(int n, uint32_t spin, bool pin) {
  static std::mutex mutex;
  static std::unordered_map<uint64_t, std::weak_ptr<Threadpool>> pools;
  std::lock_guard lock(mutex);
  uint64_t key =
      (static_cast<uint64_t>(n) << 32) | (uint64_t(pin) << 16) | spin;
  if (auto p = pools[key].lock())
    return p;
  auto p = std::make_shared<Threadpool>(n, spin, pin);
  pools[key] = p;
  return p;
}
} // namespace vocal::cpu
