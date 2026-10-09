#pragma once
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// Experimental C ABI; canonical tensors use contiguous NCHW or row-major order.
struct Spec {
  uint32_t op, x_rank, w_rank, y_rank;
  int64_t x[8], w[8], y[8];
  int32_t attrs[12];
  uint32_t bias;
};
inline size_t elements(const int64_t *dims, size_t rank) {
  size_t size = 1;
  for (size_t i = 0; i < rank; ++i) {
    if (dims[i] <= 0)
      throw std::runtime_error("invalid tensor dimension");
    size *= dims[i];
  }
  return size;
}
struct Aligned {
  void *ptr = nullptr;
  explicit Aligned(size_t bytes = 0) {
    ptr = _aligned_malloc(std::max<size_t>(bytes, 64), 64);
    if (!ptr)
      throw std::bad_alloc();
    std::memset(ptr, 0, std::max<size_t>(bytes, 64));
  }
  ~Aligned() { _aligned_free(ptr); }
  Aligned(const Aligned &) = delete;
  Aligned &operator=(const Aligned &) = delete;
  float *floats() { return static_cast<float *>(ptr); }
};
struct Operator {
  virtual ~Operator() = default;
  virtual void run(bool boundary) = 0;
  virtual void output(float *dst) = 0;
  virtual std::string implementation() const = 0;
};
std::unique_ptr<Operator> make_operator(const Spec &, const float *,
                                        const float *, const float *, int, int);
