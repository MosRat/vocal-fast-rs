#pragma once
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <stdexcept>
#include <vector>
namespace vocal::cpu {
using Shape = std::vector<int64_t>;
inline void require(bool ok, const char *message) {
  if (!ok)
    throw std::runtime_error(message);
}
inline size_t dtype_bytes(uint32_t d) { return d == 2 ? 2 : d == 5 ? 1 : 4; }
struct Storage {
  size_t bytes;
  void *data;
  explicit Storage(size_t n)
      : bytes(n),
        data(::operator new(std::max<size_t>(64, n), std::align_val_t(64))) {}
  ~Storage() { ::operator delete(data, std::align_val_t(64)); }
  Storage(const Storage &) = delete;
};
struct Tensor {
  uint32_t dtype = 1;
  Shape shape, strides;
  std::shared_ptr<Storage> storage;
  size_t offset = 0;
  size_t elements() const;
  size_t bytes() const { return elements() * dtype_bytes(dtype); }
  void *data() const { return static_cast<uint8_t *>(storage->data) + offset; }
  bool contiguous() const;
  bool dense() const;
  Shape order() const;
  size_t position(size_t) const;
  double read(size_t) const;
  void write(size_t, double) const;
};
Shape contiguous_strides(const Shape &);
Shape layout_strides(const Shape &, const Shape &);
bool reshape_view(const Tensor &, const Shape &, Tensor &);
Tensor transpose_view(const Tensor &, const Shape &);
void copy_tensor(const Tensor &, const Tensor &);
} // namespace vocal::cpu
