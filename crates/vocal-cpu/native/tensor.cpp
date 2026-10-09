#include "tensor.h"
#include <cmath>
#include <numeric>
namespace vocal::cpu {
size_t Tensor::elements() const {
  size_t n = 1;
  for (auto d : shape) {
    require(d > 0, "invalid tensor shape");
    require(n <= SIZE_MAX / d, "tensor size overflow");
    n *= d;
  }
  return n;
}
Shape contiguous_strides(const Shape &shape) {
  Shape s(shape.size());
  int64_t n = 1;
  for (size_t i = shape.size(); i-- > 0;) {
    s[i] = n;
    n *= shape[i];
  }
  return s;
}
Shape layout_strides(const Shape &shape, const Shape &axes) {
  Shape s(shape.size());
  int64_t n = 1;
  for (auto a : axes) {
    s[a] = n;
    n *= shape[a];
  }
  return s;
}
bool Tensor::contiguous() const {
  auto s = contiguous_strides(shape);
  for (size_t i = 0; i < s.size(); ++i)
    if (shape[i] > 1 && strides[i] != s[i])
      return false;
  return true;
}
Shape Tensor::order() const {
  Shape a(shape.size());
  std::iota(a.begin(), a.end(), 0);
  std::stable_sort(a.begin(), a.end(),
                   [&](auto x, auto y) { return strides[x] < strides[y]; });
  return a;
}
bool Tensor::dense() const {
  int64_t n = 1;
  for (auto a : order()) {
    if (shape[a] == 1)
      continue;
    if (strides[a] != n)
      return false;
    n *= shape[a];
  }
  return true;
}
size_t Tensor::position(size_t i) const {
  size_t p = 0;
  for (size_t a = shape.size(); a-- > 0;) {
    p += (i % shape[a]) * strides[a];
    i /= shape[a];
  }
  return p;
}
static float half_float(uint16_t v) {
  uint32_t sign = (v & 0x8000u) << 16, e = (v >> 10) & 31, m = v & 1023, bits;
  if (!e) {
    if (!m)
      bits = sign;
    else {
      int shift = 0;
      while (!(m & 1024)) {
        m <<= 1;
        ++shift;
      }
      bits = sign | ((127 - 14 - shift) << 23) | ((m & 1023) << 13);
    }
  } else
    bits = sign | ((e == 31 ? 255 : e + 112) << 23) | (m << 13);
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}
static uint16_t float_half(float f) {
  uint32_t b;
  std::memcpy(&b, &f, 4);
  uint32_t sign = (b >> 16) & 0x8000, m = b & 0x7fffff;
  int e = ((b >> 23) & 255) - 112;
  if (e >= 31)
    return static_cast<uint16_t>(sign | 0x7c00 |
                                 ((((b >> 23) & 255) == 255 && m) ? 0x200 : 0));
  if (e <= 0) {
    if (e < -10)
      return static_cast<uint16_t>(sign);
    m |= 0x800000;
    int s = 14 - e;
    return static_cast<uint16_t>(
        sign | ((m + ((1u << (s - 1)) - 1) + ((m >> s) & 1)) >> s));
  }
  uint32_t rounded = m + 0xfff + ((m >> 13) & 1);
  if (rounded & 0x800000) {
    rounded = 0;
    ++e;
  }
  return static_cast<uint16_t>(sign | (e << 10) | (rounded >> 13));
}
double Tensor::read(size_t i) const {
  switch (dtype) {
  case 1:
    return static_cast<float *>(data())[i];
  case 2:
    return half_float(static_cast<uint16_t *>(data())[i]);
  case 3:
    return static_cast<int32_t *>(data())[i];
  case 4:
    return static_cast<uint32_t *>(data())[i];
  case 5:
    return static_cast<uint8_t *>(data())[i];
  default:
    throw std::runtime_error("unsupported dtype");
  }
}
void Tensor::write(size_t i, double v) const {
  switch (dtype) {
  case 1:
    static_cast<float *>(data())[i] = static_cast<float>(v);
    break;
  case 2:
    static_cast<uint16_t *>(data())[i] = float_half(static_cast<float>(v));
    break;
  case 3:
    require(std::isfinite(v) && v >= INT32_MIN && v <= INT32_MAX,
            "INT32 cast out of range");
    static_cast<int32_t *>(data())[i] = static_cast<int32_t>(v);
    break;
  case 4:
    require(std::isfinite(v) && v >= 0 && v <= UINT32_MAX,
            "U32 cast out of range");
    static_cast<uint32_t *>(data())[i] = static_cast<uint32_t>(v);
    break;
  case 5:
    static_cast<uint8_t *>(data())[i] = static_cast<uint8_t>(v);
    break;
  default:
    throw std::runtime_error("unsupported dtype");
  }
}
bool reshape_view(const Tensor &x, const Shape &shape, Tensor &y) {
  y = x;
  y.shape = shape;
  y.strides.resize(shape.size());
  if (x.contiguous()) {
    y.strides = contiguous_strides(shape);
    return true;
  }
  int target = static_cast<int>(shape.size()) - 1,
      old = static_cast<int>(x.shape.size()) - 1;
  while (old >= 0) {
    while (old >= 0 && x.shape[old] == 1)
      --old;
    if (old < 0)
      break;
    int64_t base = x.strides[old], chunk = x.shape[old];
    --old;
    while (old >= 0) {
      if (x.shape[old] == 1) {
        --old;
        continue;
      }
      if (x.strides[old] != chunk * base)
        break;
      chunk *= x.shape[old--];
    }
    int64_t covered = 1;
    while (target >= 0 && (covered < chunk || shape[target] == 1)) {
      y.strides[target] = covered * base;
      covered *= shape[target--];
    }
    if (covered != chunk)
      return false;
  }
  while (target >= 0 && shape[target] == 1)
    y.strides[target--] = x.elements();
  return target < 0;
}
Tensor transpose_view(const Tensor &x, const Shape &axes) {
  Tensor y = x;
  for (size_t i = 0; i < axes.size(); ++i) {
    y.shape[i] = x.shape[axes[i]];
    y.strides[i] = x.strides[axes[i]];
  }
  return y;
}
void copy_tensor(const Tensor &x, const Tensor &y) {
  require(x.elements() == y.elements(), "copy size mismatch");
  if (x.dtype == y.dtype && x.contiguous() && y.contiguous())
    std::memcpy(y.data(), x.data(), x.bytes());
  else
    for (size_t i = 0; i < x.elements(); ++i)
      y.write(y.position(i), x.read(x.position(i)));
}
} // namespace vocal::cpu
