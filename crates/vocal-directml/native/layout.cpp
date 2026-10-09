#include "layout.h"

namespace vdml {
DML_TENSOR_DATA_TYPE dtype(uint32_t type) {
  switch (type) {
  case 1:
    return DML_TENSOR_DATA_TYPE_FLOAT32;
  case 2:
    return DML_TENSOR_DATA_TYPE_FLOAT16;
  case 3:
    return DML_TENSOR_DATA_TYPE_INT32;
  case 4:
    return DML_TENSOR_DATA_TYPE_UINT32;
  case 5:
    return DML_TENSOR_DATA_TYPE_UINT8;
  default:
    throw std::runtime_error("invalid tensor dtype");
  }
}
Shape sizes(E x) { return x.Impl()->GetOutputDesc().sizes; }
dml::TensorStrides dense(Shape const &shape) {
  dml::TensorStrides s(shape.size());
  uint32_t stride = 1;
  for (size_t i = shape.size(); i-- > 0;) {
    s[i] = stride;
    stride *= shape[i];
  }
  return s;
}
E reshape(E x, Shape const &shape, bool contiguous_views) {
  auto const &desc = x.Impl()->GetOutputDesc();
  auto strides = desc.strides ? *desc.strides : dense(desc.sizes);
  auto expected = dense(desc.sizes);
  bool contiguous = true;
  for (size_t i = 0; i < strides.size(); ++i)
    if (desc.sizes[i] != 1 && strides[i] != expected[i])
      contiguous = false;
  // A broadcast/strided tensor cannot be flattened by changing its sizes.
  // Keep initializer-owned parameter materialization: removing it can change
  // DirectML weight packing/tactics even when a dense view is mathematically
  // equivalent. Elide only activation views and ordinary dynamic inputs.
  if (!contiguous_views || !contiguous ||
      (desc.flags & DML_TENSOR_FLAG_OWNED_BY_DML))
    x = dml::Identity(x);
  return dml::Reinterpret(x, shape, dense(shape));
}
E padded(E x, size_t rank, bool contiguous_views) {
  auto s = sizes(x);
  s.insert(s.begin(), rank - s.size(), 1);
  return reshape(x, s, contiguous_views);
}
E broadcast(E x, Shape const &target) {
  auto src = sizes(x);
  auto const &desc = x.Impl()->GetOutputDesc();
  auto strides = desc.strides ? *desc.strides : dense(src);
  strides.insert(strides.begin(), target.size() - src.size(), 0);
  src.insert(src.begin(), target.size() - src.size(), 1);
  for (size_t i = 0; i < target.size(); ++i) {
    require(src[i] == 1 || src[i] == target[i],
            "invalid native tensor broadcast");
    if (src[i] == 1)
      strides[i] = 0;
  }
  return dml::Reinterpret(x, target, strides);
}
E gemm(E a, E b, bool transpose, Shape const &wanted,
       CompileOptions const &options, dml::Optional<E> bias) {
  require(std::max(sizes(a).size(), sizes(b).size()) <= 4,
          "DirectML GEMM rank exceeds four");
  a = padded(a, 4, options.contiguous_views != 0);
  b = padded(b, 4, options.contiguous_views != 0);
  auto s = sizes(a), t = sizes(b);
  for (int i = 0; i < 2; ++i)
    s[i] = t[i] = std::max(s[i], t[i]);
  if (bias) {
    auto shape = wanted;
    shape.insert(shape.begin(), 4 - shape.size(), 1);
    bias = broadcast(*bias, shape);
  }
  auto y = dml::Gemm(
      broadcast(a, s), broadcast(b, t), bias, DML_MATRIX_TRANSFORM_NONE,
      transpose ? DML_MATRIX_TRANSFORM_TRANSPOSE : DML_MATRIX_TRANSFORM_NONE);
  return reshape(y, wanted, options.contiguous_views != 0);
}
} // namespace vdml
