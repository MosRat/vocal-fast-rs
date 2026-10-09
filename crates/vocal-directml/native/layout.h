#pragma once
#include "bridge.h"
#include <DirectMLX.h>

namespace vdml {
using E = dml::Expression;
using Shape = dml::TensorDimensions;
DML_TENSOR_DATA_TYPE dtype(uint32_t type);
Shape sizes(E x);
dml::TensorStrides dense(Shape const &shape);
E reshape(E x, Shape const &shape, bool contiguous_views = false);
E padded(E x, size_t rank, bool contiguous_views = false);
E broadcast(E x, Shape const &target);
E gemm(E a, E b, bool transpose, Shape const &wanted,
       CompileOptions const &options, dml::Optional<E> bias = dml::NullOpt);
} // namespace vdml
