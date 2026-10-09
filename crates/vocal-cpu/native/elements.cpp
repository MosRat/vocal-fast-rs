#include "internal.h"
#include <cmath>
#include <numeric>
namespace vocal::cpu {
static double unary(double value, uint32_t op, float parameter) {
  float x = static_cast<float>(value);
  switch (op) {
  case 4:
    return value;
  case 12:
    return std::max(x, 0.f);
  case 13:
    return x < 0 ? x * parameter : x;
  case 14:
    return std::tanh(x);
  case 15:
    return 1.f / (1.f + std::exp(-x));
  case 16:
    return x * 0.5f * (1.f + std::erf(x * 0.7071067811865475f));
  case 17:
    return std::exp(x);
  case 18:
    return std::log(x);
  case 19:
    return std::sqrt(x);
  case 21:
    return std::floor(x);
  case 22:
    return std::sin(x);
  case 43:
    return std::nearbyint(x);
  default:
    throw std::runtime_error("unsupported unary operator");
  }
}
static double binary(double left, double right, uint32_t op, float parameter) {
  float a = static_cast<float>(left), b = static_cast<float>(right);
  switch (op) {
  case 5:
    return a + b;
  case 6:
    return a - b;
  case 7:
    return a * b;
  case 8:
    return a / b;
  case 10:
    return std::max(a, b);
  case 11:
    return std::min(a, b);
  case 36:
    return left > right;
  case 37:
    return left < right;
  case 45:
    return left == right;
  case 52: {
    float x = a + b;
    return x < 0 ? x * parameter : x;
  }
  default:
    throw std::runtime_error("unsupported binary operator");
  }
}
extern "C" void vocal_cpu_element_avx2(uint32_t, const float *, size_t,
                                       const float *, size_t, float *, size_t,
                                       float);
static void elementwise(Graph &g, size_t id) {
  auto &n = g.nodes[id];
  auto &y = g.tensors[id];
  auto axes = y.order();
  size_t rank = y.shape.size(), count = n.inputs.size(), inner = 1,
         coalesced = 0;
  size_t strides[3][8] = {}, slopes[3] = {};
  for (size_t j = 0; j < count; ++j) {
    auto &x = g.tensors[n.inputs[j]];
    for (size_t p = 0; p < rank; ++p) {
      int axis = static_cast<int>(axes[p]) + static_cast<int>(x.shape.size()) -
                 static_cast<int>(rank);
      strides[j][p] = axis < 0 || x.shape[axis] == 1 ? 0 : x.strides[axis];
    }
  }
  for (size_t p = 0; p < rank; ++p) {
    if (y.shape[axes[p]] == 1) {
      ++coalesced;
      continue;
    }
    bool compatible = true;
    for (size_t j = 0; j < count; ++j) {
      if (inner == 1)
        slopes[j] = strides[j][p];
      else if (strides[j][p] != inner * slopes[j])
        compatible = false;
    }
    if (!compatible)
      break;
    inner *= y.shape[axes[p]];
    ++coalesced;
  }
  bool fp32 = y.dtype == 1;
  for (auto input : n.inputs)
    fp32 = fp32 && g.tensors[input].dtype == 1;
  g.pool->range(y.elements(), 32768, [&](size_t begin, size_t end) {
    for (size_t index = begin; index < end;) {
      size_t row = index / inner, within = index % inner,
             length = std::min(inner - within, end - index);
      size_t offsets[3] = {};
      for (size_t p = coalesced; p < rank; ++p) {
        size_t coord = row % y.shape[axes[p]];
        row /= y.shape[axes[p]];
        for (size_t j = 0; j < count; ++j)
          offsets[j] += coord * strides[j][p];
      }
      for (size_t j = 0; j < count; ++j)
        offsets[j] += within * slopes[j];
#if defined(VOCAL_CPU_AVX2)
      bool simd = fp32 && g.avx2 &&
                  (n.op >= 5 && n.op <= 8 || n.op == 10 || n.op == 11 ||
                   n.op == 12 || n.op == 13 || n.op == 52);
      for (size_t j = 0; j < count; ++j)
        simd = simd && slopes[j] <= 1;
      if (simd) {
        auto *a =
            static_cast<float *>(g.tensors[n.inputs[0]].data()) + offsets[0];
        auto *b = count > 1
                      ? static_cast<float *>(g.tensors[n.inputs[1]].data()) +
                            offsets[1]
                      : a;
        vocal_cpu_element_avx2(
            n.op, a, slopes[0], b, count > 1 ? slopes[1] : slopes[0],
            static_cast<float *>(y.data()) + index, length, n.values[0]);
        index += length;
        continue;
      }
#endif
      for (size_t k = 0; k < length; ++k) {
        auto &x = g.tensors[n.inputs[0]];
        double a = x.read(offsets[0] + k * slopes[0]), value;
        if (count == 1)
          value = unary(a, n.op, n.values[0]);
        else if (n.op == 38) {
          size_t selected = a != 0 ? 1 : 2;
          value = g.tensors[n.inputs[selected]].read(offsets[selected] +
                                                     k * slopes[selected]);
        } else
          value =
              binary(a, g.tensors[n.inputs[1]].read(offsets[1] + k * slopes[1]),
                     n.op, n.values[0]);
        y.write(index + k, value);
      }
      index += length;
    }
  });
}
static void reduce(Graph &g, size_t id) {
  auto &n = g.nodes[id];
  auto &x = g.tensors[n.inputs[0]];
  auto &y = g.tensors[id];
  Shape reduced;
  size_t reduction = 1;
  for (auto axis : n.attrs) {
    reduced.push_back(axis);
    reduction *= x.shape[axis];
  }
  g.pool->range(y.elements(), 256, [&](size_t begin, size_t end) {
    for (size_t i = begin; i < end; ++i) {
      size_t base = 0, index = i;
      for (size_t a = y.shape.size(); a-- > 0;) {
        base += (index % y.shape[a]) * x.strides[a];
        index /= y.shape[a];
      }
      double value = n.op == 29 ? -INFINITY : 0;
      if (x.dtype == 1 && reduced.size() == 1) {
        auto *data = static_cast<float *>(x.data()) + base;
        auto stride = x.strides[reduced[0]];
        for (size_t j = 0; j < reduction; ++j) {
          double v = data[j * stride];
          value = n.op == 29 ? std::max(value, v) : value + v;
        }
        if (n.op == 28)
          value /= reduction;
        y.write(y.position(i), value);
        continue;
      }
      for (size_t j = 0; j < reduction; ++j) {
        size_t at = base, r = j;
        for (size_t a = reduced.size(); a-- > 0;) {
          auto axis = reduced[a];
          at += (r % x.shape[axis]) * x.strides[axis];
          r /= x.shape[axis];
        }
        double v = x.read(at);
        value = n.op == 29 ? std::max(value, v) : value + v;
      }
      if (n.op == 28)
        value /= reduction;
      y.write(y.position(i), value);
    }
  });
}
static void norm(Graph &g, size_t id) {
  auto &n = g.nodes[id];
  auto &x = g.tensors[n.inputs[0]];
  auto &y = g.tensors[id];
  auto &gamma = g.tensors[n.inputs[1]];
  auto &beta = g.tensors[n.inputs[2]];
  if (n.attrs.size() == 2) {
    channel_temporal_norm(g, id);
    return;
  }
  size_t axis = n.attrs[0], width = x.shape[axis], rows = x.elements() / width;
  g.pool->range(rows, 64, [&](size_t begin, size_t end) {
    for (size_t row = begin; row < end; ++row) {
      size_t ix = 0, iy = 0, q = row;
      for (size_t a = x.shape.size(); a-- > 0;)
        if (a != axis) {
          size_t coord = q % x.shape[a];
          q /= x.shape[a];
          ix += coord * x.strides[a];
          iy += coord * y.strides[a];
        }
      double mean = 0, var = 0;
      for (size_t j = 0; j < width; ++j)
        mean += x.read(ix + j * x.strides[axis]);
      mean /= width;
      for (size_t j = 0; j < width; ++j) {
        double d = x.read(ix + j * x.strides[axis]) - mean;
        var += d * d;
      }
      float inv =
          1.f / std::sqrt(static_cast<float>(var / width) + n.values[0]);
      float center = static_cast<float>(mean);
      for (size_t j = 0; j < width; ++j) {
        float value =
            (static_cast<float>(x.read(ix + j * x.strides[axis])) - center) *
            inv;
        value = value * static_cast<float>(gamma.read(gamma.position(j))) +
                static_cast<float>(beta.read(beta.position(j)));
        y.write(iy + j * y.strides[axis], value);
      }
    }
  });
}
static void concat(Graph &g, size_t id) {
  auto &n = g.nodes[id];
  auto &y = g.tensors[id];
  size_t axis = n.attrs[0];
  auto axes = y.order();
  size_t inner = 1, p = 0;
  while (p < axes.size() && axes[p] != static_cast<int64_t>(axis)) {
    inner *= y.shape[axes[p++]];
  }
  size_t rows = y.elements() / (inner * y.shape[axis]);
  g.pool->range(rows, 128, [&](size_t begin, size_t end) {
    for (size_t row = begin; row < end; ++row) {
      size_t remaining = row, coords[8] = {};
      for (size_t a = p + 1; a < axes.size(); ++a) {
        coords[axes[a]] = remaining % y.shape[axes[a]];
        remaining /= y.shape[axes[a]];
      }
      size_t output = row * inner * y.shape[axis];
      for (auto input : n.inputs) {
        auto &x = g.tensors[input];
        size_t base = 0;
        for (size_t a = p + 1; a < axes.size(); ++a)
          base += coords[axes[a]] * x.strides[axes[a]];
        bool dense = x.dense() && x.order() == axes;
        if (dense) {
          size_t length = inner * x.shape[axis];
          std::memcpy(
              static_cast<uint8_t *>(y.data()) + output * dtype_bytes(y.dtype),
              static_cast<uint8_t *>(x.data()) + base * dtype_bytes(x.dtype),
              length * dtype_bytes(x.dtype));
          output += length;
        } else
          for (size_t j = 0; j < static_cast<size_t>(x.shape[axis]); ++j)
            for (size_t k = 0; k < inner; ++k) {
              size_t at = base + j * x.strides[axis], q = k;
              for (size_t a = 0; a < p; ++a) {
                at += (q % y.shape[axes[a]]) * x.strides[axes[a]];
                q /= y.shape[axes[a]];
              }
              y.write(output++, x.read(at));
            }
      }
    }
  });
}
void execute_elements(Graph &g, size_t id) {
  auto &n = g.nodes[id];
  auto &y = g.tensors[id];
  auto &x = g.tensors[n.inputs[0]];
  auto &a = n.attrs;
  if (n.op == 2) {
    copy_tensor(x, y);
    return;
  }
  if (n.op == 26) {
    size_t axis = a[0], width = x.shape[axis], rows = x.elements() / width;
    g.pool->range(rows, 64, [&](size_t begin, size_t end) {
      for (size_t row = begin; row < end; ++row) {
        size_t bx = 0, by = 0, q = row;
        for (size_t p = x.shape.size(); p-- > 0;)
          if (p != axis) {
            size_t c = q % x.shape[p];
            q /= x.shape[p];
            bx += c * x.strides[p];
            by += c * y.strides[p];
          }
        float maximum = -INFINITY;
        for (size_t j = 0; j < width; ++j)
          maximum = std::max(
              maximum, static_cast<float>(x.read(bx + j * x.strides[axis])));
        float sum = 0;
        for (size_t j = 0; j < width; ++j) {
          float v = std::exp(
              static_cast<float>(x.read(bx + j * x.strides[axis])) - maximum);
          y.write(by + j * y.strides[axis], v);
          sum += v;
        }
        for (size_t j = 0; j < width; ++j)
          y.write(by + j * y.strides[axis],
                  y.read(by + j * y.strides[axis]) / sum);
      }
    });
    return;
  }
  if (n.op == 27 || n.op == 28 || n.op == 29) {
    reduce(g, id);
    return;
  }
  if (n.op == 51) {
    norm(g, id);
    return;
  }
  if (n.op == 33) {
    concat(g, id);
    return;
  }
  if (n.op == 30 || n.op == 31) {
    if (n.op == 31 && g.fast_tensor_io && gather_elements_rows(g, id))
      return;
    auto &indices = g.tensors[n.inputs[1]];
    size_t axis = a[0];
    g.pool->range(y.elements(), 32768, [&](size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) {
        size_t coords[8] = {}, q = i;
        for (size_t p = y.shape.size(); p-- > 0;) {
          coords[p] = q % y.shape[p];
          q /= y.shape[p];
        }
        size_t at = 0, index = 0;
        if (n.op == 31) {
          index = i;
          for (size_t p = 0; p < x.shape.size(); ++p)
            if (p != axis)
              at += coords[p] * x.strides[p];
        } else {
          size_t inserted = indices.shape.size();
          for (size_t p = 0; p < axis; ++p)
            at += coords[p] * x.strides[p];
          for (size_t p = 0; p < inserted; ++p)
            index = index * indices.shape[p] + coords[axis + p];
          for (size_t p = axis + 1; p < x.shape.size(); ++p)
            at += coords[p + inserted - 1] * x.strides[p];
        }
        int64_t selected =
            static_cast<int64_t>(indices.read(indices.position(index)));
        if (selected < 0)
          selected += x.shape[axis];
        require(selected >= 0 && selected < x.shape[axis],
                "gather index out of range");
        at += selected * x.strides[axis];
        y.write(y.position(i), x.read(at));
      }
    });
    return;
  }
  if (n.op == 32) {
    size_t axis = a[0];
    g.pool->range(y.elements(), 32768, [&](size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) {
        size_t at = 0, q = i;
        for (size_t p = y.shape.size(); p-- > 0;) {
          int64_t coord = q % y.shape[p];
          q /= y.shape[p];
          if (p == axis)
            coord = a[1] + coord * a[2];
          require(coord >= 0 && coord < x.shape[p], "slice out of range");
          at += coord * x.strides[p];
        }
        y.write(y.position(i), x.read(at));
      }
    });
    return;
  }
  if (n.op == 35) {
    size_t axis = a[0], width = x.shape[axis], rows = x.elements() / width;
    bool exclusive = a.size() > 1 && a[1];
    g.pool->range(rows, 128, [&](size_t begin, size_t end) {
      for (size_t row = begin; row < end; ++row) {
        size_t bx = 0, by = 0, q = row;
        for (size_t p = x.shape.size(); p-- > 0;)
          if (p != axis) {
            size_t c = q % x.shape[p];
            q /= x.shape[p];
            bx += c * x.strides[p];
            by += c * y.strides[p];
          }
        double sum = 0;
        for (size_t j = 0; j < width; ++j) {
          if (exclusive)
            y.write(by + j * y.strides[axis], sum);
          sum += x.read(bx + j * x.strides[axis]);
          if (!exclusive)
            y.write(by + j * y.strides[axis], sum);
        }
      }
    });
    return;
  }
  if (n.op == 41) {
    g.pool->range(y.elements(), 32768, [&](size_t begin, size_t end) {
      for (size_t i = begin; i < end; ++i) {
        size_t q = i, w = q % y.shape[3];
        q /= y.shape[3];
        size_t h = q % y.shape[2];
        q /= y.shape[2];
        size_t c = q % y.shape[1], b = q / y.shape[1],
               base = b * x.strides[0] + c * x.strides[1] +
                      2 * h * x.strides[2] + 2 * w * x.strides[3];
        float v = 0;
        for (size_t u = 0; u < 2; ++u)
          for (size_t z = 0; z < 2; ++z)
            v += static_cast<float>(
                x.read(base + u * x.strides[2] + z * x.strides[3]));
        y.write(y.position(i), v * 0.25f);
      }
    });
    return;
  }
  if (n.op == 42) {
    size_t width = x.shape.back(), k = a[0], rows = x.elements() / width;
    require(k <= width, "topk exceeds input");
    g.pool->range(rows, 64, [&](size_t begin, size_t end) {
      std::vector<std::pair<double, size_t>> values(width);
      for (size_t row = begin; row < end; ++row) {
        for (size_t j = 0; j < width; ++j)
          values[j] = {x.read(x.position(row * width + j)), j};
        std::partial_sort(
            values.begin(), values.begin() + k, values.end(),
            [&](auto l, auto r) {
              return a[1] ? (l.first > r.first ||
                             (l.first == r.first && l.second < r.second))
                          : (l.first < r.first ||
                             (l.first == r.first && l.second < r.second));
            });
        for (size_t j = 0; j < k; ++j)
          y.write(y.position(row * k + j),
                  a[2] ? static_cast<double>(values[j].second)
                       : values[j].first);
      }
    });
    return;
  }
  elementwise(g, id);
}
} // namespace vocal::cpu
