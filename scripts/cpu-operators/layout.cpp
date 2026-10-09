#include "layout.h"
#include <algorithm>
#include <immintrin.h>

namespace {
void tile8(const float *input, float *output, size_t in_stride,
           size_t out_stride) {
  __m256 r[8];
  for (int i = 0; i < 8; ++i)
    r[i] = _mm256_loadu_ps(input + i * in_stride);
  __m256 t[8];
  for (int i = 0; i < 4; ++i) {
    t[2 * i] = _mm256_unpacklo_ps(r[2 * i], r[2 * i + 1]);
    t[2 * i + 1] = _mm256_unpackhi_ps(r[2 * i], r[2 * i + 1]);
  }
  __m256 u[8];
  for (int i = 0; i < 2; ++i) {
    u[4 * i] = _mm256_shuffle_ps(t[4 * i], t[4 * i + 2], 0x44);
    u[4 * i + 1] = _mm256_shuffle_ps(t[4 * i], t[4 * i + 2], 0xee);
    u[4 * i + 2] = _mm256_shuffle_ps(t[4 * i + 1], t[4 * i + 3], 0x44);
    u[4 * i + 3] = _mm256_shuffle_ps(t[4 * i + 1], t[4 * i + 3], 0xee);
  }
  for (int i = 0; i < 4; ++i) {
    _mm256_storeu_ps(output + i * out_stride,
                     _mm256_permute2f128_ps(u[i], u[i + 4], 0x20));
    _mm256_storeu_ps(output + (i + 4) * out_stride,
                     _mm256_permute2f128_ps(u[i], u[i + 4], 0x31));
  }
}
struct Transpose {
  const float *source;
  float *destination;
  size_t rows, cols, row_tiles, col_tiles;
};
void execute_tile(void *pointer, size_t index) {
  const auto &t = *static_cast<Transpose *>(pointer);
  size_t batch = index / (t.row_tiles * t.col_tiles);
  size_t tile = index % (t.row_tiles * t.col_tiles);
  size_t row = (tile / t.col_tiles) * 8, col = (tile % t.col_tiles) * 256;
  auto source = t.source + batch * t.rows * t.cols;
  auto destination = t.destination + batch * t.rows * t.cols;
  size_t row_end = std::min(row + 8, t.rows),
         col_end = std::min(col + 256, t.cols);
  if (row_end - row == 8) {
    for (; col + 8 <= col_end; col += 8)
      tile8(source + row * t.cols + col, destination + col * t.rows + row,
            t.cols, t.rows);
  }
  for (size_t r = row; r < row_end; ++r)
    for (size_t c = col; c < col_end; ++c)
      destination[c * t.rows + r] = source[r * t.cols + c];
}
void transpose(const float *source, float *destination, size_t batch,
               size_t rows, size_t cols, pthreadpool_t pool) {
  Transpose task{source, destination,    rows,
                 cols,   (rows + 7) / 8, (cols + 255) / 256};
  pthreadpool_parallelize_1d(pool, execute_tile, &task,
                             batch * task.row_tiles * task.col_tiles, 0);
}
} // namespace
void nchw_to_nhwc(const float *source, float *destination, const int64_t *shape,
                  pthreadpool_t pool) {
  transpose(source, destination, shape[0], shape[1], shape[2] * shape[3], pool);
}
void nhwc_to_nchw(const float *source, float *destination, const int64_t *shape,
                  pthreadpool_t pool) {
  transpose(source, destination, shape[0], shape[2] * shape[3], shape[1], pool);
}
