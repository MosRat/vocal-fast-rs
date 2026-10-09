#pragma once
#include <cstddef>
#include <cstdint>
extern "C" {
struct CpuNode {
  uint32_t op, dtype, rank;
  uint32_t shape[8];
  const uint32_t *inputs;
  size_t input_count;
  const int32_t *attrs;
  size_t attr_count;
  float values[4];
  const uint8_t *data;
  size_t data_bytes;
};
struct CpuBinding {
  uint32_t id;
  const uint8_t *data;
  size_t bytes;
};
struct CpuMemory {
  size_t tensor_bytes, workspace_bytes, packed_bytes, auxiliary_bytes;
  size_t tensor_allocations, workspace_allocations;
};
struct CpuQuantization {
  uint32_t id, mode; // 1: observe FP32 inputs; 2: static W8A8.
  float smooth_alpha;
  const float *input_max;
  size_t channels;
};
struct CpuAttention {
  uint32_t mode, query_tile, key_tile;
  bool token_major_output;
};
struct CpuConvolution {
  uint32_t mode;
  bool fuse_gelu;
  bool fuse_residual;
};
void *vocal_cpu_create(const CpuNode *, size_t, const uint32_t *, size_t,
                       size_t, uint32_t, bool, bool, bool, bool, bool, bool, bool,
                       uint32_t, const CpuConvolution *, const CpuAttention *,
                       const CpuQuantization *, size_t);
int vocal_cpu_calibration(void *, uint32_t, float *, size_t);
const char *vocal_cpu_implementation(void *, uint32_t);
int vocal_cpu_run(void *, const CpuBinding *, size_t);
int vocal_cpu_output(void *, uint32_t, uint8_t *, size_t);
double vocal_cpu_node_ms(void *, uint32_t);
void vocal_cpu_memory(void *, CpuMemory *);
const char *vocal_cpu_error();
void vocal_cpu_version(uint32_t *);
void vocal_cpu_destroy(void *);
}
