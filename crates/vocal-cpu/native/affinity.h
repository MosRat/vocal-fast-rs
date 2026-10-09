#pragma once
#include <cstdint>
#include <vector>
namespace vocal::cpu {
std::vector<uintptr_t> performance_core_masks(bool enable);
class ThreadAffinity final {
  uintptr_t previous = 0;

public:
  explicit ThreadAffinity(uintptr_t mask);
  ~ThreadAffinity();
};
} // namespace vocal::cpu
