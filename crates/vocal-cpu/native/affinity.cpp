#include "affinity.h"
#include <algorithm>
#ifdef _WIN32
#include <unordered_set>
#include <windows.h>
#endif
namespace vocal::cpu {
std::vector<uintptr_t> performance_core_masks(bool enable) {
  std::vector<uintptr_t> result;
#ifdef _WIN32
  if (!enable)
    return result;
  ULONG bytes = 0;
  GetSystemCpuSetInformation(nullptr, 0, &bytes, GetCurrentProcess(), 0);
  if (!bytes)
    return result;
  std::vector<uint8_t> data(bytes);
  if (!GetSystemCpuSetInformation(
          reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(data.data()), bytes,
          &bytes, GetCurrentProcess(), 0))
    return result;
  DWORD_PTR allowed = 0, system = 0;
  if (!GetProcessAffinityMask(GetCurrentProcess(), &allowed, &system))
    return result;
  std::vector<std::pair<unsigned, unsigned>> cores;
  unsigned best = 0;
  for (size_t offset = 0; offset < bytes;) {
    auto *entry =
        reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(data.data() + offset);
    if (entry->Size < sizeof(*entry) || entry->Size > bytes - offset)
      return {};
    if (entry->Type == CpuSetInformation && entry->CpuSet.Group == 0 &&
        entry->CpuSet.LogicalProcessorIndex < sizeof(uintptr_t) * 8 &&
        (allowed & (uintptr_t(1) << entry->CpuSet.LogicalProcessorIndex)))
      best = std::max<unsigned>(best, entry->CpuSet.EfficiencyClass);
    offset += entry->Size;
  }
  for (size_t offset = 0; offset < bytes;) {
    auto *entry =
        reinterpret_cast<PSYSTEM_CPU_SET_INFORMATION>(data.data() + offset);
    if (entry->Type == CpuSetInformation && entry->CpuSet.Group == 0 &&
        entry->CpuSet.LogicalProcessorIndex < sizeof(uintptr_t) * 8 &&
        (allowed & (uintptr_t(1) << entry->CpuSet.LogicalProcessorIndex)) &&
        entry->CpuSet.EfficiencyClass == best)
      cores.emplace_back(entry->CpuSet.CoreIndex,
                         entry->CpuSet.LogicalProcessorIndex);
    offset += entry->Size;
  }
  std::unordered_set<unsigned> seen;
  std::vector<uintptr_t> siblings;
  for (auto [core, logical] : cores) {
    auto mask = uintptr_t(1) << logical;
    if (seen.insert(core).second)
      result.push_back(mask);
    else
      siblings.push_back(mask);
  }
  result.insert(result.end(), siblings.begin(), siblings.end());
#else
  (void)enable;
#endif
  return result;
}
ThreadAffinity::ThreadAffinity(uintptr_t mask) {
#ifdef _WIN32
  if (mask)
    previous = SetThreadAffinityMask(GetCurrentThread(), mask);
#else
  (void)mask;
#endif
}
ThreadAffinity::~ThreadAffinity() {
#ifdef _WIN32
  if (previous)
    SetThreadAffinityMask(GetCurrentThread(), previous);
#endif
}
} // namespace vocal::cpu
