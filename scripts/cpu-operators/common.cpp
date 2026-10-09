#include "common.h"
static thread_local std::string error;
extern "C" __declspec(dllexport) const char *bench_error() {
  return error.c_str();
}
extern "C" __declspec(dllexport) void *
bench_create(const Spec *s, const float *x, const float *w, const float *b,
             int threads, int layout) {
  try {
    if (!s || !x || !w || threads < 1 || threads > 64)
      throw std::runtime_error("invalid benchmark arguments");
    return make_operator(*s, x, w, b, threads, layout).release();
  } catch (const std::exception &e) {
    error = e.what();
    return nullptr;
  }
}
extern "C" __declspec(dllexport) int bench_measure(void *p, int boundary,
                                                   int warmup, int repeats,
                                                   int loops, double *ms,
                                                   float *output) {
  try {
    if (!p || warmup < 0 || repeats < 1 || loops < 1)
      throw std::runtime_error("invalid measurement arguments");
    auto &op = *static_cast<Operator *>(p);
    for (int i = 0; i < warmup; ++i)
      op.run(boundary != 0);
    for (int r = 0; r < repeats; ++r) {
      auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < loops; ++i)
        op.run(boundary != 0);
      ms[r] = std::chrono::duration<double, std::milli>(
                  std::chrono::steady_clock::now() - start)
                  .count() /
              loops;
    }
    op.output(output); // Numerical checks occur outside the timed interval.
    return 0;
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}
extern "C" __declspec(dllexport) const char *bench_implementation(void *p) {
  static thread_local std::string name;
  name = static_cast<Operator *>(p)->implementation();
  return name.c_str();
}
extern "C" __declspec(dllexport) void bench_destroy(void *p) {
  delete static_cast<Operator *>(p);
}
