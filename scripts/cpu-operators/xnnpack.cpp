#include "common.h"
#include "layout.h"
#include <limits>
#include <xnnpack.h>

static void check(xnn_status status) {
  if (status != xnn_status_success)
    throw std::runtime_error("XNNPACK status " + std::to_string(status));
}
class XnnOperator final : public Operator {
  Spec s;
  xnn_operator_t op = nullptr;
  pthreadpool_t pool = nullptr;
  Aligned input, output_buffer, canonical_output;
  std::unique_ptr<Aligned> workspace;
  std::vector<float> canonical, weights;

public:
  XnnOperator(const Spec &spec, const float *x, const float *w, const float *b,
              int threads)
      : s(spec), input(elements(s.x, s.x_rank) * 4 + XNN_EXTRA_BYTES),
        output_buffer(elements(s.y, s.y_rank) * 4 + XNN_EXTRA_BYTES),
        canonical_output(elements(s.y, s.y_rank) * 4) {
    check(xnn_initialize(nullptr));
    pool = pthreadpool_create(threads);
    if (!pool)
      throw std::runtime_error("XNNPACK threadpool allocation failed");
    try {
      canonical.assign(x, x + elements(s.x, s.x_rank));
      auto infinity = std::numeric_limits<float>::infinity();
      if (s.op == 24 || s.op == 25) {
        nchw_to_nhwc(x, input.floats(), s.x, pool);
        size_t groups = s.attrs[8], ic = s.x[1] / groups, oc = s.y[1] / groups,
               kh = s.w[2], kw = s.w[3];
        weights.resize(elements(s.w, s.w_rank) + XNN_EXTRA_BYTES / 4);
        // XNNPACK uses GOHWI; Torch deconv uses GIOHW.
        for (size_t g = 0; g < groups; ++g)
          for (size_t o = 0; o < oc; ++o)
            for (size_t i = 0; i < ic; ++i)
              for (size_t h = 0; h < kh; ++h)
                for (size_t k = 0; k < kw; ++k) {
                  size_t dst = ((((g * oc + o) * kh + h) * kw + k) * ic + i);
                  size_t src =
                      s.op == 24
                          ? ((((g * oc + o) * ic + i) * kh + h) * kw + k)
                          : ((((g * ic + i) * oc + o) * kh + h) * kw + k);
                  weights[dst] = w[src];
                }
        auto a = s.attrs;
        size_t oh = 0, ow = 0;
        if (s.op == 24) {
          check(xnn_create_convolution2d_nhwc_f32(
              a[4], a[5], a[4], a[5], kh, kw, a[0], a[1], a[2], a[3], groups,
              ic, oc, s.x[1], s.y[1], weights.data(), s.bias ? b : nullptr,
              -infinity, infinity, 0, nullptr, nullptr, &op));
          size_t bytes = 0, alignment = 0;
          check(xnn_reshape_convolution2d_nhwc_f32(
              op, s.x[0], s.x[2], s.x[3], &bytes, &alignment, &oh, &ow, pool));
          workspace = std::make_unique<Aligned>(bytes);
          check(xnn_setup_convolution2d_nhwc_f32(
              op, workspace->ptr, input.floats(), output_buffer.floats()));
        } else {
          check(xnn_create_deconvolution2d_nhwc_f32(
              a[4], a[5], a[4], a[5], kh, kw, a[0], a[1], a[2], a[3], groups,
              ic, oc, s.x[1], s.y[1], weights.data(), s.bias ? b : nullptr,
              -infinity, infinity, 0, nullptr, nullptr, &op));
          check(xnn_reshape_deconvolution2d_nhwc_f32(
              op, s.x[0], s.x[2], s.x[3], a[6], a[7], &oh, &ow, pool));
          check(xnn_setup_deconvolution2d_nhwc_f32(op, input.floats(),
                                                   output_buffer.floats()));
        }
        if (oh != s.y[2] || ow != s.y[3])
          throw std::runtime_error("XNNPACK output shape mismatch");
      } else {
        size_t k = s.x[s.x_rank - 1], n = s.y[s.y_rank - 1],
               m = canonical.size() / k;
        std::memcpy(input.ptr, x, canonical.size() * 4);
        weights.resize(k * n + XNN_EXTRA_BYTES / 4);
        for (size_t i = 0; i < k; ++i)
          for (size_t j = 0; j < n; ++j)
            weights[j * k + i] = s.attrs[0] ? w[j * k + i] : w[i * n + j];
        check(xnn_create_fully_connected_nc_f32(
            k, n, k, n, weights.data(), s.bias ? b : nullptr, -infinity,
            infinity, 0, nullptr, nullptr, &op));
        check(xnn_reshape_fully_connected_nc_f32(op, m, pool));
        check(xnn_setup_fully_connected_nc_f32(op, input.floats(),
                                               output_buffer.floats()));
      }
    } catch (...) {
      if (op)
        xnn_delete_operator(op);
      pthreadpool_destroy(pool);
      throw;
    }
  }
  ~XnnOperator() override {
    xnn_delete_operator(op);
    pthreadpool_destroy(pool);
  }
  void run(bool boundary) override {
    bool conv = s.op == 24 || s.op == 25;
    if (boundary && conv)
      nchw_to_nhwc(canonical.data(), input.floats(), s.x, pool);
    check(xnn_run_operator(op, pool));
    if (boundary && conv)
      nhwc_to_nchw(output_buffer.floats(), canonical_output.floats(), s.y,
                   pool);
  }
  void output(float *dst) override {
    if (s.op == 24 || s.op == 25)
      nhwc_to_nchw(output_buffer.floats(), dst, s.y, pool);
    else
      std::memcpy(dst, output_buffer.ptr, elements(s.y, s.y_rank) * 4);
  }
  std::string implementation() const override {
    return "XNNPACK static weights, NHWC, pthreadpool";
  }
};
std::unique_ptr<Operator> make_operator(const Spec &s, const float *x,
                                        const float *w, const float *b,
                                        int threads, int) {
  return std::make_unique<XnnOperator>(s, x, w, b, threads);
}
