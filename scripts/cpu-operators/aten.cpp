#include "common.h"
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <c10/core/InferenceMode.h>

class AtenOperator final : public Operator {
  Spec s;
  at::Tensor canonical, input, weight, bias, result;
  bool channels_last;

public:
  AtenOperator(const Spec &spec, const float *x, const float *w, const float *b,
               int threads, int layout)
      : s(spec), channels_last(layout != 0) {
    c10::InferenceMode guard;
    at::set_num_threads(threads);
    auto dims = [](const int64_t *p, size_t n) {
      return std::vector<int64_t>(p, p + n);
    };
    canonical =
        at::from_blob(const_cast<float *>(x), dims(s.x, s.x_rank), at::kFloat)
            .clone();
    weight =
        at::from_blob(const_cast<float *>(w), dims(s.w, s.w_rank), at::kFloat)
            .clone();
    if (s.bias)
      bias =
          at::from_blob(const_cast<float *>(b),
                        {s.op == 24 || s.op == 25 ? s.y[1] : s.y[s.y_rank - 1]},
                        at::kFloat)
              .clone();
    if (s.op == 24 || s.op == 25) {
      auto format = channels_last ? at::MemoryFormat::ChannelsLast
                                  : at::MemoryFormat::Contiguous;
      input = canonical.contiguous(format);
      weight = weight.contiguous(format);
    } else
      input = canonical;
  }
  void run(bool boundary) override {
    c10::InferenceMode guard;
    if (s.op == 24 || s.op == 25) {
      if (boundary)
        input =
            canonical.contiguous(channels_last ? at::MemoryFormat::ChannelsLast
                                               : at::MemoryFormat::Contiguous);
      auto a = s.attrs;
      std::optional<at::Tensor> b;
      if (bias.defined())
        b = bias;
      result = at::convolution(input, weight, b, {a[0], a[1]}, {a[4], a[5]},
                               {a[2], a[3]}, s.op == 25, {a[6], a[7]}, a[8]);
      if (boundary)
        result = result.contiguous();
    } else {
      auto lhs = input.reshape({-1, s.x[s.x_rank - 1]});
      auto rhs = s.attrs[0] ? weight.transpose(-1, -2) : weight;
      result = bias.defined() ? at::addmm(bias, lhs, rhs) : at::mm(lhs, rhs);
    }
  }
  void output(float *dst) override {
    auto out = result.contiguous();
    std::memcpy(dst, out.const_data_ptr(), out.numel() * sizeof(float));
  }
  std::string implementation() const override {
    return channels_last ? "ATen CPU channels_last (allocation included)"
                         : "ATen CPU contiguous (allocation included)";
  }
};
std::unique_ptr<Operator> make_operator(const Spec &s, const float *x,
                                        const float *w, const float *b,
                                        int threads, int layout) {
  return std::make_unique<AtenOperator>(s, x, w, b, threads, layout);
}
