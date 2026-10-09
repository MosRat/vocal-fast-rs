#include "common.h"
#include "threadpool.h"
#include <cmath>
#include <cstdlib>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_threadpool.hpp>
#include <unordered_map>
using namespace dnnl;

class DnnlOperator final : public Operator {
  Threadpool pool;
  engine eng{engine::kind::cpu, 0};
  stream execution;
  memory canonical_x, canonical_y, input, weight, destination_memory, bias,
      scratch;
  primitive operation;
  std::unique_ptr<reorder> upload, download;
  std::unordered_map<int, memory> arguments;
  std::vector<float> source, weights, biases, destination;
  std::string implementation_name;
#ifdef VOCAL_BENCH_INT8
  std::vector<float> weight_scales;
  std::vector<int8_t> quantized_weights;
  float source_scale = 1;
  memory source_scales, packed_scales;
  std::unordered_map<int, memory> upload_args;
#endif

public:
  DnnlOperator(const Spec &s, const float *x, const float *w, const float *b,
               int threads, int layout)
      : pool(threads), execution(threadpool_interop::make_stream(eng, &pool)) {
    using dt = memory::data_type;
    using tag = memory::format_tag;
    dnnl_threadpool_interop_set_max_concurrency(threads);
    source.assign(x, x + elements(s.x, s.x_rank));
    weights.assign(w, w + elements(s.w, s.w_rank));
    destination.resize(elements(s.y, s.y_rank));
    primitive_attr attr;
    attr.set_scratchpad_mode(scratchpad_mode::user);
    memory::desc xm, wm, ym, bm, sm;
    memory::desc original_x, original_w, original_y;
#ifdef VOCAL_BENCH_INT8
    const auto activation_type = dt::s8, weight_type = dt::s8;
    const int weight_mask =
        s.op == 24 || s.op == 25 ? (s.attrs[8] == 1 ? 1 : 3) : 2;
    attr.set_scales_mask(DNNL_ARG_SRC, 0);
    attr.set_scales_mask(DNNL_ARG_WEIGHTS, weight_mask);
#else
    const auto activation_type = dt::f32, weight_type = dt::f32;
#endif
    auto finish = [&](const auto &pd) {
      xm = pd.src_desc();
      wm = pd.weights_desc();
      ym = pd.dst_desc();
      sm = pd.scratchpad_desc();
      if (s.bias)
        bm = pd.bias_desc();
      implementation_name = pd.impl_info_str();
    };
    if (s.op == 24 || s.op == 25) {
      memory::dims xd(s.x, s.x + 4), yd(s.y, s.y + 4), wd;
      int groups = s.attrs[8];
      if (s.op == 24)
        wd.assign(s.w, s.w + 4);
      else
        wd = {s.y[1], s.x[1] / groups, s.w[2], s.w[3]};
      if (groups != 1)
        wd = {groups, yd[1] / groups, xd[1] / groups, s.w[2], s.w[3]};
      original_x = {xd, dt::f32, tag::nchw};
      original_y = {yd, dt::f32, tag::nchw};
      original_w = {wd, dt::f32,
                    groups == 1 ? (s.op == 24 ? tag::oihw : tag::iohw)
                                : (s.op == 24 ? tag::goihw : tag::giohw)};
      auto preferred = layout == 2   ? tag::any
                       : layout == 1 ? tag::nhwc
                                     : tag::nchw;
      auto xdesc = memory::desc(xd, activation_type, preferred),
           wdesc = memory::desc(wd, weight_type, tag::any),
           ydesc = memory::desc(yd, dt::f32, preferred);
      auto bdesc = memory::desc({yd[1]}, dt::f32, tag::x);
      memory::dims stride = {s.attrs[0], s.attrs[1]},
                   dilation = {s.attrs[2] - 1, s.attrs[3] - 1};
      memory::dims left = {s.attrs[4], s.attrs[5]},
                   right = {s.attrs[4] - s.attrs[6], s.attrs[5] - s.attrs[7]};
      if (s.op == 24) {
        auto pd = s.bias
                      ? convolution_forward::primitive_desc(
                            eng, prop_kind::forward_inference,
                            algorithm::convolution_direct, xdesc, wdesc, bdesc,
                            ydesc, stride, dilation, left, right, attr)
                      : convolution_forward::primitive_desc(
                            eng, prop_kind::forward_inference,
                            algorithm::convolution_direct, xdesc, wdesc, ydesc,
                            stride, dilation, left, right, attr);
        finish(pd);
        operation = convolution_forward(pd);
      } else {
        auto pd = s.bias
                      ? deconvolution_forward::primitive_desc(
                            eng, prop_kind::forward_inference,
                            algorithm::deconvolution_direct, xdesc, wdesc,
                            bdesc, ydesc, stride, dilation, left, right, attr)
                      : deconvolution_forward::primitive_desc(
                            eng, prop_kind::forward_inference,
                            algorithm::deconvolution_direct, xdesc, wdesc,
                            ydesc, stride, dilation, left, right, attr);
        finish(pd);
        operation = deconvolution_forward(pd);
      }
    } else {
      int64_t k = s.x[s.x_rank - 1],
              m = static_cast<int64_t>(source.size()) / k,
              n = s.y[s.y_rank - 1];
      original_x = {{static_cast<int64_t>(m), k}, dt::f32, tag::ab};
      original_y = {{static_cast<int64_t>(m), n}, dt::f32, tag::ab};
      original_w =
          memory::desc({k, n}, dt::f32,
                       s.attrs[0] ? memory::dims{1, k} : memory::dims{n, 1});
      auto wdesc = memory::desc({k, n}, weight_type, tag::any);
      auto xdesc = memory::desc({m, k}, activation_type, tag::ab);
      auto pd =
          s.bias
              ? matmul::primitive_desc(eng, xdesc, wdesc,
                                       memory::desc({1, n}, dt::f32, tag::ab),
                                       original_y, attr)
              : matmul::primitive_desc(eng, xdesc, wdesc, original_y, attr);
      finish(pd);
      operation = matmul(pd);
    }
    canonical_x = memory(original_x, eng, source.data());
    canonical_y = memory(original_y, eng, destination.data());
    input = xm == original_x ? canonical_x : memory(xm, eng);
    destination_memory = ym == original_y ? canonical_y : memory(ym, eng);
    weight = memory(wm, eng);
    auto rawweight = memory(original_w, eng, weights.data());
#ifdef VOCAL_BENCH_INT8
    // Benchmark-only calibration uses this captured input. Production accuracy
    // experiments use disjoint calibration windows in experiment-cpu-w8a8.py.
    int range = std::getenv("VOCAL_INT8_REDUCED_RANGE") ? 63 : 127;
    bool even_grid = std::getenv("VOCAL_INT8_EVEN_WEIGHT_GRID") != nullptr;
    int weight_range = even_grid ? 126 : range;
    float weight_step = even_grid ? 2.f : 1.f;
    source_scale = 1e-8f;
    for (float value : source)
      source_scale = std::max(source_scale, std::abs(value));
    source_scale /= range;
    auto dims = original_w.get_dims(), strides = original_w.get_strides();
    bool conv = s.op == 24 || s.op == 25;
    size_t channels = conv ? s.y[1] : s.y[s.y_rank - 1];
    weight_scales.assign(channels, 1e-8f);
    for (size_t i = 0; i < weights.size(); ++i) {
      size_t rest = i, physical = 0, channel = 0;
      size_t coords[8] = {};
      for (size_t j = dims.size(); j-- > 0;) {
        coords[j] = rest % dims[j];
        rest /= dims[j];
        physical += coords[j] * strides[j];
      }
      channel = !conv              ? coords[1]
                : dims.size() == 5 ? coords[0] * dims[1] + coords[1]
                                   : coords[0];
      weight_scales[channel] =
          std::max(weight_scales[channel], std::abs(weights[physical]));
    }
    for (auto &scale : weight_scales)
      scale /= weight_range;
    source_scales = memory({{1}, dt::f32, tag::x}, eng, &source_scale);
    packed_scales = memory({{static_cast<int64_t>(channels)}, dt::f32, tag::x},
                           eng, weight_scales.data());
    // Some packed GEMM formats reject per-channel FP32->S8 reorders. Quantize
    // canonical weights once, then use the normal S8 layout reorder.
    quantized_weights.resize(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
      size_t rest = i, physical = 0, coords[8] = {};
      for (size_t j = dims.size(); j-- > 0;) {
        coords[j] = rest % dims[j];
        rest /= dims[j];
        physical += coords[j] * strides[j];
      }
      size_t channel = !conv              ? coords[1]
                       : dims.size() == 5 ? coords[0] * dims[1] + coords[1]
                                          : coords[0];
      quantized_weights[physical] = static_cast<int8_t>(std::clamp(
          std::nearbyint(weights[physical] / weight_scales[channel] /
                         weight_step) *
              weight_step,
          -static_cast<float>(weight_range), static_cast<float>(weight_range)));
    }
    auto quantized_md = memory::desc(dims, dt::s8, strides);
    auto quantized_memory = memory(quantized_md, eng, quantized_weights.data());
    reorder(quantized_memory, weight)
        .execute(execution, quantized_memory, weight);
    primitive_attr xa;
    xa.set_scales_mask(DNNL_ARG_DST, 0);
    upload = std::make_unique<reorder>(
        reorder::primitive_desc(eng, original_x, eng, xm, xa));
    upload_args = {{DNNL_ARG_FROM, canonical_x},
                   {DNNL_ARG_TO, input},
                   { DNNL_ARG_ATTR_SCALES | DNNL_ARG_DST,
                     source_scales }};
    upload->execute(execution, upload_args);
#else
    reorder(rawweight, weight).execute(execution, rawweight, weight);
    if (xm != original_x) {
      upload = std::make_unique<reorder>(canonical_x, input);
      upload->execute(execution, canonical_x, input);
    }
#endif
    if (ym != original_y)
      download = std::make_unique<reorder>(destination_memory, canonical_y);
    arguments = {{DNNL_ARG_SRC, input},
                 {DNNL_ARG_WEIGHTS, weight},
                 {DNNL_ARG_DST, destination_memory}};
#ifdef VOCAL_BENCH_INT8
    arguments.emplace(DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC, source_scales);
    arguments.emplace(DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, packed_scales);
    implementation_name +=
        ";s8xs8->s32->f32;isa=" + std::to_string(dnnl_get_effective_cpu_isa());
#endif
    if (s.bias) {
      size_t count = s.op == 24 || s.op == 25 ? s.y[1] : s.y[s.y_rank - 1];
      biases.assign(b, b + count);
      bias = memory(bm, eng, biases.data());
      arguments.emplace(DNNL_ARG_BIAS, bias);
    }
    scratch = memory(sm, eng);
    arguments.emplace(DNNL_ARG_SCRATCHPAD, scratch);
    execution.wait();
  }
  void run(bool boundary) override {
    if (boundary && upload)
#ifdef VOCAL_BENCH_INT8
      upload->execute(execution, upload_args);
#else
      upload->execute(execution, canonical_x, input);
#endif
    operation.execute(execution, arguments);
    if (boundary && download)
      download->execute(execution, destination_memory, canonical_y);
    execution.wait();
  }
  void output(float *dst) override {
    if (download)
      download->execute(execution, destination_memory, canonical_y);
    execution.wait();
    std::memcpy(dst, destination.data(), destination.size() * sizeof(float));
  }
  std::string implementation() const override { return implementation_name; }
};
std::unique_ptr<Operator> make_operator(const Spec &s, const float *x,
                                        const float *w, const float *b,
                                        int threads, int layout) {
  return std::make_unique<DnnlOperator>(s, x, w, b, threads, layout);
}
