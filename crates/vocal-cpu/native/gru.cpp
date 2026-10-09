#include "internal.h"
#include <unordered_map>
namespace vocal::cpu {
using namespace dnnl;
class Gru final : public Kernel {
  primitive operation;
  std::unordered_map<int, memory> args;
  memory input, output, weights, recurrent, bias, scratch;
  Tensor canonical;
  uint32_t source;

public:
  Gru(Graph &g, size_t id) {
    auto &n = g.nodes[id];
    source = n.inputs[0];
    auto &x = g.tensors[source];
    auto &y = g.tensors[id];
    size_t steps = y.shape[0], hidden = y.shape[3],
           width = x.elements() / steps;
    require(y.shape[1] == 2 && y.shape[2] == 1,
            "GRU requires batch one, bidirectional output");
    for (size_t j = 1; j < 4; ++j)
      require(g.nodes[n.inputs[j]].folded, "GRU weights must be constant");
    auto xm = memory::desc(
        {static_cast<int64_t>(steps), 1, static_cast<int64_t>(width)},
        memory::data_type::f32, memory::format_tag::tnc);
    auto ym = memory::desc(
        {static_cast<int64_t>(steps), 1, static_cast<int64_t>(2 * hidden)},
        memory::data_type::f32, memory::format_tag::tnc);
    Shape wd = {1, 2, static_cast<int64_t>(width), 3,
                static_cast<int64_t>(hidden)},
          rd = {1, 2, static_cast<int64_t>(hidden), 3,
                static_cast<int64_t>(hidden)},
          bd = {1, 2, 4, static_cast<int64_t>(hidden)};
    primitive_attr attr;
    attr.set_scratchpad_mode(scratchpad_mode::user);
    auto pd = lbr_gru_forward::primitive_desc(
        g.engine, prop_kind::forward_inference,
        rnn_direction::bidirectional_concat, xm, memory::desc(),
        memory::desc(wd, memory::data_type::f32, memory::format_tag::any),
        memory::desc(rd, memory::data_type::f32, memory::format_tag::any),
        memory::desc(bd, memory::data_type::f32, memory::format_tag::ldgo), ym,
        memory::desc(), attr);
    operation = lbr_gru_forward(pd);
    output = memory(ym, g.engine, y.data());
    canonical.dtype = 1;
    canonical.shape = x.shape;
    canonical.strides = contiguous_strides(x.shape);
    canonical.storage = std::make_shared<Storage>(x.bytes());
    input = memory(xm, g.engine, canonical.data());
    auto convert = [&](const Tensor &t, size_t channels, const Shape &dims,
                       const memory::desc &target) {
      std::vector<float> values(2 * channels * 3 * hidden);
      for (size_t d = 0; d < 2; ++d)
        for (size_t i = 0; i < channels; ++i)
          for (size_t gate = 0; gate < 3; ++gate)
            for (size_t o = 0; o < hidden; ++o)
              values[(((d * channels + i) * 3 + gate) * hidden + o)] =
                  static_cast<float>(t.read(t.position(
                      (d * 3 * hidden + gate * hidden + o) * channels + i)));
      auto raw = memory(
          memory::desc(dims, memory::data_type::f32, memory::format_tag::ldigo),
          g.engine, values.data());
      auto packed = memory(target, g.engine);
      reorder(raw, packed).execute(g.stream, raw, packed);
      g.stream.wait();
      return packed;
    };
    weights =
        convert(g.tensors[n.inputs[1]], width, wd, pd.weights_layer_desc());
    recurrent =
        convert(g.tensors[n.inputs[2]], hidden, rd, pd.weights_iter_desc());
    bias = memory(pd.bias_desc(), g.engine);
    auto *biases = static_cast<float *>(bias.get_data_handle());
    auto &b = g.tensors[n.inputs[3]];
    for (size_t d = 0; d < 2; ++d)
      for (size_t o = 0; o < hidden; ++o) {
        auto value = [&](size_t gate) {
          return static_cast<float>(
              b.read(b.position(d * 6 * hidden + gate * hidden + o)));
        };
        biases[(d * 4 + 0) * hidden + o] = value(0) + value(3);
        biases[(d * 4 + 1) * hidden + o] = value(1) + value(4);
        biases[(d * 4 + 2) * hidden + o] = value(2);
        biases[(d * 4 + 3) * hidden + o] = value(5);
      }
    packed_inputs = {1, 2, 3};
    packed_bytes = pd.weights_layer_desc().get_size() +
                   pd.weights_iter_desc().get_size() +
                   pd.bias_desc().get_size();
    auxiliary_bytes = canonical.storage->bytes;
    scratch = g.workspace(0, pd.scratchpad_desc());
    args = {{DNNL_ARG_SRC_LAYER, input},
            {DNNL_ARG_DST_LAYER, output},
            {DNNL_ARG_WEIGHTS_LAYER, weights},
            {DNNL_ARG_WEIGHTS_ITER, recurrent},
            {DNNL_ARG_BIAS, bias},
            {DNNL_ARG_SCRATCHPAD, scratch}};
  }
  void run(Graph &g) override {
    copy_tensor(g.tensors[source], canonical);
    operation.execute(g.stream, args);
  }
};
std::unique_ptr<Kernel> prepare_gru(Graph &g, size_t id) {
  return std::make_unique<Gru>(g, id);
}
} // namespace vocal::cpu
