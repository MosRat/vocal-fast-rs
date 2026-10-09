#include "bridge.h"
#include "layout.h"
#include <d3d12sdklayers.h>

namespace vdml {
// DirectMLX's Compile helper drops the HRESULT when it throws. Preserve it at
// this native boundary so invalid graphs and allocation failures are distinct.
Microsoft::WRL::ComPtr<IDMLCompiledOperator>
compile_graph(Device &context, dml::Graph &graph, std::vector<E> const &outputs,
              uint32_t input_count) {
  auto description = graph.Impl()->GetGraphDesc(outputs);
  auto nodes = description.Nodes();
  auto inputs = description.InputEdges();
  auto results = description.OutputEdges();
  auto edges = description.IntermediateEdges();
  DML_GRAPH_DESC desc{};
  desc.InputCount = input_count;
  desc.OutputCount = description.outputCount;
  desc.NodeCount = static_cast<UINT>(nodes.size());
  desc.Nodes = nodes.data();
  desc.InputEdgeCount = static_cast<UINT>(inputs.size());
  desc.InputEdges = inputs.data();
  desc.OutputEdgeCount = static_cast<UINT>(results.size());
  desc.OutputEdges = results.data();
  desc.IntermediateEdgeCount = static_cast<UINT>(edges.size());
  desc.IntermediateEdges = edges.data();
  Microsoft::WRL::ComPtr<IDMLDevice1> device;
  check(graph.Impl()->GetDevice()->QueryInterface(IID_PPV_ARGS(&device)),
        "query DirectML graph compiler");
  Microsoft::WRL::ComPtr<IDMLCompiledOperator> compiled;
  auto hr = device->CompileGraph(&desc, DML_EXECUTION_FLAG_NONE,
                                 IID_PPV_ARGS(&compiled));
  if (FAILED(hr)) {
    std::string operation = "compile DirectML graph";
    Microsoft::WRL::ComPtr<ID3D12InfoQueue> queue;
    if (SUCCEEDED(context.d3d.As(&queue))) {
      auto count = queue->GetNumStoredMessagesAllowedByRetrievalFilter();
      for (UINT64 i = 0; i < count; ++i) {
        SIZE_T bytes = 0;
        if (FAILED(queue->GetMessage(i, nullptr, &bytes)))
          continue;
        std::vector<uint8_t> storage(bytes);
        auto message = reinterpret_cast<D3D12_MESSAGE *>(storage.data());
        if (SUCCEEDED(queue->GetMessage(i, message, &bytes)) &&
            message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR) {
          operation += ": ";
          auto length = message->DescriptionByteLength;
          if (length && message->pDescription[length - 1] == '\0')
            --length;
          operation.append(message->pDescription, length);
        }
      }
    }
    check(hr, operation.c_str());
  }
  return compiled;
}
E apply(Node const &n, std::vector<E> const &tensors, dml::Graph &graph,
        Graph &result, std::vector<ConstantView> &constants,
        Node const *fused_matmul = nullptr, size_t bias_index = 0) {
  Shape wanted(n.shape, n.shape + n.rank);
  auto get = [&](size_t i) {
    require(i < n.input_count && n.inputs[i] < tensors.size(),
            "invalid graph edge");
    return tensors[n.inputs[i]];
  };
  if (n.op <= 1) {
    auto desc = dml::TensorDesc(dtype(n.dtype),
                                n.op == 1 ? DML_TENSOR_FLAG_OWNED_BY_DML
                                          : DML_TENSOR_FLAG_NONE,
                                wanted);
    result.inputs.push_back({desc.totalTensorSizeInBytes, 0, n.op == 1});
    require(n.op != 1 || (n.data && n.data_bytes &&
                          n.data_bytes <= desc.totalTensorSizeInBytes),
            "invalid constant data extent");
    constants.push_back(n.op == 1 ? ConstantView{n.data, n.data_bytes}
                                  : ConstantView{});
    return dml::InputTensor(
        graph, static_cast<uint32_t>(result.inputs.size() - 1), desc);
  }
  auto a = get(0);
  auto attr = [&](size_t i) {
    require(i < n.attr_count, "missing operator attribute");
    return n.attrs[i];
  };
  auto binary = [&](auto fn) {
    return fn(broadcast(a, wanted), broadcast(get(1), wanted));
  };
  switch (n.op) {
  case 2:
    return reshape(a, wanted, result.options.contiguous_views != 0);
  case 3: {
    auto src = sizes(a);
    auto stride = dense(src);
    dml::TensorStrides perm;
    for (size_t i = 0; i < n.rank; ++i)
      perm.push_back(stride[attr(i)]);
    return dml::Identity(dml::Reinterpret(a, wanted, perm));
  }
  case 4:
    return dml::Cast(a, dtype(n.dtype));
  case 5:
    if (fused_matmul) {
      require(fused_matmul->input_count == 2 && fused_matmul->attr_count == 1,
              "invalid fused GEMM geometry");
      ++result.memory.gemm_bias_fusions;
      return gemm(tensors[fused_matmul->inputs[0]],
                  tensors[fused_matmul->inputs[1]], fused_matmul->attrs[0] != 0,
                  wanted, result.options, get(bias_index));
    }
    return binary([](E a, E b) { return dml::Add(a, b); });
  case 6:
    return binary([](E a, E b) { return dml::Subtract(a, b); });
  case 7:
    return binary([](E a, E b) { return dml::Multiply(a, b); });
  case 8:
    return binary([](E a, E b) { return dml::Divide(a, b); });
  case 9:
    return binary([](E a, E b) { return dml::Pow(a, b); });
  case 10:
    return binary([](E a, E b) { return dml::Max(a, b); });
  case 11:
    return binary([](E a, E b) { return dml::Min(a, b); });
  case 12:
    return dml::ActivationRelu(a);
  case 13:
    return dml::ActivationLeakyRelu(a, n.values[0]);
  case 14:
    return dml::ActivationTanh(a);
  case 15:
    return dml::ActivationSigmoid(a);
  case 16:
    return dml::ActivationGelu(a);
  case 17:
    return dml::Exp(a);
  case 18:
    return dml::Log(a);
  case 19:
    return dml::Sqrt(a);
  case 20:
    return dml::Abs(a);
  case 21:
    return dml::Floor(a);
  case 22:
    return dml::Sin(a);
  case 23: {
    return gemm(a, get(1), attr(0) != 0, wanted, result.options);
  }
  case 24:
  case 25: {
    auto w = get(1);
    dml::Optional<E> bias = dml::NullOpt;
    if (n.input_count > 2)
      bias = get(2);
    std::vector<uint32_t> s{static_cast<uint32_t>(attr(0)),
                            static_cast<uint32_t>(attr(1))};
    std::vector<uint32_t> d{static_cast<uint32_t>(attr(2)),
                            static_cast<uint32_t>(attr(3))};
    std::vector<uint32_t> p{static_cast<uint32_t>(attr(4)),
                            static_cast<uint32_t>(attr(5))};
    std::vector<uint32_t> op{static_cast<uint32_t>(attr(6)),
                             static_cast<uint32_t>(attr(7))};
    auto end = p;
    if (n.op == 25) {
      // Torch output_padding extends the computed region; DML's appends zeros.
      for (size_t i = 0; i < 2; ++i) {
        require(op[i] <= p[i],
                "deconvolution output padding exceeds ending crop");
        end[i] -= op[i];
        op[i] = 0;
      }
    }
    return dml::Convolution(a, w, bias, DML_CONVOLUTION_MODE_CROSS_CORRELATION,
                            n.op == 25 ? DML_CONVOLUTION_DIRECTION_BACKWARD
                                       : DML_CONVOLUTION_DIRECTION_FORWARD,
                            s, d, p, end, op, attr(8),
                            dml::FusedActivation::None(), wanted);
  }
  case 26: {
    std::vector<uint32_t> axes{static_cast<uint32_t>(attr(0))};
    return dml::ActivationSoftmax(a, axes);
  }
  case 27:
  case 28:
  case 29: {
    std::vector<uint32_t> axes(n.attrs, n.attrs + n.attr_count);
    return dml::Reduce(a,
                       n.op == 27   ? DML_REDUCE_FUNCTION_SUM
                       : n.op == 28 ? DML_REDUCE_FUNCTION_AVERAGE
                                    : DML_REDUCE_FUNCTION_MAX,
                       axes);
  }
  case 30: {
    auto idx = get(1);
    auto in = sizes(a);
    size_t r = wanted.size();
    auto axis = static_cast<uint32_t>(attr(0) + r - in.size());
    return dml::Gather(padded(a, r),
                       padded(dml::Cast(idx, DML_TENSOR_DATA_TYPE_UINT32), r),
                       axis, attr(1));
  }
  case 31:
    return dml::GatherElements(
        a, dml::Cast(get(1), DML_TENSOR_DATA_TYPE_UINT32), attr(0));
  case 32: {
    auto src = sizes(a);
    std::vector<uint32_t> offset(src.size(), 0), window(src);
    std::vector<int32_t> strides(src.size(), 1);
    size_t axis = attr(0);
    int step = attr(2);
    offset[axis] =
        step < 0 ? static_cast<uint32_t>(attr(1) - (wanted[axis] - 1) * -step)
                 : static_cast<uint32_t>(attr(1));
    window[axis] = (wanted[axis] - 1) * std::abs(step) + 1;
    strides[axis] = step;
    return dml::Slice(a, offset, window, strides);
  }
  case 33: {
    std::vector<E> xs;
    for (size_t i = 0; i < n.input_count; ++i)
      xs.push_back(get(i));
    return dml::Join(xs, attr(0));
  }
  case 34: {
    std::vector<float> scales;
    auto src = sizes(a);
    for (size_t i = 0; i < wanted.size(); ++i)
      scales.push_back(static_cast<float>(wanted[i]) / src[i]);
    return dml::Resample(a, wanted, DML_INTERPOLATION_MODE_NEAREST_NEIGHBOR,
                         DML_AXIS_DIRECTION_INCREASING, scales);
  }
  case 35:
    return dml::CumulativeSummation(a, attr(0), DML_AXIS_DIRECTION_INCREASING,
                                    attr(1) != 0);
  case 36:
    return binary([](E a, E b) { return dml::GreaterThan(a, b); });
  case 37:
    return binary([](E a, E b) { return dml::LessThan(a, b); });
  case 38:
    return dml::If(broadcast(a, wanted), broadcast(get(1), wanted),
                   broadcast(get(2), wanted));
  case 39: {
    std::vector<uint32_t> begin, end;
    for (size_t i = 0; i < n.rank; ++i) {
      begin.push_back(attr(i));
      end.push_back(attr(n.rank + i));
    }
    return dml::Padding(a, DML_PADDING_MODE_CONSTANT, n.values[0], begin, end);
  }
  case 40: {
    // GRU needs materialized parameters: view elision must not propagate
    // OWNED_BY_DML flags into unsupported parameter descriptors.
    auto parameter = [&](size_t index) {
      auto x = get(index);
      return (x.Impl()->GetOutputDesc().flags & DML_TENSOR_FLAG_OWNED_BY_DML)
                 ? dml::Identity(x)
                 : x;
    };
    std::vector<dml::FusedActivation> acts{
        dml::FusedActivation::Sigmoid(), dml::FusedActivation::Tanh(),
        dml::FusedActivation::Sigmoid(), dml::FusedActivation::Tanh()};
    return dml::GRU(a, parameter(1), parameter(2), parameter(3), dml::NullOpt,
                    dml::NullOpt, acts,
                    DML_RECURRENT_NETWORK_DIRECTION_BIDIRECTIONAL, true,
                    dml::GRUOutputOptions::Sequence)
        .sequence;
  }
  case 41: {
    std::vector<uint32_t> two{2, 2}, zero{0, 0}, one{1, 1};
    return dml::AveragePooling(a, two, two, zero, zero, one, false, wanted);
  }
  case 42: {
    auto outputs =
        dml::TopK(a, static_cast<uint32_t>(sizes(a).size() - 1), attr(0),
                  attr(1) ? DML_AXIS_DIRECTION_DECREASING
                          : DML_AXIS_DIRECTION_INCREASING);
    return attr(2) ? outputs.index : outputs.value;
  }
  case 43:
    return dml::Round(a, DML_ROUNDING_MODE_HALVES_TO_NEAREST_EVEN);
  case 44:
    return binary([](E a, E b) { return dml::LogicalAnd(a, b); });
  case 45:
    return binary([](E a, E b) { return dml::Equals(a, b); });
  default:
    throw std::runtime_error("unsupported DirectML operator code");
  }
}
} // namespace vdml
extern "C" int vdml_compile(vdml::Device *device, vdml::Node const *nodes,
                            size_t count, uint32_t const *outputs,
                            size_t output_count, vdml::Feedback const *feedback,
                            size_t feedback_count,
                            vdml::CompileOptions const *options,
                            vdml::Graph **out, char **error) noexcept {
  return vdml::guarded(error, [&] {
    vdml::require(device && nodes && count && outputs && output_count && out,
                  "invalid graph compilation arguments");
    *out = nullptr;
    auto result = std::make_unique<vdml::Graph>();
    result->device = device;
    vdml::require(options, "null DirectML compile options");
    result->options = *options;
    result->memory.source_nodes = count;
    std::vector<size_t> users(count, 0);
    for (size_t i = 0; i < count; ++i)
      for (size_t j = 0; j < nodes[i].input_count; ++j) {
        vdml::require(nodes[i].inputs[j] < i, "invalid graph node order");
        ++users[nodes[i].inputs[j]];
      }
    // Marked outputs and feedback branches must retain the unfused result.
    for (size_t i = 0; i < output_count; ++i) {
      vdml::require(outputs[i] < count, "invalid graph output");
      ++users[outputs[i]];
    }
    std::vector<vdml::Node const *> fusion(count, nullptr);
    std::vector<size_t> bias_indices(count, 0);
    std::vector<bool> skipped(count, false);
    if (options->fuse_gemm_bias)
      for (size_t i = 0; i < count; ++i)
        if (nodes[i].op == 5 && nodes[i].input_count == 2)
          for (size_t j = 0; j < 2; ++j) {
            auto id = nodes[i].inputs[j];
            auto const &candidate = nodes[id];
            if (candidate.op == 23 && users[id] == 1 &&
                candidate.dtype == nodes[i].dtype &&
                candidate.rank == nodes[i].rank &&
                std::equal(candidate.shape, candidate.shape + candidate.rank,
                           nodes[i].shape)) {
              fusion[i] = &candidate;
              bias_indices[i] = 1 - j;
              skipped[id] = true;
              break;
            }
          }
    dml::Graph graph(device->dml.Get());
    std::vector<vdml::E> tensors;
    std::vector<vdml::ConstantView> constants;
    for (size_t i = 0; i < count; ++i) {
      try {
        if (skipped[i]) {
          tensors.emplace_back();
          continue;
        }
        auto e = vdml::apply(nodes[i], tensors, graph, *result, constants,
                             fusion[i], bias_indices[i]);
        auto actual = vdml::sizes(e);
        vdml::require(actual == vdml::Shape(nodes[i].shape,
                                            nodes[i].shape + nodes[i].rank),
                      "operator output shape mismatch");
        tensors.push_back(e);
      } catch (std::exception const &e) {
        throw std::runtime_error("node " + std::to_string(i) + " op " +
                                 std::to_string(nodes[i].op) + ": " + e.what());
      }
    }
    std::vector<vdml::E> values;
    for (size_t i = 0; i < output_count; ++i) {
      vdml::require(outputs[i] < tensors.size(), "invalid graph output");
      auto e = dml::Identity(tensors[outputs[i]]);
      values.push_back(e);
      result->outputs.push_back(
          {e.Impl()->GetOutputDesc().totalTensorSizeInBytes});
    }
    result->compiled = vdml::compile_graph(
        *device, graph, values, static_cast<uint32_t>(result->inputs.size()));
    result->memory.lowered_nodes =
        graph.Impl()->GetGraphDesc(values).Nodes().size();
    vdml::require(feedback_count == 0 || feedback, "null graph feedback");
    for (size_t i = 0; i < feedback_count; ++i) {
      auto const &f = feedback[i];
      vdml::require(f.input < result->inputs.size() &&
                        f.output < result->outputs.size(),
                    "invalid feedback binding index");
      auto &input = result->inputs[f.input];
      auto &output = result->outputs[f.output];
      vdml::require(!input.owned && !input.state && output.readback &&
                        input.bytes == output.bytes,
                    "invalid feedback tensor geometry or duplicate state");
      input.state = true;
      output.readback = false;
      result->feedback.push_back(f);
    }
    result->initialize(constants);
    *out = result.release();
  });
}
