#include "internal.h"
#include <chrono>
#include <string>
#include <unordered_set>
using namespace vocal::cpu;
static thread_local std::string error;
extern "C" const char *vocal_cpu_error() { return error.c_str(); }
extern "C" void vocal_cpu_version(uint32_t *output) {
  auto *version = dnnl_version();
  output[0] = version->major;
  output[1] = version->minor;
  output[2] = version->patch;
}
extern "C" void *vocal_cpu_create(const CpuNode *raw, size_t count,
                                  const uint32_t *outputs, size_t output_count,
                                  size_t threads, uint32_t spin_us, bool pin,
                                  bool avx2, bool profile, bool channels_last,
                                  bool fuse, bool fuse_linear_gelu,
                                  bool fast_tensor_io,
                                  uint32_t linear_mode,
                                  const CpuConvolution *convolution,
                                  const CpuAttention *attention,
                                  const CpuQuantization *quantization,
                                  size_t quantization_count) {
  size_t current = 0;
  try {
    require(threads > 0 && threads <= 256, "invalid thread count");
    require(spin_us <= 1000, "invalid thread spin budget");
    require(convolution && convolution->mode <= 2,
            "invalid convolution configuration");
    require(linear_mode <= 3, "invalid linear implementation");
    require(attention && attention->mode <= 1 && attention->query_tile > 0 &&
                attention->query_tile <= 32 && attention->key_tile > 0 &&
                attention->key_tile <= 256, "invalid attention configuration");
    Concurrency concurrency(static_cast<int>(threads));
    auto graph =
        std::make_unique<Graph>(static_cast<int>(threads), spin_us, pin);
    auto &g = *graph;
    ThreadAffinity affinity(g.pool->caller_affinity());
    g.avx2 = avx2;
    g.profile = profile;
    g.channels_last = channels_last;
    g.fast_tensor_io = fast_tensor_io;
    g.convolution = *convolution;
    g.linear_mode = linear_mode;
    g.attention = *attention;
    g.outputs.assign(outputs, outputs + output_count);
    g.nodes.resize(count);
    g.tensors.resize(count);
    g.timings.resize(count, 0);
    for (size_t i = 0; i < quantization_count; ++i) {
      const auto &q = quantization[i];
      require(q.id < count && (q.mode == 1 || q.mode == 2), "invalid quantization ID/mode");
      auto &node = g.nodes[q.id];
      require(node.quantization == 0, "duplicate quantization ID");
      node.quantization = q.mode;
      node.smooth_alpha = q.smooth_alpha;
      node.input_max.assign(q.channels, 0);
      if (q.mode == 2) {
        require(q.input_max != nullptr, "missing static calibration");
        std::copy(q.input_max, q.input_max + q.channels, node.input_max.begin());
      }
    }
    std::vector<size_t> last(count), root(count), consumer(count, count),
        users(count, 0), activation_source(count, count);
    std::vector<bool> reusable(count, false), released(count, false);
    for (size_t i = 0; i < count; ++i) {
      last[i] = i;
      root[i] = i;
    }
    for (size_t i = 0; i < count; ++i)
      for (size_t j = 0; j < raw[i].input_count; ++j) {
        require(raw[i].inputs[j] < i, "non-topological graph");
        auto input = raw[i].inputs[j];
        last[input] = i;
        consumer[input] = i;
        ++users[input];
      }
    for (auto id : g.outputs) {
      require(id < count, "invalid output ID");
      last[id] = count;
      ++users[id];
    }
    std::vector<std::shared_ptr<Storage>> available;
    for (current = 0; current < count; ++current) {
      for (size_t i = 0; i < current; ++i)
        if (reusable[i] && !released[i] && last[i] < current) {
          available.push_back(g.tensors[i].storage);
          released[i] = true;
        }
      const auto &r = raw[current];
      size_t residual_target = count;
      auto &n = g.nodes[current];
      auto &y = g.tensors[current];
      n.op = r.op;
      if (r.input_count)
        n.inputs.assign(r.inputs, r.inputs + r.input_count);
      if (r.attr_count)
        n.attrs.assign(r.attrs, r.attrs + r.attr_count);
      std::copy(r.values, r.values + 4, n.values);
      y.dtype = r.dtype;
      y.shape.assign(r.shape, r.shape + r.rank);
      y.strides = contiguous_strides(y.shape);
      bool folded = n.op == 1;
      if (n.op > 1) {
        folded = true;
        for (auto id : n.inputs)
          folded = folded && g.nodes[id].folded;
      }
      n.folded = folded;
      if (n.op == 2) {
        Tensor view;
        if (reshape_view(g.tensors[n.inputs[0]], y.shape, view)) {
          y = view;
          n.alias = true;
        }
      } else if (n.op == 3) {
        Shape axes(n.attrs.begin(), n.attrs.end());
        y = transpose_view(g.tensors[n.inputs[0]], axes);
        n.alias = true;
      } else if (n.op == 4 && g.tensors[n.inputs[0]].dtype == y.dtype) {
        y = g.tensors[n.inputs[0]];
        n.alias = true;
      } else if (activation_source[current] != count) {
        y = g.tensors[activation_source[current]];
        n.alias = true;
      } else if (n.op == 32 && n.attrs[2] > 0) {
        y = g.tensors[n.inputs[0]];
        auto axis = n.attrs[0];
        y.offset += n.attrs[1] * y.strides[axis] * dtype_bytes(y.dtype);
        y.strides[axis] *= n.attrs[2];
        y.shape.assign(r.shape, r.shape + r.rank);
        n.alias = true;
      }
      if (n.alias) {
        root[current] = root[activation_source[current] != count
                                 ? activation_source[current]
                                 : n.inputs[0]];
        last[root[current]] = std::max(last[root[current]], last[current]);
      } else {
        if (n.op == 54 && g.attention.token_major_output && y.shape.size() == 4)
          y.strides = layout_strides(y.shape, {3, 1, 2, 0});
        else if ((n.op == 24 || n.op == 25) && channels_last)
          y.strides = layout_strides(y.shape, {1, 3, 2, 0});
        else if ((n.op >= 5 && n.op <= 22) || n.op == 26 || n.op == 38 ||
                 n.op == 51 || n.op == 52)
          for (auto id : n.inputs) {
            auto &x = g.tensors[id];
            if (x.shape == y.shape && x.dense()) {
              y.strides = layout_strides(y.shape, x.order());
              break;
            }
          }
        else if (n.op == 41 || n.op == 33) {
          auto &x = g.tensors[n.inputs[0]];
          if (x.dense() && x.shape.size() == y.shape.size())
            y.strides = layout_strides(y.shape, x.order());
        }
        auto allocation = available.end();
        if (!folded && n.op != 0)
          for (auto it = available.begin(); it != available.end(); ++it)
            if ((*it)->bytes >= y.bytes() &&
                (allocation == available.end() ||
                 (*it)->bytes < (*allocation)->bytes))
              allocation = it;
        if (allocation != available.end()) {
          y.storage = *allocation;
          available.erase(allocation);
        } else {
          y.storage = std::make_shared<Storage>(y.bytes());
          g.slots.push_back(y.storage);
        }
        reusable[current] = !folded && n.op != 0;
      }
      if (n.op == 1) {
        require(r.data_bytes == y.bytes(), "constant byte size mismatch");
        std::memcpy(y.data(), r.data, r.data_bytes);
      }
      if (fuse && (n.op == 24 || n.op == 25)) {
        size_t source = current, next = consumer[source];
        Tensor view = y;
        while (users[source] == 1 && next < count && raw[next].op == 2) {
          Shape shape(raw[next].shape, raw[next].shape + raw[next].rank);
          Tensor reshaped;
          if (!reshape_view(view, shape, reshaped))
            break;
          view = reshaped;
          source = next;
          next = consumer[source];
        }
        if (users[source] == 1 && next < count &&
            (raw[next].op == 12 || raw[next].op == 13 ||
             (g.convolution.fuse_gelu && raw[next].op == 16))) {
          n.activation = raw[next].op == 16 ? ActivationPostOp::Gelu
                                           : ActivationPostOp::Relu;
          n.activation_alpha = raw[next].op == 13 ? raw[next].values[0] : 0;
          activation_source[next] = source;
        } else if (g.convolution.fuse_residual && users[source] == 1 &&
                   next < count && (raw[next].op == 5 || raw[next].op == 52) &&
                   raw[next].input_count == 2 && !folded && y.dtype == 1 &&
                   n.quantization == 0) {
          const auto other = raw[next].inputs[0] == source
                                 ? raw[next].inputs[1]
                                 : raw[next].inputs[0];
          Tensor residual;
          // Do not move an addition before an unavailable producer, expand a
          // broadcast input, or reinterpret a materialized layout.
          if (other < current && g.tensors[other].dtype == 1 &&
              g.tensors[other].shape == view.shape &&
              reshape_view(g.tensors[other], y.shape, residual)) {
            n.residual = residual;
            // Singleton strides are physically irrelevant, but noncanonical
            // values can make oneDNN's binary post-op select ref:any.
            auto preferred = layout_strides(y.shape, {1, 3, 2, 0});
            bool channels_last_residual = true;
            for (size_t axis = 0; axis < y.shape.size(); ++axis)
              if (y.shape[axis] > 1 && residual.strides[axis] != preferred[axis])
                channels_last_residual = false;
            if (channels_last_residual)
              n.residual.strides = preferred;
            if (raw[next].op == 52) {
              n.activation = ActivationPostOp::Relu;
              n.activation_alpha = raw[next].values[0];
            }
            activation_source[next] = source;
            residual_target = next;
          }
        }
      }
      // A marked pre-activation or another consumer must retain its original
      // value. Bias has already been lowered into LinearBias by the Rust planner.
      if (fuse && fuse_linear_gelu && (n.op == 23 || n.op == 53) &&
          !folded && users[current] == 1 && consumer[current] < count &&
          raw[consumer[current]].op == 16) {
        n.activation = ActivationPostOp::Gelu;
        activation_source[consumer[current]] = current;
      }
      if (!n.alias && n.op > 1) {
        if (n.quantization == 2 && !folded)
          n.kernel = prepare_int8(g, current);
        else if (n.op == 40)
          n.kernel = prepare_gru(g, current);
        else if (n.op == 50)
          n.kernel = prepare_stft(g, current);
        else if (n.op == 54)
          n.kernel = prepare_attention(g, current);
        else if (n.op == 2 && g.fast_tensor_io)
          n.kernel = prepare_tensor_reorder(g, current);
        else
          n.kernel = prepare_dnnl(g, current);
        // Supporting a post-op semantically does not guarantee a fast CPU
        // implementation. Restore the original Add before its alias is built.
        if (residual_target != count && n.kernel &&
            n.kernel->implementation.find("ref:") != std::string::npos) {
          n.residual = Tensor{};
          n.activation = ActivationPostOp::None;
          activation_source[residual_target] = count;
          n.kernel = prepare_dnnl(g, current);
        }
        if (folded) {
          if (n.kernel)
            n.kernel->run(g);
          else
            execute_elements(g, current);
          n.kernel.reset();
        }
      }
    }
    g.finalize_memory();
    return graph.release();
  } catch (const std::exception &e) {
    error = "compile node " + std::to_string(current) + ": " + e.what();
    return nullptr;
  }
}
extern "C" int vocal_cpu_run(void *ptr, const CpuBinding *bindings,
                             size_t count) {
  auto &g = *static_cast<Graph *>(ptr);
  g.ready = false;
  size_t current = 0;
  try {
    Concurrency concurrency(g.pool->get_num_threads());
    ThreadAffinity affinity(g.pool->caller_affinity());
    std::unordered_set<uint32_t> bound;
    for (size_t i = 0; i < count; ++i) {
      auto id = bindings[i].id;
      require(id < g.nodes.size(), "binding ID out of range");
      require(g.nodes[id].op == 0 && bound.insert(id).second,
              "duplicate or non-input binding");
      auto &x = g.tensors[id];
      require(bindings[i].bytes == x.bytes(), "binding size mismatch");
      std::memcpy(x.data(), bindings[i].data, x.bytes());
    }
    for (current = 0; current < g.nodes.size(); ++current) {
      auto &n = g.nodes[current];
      if (n.op == 0)
        require(bound.contains(static_cast<uint32_t>(current)),
                "missing binding");
      if (n.op > 1 && !n.folded && !n.alias) {
        auto start = g.profile ? std::chrono::steady_clock::now()
                               : std::chrono::steady_clock::time_point{};
        if (n.quantization == 1)
          observe_input(g, current);
        if (n.kernel)
          n.kernel->run(g);
        else
          execute_elements(g, current);
        if (g.profile)
          g.timings[current] = std::chrono::duration<double, std::milli>(
                                   std::chrono::steady_clock::now() - start)
                                   .count();
      }
    }
    g.stream.wait();
    g.ready = true;
    return 0;
  } catch (const std::exception &e) {
    error = "node " + std::to_string(current) + ": " + e.what();
    return -1;
  }
}
extern "C" int vocal_cpu_output(void *ptr, uint32_t id, uint8_t *data,
                                size_t bytes) {
  try {
    auto &g = *static_cast<Graph *>(ptr);
    require(g.ready, "no valid output");
    require(id < g.nodes.size() && std::find(g.outputs.begin(), g.outputs.end(),
                                             id) != g.outputs.end(),
            "unmarked output");
    auto &x = g.tensors[id];
    require(bytes == x.bytes(), "output size mismatch");
    if (x.contiguous())
      std::memcpy(data, x.data(), bytes);
    else {
      auto width = dtype_bytes(x.dtype);
      for (size_t i = 0; i < x.elements(); ++i)
        std::memcpy(data + i * width,
                    static_cast<uint8_t *>(x.data()) + x.position(i) * width,
                    width);
    }
    return 0;
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}
extern "C" double vocal_cpu_node_ms(void *ptr, uint32_t id) {
  auto &g = *static_cast<Graph *>(ptr);
  return id < g.timings.size() ? g.timings[id] : 0;
}
extern "C" int vocal_cpu_calibration(void *ptr, uint32_t id, float *data, size_t channels) {
  try {
    auto &g = *static_cast<Graph *>(ptr);
    require(g.ready, "no complete calibration run");
    require(id < g.nodes.size(), "invalid calibration ID");
    const auto &n = g.nodes[id];
    require(n.quantization == 1 && n.observations > 0 && channels == n.input_max.size(),
            "missing calibration observations or channel mismatch");
    std::copy(n.input_max.begin(), n.input_max.end(), data);
    return 0;
  } catch (const std::exception &e) {
    error = e.what();
    return -1;
  }
}
extern "C" const char *vocal_cpu_implementation(void *ptr, uint32_t id) {
  auto &g = *static_cast<Graph *>(ptr);
  return id < g.nodes.size() && g.nodes[id].kernel
             ? g.nodes[id].kernel->implementation.c_str() : "";
}
extern "C" void vocal_cpu_destroy(void *ptr) {
  delete static_cast<Graph *>(ptr);
}
