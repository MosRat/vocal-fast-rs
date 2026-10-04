#include "bridge.h"
#include <atomic>
#include <cstdio>

namespace vocal {
namespace {
std::atomic<LogCallback> g_callback{nullptr};
}

void Logger::log(Severity severity, char const* message) noexcept {
    if (auto callback = g_callback.load(std::memory_order_acquire)) {
        callback(static_cast<int>(severity), message ? message : "<null>");
    } else if (severity <= Severity::kWARNING) {
        std::fprintf(stderr, "[TensorRT] %s\n", message ? message : "<null>");
    }
}

Logger& logger() {
    static Logger instance;
    return instance;
}

void log_message(nvinfer1::ILogger::Severity severity, std::string const& message) noexcept {
    logger().log(severity, message.c_str());
}

void set_log_callback(LogCallback callback) noexcept {
    g_callback.store(callback, std::memory_order_release);
}

char* make_error(char const* message) noexcept {
    if (!message) message = "native TensorRT error";
    auto bytes = std::strlen(message) + 1;
    auto* result = static_cast<char*>(std::malloc(bytes));
    if (result) std::memcpy(result, message, bytes);
    return result;
}
}

extern "C" void vocal_trt_set_log_callback(vocal::LogCallback callback) noexcept {
    vocal::set_log_callback(callback);
}

extern "C" void vocal_trt_free_error(char* message) { std::free(message); }
