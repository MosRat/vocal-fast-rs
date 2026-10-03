#include "bridge.h"
#include <cstdio>

namespace vocal {
void Logger::log(Severity severity, char const* message) noexcept {
    if (severity <= Severity::kWARNING) {
        std::fprintf(stderr, "[TensorRT] %s\n", message ? message : "<null>");
    }
}

Logger& logger() {
    static Logger instance;
    return instance;
}

char* make_error(char const* message) noexcept {
    if (!message) message = "native TensorRT error";
    auto bytes = std::strlen(message) + 1;
    auto* result = static_cast<char*>(std::malloc(bytes));
    if (result) std::memcpy(result, message, bytes);
    return result;
}
}

extern "C" void vocal_trt_free_error(char* message) { std::free(message); }
