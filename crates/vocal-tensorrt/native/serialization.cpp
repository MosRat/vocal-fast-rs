#include "bridge.h"

extern "C" int vocal_trt_build_view(
    vocal::Build* build, unsigned char const** data, size_t* bytes) noexcept {
    if (!build || !build->mem || !data || !bytes) return 1;
    *data = static_cast<unsigned char const*>(build->mem->data());
    *bytes = build->mem->size();
    return 0;
}

extern "C" int vocal_trt_build_bytes(
    vocal::Build* build, unsigned char** data, size_t* bytes) noexcept {
    if (!build || !build->mem || !data || !bytes) return 1;
    *data = static_cast<unsigned char*>(std::malloc(build->mem->size()));
    if (!*data) return 1;
    *bytes = build->mem->size();
    std::memcpy(*data, build->mem->data(), *bytes);
    return 0;
}

extern "C" void vocal_trt_build_free(vocal::Build* build) noexcept {
    if (!build) return;
    delete build->mem;
    delete build;
}

extern "C" void vocal_trt_free_bytes(unsigned char* data, size_t) noexcept {
    std::free(data);
}
