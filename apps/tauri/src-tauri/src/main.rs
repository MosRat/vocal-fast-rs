#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

// The GUI is a long-lived, multi-threaded controller. Mimalloc reduces
// allocator contention for control-plane work while audio callbacks remain
// allocation-free. Keep this in the binary rather than a shared library.
#[cfg(windows)]
#[global_allocator]
static GLOBAL_ALLOCATOR: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    vocal_fast_tauri_lib::run();
}
