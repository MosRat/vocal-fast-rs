//! Return freed model/build allocations before the control actor waits again.
//! This is deliberately limited to resource changes, never audio processing,
//! snapshots, parameter edits, or device polling. GUI and CLI use mimalloc on
//! Windows; other allocators simply have no mimalloc allocations to reclaim.
pub(crate) fn collect_idle() {
    #[cfg(windows)]
    unsafe {
        libmimalloc_sys::mi_collect(true);
    }
}
