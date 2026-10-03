//! Low-latency audio I/O primitives and device backends.
//!
//! The public API is split into the allocation-free queue/stream core and
//! optional platform adapters. Device code stays at the edge; inference code
//! only sees [`WorkerIo`] and its slot leases.

mod audio_core;

pub use audio_core::*;

#[cfg(feature = "cpal-backend")]
pub mod cpal_backend;

#[cfg(all(feature = "wasapi-backend", target_os = "windows"))]
pub mod wasapi_backend;
