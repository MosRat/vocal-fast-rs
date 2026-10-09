//! Low-latency audio I/O primitives and device backends.
//!
//! The public API is split into the allocation-free queue/stream core and
//! optional platform adapters. Device code stays at the edge; inference code
//! only sees [`WorkerIo`] and its slot leases.

mod audio_core;
pub mod wav;

pub use audio_core::*;
pub use wav::{WavData, WavError, read_wav, read_wav_bytes, resample_linear, write_wav_pcm16};

#[cfg(feature = "cpal-backend")]
pub mod cpal_backend;

#[cfg(feature = "cpal-backend")]
mod cpal_samples;

#[cfg(all(feature = "wasapi-backend", target_os = "windows"))]
pub mod wasapi_backend;

#[cfg(all(feature = "wasapi-backend", target_os = "windows"))]
mod threading;

#[cfg(all(feature = "wasapi-backend", target_os = "windows"))]
mod backend_error;

#[cfg(all(feature = "wasapi-backend", target_os = "windows"))]
mod wasapi_native;

#[cfg(all(feature = "wasapi-backend", target_os = "windows"))]
mod wasapi_devices;
