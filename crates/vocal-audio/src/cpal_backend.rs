//! `cpal` device adapter.
//!
//! On Windows, cpal's native host is WASAPI. The adapter deliberately keeps
//! cpal at the edge: callback code only feeds [`BlockAssembler`] and
//! [`BlockDrainer`], while the worker communicates through [`WorkerIo`].

use std::sync::Arc;

use cpal::traits::{DeviceTrait, StreamTrait};
use thiserror::Error;

use crate::{AudioError, BlockAssembler, BlockDrainer, DeviceIo, StreamConfig, WorkerIo};

#[derive(Debug, Error)]
pub enum CpalError {
    #[error("cpal input device has no default stream configuration: {0}")]
    InputConfig(#[source] cpal::DefaultStreamConfigError),
    #[error("cpal output device has no default stream configuration: {0}")]
    OutputConfig(#[source] cpal::DefaultStreamConfigError),
    #[error("cpal callback format must be f32 (input={input:?}, output={output:?})")]
    UnsupportedSampleFormat {
        input: cpal::SampleFormat,
        output: cpal::SampleFormat,
    },
    #[error("invalid cpal stream configuration: {0}")]
    InvalidConfig(#[from] AudioError),
    #[error("failed to build input stream: {0}")]
    BuildInput(#[source] cpal::BuildStreamError),
    #[error("failed to build output stream: {0}")]
    BuildOutput(#[source] cpal::BuildStreamError),
    #[error("failed to start input stream: {0}")]
    PlayInput(#[source] cpal::PlayStreamError),
    #[error("failed to start output stream: {0}")]
    PlayOutput(#[source] cpal::PlayStreamError),
}

/// The two native streams. The callback closures own all mutable state, so no
/// mutex is taken from the real-time thread.
pub struct CpalDuplexStream {
    input: cpal::Stream,
    output: cpal::Stream,
}

impl CpalDuplexStream {
    pub fn input(&self) -> &cpal::Stream {
        &self.input
    }

    pub fn output(&self) -> &cpal::Stream {
        &self.output
    }
}

/// Open separate capture and render streams using f32 interleaved samples.
///
/// `StreamConfig::block_frames` is requested as the native callback buffer
/// size. Drivers may still return another size; the assembler and drainer make
/// that case safe without allocating in callbacks. The returned `WorkerIo`
/// must be moved to the inference thread.
pub fn build_duplex(
    input_device: &cpal::Device,
    output_device: &cpal::Device,
    config: StreamConfig,
    device_io: DeviceIo,
    worker_io: WorkerIo,
    error_handler: impl Fn(cpal::StreamError) + Send + Sync + 'static,
) -> Result<(CpalDuplexStream, WorkerIo), CpalError> {
    let config = config.validate()?;
    let input_default = input_device
        .default_input_config()
        .map_err(CpalError::InputConfig)?;
    let output_default = output_device
        .default_output_config()
        .map_err(CpalError::OutputConfig)?;
    if input_default.sample_format() != cpal::SampleFormat::F32
        || output_default.sample_format() != cpal::SampleFormat::F32
    {
        return Err(CpalError::UnsupportedSampleFormat {
            input: input_default.sample_format(),
            output: output_default.sample_format(),
        });
    }

    let native_config = cpal::StreamConfig {
        channels: config.format.channels,
        sample_rate: cpal::SampleRate(config.format.sample_rate),
        buffer_size: cpal::BufferSize::Fixed(config.block_frames as u32),
    };
    let callback_error = Arc::new(error_handler);

    let DeviceIo {
        input: input_producer,
        output: output_consumer,
    } = device_io;
    let mut assembler = BlockAssembler::new(config)?;
    let input_error = Arc::clone(&callback_error);
    let input = input_device
        .build_input_stream(
            &native_config,
            move |data: &[f32], _| {
                // A full queue is a real-time overflow. Dropping the newest
                // block bounds latency; the ring and assembler expose counters
                // for diagnostics.
                let _ = assembler.push(data, &input_producer);
            },
            move |error| input_error(error),
            None,
        )
        .map_err(CpalError::BuildInput)?;

    let mut drainer = BlockDrainer::new(config)?;
    let output_error = Arc::clone(&callback_error);
    let output = output_device
        .build_output_stream(
            &native_config,
            move |data: &mut [f32], _| {
                // Silence is written on underflow so the native callback never
                // blocks behind inference.
                let _ = drainer.drain(data, &output_consumer);
            },
            move |error| output_error(error),
            None,
        )
        .map_err(CpalError::BuildOutput)?;

    input.play().map_err(CpalError::PlayInput)?;
    output.play().map_err(CpalError::PlayOutput)?;

    Ok((CpalDuplexStream { input, output }, worker_io))
}

/// Prefer the native Windows host. On other platforms this returns cpal's
/// default host so the same application-level stream code remains portable.
pub fn recommended_host() -> cpal::Host {
    #[cfg(windows)]
    {
        cpal::host_from_id(cpal::HostId::Wasapi).unwrap_or_else(|_| cpal::default_host())
    }
    #[cfg(not(windows))]
    {
        cpal::default_host()
    }
}
