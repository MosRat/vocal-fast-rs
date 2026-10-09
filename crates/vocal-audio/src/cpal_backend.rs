//! `cpal` device adapter.
//!
//! On Windows, cpal's native host is WASAPI. The adapter deliberately keeps
//! cpal at the edge: callback code only feeds [`BlockAssembler`] and
//! [`BlockDrainer`], while the worker communicates through [`WorkerIo`].

use std::sync::Arc;
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, StreamTrait};
use thiserror::Error;

use crate::logging;
use crate::{
    AudioClock, AudioDirection, AudioError, AudioFault, AudioLatencySnapshot, AudioLatencyStats,
    AudioTimestamp, BlockAssembler, BlockDrainer, DeviceIo, StreamConfig, StreamLifecycle,
    StreamState, StreamStatus, WorkerIo,
};

#[derive(Debug, Error)]
pub enum CpalError {
    #[error("cpal input device has no default stream configuration: {0}")]
    InputConfig(#[source] cpal::Error),
    #[error("cpal output device has no default stream configuration: {0}")]
    OutputConfig(#[source] cpal::Error),
    #[error("unsupported cpal PCM sample format (input={input:?}, output={output:?})")]
    UnsupportedSampleFormat {
        input: cpal::SampleFormat,
        output: cpal::SampleFormat,
    },
    #[error(
        "cpal {direction} device format is {sample_rate} Hz/{channels} channels; expected {expected_sample_rate} Hz/{expected_channels} channels"
    )]
    DeviceFormatMismatch {
        direction: &'static str,
        sample_rate: u32,
        channels: u16,
        expected_sample_rate: u32,
        expected_channels: u16,
    },
    #[error("invalid cpal stream configuration: {0}")]
    InvalidConfig(#[from] AudioError),
    #[error("failed to build input stream: {0}")]
    BuildInput(#[source] cpal::Error),
    #[error("failed to build output stream: {0}")]
    BuildOutput(#[source] cpal::Error),
    #[error("failed to start input stream: {0}")]
    PlayInput(#[source] cpal::Error),
    #[error("failed to start output stream: {0}")]
    PlayOutput(#[source] cpal::Error),
}

/// The two native streams. The callback closures own all mutable state, so no
/// mutex is taken from the real-time thread.
pub struct CpalDuplexStream {
    input: cpal::Stream,
    output: cpal::Stream,
    latency: Option<Arc<AudioLatencyStats>>,
    state: Arc<StreamState>,
}

impl CpalDuplexStream {
    pub fn input(&self) -> &cpal::Stream {
        &self.input
    }

    pub fn output(&self) -> &cpal::Stream {
        &self.output
    }

    pub fn latency(&self) -> Option<AudioLatencySnapshot> {
        self.latency.as_deref().map(AudioLatencyStats::snapshot)
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
    error_handler: impl Fn(cpal::Error) + Send + Sync + 'static,
) -> Result<(CpalDuplexStream, WorkerIo), CpalError> {
    build_duplex_inner(
        input_device,
        output_device,
        config,
        device_io,
        worker_io,
        None,
        error_handler,
    )
}

/// Profiled variant of [`build_duplex`]. Capture blocks are timestamped with
/// `clock`; the inference worker should propagate that timestamp to its output
/// lease with [`crate::AudioWriteLease::set_timestamp`].
pub fn build_duplex_profiled(
    input_device: &cpal::Device,
    output_device: &cpal::Device,
    config: StreamConfig,
    device_io: DeviceIo,
    worker_io: WorkerIo,
    clock: AudioClock,
    error_handler: impl Fn(cpal::Error) + Send + Sync + 'static,
) -> Result<(CpalDuplexStream, WorkerIo), CpalError> {
    build_duplex_inner(
        input_device,
        output_device,
        config,
        device_io,
        worker_io,
        Some(clock),
        error_handler,
    )
}

fn build_duplex_inner(
    input_device: &cpal::Device,
    output_device: &cpal::Device,
    config: StreamConfig,
    device_io: DeviceIo,
    worker_io: WorkerIo,
    clock: Option<AudioClock>,
    error_handler: impl Fn(cpal::Error) + Send + Sync + 'static,
) -> Result<(CpalDuplexStream, WorkerIo), CpalError> {
    let config = config.validate()?;
    build_duplex_formats_inner(
        input_device,
        output_device,
        config,
        config,
        device_io,
        worker_io,
        clock,
        error_handler,
    )
}

/// Open capture and render in their respective native formats. Conversion is
/// performed by `DuplexWorkerAdapter` outside callbacks, with no format equality
/// requirement between directions.
pub fn build_duplex_formats_profiled(
    input_device: &cpal::Device,
    output_device: &cpal::Device,
    capture: StreamConfig,
    render: StreamConfig,
    device_io: DeviceIo,
    worker_io: WorkerIo,
    clock: AudioClock,
    error_handler: impl Fn(cpal::Error) + Send + Sync + 'static,
) -> Result<(CpalDuplexStream, WorkerIo), CpalError> {
    build_duplex_formats_inner(
        input_device,
        output_device,
        capture,
        render,
        device_io,
        worker_io,
        Some(clock),
        error_handler,
    )
}

fn build_duplex_formats_inner(
    input_device: &cpal::Device,
    output_device: &cpal::Device,
    config: StreamConfig,
    render_config: StreamConfig,
    device_io: DeviceIo,
    worker_io: WorkerIo,
    clock: Option<AudioClock>,
    error_handler: impl Fn(cpal::Error) + Send + Sync + 'static,
) -> Result<(CpalDuplexStream, WorkerIo), CpalError> {
    let config = config.validate()?;
    let render_config = render_config.validate()?;
    if config.block_frames > u32::MAX as usize || render_config.block_frames > u32::MAX as usize {
        return Err(CpalError::InvalidConfig(AudioError::InvalidConfig(
            "cpal buffer size exceeds u32::MAX frames".into(),
        )));
    }
    let input_default = input_device
        .default_input_config()
        .map_err(CpalError::InputConfig)?;
    let output_default = output_device
        .default_output_config()
        .map_err(CpalError::OutputConfig)?;
    let expected_channels = config.format.channels;
    let expected_sample_rate = config.format.sample_rate;
    if input_default.channels() != expected_channels
        || input_default.sample_rate() != expected_sample_rate
    {
        return Err(CpalError::DeviceFormatMismatch {
            direction: "input",
            sample_rate: input_default.sample_rate(),
            channels: input_default.channels(),
            expected_sample_rate,
            expected_channels,
        });
    }
    if output_default.channels() != render_config.format.channels
        || output_default.sample_rate() != render_config.format.sample_rate
    {
        return Err(CpalError::DeviceFormatMismatch {
            direction: "output",
            sample_rate: output_default.sample_rate(),
            channels: output_default.channels(),
            expected_sample_rate: render_config.format.sample_rate,
            expected_channels: render_config.format.channels,
        });
    }

    // Native driver periods need not equal ring slot sizes. Default lets CPAL
    // select a legal callback period; assembler/drainer retain exact slot sizes.
    let capture_native = cpal::StreamConfig {
        channels: config.format.channels,
        sample_rate: config.format.sample_rate,
        buffer_size: cpal::BufferSize::Default,
    };
    let render_native = cpal::StreamConfig {
        channels: render_config.format.channels,
        sample_rate: render_config.format.sample_rate,
        buffer_size: cpal::BufferSize::Default,
    };
    let state = Arc::new(StreamState::new());
    state.set(StreamLifecycle::Starting);
    let callback_state = Arc::clone(&state);
    let callback_error = Arc::new(move |error: cpal::Error| {
        let message = error.to_string();
        let first =
            callback_state.record_fault(crate::AudioFaultKind::Unknown, None, message.clone());
        if first {
            logging::fault("cpal", crate::AudioFaultKind::Unknown, None, &message);
        }
        error_handler(error);
    });
    let latency = clock
        .as_ref()
        .map(|_| Arc::new(AudioLatencyStats::default()));

    let DeviceIo {
        input: mut input_producer,
        output: mut output_consumer,
    } = device_io;
    let mut assembler = BlockAssembler::new(config)?;
    let input_clock = clock.clone();
    let input_latency = latency.clone();
    let input_error = Arc::clone(&callback_error);
    let mut input_last_callback = None;
    let input = crate::cpal_samples::build_input(
        input_device,
        capture_native,
        input_default.sample_format(),
        move |data: &[f32], info, callback_frames, offset_frames| {
            if offset_frames == 0 {
                record_callback_tick(
                    input_latency.as_deref(),
                    &mut input_last_callback,
                    Duration::from_secs_f64(
                        callback_frames as f64 / f64::from(config.format.sample_rate),
                    ),
                    AudioDirection::Capture,
                );
            }
            // A full queue is a real-time overflow. Dropping the newest
            // block bounds latency; the ring and assembler expose counters
            // for diagnostics.
            let timestamp = input_clock
                .as_ref()
                .map(|clock| {
                    let timing = info.timestamp();
                    let age = timing.callback.duration_since(timing.capture);
                    let offset = Duration::from_secs_f64(
                        offset_frames as f64 / f64::from(config.format.sample_rate),
                    );
                    clock.now().earlier_by(age.saturating_sub(offset))
                })
                .unwrap_or(AudioTimestamp::ZERO);
            let _ = assembler.push_at(data, &mut input_producer, timestamp);
            if let Some(latency) = input_latency.as_ref() {
                latency.record_block_path(AudioDirection::Capture, false);
                latency.record_assembler_buffered(
                    assembler
                        .buffered_samples()
                        .checked_div(config.format.channels as usize)
                        .unwrap_or(0),
                );
                let stats = input_producer.stats();
                latency.record_queue_state_for(
                    AudioDirection::Capture,
                    stats.queued_blocks,
                    stats.high_water_blocks,
                    stats.dropped_full,
                    stats.dropped_stale,
                );
            }
        },
        move |error| input_error(error),
    )?;

    let mut drainer = BlockDrainer::new(render_config)?;
    let output_clock = clock;
    let output_latency = latency.clone();
    let output_error = Arc::clone(&callback_error);
    let mut output_last_callback = None;
    let output = crate::cpal_samples::build_output(
        output_device,
        render_native,
        output_default.sample_format(),
        move |data: &mut [f32], info, callback_frames, offset_frames| {
            if offset_frames == 0 {
                record_callback_tick(
                    output_latency.as_deref(),
                    &mut output_last_callback,
                    Duration::from_secs_f64(
                        callback_frames as f64 / f64::from(render_config.format.sample_rate),
                    ),
                    AudioDirection::Render,
                );
            }
            // Silence is written on underflow so the native callback never
            // blocks behind inference.
            match drainer.drain(data, &mut output_consumer) {
                Ok(report) => {
                    if let Some(latency) = output_latency.as_ref() {
                        latency.record_block_path(AudioDirection::Render, false);
                        latency.record_drainer_pending(
                            drainer
                                .pending_samples()
                                .checked_div(render_config.format.channels as usize)
                                .unwrap_or(0),
                        );
                        latency.record_underflow(report.underflow_samples);
                        let stats = output_consumer.stats();
                        latency.record_queue_state_for(
                            AudioDirection::Render,
                            stats.queued_blocks,
                            stats.high_water_blocks,
                            stats.dropped_full,
                            stats.dropped_stale,
                        );
                        if let Some(clock) = output_clock.as_ref() {
                            let age = report
                                .source_timestamp
                                .is_set()
                                .then(|| clock.elapsed(report.source_timestamp))
                                .flatten();
                            if let Some(age) = age {
                                let timing = info.timestamp();
                                let padding = timing
                                    .playback
                                    .duration_since(timing.callback)
                                    .saturating_add(Duration::from_secs_f64(
                                        offset_frames as f64
                                            / f64::from(render_config.format.sample_rate),
                                    ));
                                latency.record(age.saturating_add(padding));
                            }
                        }
                    }
                }
                Err(_) => data.fill(0.0),
            }
        },
        move |error| output_error(error),
    )?;

    input.play().map_err(CpalError::PlayInput)?;
    output.play().map_err(CpalError::PlayOutput)?;

    state.set(StreamLifecycle::Running);
    logging::state("cpal", state.status());
    logging::started("cpal");

    Ok((
        CpalDuplexStream {
            input,
            output,
            latency,
            state,
        },
        worker_io,
    ))
}

impl CpalDuplexStream {
    pub fn status(&self) -> StreamStatus {
        self.state.status()
    }

    pub fn take_faults(&self) -> Vec<AudioFault> {
        self.state.take_faults()
    }

    pub fn take_errors(&self) -> Vec<String> {
        self.take_faults()
            .into_iter()
            .flat_map(|fault| {
                if fault.occurrences > 1 {
                    vec![
                        fault.message,
                        format!(
                            "{} backend errors occurred; first error shown above",
                            fault.occurrences
                        ),
                    ]
                } else {
                    vec![fault.message]
                }
            })
            .collect()
    }
}

impl Drop for CpalDuplexStream {
    fn drop(&mut self) {
        if !matches!(self.state.lifecycle(), StreamLifecycle::Stopped) {
            self.state.begin_shutdown();
            self.state.finish_shutdown();
            logging::state("cpal", self.state.status());
            logging::stopped("cpal");
        }
    }
}

fn record_callback_tick(
    latency: Option<&AudioLatencyStats>,
    last_callback: &mut Option<Instant>,
    expected: Duration,
    direction: AudioDirection,
) {
    let Some(latency) = latency else {
        return;
    };
    let now = Instant::now();
    if let Some(previous) = last_callback.replace(now) {
        latency.record_callback_period_for(
            direction,
            now.saturating_duration_since(previous),
            expected,
        );
    }
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
