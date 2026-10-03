//! Native Windows event-driven WASAPI backend.
//!
//! This module is intentionally behind `wasapi-backend` and only compiles on
//! Windows. It keeps COM and WASAPI interfaces on their owning threads, then
//! exposes the same [`WorkerIo`] lease API as the portable adapter. No mutex is
//! used on the capture or render hot path; the mutex below only records a
//! control-plane device error for the UI.

use std::slice;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use thiserror::Error;
use wasapi::{
    AudioCaptureClient, AudioClient, AudioRenderClient, DeviceEnumerator, Direction, Handle,
    SampleType, StreamMode, WaveFormat, initialize_mta,
};

use crate::{AudioClock, AudioLatencySnapshot, AudioLatencyStats, AudioTimestamp};
use crate::{
    AudioError, BlockAssembler, BlockDrainer, DeviceIo, StreamConfig, StreamQueueConfig, WorkerIo,
};

pub use crate::threading::MmcssGuard;
pub use crate::wasapi_native::{NativeWasapiError, NativeWasapiStream};

#[cfg(not(target_endian = "little"))]
compile_error!("the WASAPI f32 path requires a little-endian Windows target");

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WasapiMode {
    /// Shared mode with event-driven scheduling. Windows chooses the engine
    /// period; this is the interoperable default.
    SharedEvent,
    /// Exclusive event-driven mode. The requested block period is passed to
    /// the driver and must be supported by the endpoint.
    ExclusiveEvent,
}

#[derive(Debug, Error)]
pub enum WasapiBackendError {
    #[error("invalid WASAPI stream configuration: {0}")]
    Audio(#[from] AudioError),
    #[error("WASAPI worker failed to start: {0}")]
    Startup(String),
    #[error("failed to spawn WASAPI worker: {0}")]
    Spawn(#[source] std::io::Error),
}

struct LatencyProfile {
    clock: Option<AudioClock>,
    stats: Option<Arc<AudioLatencyStats>>,
}

/// Handles the capture and render event threads. Device interfaces are
/// thread-affine in the `wasapi` crate, so the handles never leave their
/// owning threads.
pub struct WasapiStream {
    stop: Arc<AtomicBool>,
    errors: Arc<Mutex<Vec<String>>>,
    latency: Option<Arc<AudioLatencyStats>>,
    joins: Option<(JoinHandle<()>, JoinHandle<()>)>,
}

impl WasapiStream {
    /// Open the default capture and render endpoints and start their event
    /// loops. The returned worker endpoints are ready for a model thread.
    pub fn open(
        config: StreamConfig,
        mode: WasapiMode,
    ) -> Result<(Self, WorkerIo), WasapiBackendError> {
        Self::open_with_queue_config(
            config,
            mode,
            StreamQueueConfig::symmetric(config.queue_blocks),
        )
    }

    pub fn open_with_queue_config(
        config: StreamConfig,
        mode: WasapiMode,
        queues: StreamQueueConfig,
    ) -> Result<(Self, WorkerIo), WasapiBackendError> {
        Self::open_inner(config, mode, queues, None)
    }

    /// Profiled variant of [`Self::open`]. Capture blocks receive a monotonic
    /// timestamp; the worker should copy it to its output lease so the stream
    /// can report measured capture-to-render age.
    pub fn open_profiled(
        config: StreamConfig,
        mode: WasapiMode,
        clock: AudioClock,
    ) -> Result<(Self, WorkerIo), WasapiBackendError> {
        Self::open_profiled_with_queue_config(
            config,
            mode,
            StreamQueueConfig::symmetric(config.queue_blocks),
            clock,
        )
    }

    pub fn open_profiled_with_queue_config(
        config: StreamConfig,
        mode: WasapiMode,
        queues: StreamQueueConfig,
        clock: AudioClock,
    ) -> Result<(Self, WorkerIo), WasapiBackendError> {
        Self::open_inner(config, mode, queues, Some(clock))
    }

    fn open_inner(
        config: StreamConfig,
        mode: WasapiMode,
        queues: StreamQueueConfig,
        clock: Option<AudioClock>,
    ) -> Result<(Self, WorkerIo), WasapiBackendError> {
        let config = config.validate()?;
        let io = crate::StreamIo::new_with_queue_config(config, queues)?;
        let (device_io, worker_io) = io.split();
        let stop = Arc::new(AtomicBool::new(false));
        let errors = Arc::new(Mutex::new(Vec::new()));
        let (capture_ready_tx, capture_ready_rx) = mpsc::sync_channel(1);
        let (render_ready_tx, render_ready_rx) = mpsc::sync_channel(1);
        let latency = clock
            .as_ref()
            .map(|_| Arc::new(AudioLatencyStats::default()));

        let capture_stop = Arc::clone(&stop);
        let capture_errors = Arc::clone(&errors);
        let capture_latency = latency.clone();
        let DeviceIo {
            input: input_producer,
            output: output_consumer,
        } = device_io;
        let capture_clock = clock.clone();
        let capture = thread::Builder::new()
            .name("vocal-wasapi-capture".into())
            .spawn(move || {
                capture_loop(
                    config,
                    mode,
                    input_producer,
                    capture_stop,
                    capture_errors,
                    capture_ready_tx,
                    capture_clock,
                    capture_latency,
                );
            })
            .map_err(WasapiBackendError::Spawn)?;

        let render_stop = Arc::clone(&stop);
        let render_errors = Arc::clone(&errors);
        let render_clock = clock;
        let render_profile = LatencyProfile {
            clock: render_clock,
            stats: latency.clone(),
        };
        let render = match thread::Builder::new()
            .name("vocal-wasapi-render".into())
            .spawn(move || {
                render_loop(
                    config,
                    mode,
                    output_consumer,
                    render_stop,
                    render_errors,
                    render_ready_tx,
                    render_profile,
                );
            }) {
            Ok(render) => render,
            Err(error) => {
                stop.store(true, Ordering::Release);
                let _ = capture.join();
                return Err(WasapiBackendError::Spawn(error));
            }
        };

        let capture_result = capture_ready_rx
            .recv()
            .unwrap_or_else(|_| Err("capture thread exited before startup".into()));
        let render_result = render_ready_rx
            .recv()
            .unwrap_or_else(|_| Err("render thread exited before startup".into()));
        if let Err(error) = capture_result {
            stop.store(true, Ordering::Release);
            let _ = capture.join();
            let _ = render.join();
            return Err(WasapiBackendError::Startup(error));
        }
        if let Err(error) = render_result {
            stop.store(true, Ordering::Release);
            let _ = capture.join();
            let _ = render.join();
            return Err(WasapiBackendError::Startup(error));
        }

        Ok((
            Self {
                stop,
                errors,
                latency,
                joins: Some((capture, render)),
            },
            worker_io,
        ))
    }

    pub fn take_errors(&self) -> Vec<String> {
        self.errors
            .lock()
            .map(|mut errors| std::mem::take(&mut *errors))
            .unwrap_or_default()
    }

    pub fn latency(&self) -> Option<AudioLatencySnapshot> {
        self.latency.as_deref().map(AudioLatencyStats::snapshot)
    }

    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Release);
        if let Some((capture, render)) = self.joins.take() {
            let _ = capture.join();
            let _ = render.join();
        }
    }
}

impl Drop for WasapiStream {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn period_hns(config: StreamConfig) -> Result<i64, String> {
    let period =
        (config.block_frames as u128 * 10_000_000).div_ceil(config.format.sample_rate as u128);
    i64::try_from(period).map_err(|_| "WASAPI period exceeds i64::MAX 100 ns units".into())
}

fn stream_mode(
    client: &AudioClient,
    config: StreamConfig,
    mode: WasapiMode,
) -> Result<StreamMode, String> {
    match mode {
        WasapiMode::SharedEvent => {
            let (default_period, _) = client.get_device_period().map_err(|e| e.to_string())?;
            Ok(StreamMode::EventsShared {
                autoconvert: true,
                buffer_duration_hns: default_period,
            })
        }
        WasapiMode::ExclusiveEvent => Ok(StreamMode::EventsExclusive {
            period_hns: period_hns(config)?,
        }),
    }
}

fn setup_capture(
    config: StreamConfig,
    mode: WasapiMode,
) -> Result<
    (
        AudioClient,
        Handle,
        AudioCaptureClient,
        BlockAssembler,
        Vec<f32>,
    ),
    String,
> {
    let enumerator = DeviceEnumerator::new().map_err(|e| e.to_string())?;
    let device = enumerator
        .get_default_device(&Direction::Capture)
        .map_err(|e| e.to_string())?;
    let mut client = device.get_iaudioclient().map_err(|e| e.to_string())?;
    let format = WaveFormat::new(
        32,
        32,
        &SampleType::Float,
        config.format.sample_rate as usize,
        config.format.channels as usize,
        None,
    );
    let stream_mode = stream_mode(&client, config, mode)?;
    client
        .initialize_client(&format, &Direction::Capture, &stream_mode)
        .map_err(|e| e.to_string())?;
    let event = client.set_get_eventhandle().map_err(|e| e.to_string())?;
    let capture = client.get_audiocaptureclient().map_err(|e| e.to_string())?;
    let max_frames = client.get_buffer_size().map_err(|e| e.to_string())? as usize;
    let max_samples = max_frames * config.format.channels as usize;
    let assembler = BlockAssembler::new(config).map_err(|e| e.to_string())?;
    client.start_stream().map_err(|e| e.to_string())?;
    Ok((client, event, capture, assembler, vec![0.0; max_samples]))
}

fn setup_render(
    config: StreamConfig,
    mode: WasapiMode,
) -> Result<
    (
        AudioClient,
        Handle,
        AudioRenderClient,
        BlockDrainer,
        Vec<f32>,
    ),
    String,
> {
    let enumerator = DeviceEnumerator::new().map_err(|e| e.to_string())?;
    let device = enumerator
        .get_default_device(&Direction::Render)
        .map_err(|e| e.to_string())?;
    let mut client = device.get_iaudioclient().map_err(|e| e.to_string())?;
    let format = WaveFormat::new(
        32,
        32,
        &SampleType::Float,
        config.format.sample_rate as usize,
        config.format.channels as usize,
        None,
    );
    let stream_mode = stream_mode(&client, config, mode)?;
    client
        .initialize_client(&format, &Direction::Render, &stream_mode)
        .map_err(|e| e.to_string())?;
    let event = client.set_get_eventhandle().map_err(|e| e.to_string())?;
    let render = client.get_audiorenderclient().map_err(|e| e.to_string())?;
    let max_frames = client.get_buffer_size().map_err(|e| e.to_string())? as usize;
    let max_samples = max_frames * config.format.channels as usize;
    let drainer = BlockDrainer::new(config).map_err(|e| e.to_string())?;
    client.start_stream().map_err(|e| e.to_string())?;
    Ok((client, event, render, drainer, vec![0.0; max_samples]))
}

fn f32_bytes(samples: &[f32]) -> &[u8] {
    // WASAPI uses the native little-endian IEEE-754 representation on
    // Windows. Reusing the f32 allocation removes a conversion buffer and a
    // second pass over every sample from both event loops.
    unsafe {
        slice::from_raw_parts(
            samples.as_ptr().cast::<u8>(),
            std::mem::size_of_val(samples),
        )
    }
}

fn f32_bytes_mut(samples: &mut [f32]) -> &mut [u8] {
    unsafe {
        slice::from_raw_parts_mut(
            samples.as_mut_ptr().cast::<u8>(),
            std::mem::size_of_val(samples),
        )
    }
}

fn packet_timestamp(
    clock: Option<&AudioClock>,
    latency: Option<&AudioLatencyStats>,
    info: &wasapi::BufferInfo,
) -> AudioTimestamp {
    if info.flags.data_discontinuity {
        if let Some(latency) = latency {
            latency.record_discontinuity();
        }
    }
    match clock {
        Some(clock) => {
            let mapped = (!info.flags.timestamp_error && info.timestamp != 0)
                .then(|| clock.from_wasapi_qpc(info.timestamp))
                .flatten();
            if let Some(latency) = latency {
                latency.record_device_timestamp(mapped.is_some());
            }
            mapped.unwrap_or_else(|| clock.now())
        }
        None => AudioTimestamp::ZERO,
    }
}

fn record_error(errors: &Mutex<Vec<String>>, error: String) {
    if let Ok(mut errors) = errors.lock() {
        errors.push(error);
    }
}

fn record_callback_tick(
    latency: Option<&AudioLatencyStats>,
    last_callback: &mut Option<Instant>,
    expected: Duration,
) {
    let Some(latency) = latency else {
        return;
    };
    let now = Instant::now();
    if let Some(previous) = last_callback.replace(now) {
        latency.record_callback_period(now.saturating_duration_since(previous), expected);
    }
}

fn capture_loop(
    config: StreamConfig,
    mode: WasapiMode,
    mut input: crate::AudioProducer,
    stop: Arc<AtomicBool>,
    errors: Arc<Mutex<Vec<String>>>,
    ready: mpsc::SyncSender<Result<(), String>>,
    clock: Option<AudioClock>,
    latency: Option<Arc<AudioLatencyStats>>,
) {
    let com_result = initialize_mta();
    if com_result.is_err() {
        let error = format!("failed to initialize COM MTA: {com_result}");
        let _ = ready.send(Err(error.clone()));
        record_error(&errors, error);
        return;
    }
    let _mmcss = MmcssGuard::register("Pro Audio");
    let setup = setup_capture(config, mode);
    let (client, event, capture, mut assembler, mut samples) = match setup {
        Ok(value) => {
            let _ = ready.send(Ok(()));
            value
        }
        Err(error) => {
            let _ = ready.send(Err(error.clone()));
            record_error(&errors, error);
            wasapi::deinitialize();
            return;
        }
    };
    let bytes_per_frame = config.format.channels as usize * std::mem::size_of::<f32>();
    if let Some(latency) = latency.as_ref() {
        latency.record_endpoint(
            samples.len() / config.format.channels as usize,
            config.format.sample_rate,
        );
    }
    let mut last_callback = None;
    while !stop.load(Ordering::Acquire) {
        if let Err(error) = event.wait_for_event(1000) {
            if !matches!(error, wasapi::WasapiError::EventTimeout) {
                record_error(&errors, error.to_string());
                break;
            }
            continue;
        }
        record_callback_tick(
            latency.as_deref(),
            &mut last_callback,
            config.block_duration(),
        );
        loop {
            let packet_frames = match capture.get_next_packet_size() {
                Ok(Some(frames)) => frames as usize,
                Ok(None) => client.get_current_padding().unwrap_or(0) as usize,
                Err(error) => {
                    record_error(&errors, error.to_string());
                    break;
                }
            };
            if packet_frames == 0 {
                break;
            }
            let byte_count = packet_frames.saturating_mul(bytes_per_frame);
            if byte_count > samples.len() * std::mem::size_of::<f32>()
                && !(packet_frames == config.block_frames && input.available_capacity() > 0)
            {
                record_error(
                    &errors,
                    "WASAPI capture packet exceeds endpoint buffer".into(),
                );
                break;
            }

            // With a fixed engine period the endpoint packet is already one
            // model block. Read it straight into the ring lease and avoid the
            // endpoint staging buffer plus the assembler copy. If the packet
            // shape changes, or the ring is full, fall back to the general
            // assembler path below so the capture packet is still released.
            let direct_result =
                if packet_frames == config.block_frames && input.available_capacity() > 0 {
                    match input.try_acquire() {
                        Ok(mut lease) => {
                            match capture.read_from_device(f32_bytes_mut(lease.samples_mut())) {
                                Ok((frames, info)) => {
                                    let sample_count = (frames as usize)
                                        .saturating_mul(config.format.channels as usize);
                                    let timestamp =
                                        packet_timestamp(clock.as_ref(), latency.as_deref(), &info);
                                    if frames as usize == config.block_frames {
                                        if info.flags.silent {
                                            lease.samples_mut().fill(0.0);
                                        }
                                        lease.set_timestamp(timestamp);
                                        lease.commit();
                                        Ok(Some((true, sample_count, timestamp)))
                                    } else if sample_count > samples.len() {
                                        Err("WASAPI capture packet exceeds endpoint buffer".into())
                                    } else {
                                        samples[..sample_count]
                                            .copy_from_slice(&lease.samples()[..sample_count]);
                                        if info.flags.silent {
                                            samples[..sample_count].fill(0.0);
                                        }
                                        Ok(Some((false, sample_count, timestamp)))
                                    }
                                }
                                Err(error) => Err(error.to_string()),
                            }
                        }
                        Err(AudioError::QueueFull) => Ok(None),
                        Err(error) => Err(error.to_string()),
                    }
                } else {
                    Ok(None)
                };
            match direct_result {
                Ok(Some((true, _, _))) => {
                    if let Some(latency) = latency.as_ref() {
                        let stats = input.stats();
                        latency.record_queue_state(
                            stats.queued_blocks,
                            stats.high_water_blocks,
                            stats.dropped_full,
                            stats.dropped_stale,
                        );
                    }
                    continue;
                }
                Ok(Some((false, sample_count, timestamp))) => {
                    let _ = assembler.push_at(&samples[..sample_count], &mut input, timestamp);
                    if let Some(latency) = latency.as_ref() {
                        let stats = input.stats();
                        latency.record_queue_state(
                            stats.queued_blocks,
                            stats.high_water_blocks,
                            stats.dropped_full,
                            stats.dropped_stale,
                        );
                    }
                    continue;
                }
                Ok(None) => {}
                Err(error) => {
                    record_error(&errors, error);
                    break;
                }
            }

            let (frames, info) =
                match capture.read_from_device(&mut f32_bytes_mut(&mut samples)[..byte_count]) {
                    Ok(value) => value,
                    Err(error) => {
                        record_error(&errors, error.to_string());
                        break;
                    }
                };
            let sample_count = frames as usize * config.format.channels as usize;
            if info.flags.silent {
                samples[..sample_count].fill(0.0);
            }
            let timestamp = packet_timestamp(clock.as_ref(), latency.as_deref(), &info);
            let _ = assembler.push_at(&samples[..sample_count], &mut input, timestamp);
            if let Some(latency) = latency.as_ref() {
                let stats = input.stats();
                latency.record_queue_state(
                    stats.queued_blocks,
                    stats.high_water_blocks,
                    stats.dropped_full,
                    stats.dropped_stale,
                );
            }
        }
    }
    let _ = client.stop_stream();
    wasapi::deinitialize();
}

fn render_loop(
    config: StreamConfig,
    mode: WasapiMode,
    mut output: crate::AudioConsumer,
    stop: Arc<AtomicBool>,
    errors: Arc<Mutex<Vec<String>>>,
    ready: mpsc::SyncSender<Result<(), String>>,
    profile: LatencyProfile,
) {
    let com_result = initialize_mta();
    if com_result.is_err() {
        let error = format!("failed to initialize COM MTA: {com_result}");
        let _ = ready.send(Err(error.clone()));
        record_error(&errors, error);
        return;
    }
    let _mmcss = MmcssGuard::register("Pro Audio");
    let setup = setup_render(config, mode);
    let (client, event, render, mut drainer, mut samples) = match setup {
        Ok(value) => {
            let _ = ready.send(Ok(()));
            value
        }
        Err(error) => {
            let _ = ready.send(Err(error.clone()));
            record_error(&errors, error);
            wasapi::deinitialize();
            return;
        }
    };
    let channels = config.format.channels as usize;
    let mut last_callback = None;
    while !stop.load(Ordering::Acquire) {
        if let Err(error) = event.wait_for_event(1000) {
            if !matches!(error, wasapi::WasapiError::EventTimeout) {
                record_error(&errors, error.to_string());
                break;
            }
            continue;
        }
        record_callback_tick(
            profile.stats.as_deref(),
            &mut last_callback,
            config.block_duration(),
        );
        let frames = match client.get_available_space_in_frames() {
            Ok(frames) => frames as usize,
            Err(error) => {
                record_error(&errors, error.to_string());
                break;
            }
        };
        if frames == 0 {
            continue;
        }
        let frames = frames.min(samples.len() / channels);
        let sample_count = frames * channels;
        if drainer.can_drain_direct(sample_count) {
            let direct_result = match output.try_acquire() {
                Ok(lease) => {
                    let timestamp = lease.timestamp();
                    let write_result =
                        render.write_to_device(frames, f32_bytes(lease.samples()), None);
                    drop(lease);
                    write_result
                        .map(|()| Some(timestamp))
                        .map_err(|error| error.to_string())
                }
                Err(AudioError::QueueEmpty) => Ok(None),
                Err(error) => Err(error.to_string()),
            };
            match direct_result {
                Ok(Some(timestamp)) => {
                    drainer.record_direct_block();
                    if let Some(latency) = profile.stats.as_ref() {
                        let stats = output.stats();
                        latency.record_queue_state(
                            stats.queued_blocks,
                            stats.high_water_blocks,
                            stats.dropped_full,
                            stats.dropped_stale,
                        );
                        if let Ok(padding) = client.get_current_padding() {
                            latency.record_endpoint(padding as usize, config.format.sample_rate);
                        }
                        if let Some(clock) = profile.clock.as_ref()
                            && let Some(age) = timestamp
                                .is_set()
                                .then(|| clock.elapsed(timestamp))
                                .flatten()
                        {
                            latency.record(age);
                        }
                    }
                    continue;
                }
                Ok(None) => {}
                Err(error) => {
                    record_error(&errors, error);
                    break;
                }
            }
        }
        let report = match drainer.drain(&mut samples[..sample_count], &mut output) {
            Ok(report) => report,
            Err(error) => {
                record_error(&errors, error.to_string());
                break;
            }
        };
        if let Some(latency) = profile.stats.as_ref() {
            latency.record_underflow(report.underflow_samples);
            let stats = output.stats();
            latency.record_queue_state(
                stats.queued_blocks,
                stats.high_water_blocks,
                stats.dropped_full,
                stats.dropped_stale,
            );
        }
        if let Err(error) =
            render.write_to_device(frames, f32_bytes(&samples[..sample_count]), None)
        {
            record_error(&errors, error.to_string());
            break;
        }
        if let Some(latency) = profile.stats.as_ref() {
            if let Ok(padding) = client.get_current_padding() {
                latency.record_endpoint(padding as usize, config.format.sample_rate);
            }
            if let Some(clock) = profile.clock.as_ref() {
                let age = report
                    .source_timestamp
                    .is_set()
                    .then(|| clock.elapsed(report.source_timestamp))
                    .flatten();
                if let Some(age) = age {
                    latency.record(age);
                }
            }
        }
    }
    let _ = client.stop_stream();
    wasapi::deinitialize();
}
