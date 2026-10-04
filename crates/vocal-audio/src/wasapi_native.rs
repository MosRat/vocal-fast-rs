//! Windows 10+ shared-mode backend using `IAudioClient3`.
//!
//! The `wasapi` crate intentionally exposes the older `IAudioClient` setup
//! surface. This module owns the small native setup layer needed to request a
//! real shared engine period, while keeping the same SPSC/lease contract as
//! the high-level adapter.

use std::ptr;
use std::slice;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use thiserror::Error;
use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0, WAIT_TIMEOUT};
use windows::Win32::Media::Audio::{
    AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY, AUDCLNT_BUFFERFLAGS_SILENT,
    AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR, AUDCLNT_STREAMFLAGS_EVENTCALLBACK, EDataFlow,
    IAudioCaptureClient, IAudioClient, IAudioClient3, IAudioClock, IAudioRenderClient,
    MMDeviceEnumerator, eCapture, eConsole, eRender,
};
use windows::Win32::Media::KernelStreaming::WAVE_FORMAT_EXTENSIBLE;
use windows::Win32::Media::Multimedia::{KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, WAVE_FORMAT_IEEE_FLOAT};
use windows::Win32::System::Com::{
    CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoTaskMemFree,
    CoUninitialize,
};
use windows::Win32::System::Threading::{CreateEventW, INFINITE, SetEvent, WaitForMultipleObjects};
use windows::core::Interface;

use crate::threading::MmcssGuard;
use crate::{
    AudioClock, AudioDirection, AudioError, AudioLatencySnapshot, AudioLatencyStats,
    AudioTimestamp, BlockAssembler, BlockDrainer, StreamConfig, StreamQueueConfig, WorkerIo,
};

#[derive(Debug, Error)]
pub enum NativeWasapiError {
    #[error("invalid native WASAPI stream configuration: {0}")]
    Audio(#[from] AudioError),
    #[error("native WASAPI worker failed to start: {0}")]
    Startup(String),
    #[error("failed to spawn native WASAPI worker: {0}")]
    Spawn(#[source] std::io::Error),
}

/// Shared-mode stream configured through `IAudioClient3::InitializeSharedAudioStream`.
pub struct NativeWasapiStream {
    stop: Arc<AtomicBool>,
    stop_event: EventHandle,
    errors: Arc<Mutex<Vec<String>>>,
    latency: Option<Arc<AudioLatencyStats>>,
    joins: Option<(JoinHandle<()>, JoinHandle<()>)>,
}

impl NativeWasapiStream {
    pub fn open(config: StreamConfig) -> Result<(Self, WorkerIo), NativeWasapiError> {
        Self::open_with_queue_config(config, StreamQueueConfig::default())
    }

    pub fn open_with_queue_config(
        config: StreamConfig,
        queues: StreamQueueConfig,
    ) -> Result<(Self, WorkerIo), NativeWasapiError> {
        Self::open_inner(config, queues, None)
    }

    pub fn open_profiled(
        config: StreamConfig,
        clock: AudioClock,
    ) -> Result<(Self, WorkerIo), NativeWasapiError> {
        Self::open_profiled_with_queue_config(config, StreamQueueConfig::default(), clock)
    }

    pub fn open_profiled_with_queue_config(
        config: StreamConfig,
        queues: StreamQueueConfig,
        clock: AudioClock,
    ) -> Result<(Self, WorkerIo), NativeWasapiError> {
        Self::open_inner(config, queues, Some(clock))
    }

    fn open_inner(
        config: StreamConfig,
        queues: StreamQueueConfig,
        clock: Option<AudioClock>,
    ) -> Result<(Self, WorkerIo), NativeWasapiError> {
        let config = config.validate()?;
        let io = crate::StreamIo::new_with_queue_config(config, queues)?;
        let (device, worker) = io.split();
        let stop = Arc::new(AtomicBool::new(false));
        let stop_event = EventHandle(
            unsafe { CreateEventW(None, true, false, None) }
                .map_err(|error| NativeWasapiError::Startup(error.to_string()))?,
        );
        let errors = Arc::new(Mutex::new(Vec::new()));
        let latency = clock
            .as_ref()
            .map(|_| Arc::new(AudioLatencyStats::default()));
        let (capture_ready_tx, capture_ready_rx) = mpsc::sync_channel(1);
        let (render_ready_tx, render_ready_rx) = mpsc::sync_channel(1);

        let crate::DeviceIo {
            input: input_producer,
            output: output_consumer,
        } = device;

        let capture = spawn_capture(
            config,
            input_producer,
            Arc::clone(&stop),
            stop_event.0.0 as usize,
            Arc::clone(&errors),
            capture_ready_tx,
            clock.clone(),
            latency.clone(),
        )?;
        let render = match spawn_render(
            config,
            output_consumer,
            Arc::clone(&stop),
            stop_event.0.0 as usize,
            Arc::clone(&errors),
            render_ready_tx,
            clock,
            latency.clone(),
        ) {
            Ok(render) => render,
            Err(error) => {
                signal_stop(&stop, stop_event.0);
                let _ = capture.join();
                return Err(error);
            }
        };

        let capture_result = capture_ready_rx
            .recv()
            .unwrap_or_else(|_| Err("capture thread exited before startup".into()));
        let render_result = render_ready_rx
            .recv()
            .unwrap_or_else(|_| Err("render thread exited before startup".into()));
        if let Err(error) = capture_result {
            signal_stop(&stop, stop_event.0);
            let _ = capture.join();
            let _ = render.join();
            return Err(NativeWasapiError::Startup(error));
        }
        if let Err(error) = render_result {
            signal_stop(&stop, stop_event.0);
            let _ = capture.join();
            let _ = render.join();
            return Err(NativeWasapiError::Startup(error));
        }

        Ok((
            Self {
                stop,
                stop_event,
                errors,
                latency,
                joins: Some((capture, render)),
            },
            worker,
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

    pub(crate) fn shutdown(&mut self) {
        signal_stop(&self.stop, self.stop_event.0);
        if let Some((capture, render)) = self.joins.take() {
            let _ = capture.join();
            let _ = render.join();
        }
    }
}

impl Drop for NativeWasapiStream {
    fn drop(&mut self) {
        self.shutdown();
    }
}

fn spawn_capture(
    config: StreamConfig,
    input: crate::AudioProducer,
    stop: Arc<AtomicBool>,
    stop_event: usize,
    errors: Arc<Mutex<Vec<String>>>,
    ready: mpsc::SyncSender<Result<(), String>>,
    clock: Option<AudioClock>,
    latency: Option<Arc<AudioLatencyStats>>,
) -> Result<JoinHandle<()>, NativeWasapiError> {
    thread::Builder::new()
        .name("vocal-wasapi3-capture".into())
        .spawn(move || {
            run_capture(
                config,
                input,
                stop,
                handle_from_raw(stop_event),
                errors,
                ready,
                clock,
                latency,
            )
        })
        .map_err(NativeWasapiError::Spawn)
}

fn spawn_render(
    config: StreamConfig,
    output: crate::AudioConsumer,
    stop: Arc<AtomicBool>,
    stop_event: usize,
    errors: Arc<Mutex<Vec<String>>>,
    ready: mpsc::SyncSender<Result<(), String>>,
    clock: Option<AudioClock>,
    latency: Option<Arc<AudioLatencyStats>>,
) -> Result<JoinHandle<()>, NativeWasapiError> {
    thread::Builder::new()
        .name("vocal-wasapi3-render".into())
        .spawn(move || {
            run_render(
                config,
                output,
                stop,
                handle_from_raw(stop_event),
                errors,
                ready,
                clock,
                latency,
            )
        })
        .map_err(NativeWasapiError::Spawn)
}

struct EventHandle(HANDLE);

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum WaitResult {
    Audio,
    Stop,
    Error,
}

// `windows::HANDLE` wraps a raw pointer and is intentionally not `Send` in
// the bindings. A kernel event handle is process-global, so pass its integer
// representation through the thread boundary and reconstruct the wrapper on
// the target thread. The owning `EventHandle` remains in `NativeWasapiStream`
// until both threads have joined.
fn handle_from_raw(raw: usize) -> HANDLE {
    HANDLE(raw as *mut _)
}

fn wait_for_audio(stop_event: HANDLE, audio_event: HANDLE) -> WaitResult {
    let handles = [stop_event, audio_event];
    let result = unsafe { WaitForMultipleObjects(&handles, false, INFINITE) };
    if result == WAIT_OBJECT_0 {
        WaitResult::Stop
    } else if result.0 == WAIT_OBJECT_0.0.saturating_add(1) {
        WaitResult::Audio
    } else if result == WAIT_TIMEOUT {
        // INFINITE cannot time out. Treat a spurious timeout as a normal
        // wake so a future iteration can observe the stop flag or audio event.
        WaitResult::Audio
    } else {
        WaitResult::Error
    }
}

fn signal_stop(stop: &AtomicBool, stop_event: HANDLE) {
    stop.store(true, Ordering::Release);
    let _ = unsafe { SetEvent(stop_event) };
}

impl Drop for EventHandle {
    fn drop(&mut self) {
        if !self.0.is_invalid() {
            unsafe {
                let _ = CloseHandle(self.0);
            }
        }
    }
}

struct CoTaskMemFormat(*mut windows::Win32::Media::Audio::WAVEFORMATEX);

impl Drop for CoTaskMemFormat {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe {
                CoTaskMemFree(Some(self.0.cast()));
            }
        }
    }
}

struct CaptureSetup {
    client: IAudioClient,
    capture: IAudioCaptureClient,
    clock: Option<IAudioClock>,
    event: EventHandle,
    endpoint_frames: usize,
    period_frames: usize,
}

struct RenderSetup {
    client: IAudioClient,
    render: IAudioRenderClient,
    clock: Option<IAudioClock>,
    event: EventHandle,
    block_align: usize,
    endpoint_frames: usize,
    period_frames: usize,
}

fn setup_client(
    config: StreamConfig,
    flow: EDataFlow,
) -> Result<
    (
        IAudioClient,
        Option<IAudioClock>,
        EventHandle,
        usize,
        usize,
        usize,
    ),
    String,
> {
    let enumerator: windows::Win32::Media::Audio::IMMDeviceEnumerator =
        unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL) }
            .map_err(|error| error.to_string())?;
    let device = unsafe { enumerator.GetDefaultAudioEndpoint(flow, eConsole) }
        .map_err(|error| error.to_string())?;
    let client3: IAudioClient3 =
        unsafe { device.Activate(CLSCTX_ALL, None) }.map_err(|error| error.to_string())?;
    let client: IAudioClient = client3.cast().map_err(|error| error.to_string())?;
    let format_ptr = unsafe { client.GetMixFormat() }.map_err(|error| error.to_string())?;
    let _format_mem = CoTaskMemFormat(format_ptr);
    let format = unsafe { ptr::read_unaligned(format_ptr) };
    let channels = format.nChannels;
    let sample_rate = format.nSamplesPerSec;
    let block_align = format.nBlockAlign as usize;
    let bits_per_sample = format.wBitsPerSample;
    let is_float32_tag = match format.wFormatTag as u32 {
        WAVE_FORMAT_IEEE_FLOAT => true,
        WAVE_FORMAT_EXTENSIBLE => unsafe {
            let extensible =
                format_ptr.cast::<windows::Win32::Media::Audio::WAVEFORMATEXTENSIBLE>();
            let sub_format = ptr::read_unaligned(std::ptr::addr_of!((*extensible).SubFormat));
            sub_format == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT
        },
        _ => false,
    };
    let is_float32 = is_float32_tag
        && bits_per_sample == 32
        && block_align == channels as usize * std::mem::size_of::<f32>();
    if !is_float32 || channels != config.format.channels || sample_rate != config.format.sample_rate
    {
        return Err(format!(
            "native shared mix format is {} Hz, {} channels, {} bits; expected {} Hz, {} channels, f32",
            sample_rate,
            channels,
            bits_per_sample,
            config.format.sample_rate,
            config.format.channels,
        ));
    }

    let mut default_period = 0;
    let mut fundamental_period = 0;
    let mut min_period = 0;
    let mut max_period = 0;
    unsafe {
        client3
            .GetSharedModeEnginePeriod(
                format_ptr,
                &mut default_period,
                &mut fundamental_period,
                &mut min_period,
                &mut max_period,
            )
            .map_err(|error| error.to_string())?;
    }
    // IAudioClock is present on normal WASAPI endpoints, but it is telemetry
    // rather than a data-path dependency. Keep the stream usable with an
    // unusual driver that does not expose the service.
    let clock = unsafe { client.GetService::<IAudioClock>().ok() };
    let period = choose_period(config.block_frames, fundamental_period, min_period, max_period)
        .ok_or_else(|| {
            format!(
                "IAudioClient3 returned invalid engine periods default={default_period}, fundamental={fundamental_period}, min={min_period}, max={max_period}"
            )
        })?;
    unsafe {
        client3
            .InitializeSharedAudioStream(
                AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                period,
                format_ptr,
                None,
            )
            .map_err(|error| error.to_string())?;
    }
    let event = unsafe { CreateEventW(None, false, false, None) }
        .map(EventHandle)
        .map_err(|error| error.to_string())?;
    unsafe {
        client
            .SetEventHandle(event.0)
            .map_err(|error| error.to_string())?;
    }
    let endpoint_frames = unsafe { client.GetBufferSize() }.map_err(|error| error.to_string())?;
    Ok((
        client,
        clock,
        event,
        block_align,
        endpoint_frames as usize,
        period as usize,
    ))
}

fn choose_period(target: usize, fundamental: u32, min: u32, max: u32) -> Option<u32> {
    if fundamental == 0 || min == 0 || max < min || max < fundamental {
        return None;
    }
    let target = target.clamp(min as usize, max as usize) as u64;
    let fundamental = u64::from(fundamental);
    let first = u64::from(min).div_ceil(fundamental);
    let last = u64::from(max) / fundamental;
    if first > last {
        return None;
    }

    // IAUDIOCLIENT3 accepts only fundamental-period multiples. Prefer an
    // exact model block, then a period that tiles the model block (or vice
    // versa), and finally the candidate whose least common multiple reaches
    // a block boundary in the fewest callbacks. This keeps the assembler and
    // drainer on stable boundaries instead of merely picking the numerically
    // closest period.
    let mut best: Option<(u8, u64, u64)> = None;
    let mut consider = |candidate: u64| {
        if candidate < u64::from(min)
            || candidate > u64::from(max)
            || !candidate.is_multiple_of(fundamental)
        {
            return;
        }
        let alignment_class = if candidate == target {
            0
        } else if target.is_multiple_of(candidate) || candidate.is_multiple_of(target) {
            1
        } else {
            2
        };
        let gcd = gcd_u64(target, candidate);
        let callbacks_to_boundary = target / gcd;
        let score = (alignment_class, callbacks_to_boundary, candidate);
        if best.map_or(true, |current| score < current) {
            best = Some(score);
        }
    };

    // Engine periods are normally in the tens or hundreds of frames. Keep
    // malformed driver ranges bounded while still examining all legal values
    // for normal devices.
    let span = last.saturating_sub(first);
    if span <= 16_384 {
        for multiplier in first..=last {
            consider(multiplier.saturating_mul(fundamental));
        }
    } else {
        consider(first.saturating_mul(fundamental));
        consider(last.saturating_mul(fundamental));
        let nearest = target / fundamental;
        for multiplier in nearest.saturating_sub(2)..=nearest.saturating_add(2) {
            consider(multiplier.saturating_mul(fundamental));
        }
    }
    best.map(|(_, _, candidate)| candidate as u32)
}

fn gcd_u64(mut left: u64, mut right: u64) -> u64 {
    while right != 0 {
        let remainder = left % right;
        left = right;
        right = remainder;
    }
    left
}

fn setup_capture(config: StreamConfig) -> Result<CaptureSetup, String> {
    let (client, clock, event, _block_align, endpoint_frames, period_frames) =
        setup_client(config, eCapture)?;
    let capture =
        unsafe { client.GetService::<IAudioCaptureClient>() }.map_err(|error| error.to_string())?;
    unsafe { client.Start() }.map_err(|error| error.to_string())?;
    Ok(CaptureSetup {
        client,
        capture,
        clock,
        event,
        endpoint_frames,
        period_frames,
    })
}

fn setup_render(config: StreamConfig) -> Result<RenderSetup, String> {
    let (client, clock, event, block_align, endpoint_frames, period_frames) =
        setup_client(config, eRender)?;
    let render =
        unsafe { client.GetService::<IAudioRenderClient>() }.map_err(|error| error.to_string())?;
    unsafe { client.Start() }.map_err(|error| error.to_string())?;
    Ok(RenderSetup {
        client,
        render,
        clock,
        event,
        block_align,
        endpoint_frames,
        period_frames,
    })
}

fn run_capture(
    config: StreamConfig,
    mut input: crate::AudioProducer,
    stop: Arc<AtomicBool>,
    stop_event: HANDLE,
    errors: Arc<Mutex<Vec<String>>>,
    ready: mpsc::SyncSender<Result<(), String>>,
    clock: Option<AudioClock>,
    latency: Option<Arc<AudioLatencyStats>>,
) {
    let com = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
    if let Err(error) = com.ok() {
        let message = format!("failed to initialize COM MTA: {error}");
        let _ = ready.send(Err(message.clone()));
        record_error(&errors, message);
        return;
    }
    let _mmcss = MmcssGuard::register("Pro Audio");
    let setup = setup_capture(config);
    let CaptureSetup {
        client,
        capture,
        clock: device_clock,
        event,
        endpoint_frames,
        period_frames,
    } = match setup {
        Ok(setup) => setup,
        Err(error) => {
            let _ = ready.send(Err(error.clone()));
            record_error(&errors, error);
            unsafe { CoUninitialize() };
            return;
        }
    };
    if let Some(latency) = latency.as_ref() {
        latency.record_endpoint_capacity_for(
            AudioDirection::Capture,
            endpoint_frames,
            config.format.sample_rate,
        );
    }
    let mut assembler = match BlockAssembler::new(config) {
        Ok(assembler) => assembler,
        Err(error) => {
            let message = error.to_string();
            let _ = ready.send(Err(message.clone()));
            record_error(&errors, message);
            let _ = unsafe { client.Stop() };
            unsafe { CoUninitialize() };
            return;
        }
    };
    let mut silence = match crate::allocate_samples(
        endpoint_frames.saturating_mul(config.format.channels as usize),
    ) {
        Ok(samples) => samples,
        Err(error) => {
            let message = error.to_string();
            let _ = ready.send(Err(message.clone()));
            record_error(&errors, message);
            let _ = unsafe { client.Stop() };
            unsafe { CoUninitialize() };
            return;
        }
    };
    let _ = ready.send(Ok(()));
    let mut last_callback = None;
    let expected_period = period_duration(period_frames, config.format.sample_rate);
    'capture_loop: while !stop.load(Ordering::Acquire) {
        let wait = wait_for_audio(stop_event, event.0);
        if wait == WaitResult::Stop {
            break;
        }
        if wait == WaitResult::Error {
            record_error(&errors, "native WASAPI capture event wait failed".into());
            break;
        }
        record_callback_tick(
            latency.as_deref(),
            &mut last_callback,
            expected_period,
            AudioDirection::Capture,
        );
        sample_device_clock(
            device_clock.as_ref(),
            AudioDirection::Capture,
            latency.as_deref(),
            config.format.sample_rate,
        );
        if let Some(latency) = latency.as_ref()
            && let Ok(padding) = unsafe { client.GetCurrentPadding() }
        {
            latency.record_endpoint_for(
                AudioDirection::Capture,
                padding as usize,
                config.format.sample_rate,
            );
        }
        loop {
            let packet_frames = match unsafe { capture.GetNextPacketSize() } {
                Ok(frames) => frames as usize,
                Err(error) => {
                    record_error(&errors, error.to_string());
                    break 'capture_loop;
                }
            };
            if packet_frames == 0 {
                break;
            }
            let mut data = ptr::null_mut();
            let mut frames = 0_u32;
            let mut flags = 0_u32;
            let mut device_position = 0_u64;
            let mut qpc_position = 0_u64;
            if let Err(error) = unsafe {
                capture.GetBuffer(
                    &mut data,
                    &mut frames,
                    &mut flags,
                    Some(&mut device_position),
                    Some(&mut qpc_position),
                )
            } {
                record_error(&errors, error.to_string());
                break 'capture_loop;
            }
            let sample_count = frames as usize * config.format.channels as usize;
            if sample_count > silence.len() {
                let _ = unsafe { capture.ReleaseBuffer(frames) };
                record_error(
                    &errors,
                    "native WASAPI capture packet exceeds endpoint buffer".into(),
                );
                break;
            }
            let timestamp =
                packet_timestamp(clock.as_ref(), latency.as_deref(), flags, qpc_position);
            if flags & AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY.0 as u32 != 0 {
                assembler.reset();
            }
            let direct = if assembler.buffered_samples() == 0
                && frames as usize == config.block_frames
                && input.available_capacity() > 0
            {
                match input.try_acquire() {
                    Ok(mut lease) => {
                        if flags & AUDCLNT_BUFFERFLAGS_SILENT.0 as u32 != 0 {
                            lease.samples_mut().fill(0.0);
                        } else {
                            let source =
                                unsafe { slice::from_raw_parts(data.cast::<f32>(), sample_count) };
                            lease.samples_mut().copy_from_slice(source);
                        }
                        lease.set_timestamp(timestamp);
                        lease.commit();
                        true
                    }
                    Err(AudioError::QueueFull) => false,
                    Err(error) => {
                        record_error(&errors, error.to_string());
                        false
                    }
                }
            } else {
                false
            };
            if direct {
                if let Err(error) = unsafe { capture.ReleaseBuffer(frames) } {
                    record_error(&errors, error.to_string());
                    break 'capture_loop;
                }
                if let Some(latency) = latency.as_ref() {
                    let stats = input.stats();
                    latency.record_queue_state_for(
                        AudioDirection::Capture,
                        stats.queued_blocks,
                        stats.high_water_blocks,
                        stats.dropped_full,
                        stats.dropped_stale,
                    );
                }
                continue;
            }
            let read_result = if flags & AUDCLNT_BUFFERFLAGS_SILENT.0 as u32 != 0 {
                silence[..sample_count].fill(0.0);
                assembler.push_at(&silence[..sample_count], &mut input, timestamp)
            } else {
                let samples = unsafe { slice::from_raw_parts(data.cast::<f32>(), sample_count) };
                assembler.push_at(samples, &mut input, timestamp)
            };
            if let Err(error) = unsafe { capture.ReleaseBuffer(frames) } {
                record_error(&errors, error.to_string());
                break 'capture_loop;
            }
            if let Err(error) = read_result {
                if !matches!(error, AudioError::QueueFull) {
                    record_error(&errors, error.to_string());
                    break 'capture_loop;
                }
            }
            if let Some(latency) = latency.as_ref() {
                latency.record_assembler_buffered(
                    assembler
                        .buffered_samples()
                        .checked_div(config.format.channels as usize)
                        .unwrap_or(0),
                );
                let stats = input.stats();
                latency.record_queue_state_for(
                    AudioDirection::Capture,
                    stats.queued_blocks,
                    stats.high_water_blocks,
                    stats.dropped_full,
                    stats.dropped_stale,
                );
            }
        }
    }
    let _ = unsafe { client.Stop() };
    unsafe { CoUninitialize() };
}

fn run_render(
    config: StreamConfig,
    mut output: crate::AudioConsumer,
    stop: Arc<AtomicBool>,
    stop_event: HANDLE,
    errors: Arc<Mutex<Vec<String>>>,
    ready: mpsc::SyncSender<Result<(), String>>,
    clock: Option<AudioClock>,
    latency: Option<Arc<AudioLatencyStats>>,
) {
    let com = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
    if let Err(error) = com.ok() {
        let message = format!("failed to initialize COM MTA: {error}");
        let _ = ready.send(Err(message.clone()));
        record_error(&errors, message);
        return;
    }
    let _mmcss = MmcssGuard::register("Pro Audio");
    let setup = setup_render(config);
    let RenderSetup {
        client,
        render,
        clock: device_clock,
        event,
        block_align,
        endpoint_frames,
        period_frames,
    } = match setup {
        Ok(setup) => setup,
        Err(error) => {
            let _ = ready.send(Err(error.clone()));
            record_error(&errors, error);
            unsafe { CoUninitialize() };
            return;
        }
    };
    if let Some(latency) = latency.as_ref() {
        latency.record_endpoint_capacity_for(
            AudioDirection::Render,
            endpoint_frames,
            config.format.sample_rate,
        );
    }
    let mut drainer = match BlockDrainer::new(config) {
        Ok(drainer) => drainer,
        Err(error) => {
            let message = error.to_string();
            let _ = ready.send(Err(message.clone()));
            record_error(&errors, message);
            let _ = unsafe { client.Stop() };
            unsafe { CoUninitialize() };
            return;
        }
    };
    let mut samples = match crate::allocate_samples(
        endpoint_frames.saturating_mul(config.format.channels as usize),
    ) {
        Ok(samples) => samples,
        Err(error) => {
            let message = error.to_string();
            let _ = ready.send(Err(message.clone()));
            record_error(&errors, message);
            let _ = unsafe { client.Stop() };
            unsafe { CoUninitialize() };
            return;
        }
    };
    let _ = ready.send(Ok(()));
    let mut last_callback = None;
    let expected_period = period_duration(period_frames, config.format.sample_rate);
    while !stop.load(Ordering::Acquire) {
        let wait = wait_for_audio(stop_event, event.0);
        if wait == WaitResult::Stop {
            break;
        }
        if wait == WaitResult::Error {
            record_error(&errors, "native WASAPI render event wait failed".into());
            break;
        }
        record_callback_tick(
            latency.as_deref(),
            &mut last_callback,
            expected_period,
            AudioDirection::Render,
        );
        sample_device_clock(
            device_clock.as_ref(),
            AudioDirection::Render,
            latency.as_deref(),
            config.format.sample_rate,
        );
        let available = match unsafe { client.GetCurrentPadding() } {
            Ok(padding) => endpoint_frames.saturating_sub(padding as usize),
            Err(error) => {
                record_error(&errors, error.to_string());
                break;
            }
        };
        if available == 0 {
            continue;
        }
        let frames = available.min(samples.len() / config.format.channels as usize);
        let sample_count = frames * config.format.channels as usize;
        if drainer.can_drain_direct(sample_count) {
            let direct_result = match output.try_acquire() {
                Ok(lease) => {
                    let timestamp = lease.timestamp();
                    let buffer = match unsafe { render.GetBuffer(frames as u32) } {
                        Ok(buffer) => buffer,
                        Err(error) => {
                            drop(lease);
                            record_error(&errors, error.to_string());
                            break;
                        }
                    };
                    unsafe {
                        let destination = slice::from_raw_parts_mut(buffer, frames * block_align);
                        let source = slice::from_raw_parts(
                            lease.samples().as_ptr().cast::<u8>(),
                            frames * block_align,
                        );
                        destination.copy_from_slice(source);
                    }
                    drop(lease);
                    if let Err(error) = unsafe { render.ReleaseBuffer(frames as u32, 0) } {
                        record_error(&errors, error.to_string());
                        break;
                    }
                    Some(timestamp)
                }
                Err(AudioError::QueueEmpty) => None,
                Err(error) => {
                    record_error(&errors, error.to_string());
                    break;
                }
            };
            if let Some(timestamp) = direct_result {
                drainer.record_direct_block();
                if let Some(latency) = latency.as_ref() {
                    let stats = output.stats();
                    latency.record_queue_state_for(
                        AudioDirection::Render,
                        stats.queued_blocks,
                        stats.high_water_blocks,
                        stats.dropped_full,
                        stats.dropped_stale,
                    );
                    if let Ok(padding) = unsafe { client.GetCurrentPadding() } {
                        latency.record_endpoint_for(
                            AudioDirection::Render,
                            padding as usize,
                            config.format.sample_rate,
                        );
                    }
                    if let Some(clock) = clock.as_ref()
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
        }
        let report = match drainer.drain(&mut samples[..sample_count], &mut output) {
            Ok(report) => report,
            Err(error) => {
                record_error(&errors, error.to_string());
                break;
            }
        };
        if let Some(latency) = latency.as_ref() {
            latency.record_drainer_pending(
                drainer
                    .pending_samples()
                    .checked_div(config.format.channels as usize)
                    .unwrap_or(0),
            );
        }
        let buffer = match unsafe { render.GetBuffer(frames as u32) } {
            Ok(buffer) => buffer,
            Err(error) => {
                record_error(&errors, error.to_string());
                break;
            }
        };
        unsafe {
            let destination = slice::from_raw_parts_mut(buffer, frames * block_align);
            let source = slice::from_raw_parts(samples.as_ptr().cast::<u8>(), frames * block_align);
            destination.copy_from_slice(source);
        }
        if let Err(error) = unsafe { render.ReleaseBuffer(frames as u32, 0) } {
            record_error(&errors, error.to_string());
            break;
        }
        if let Some(latency) = latency.as_ref() {
            latency.record_underflow(report.underflow_samples);
            let stats = output.stats();
            latency.record_queue_state_for(
                AudioDirection::Render,
                stats.queued_blocks,
                stats.high_water_blocks,
                stats.dropped_full,
                stats.dropped_stale,
            );
            if let Ok(padding) = unsafe { client.GetCurrentPadding() } {
                latency.record_endpoint_for(
                    AudioDirection::Render,
                    padding as usize,
                    config.format.sample_rate,
                );
            }
            if let Some(clock) = clock.as_ref()
                && let Some(age) = report
                    .source_timestamp
                    .is_set()
                    .then(|| clock.elapsed(report.source_timestamp))
                    .flatten()
            {
                latency.record(age);
            }
        }
    }
    let _ = unsafe { client.Stop() };
    unsafe { CoUninitialize() };
}

fn sample_device_clock(
    clock: Option<&IAudioClock>,
    direction: AudioDirection,
    latency: Option<&AudioLatencyStats>,
    sample_rate: u32,
) {
    let (Some(clock), Some(latency)) = (clock, latency) else {
        return;
    };
    let mut position = 0_u64;
    let mut qpc = 0_u64;
    if unsafe { clock.GetPosition(&mut position, Some(&mut qpc)) }.is_ok() {
        latency.record_device_clock(direction, position, qpc, sample_rate);
    }
}

fn packet_timestamp(
    clock: Option<&AudioClock>,
    latency: Option<&AudioLatencyStats>,
    flags: u32,
    qpc_position: u64,
) -> AudioTimestamp {
    if flags & AUDCLNT_BUFFERFLAGS_DATA_DISCONTINUITY.0 as u32 != 0
        && let Some(latency) = latency
    {
        latency.record_discontinuity();
    }
    match clock {
        Some(clock) => {
            let mapped = (flags & AUDCLNT_BUFFERFLAGS_TIMESTAMP_ERROR.0 as u32 == 0)
                .then(|| clock.from_wasapi_qpc(qpc_position))
                .flatten();
            if let Some(latency) = latency {
                latency.record_device_timestamp(mapped.is_some());
            }
            mapped.unwrap_or_else(|| clock.now())
        }
        None => AudioTimestamp::ZERO,
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

fn period_duration(frames: usize, sample_rate: u32) -> Duration {
    if frames == 0 || sample_rate == 0 {
        return Duration::ZERO;
    }
    let nanos = ((frames as u128) * 1_000_000_000)
        .div_ceil(u128::from(sample_rate))
        .min(u128::from(u64::MAX)) as u64;
    Duration::from_nanos(nanos)
}

fn record_error(errors: &Mutex<Vec<String>>, error: String) {
    if let Ok(mut errors) = errors.lock() {
        errors.push(error);
    }
}
