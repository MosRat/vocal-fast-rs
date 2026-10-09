use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use crate::audio_error::AudioStartError;
use crate::audio_telemetry::{AudioSample, AudioSampler, WorkerCounters};
use cpal::traits::{DeviceTrait, HostTrait};
use serde::Serialize;
use vocal_audio::cpal_backend::{
    CpalDuplexStream, build_duplex_formats_profiled, recommended_host,
};
#[cfg(windows)]
use vocal_audio::wasapi_backend::WasapiStream;
use vocal_audio::{
    AudioClock, AudioError, DuplexWorkerAdapter, RealtimeBufferConfig, StreamConfig, StreamIo,
    StreamLifecycle, StreamQueueConfig, StreamStatus, WorkerInputPolicy, WorkerIo, WorkerSchedule,
};
use vocal_runtime::{RuntimeConfig, RuntimeContext};
use vocal_types::{AudioFormat, LifecycleState};

const WORKER_WAIT: Duration = Duration::from_millis(20);
static SESSION_ID: AtomicU64 = AtomicU64::new(0);

/// A device graph selected by the control plane. Windows uses the native
/// `IAudioClient3` graph with default or explicitly selected endpoints. CPAL is
/// a reported fallback. The worker receives native-format rings, not clients.
enum AudioStream {
    Cpal(CpalDuplexStream),
    #[cfg(windows)]
    Wasapi(WasapiStream),
}

impl AudioStream {
    fn latency(&self) -> Option<vocal_audio::AudioLatencySnapshot> {
        match self {
            Self::Cpal(stream) => stream.latency(),
            #[cfg(windows)]
            Self::Wasapi(stream) => stream.latency(),
        }
    }

    fn status(&self) -> StreamStatus {
        match self {
            Self::Cpal(stream) => stream.status(),
            #[cfg(windows)]
            Self::Wasapi(stream) => stream.status(),
        }
    }

    fn take_errors(&self) -> Vec<String> {
        match self {
            Self::Cpal(stream) => stream.take_errors(),
            #[cfg(windows)]
            Self::Wasapi(stream) => stream.take_errors(),
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioDeviceInfo {
    pub id: String,
    pub name: String,
    pub input: bool,
    pub output: bool,
    pub default_input: bool,
    pub default_output: bool,
    pub sample_rate: Option<u32>,
    pub channels: Option<u16>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioDeviceCatalog {
    pub host: String,
    pub devices: Vec<AudioDeviceInfo>,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioTelemetry {
    pub active: bool,
    pub state: LifecycleState,
    pub input_device: Option<String>,
    pub output_device: Option<String>,
    pub session_id: Option<String>,
    pub sample: Option<AudioSample>,
    pub stream_state: String,
    pub route: Option<AudioRouteInfo>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioRouteInfo {
    pub backend: &'static str,
    pub fallback_reason: Option<String>,
    pub capture_format: AudioFormat,
    pub render_format: AudioFormat,
    pub processing_format: AudioFormat,
    pub conversion_delay_micros: u64,
    pub processing_batch_frames: usize,
    pub playback_prefill_micros: u64,
}

struct OpenedAudioGraph {
    stream: AudioStream,
    worker: WorkerIo,
    input_name: String,
    output_name: String,
    route: AudioRouteInfo,
}

impl AudioTelemetry {
    pub fn stopped() -> Self {
        Self {
            state: LifecycleState::Stopped,
            stream_state: "Stopped".into(),
            ..Self::default()
        }
    }
}

#[derive(Debug, Default)]
struct WorkerTelemetry {
    input_frames: AtomicU64,
    output_frames: AtomicU64,
    processing_micros: AtomicU64,
    processed_blocks: AtomicU64,
    processed_audio_micros: AtomicU64,
    failures: AtomicU64,
}

pub struct AudioSession {
    stream: Option<AudioStream>,
    stop: Arc<AtomicBool>,
    worker_alive: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
    telemetry: Arc<WorkerTelemetry>,
    input_device: String,
    output_device: String,
    route: AudioRouteInfo,
    session_id: String,
    sampler: AudioSampler,
}

impl AudioSession {
    pub fn enumerate() -> Result<AudioDeviceCatalog, String> {
        let host = recommended_host();
        let default_input = host
            .default_input_device()
            .and_then(|device| device.id().ok())
            .map(|id| id.to_string());
        let default_output = host
            .default_output_device()
            .and_then(|device| device.id().ok())
            .map(|id| id.to_string());
        let mut devices = BTreeMap::<String, AudioDeviceInfo>::new();

        for device in host.devices().map_err(|error| error.to_string())? {
            let Ok(id) = device.id() else {
                continue;
            };
            let Ok(description) = device.description() else {
                continue;
            };
            let id = id.to_string();
            let name = description.name().to_owned();
            let entry = devices
                .entry(id.clone())
                .or_insert_with(|| AudioDeviceInfo {
                    id: id.clone(),
                    name: name.clone(),
                    input: false,
                    output: false,
                    default_input: default_input.as_deref() == Some(id.as_str()),
                    default_output: default_output.as_deref() == Some(id.as_str()),
                    sample_rate: None,
                    channels: None,
                });
            if let Ok(config) = device.default_input_config() {
                entry.input = true;
                entry.sample_rate.get_or_insert(config.sample_rate());
                entry.channels.get_or_insert(config.channels());
            }
            if let Ok(config) = device.default_output_config() {
                entry.output = true;
                entry.sample_rate.get_or_insert(config.sample_rate());
                entry.channels.get_or_insert(config.channels());
            }
        }
        Ok(AudioDeviceCatalog {
            host: format!("{:?}", host.id()),
            devices: devices.into_values().collect(),
        })
    }

    pub fn start(
        config: &RuntimeConfig,
        runtime: RuntimeContext,
        report: Arc<dyn Fn(String) + Send + Sync>,
    ) -> Result<Self, AudioStartError> {
        let started = Instant::now();
        let stream_config = StreamConfig {
            format: config.input_format,
            block_frames: config.block_frames,
            queue_blocks: config.queue_blocks,
        };
        let batch_frames = runtime
            .processing_batch_frames()
            .map_err(|error| error.to_string())?;
        let buffering = RealtimeBufferConfig::for_batch(stream_config, batch_frames)
            .map_err(|error| error.to_string())?;
        let queues = buffering.queues;
        let OpenedAudioGraph {
            stream,
            mut worker,
            input_name,
            output_name,
            mut route,
        } = open_audio_stream(config, stream_config, queues, Arc::clone(&report))?;
        let mut adapter = DuplexWorkerAdapter::with_batch_frames(
            &worker,
            config.input_format,
            config.output_format,
            buffering.batch_frames,
        )
        .map_err(|error| error.to_string())?;
        route.conversion_delay_micros = adapter.conversion_delay_micros();
        route.processing_batch_frames = buffering.batch_frames.max(config.block_frames);
        route.playback_prefill_micros = worker.output.config().block_duration().as_micros() as u64
            * queues.render_prefill_blocks as u64;
        // Once the device graph is ready, transfer the mutable engine to the
        // dedicated model worker. The control actor keeps only metadata and
        // receives the engine back after this worker exits.
        // Spawn first. A failed OS thread creation must leave the idle engine
        // in the runtime, rather than dropping an already transferred lease.
        let (engine_tx, engine_rx) =
            std::sync::mpsc::sync_channel::<(u64, Box<dyn vocal_engine::InferenceEngine>)>(1);
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let worker_alive = Arc::new(AtomicBool::new(true));
        let worker_alive_flag = Arc::clone(&worker_alive);
        let telemetry = Arc::new(WorkerTelemetry::default());
        let worker_telemetry = Arc::clone(&telemetry);
        let conversion = config.conversion;
        let worker_report = Arc::clone(&report);
        let worker_runtime = runtime.clone();
        let join = thread::Builder::new()
            .name("vocal-gui-model-worker".into())
            .spawn(move || {
                let Ok((worker_generation, engine)) = engine_rx.recv() else {
                    worker_alive_flag.store(false, Ordering::Release);
                    return;
                };
                #[cfg(windows)]
                let _mmcss = worker.register_mmcss("Pro Audio");
                let mut engine = Some(engine);
                let mut failure: Option<(String, String)> = None;
                let mut report_message: Option<String> = None;
                if let Err(error) = worker_runtime.activate_worker(worker_generation) {
                    worker_telemetry.failures.fetch_add(1, Ordering::Relaxed);
                    report_message = Some(error.to_string());
                    failure = Some((error.code().into(), error.to_string()));
                    worker_alive_flag.store(false, Ordering::Release);
                    worker_stop.store(true, Ordering::Release);
                }
                let schedule = WorkerSchedule {
                    // The model context must advance through continuous audio.
                    // Bounded queues absorb inference bursts without silently
                    // compressing captured time by discarding stale blocks.
                    input_policy: WorkerInputPolicy::Fifo,
                    timeout: Some(WORKER_WAIT),
                };
                let mut sequence = 0_u64;
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    if failure.is_some() {
                        return;
                    }
                    if let Some(worker_engine) = engine.as_mut() {
                        if let Err(error) = worker_engine.reset_stream() {
                            failure = Some((error.code().into(), error.to_string()));
                            report_message = Some(error.to_string());
                            worker_telemetry.failures.fetch_add(1, Ordering::Relaxed);
                            worker_stop.store(true, Ordering::Release);
                            return;
                        }
                    }
                    while !worker_stop.load(Ordering::Acquire) {
                        let result = adapter.process_batch_once(
                            &mut worker,
                            schedule,
                            |format, input, discontinuity| {
                                let started = Instant::now();
                                let Some(worker_engine) = engine.as_mut() else {
                                    return Err(AudioError::Processing(
                                        "model worker lost its engine ownership".into(),
                                    ));
                                };
                                let converted = match (|| {
                                    if discontinuity {
                                        worker_engine.reset_stream()?;
                                    }
                                    worker_engine
                                        .process_samples(format, input, sequence, conversion)
                                })() {
                                    Ok(converted) => converted,
                                    Err(error) => {
                                        failure = Some((error.code().into(), error.to_string()));
                                        return Err(AudioError::Processing(error.to_string()));
                                    }
                                };
                                worker_telemetry.processing_micros.fetch_add(
                                    started.elapsed().as_micros() as u64,
                                    Ordering::Relaxed,
                                );
                                worker_telemetry.processed_audio_micros.fetch_add(
                                    (input.len() / usize::from(format.channels)) as u64 * 1_000_000
                                        / u64::from(format.sample_rate),
                                    Ordering::Relaxed,
                                );
                                worker_telemetry
                                    .processed_blocks
                                    .fetch_add(1, Ordering::Relaxed);
                                sequence = sequence.wrapping_add(1);
                                Ok(converted.samples)
                            },
                        );
                        match result {
                            Ok((input_frames, output_frames)) => {
                                worker_telemetry
                                    .input_frames
                                    .fetch_add(input_frames as u64, Ordering::Relaxed);
                                worker_telemetry
                                    .output_frames
                                    .fetch_add(output_frames as u64, Ordering::Relaxed);
                            }
                            Err(
                                AudioError::Timeout
                                | AudioError::QueueEmpty
                                | AudioError::QueueFull,
                            ) => {}
                            Err(error) => {
                                worker_telemetry.failures.fetch_add(1, Ordering::Relaxed);
                                if failure.is_none() {
                                    failure = Some(("audio.worker".into(), error.to_string()));
                                }
                                worker_alive_flag.store(false, Ordering::Release);
                                report_message = Some(error.to_string());
                                worker_stop.store(true, Ordering::Release);
                                break;
                            }
                        }
                    }
                }));
                if result.is_err() {
                    worker_telemetry.failures.fetch_add(1, Ordering::Relaxed);
                    failure = Some((
                        "audio.worker_panic".into(),
                        "audio model worker panicked".into(),
                    ));
                    worker_alive_flag.store(false, Ordering::Release);
                    report_message = Some("audio worker panicked".into());
                    worker_stop.store(true, Ordering::Release);
                }
                worker_alive_flag.store(false, Ordering::Release);
                if let Some(engine) = engine.take() {
                    let _ = worker_runtime.complete_worker(worker_generation, engine, failure);
                }
                if let Some(message) = report_message {
                    worker_report(message);
                }
            })
            .map_err(|error| error.to_string())?;

        let lease = match runtime.take_engine_for_worker() {
            Ok(lease) => lease,
            Err(error) => {
                drop(engine_tx);
                let _ = join.join();
                return Err(error.to_string().into());
            }
        };
        if let Err(error) = engine_tx.send(lease) {
            let (generation, engine) = error.0;
            let _ = runtime.complete_worker(generation, engine, None);
            let _ = join.join();
            return Err("model worker exited before receiving its engine"
                .to_owned()
                .into());
        }

        Ok(Self {
            stream: Some(stream),
            stop,
            worker_alive,
            join: Some(join),
            telemetry,
            input_device: input_name,
            output_device: output_name,
            route,
            session_id: SESSION_ID.fetch_add(1, Ordering::Relaxed).to_string(),
            sampler: AudioSampler::new(started),
        })
    }

    pub fn snapshot(&mut self) -> AudioTelemetry {
        let stats = &self.telemetry;
        let sample = self.sampler.sample(
            Instant::now(),
            WorkerCounters {
                input_frames: stats.input_frames.load(Ordering::Relaxed),
                output_frames: stats.output_frames.load(Ordering::Relaxed),
                processing_micros: stats.processing_micros.load(Ordering::Relaxed),
                processed_blocks: stats.processed_blocks.load(Ordering::Relaxed),
                processed_audio_micros: stats.processed_audio_micros.load(Ordering::Relaxed),
                failures: stats.failures.load(Ordering::Relaxed),
            },
            self.stream.as_ref().and_then(AudioStream::latency),
            self.route.render_format,
        );
        let stream_state = self
            .stream
            .as_ref()
            .map(|stream| format!("{:?}", stream.status().lifecycle))
            .unwrap_or_else(|| "Stopped".into());
        let worker_alive = self.worker_alive.load(Ordering::Acquire);
        let stream_running = self
            .stream
            .as_ref()
            .is_some_and(|stream| stream.status().lifecycle == StreamLifecycle::Running);
        let state = if !worker_alive {
            LifecycleState::Failed
        } else {
            self.stream
                .as_ref()
                .map(|stream| map_stream_state(stream.status().lifecycle))
                .unwrap_or(LifecycleState::Stopped)
        };
        AudioTelemetry {
            active: self.stream.is_some() && worker_alive && stream_running,
            state,
            input_device: Some(self.input_device.clone()),
            output_device: Some(self.output_device.clone()),
            session_id: Some(self.session_id.clone()),
            sample,
            stream_state,
            route: Some(self.route.clone()),
        }
    }

    /// Stop a stream whose worker or device graph has already exited. This is
    /// called from the control-plane snapshot path, and also from a terminal
    /// worker error report; the caller may then drop the session to reclaim its
    /// join handle.
    pub fn reap_if_unhealthy(&mut self) -> Option<String> {
        let worker_dead = !self.worker_alive.load(Ordering::Acquire);
        let stream_status = self.stream.as_ref().map(AudioStream::status);
        let stream_dead = stream_status
            .is_some_and(|status| !matches!(status.lifecycle, StreamLifecycle::Running));
        if !worker_dead && !stream_dead {
            return None;
        }
        let message = if worker_dead {
            "audio model worker exited unexpectedly".to_owned()
        } else {
            let detail = self
                .stream
                .as_ref()
                .map(AudioStream::take_errors)
                .into_iter()
                .flatten()
                .next();
            detail.unwrap_or_else(|| {
                let state = stream_status
                    .map(|status| format!("{:?}", status.lifecycle))
                    .unwrap_or_else(|| "unknown".into());
                format!("audio device stream stopped unexpectedly ({state})")
            })
        };
        self.stop.store(true, Ordering::Release);
        self.stream.take();
        Some(message)
    }
}

fn open_audio_stream(
    config: &RuntimeConfig,
    stream_config: StreamConfig,
    queues: StreamQueueConfig,
    report: Arc<dyn Fn(String) + Send + Sync>,
) -> Result<OpenedAudioGraph, AudioStartError> {
    #[cfg(windows)]
    let fallback_reason = {
        let native = (|| {
            let endpoint_id = |value: Option<&str>| -> Result<Option<String>, String> {
                value
                    .map(|value| {
                        let id: cpal::DeviceId = value
                            .parse()
                            .map_err(|error| format!("invalid audio device ID: {error}"))?;
                        if id.host() != cpal::HostId::Wasapi {
                            return Err("selected device uses a different audio host".into());
                        }
                        Ok(id.id().to_owned())
                    })
                    .transpose()
            };
            WasapiStream::open_devices_profiled(
                stream_config,
                queues,
                AudioClock::new(),
                endpoint_id(config.input_device.as_deref())?,
                endpoint_id(config.output_device.as_deref())?,
            )
            .map_err(AudioStartError::native)
        })();
        match native {
            Ok((stream, worker_io)) => {
                let host = recommended_host();
                let input_name = select_device(&host, config.input_device.as_deref(), true)
                    .ok()
                    .and_then(|device| device.description().ok())
                    .map(|description| description.name().to_owned())
                    .unwrap_or_else(|| "Default input".into());
                let output_name = select_device(&host, config.output_device.as_deref(), false)
                    .ok()
                    .and_then(|device| device.description().ok())
                    .map(|description| description.name().to_owned())
                    .unwrap_or_else(|| "Default output".into());
                let route = AudioRouteInfo {
                    backend: "nativeWasapi",
                    fallback_reason: None,
                    capture_format: worker_io.input.config().format,
                    render_format: worker_io.output.config().format,
                    processing_format: config.input_format,
                    conversion_delay_micros: 0,
                    processing_batch_frames: 0,
                    playback_prefill_micros: 0,
                };
                return Ok(OpenedAudioGraph {
                    stream: AudioStream::Wasapi(stream),
                    worker: worker_io,
                    input_name,
                    output_name,
                    route,
                });
            }
            Err(mut error) => {
                describe_audio_error(config, &mut error);
                if !error.allows_fallback() {
                    return Err(error);
                }
                // Keep the portable adapter as an explicit runtime
                // fallback for drivers that do not expose a compatible
                // shared mix format or IAudioClient3 period range.
                tracing::warn!(
                    target: "vocal.gui.audio",
                    error = %error,
                    "native WASAPI unavailable; falling back to cpal"
                );
                Some(error)
            }
        }
    };
    #[cfg(not(windows))]
    let fallback_reason = None;

    open_cpal_stream(
        config,
        stream_config,
        queues,
        report,
        fallback_reason.as_ref().map(ToString::to_string),
    )
    .map_err(|mut error| {
        describe_audio_error(config, &mut error);
        if let Some(reason) = fallback_reason {
            error.technical_details.extend(reason.technical_details);
        }
        error
    })
}

fn describe_audio_error(config: &RuntimeConfig, error: &mut AudioStartError) {
    let capture = match error.direction.as_deref() {
        Some("capture") => true,
        Some("render") => false,
        _ => return,
    };
    let selected = if capture {
        config.input_device.as_deref()
    } else {
        config.output_device.as_deref()
    };
    if error.device_id.is_none() {
        error.device_id = selected.map(str::to_owned);
    }
    error.device_name = select_device(&recommended_host(), selected, capture)
        .ok()
        .and_then(|device| device.description().ok())
        .map(|description| description.name().to_owned());
}

fn open_cpal_stream(
    config: &RuntimeConfig,
    stream_config: StreamConfig,
    queues: StreamQueueConfig,
    report: Arc<dyn Fn(String) + Send + Sync>,
    fallback_reason: Option<String>,
) -> Result<OpenedAudioGraph, AudioStartError> {
    let host = recommended_host();
    let input = select_device(&host, config.input_device.as_deref(), true).map_err(|detail| {
        AudioStartError::unavailable("capture", config.input_device.as_deref(), detail)
    })?;
    let output =
        select_device(&host, config.output_device.as_deref(), false).map_err(|detail| {
            AudioStartError::unavailable("render", config.output_device.as_deref(), detail)
        })?;
    let input_name = input
        .description()
        .map_err(|error| error.to_string())?
        .name()
        .to_owned();
    let output_name = output
        .description()
        .map_err(|error| error.to_string())?
        .name()
        .to_owned();
    let input_default = input.default_input_config().map_err(|error| {
        AudioStartError::cpal(vocal_audio::cpal_backend::CpalError::InputConfig(error))
    })?;
    let output_default = output.default_output_config().map_err(|error| {
        AudioStartError::cpal(vocal_audio::cpal_backend::CpalError::OutputConfig(error))
    })?;
    let capture = stream_config
        .for_endpoint(AudioFormat {
            sample_rate: input_default.sample_rate(),
            channels: input_default.channels(),
        })
        .map_err(|error| error.to_string())?;
    let render = stream_config
        .for_endpoint(AudioFormat {
            sample_rate: output_default.sample_rate(),
            channels: output_default.channels(),
        })
        .map_err(|error| error.to_string())?;
    let io = StreamIo::new_duplex(capture, render, queues).map_err(|error| error.to_string())?;
    let (device_io, worker_io) = io.split();
    let callback_report = Arc::clone(&report);
    let (stream, worker_io) = build_duplex_formats_profiled(
        &input,
        &output,
        capture,
        render,
        device_io,
        worker_io,
        AudioClock::new(),
        move |error| callback_report(error.to_string()),
    )
    .map_err(|error| {
        let mut error = AudioStartError::cpal(error);
        let capture = error.direction.as_deref() == Some("capture");
        error.device_name = Some(if capture {
            input_name.clone()
        } else {
            output_name.clone()
        });
        error.device_id = if capture {
            config.input_device.clone()
        } else {
            config.output_device.clone()
        };
        error
    })?;
    Ok(OpenedAudioGraph {
        stream: AudioStream::Cpal(stream),
        worker: worker_io,
        input_name,
        output_name,
        route: AudioRouteInfo {
            backend: "cpal",
            fallback_reason,
            capture_format: capture.format,
            render_format: render.format,
            processing_format: config.input_format,
            conversion_delay_micros: 0,
            processing_batch_frames: 0,
            playback_prefill_micros: 0,
        },
    })
}

fn map_stream_state(state: vocal_audio::StreamLifecycle) -> LifecycleState {
    match state {
        vocal_audio::StreamLifecycle::Created => LifecycleState::Created,
        vocal_audio::StreamLifecycle::Starting => LifecycleState::Loading,
        vocal_audio::StreamLifecycle::Running => LifecycleState::Running,
        vocal_audio::StreamLifecycle::Stopping => LifecycleState::Stopping,
        vocal_audio::StreamLifecycle::Stopped => LifecycleState::Stopped,
        vocal_audio::StreamLifecycle::Failed => LifecycleState::Failed,
    }
}

impl Drop for AudioSession {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        // Stop device callbacks before waiting for the model worker. This
        // prevents a final callback burst from filling the SPSC rings while
        // the worker is unwinding and bounds shutdown work to the already
        // queued audio.
        self.stream.take();
        if let Some(join) = self.join.take() {
            // A worker can publish its terminal error through AppState before
            // its closure returns. That report may reap this session from the
            // worker thread itself; joining the current thread would panic.
            if join.thread().id() != thread::current().id() {
                let _ = join.join();
            }
        }
    }
}

fn select_device(
    host: &cpal::Host,
    requested: Option<&str>,
    input: bool,
) -> Result<cpal::Device, String> {
    if let Some(requested) = requested {
        let id = requested
            .parse::<cpal::DeviceId>()
            .map_err(|error| format!("invalid audio device id '{requested}': {error}"))?;
        return host.device_by_id(&id).ok_or_else(|| {
            format!(
                "{} device '{requested}' was not found",
                if input { "input" } else { "output" }
            )
        });
    }
    if input {
        host.default_input_device()
            .ok_or_else(|| "no default input device is available".into())
    } else {
        host.default_output_device()
            .ok_or_else(|| "no default output device is available".into())
    }
}
