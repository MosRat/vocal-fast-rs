use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait};
use serde::Serialize;
use vocal_audio::cpal_backend::{CpalDuplexStream, build_duplex_profiled, recommended_host};
use vocal_audio::{
    AudioClock, AudioError, StreamConfig, StreamIo, StreamQueueConfig, WorkerInputPolicy,
    WorkerSchedule,
};
use vocal_engine::InferenceEngine;
use vocal_types::{AudioChunk, LifecycleState};

use crate::config::GuiConfig;

const WORKER_WAIT: Duration = Duration::from_millis(100);

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
    pub input_frames: u64,
    pub output_frames: u64,
    pub processing_micros: u64,
    pub failures: u64,
    pub underflow_samples: u64,
    pub queue_overflows: u64,
    pub latency_p95_micros: Option<u64>,
    pub stream_state: String,
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
    failures: AtomicU64,
}

pub struct AudioSession {
    stream: Option<CpalDuplexStream>,
    stop: Arc<AtomicBool>,
    worker_alive: Arc<AtomicBool>,
    join: Option<JoinHandle<()>>,
    telemetry: Arc<WorkerTelemetry>,
    input_device: String,
    output_device: String,
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
        config: &GuiConfig,
        engine: Box<dyn InferenceEngine>,
        report: Arc<dyn Fn(String) + Send + Sync>,
    ) -> Result<Self, String> {
        let host = recommended_host();
        let input = select_device(&host, config.input_device.as_deref(), true)?;
        let output = select_device(&host, config.output_device.as_deref(), false)?;
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
        let stream_config = StreamConfig {
            format: config.input_format,
            block_frames: config.block_frames,
            queue_blocks: config.queue_blocks,
        };
        let queues = StreamQueueConfig::new(config.queue_blocks, config.queue_blocks.min(2).max(1));
        let io = StreamIo::new_with_queue_config(stream_config, queues)
            .map_err(|error| error.to_string())?;
        let (device_io, worker_io) = io.split();
        let callback_report = Arc::clone(&report);
        let clock = AudioClock::new();
        let (stream, mut worker_io) = build_duplex_profiled(
            &input,
            &output,
            stream_config,
            device_io,
            worker_io,
            clock,
            move |error| callback_report(error.to_string()),
        )
        .map_err(|error| error.to_string())?;
        let stop = Arc::new(AtomicBool::new(false));
        let worker_stop = Arc::clone(&stop);
        let worker_alive = Arc::new(AtomicBool::new(true));
        let worker_alive_flag = Arc::clone(&worker_alive);
        let telemetry = Arc::new(WorkerTelemetry::default());
        let worker_telemetry = Arc::clone(&telemetry);
        let conversion = config.conversion;
        let format = config.input_format;
        let worker_report = Arc::clone(&report);
        let join = thread::Builder::new()
            .name("vocal-gui-audio-worker".into())
            .spawn(move || {
                let schedule = WorkerSchedule {
                    input_policy: WorkerInputPolicy::Fifo,
                    timeout: Some(WORKER_WAIT),
                };
                let mut sequence = 0_u64;
                let mut engine = engine;
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    while !worker_stop.load(Ordering::Acquire) {
                    let result = worker_io.process_once(schedule, |input, output| {
                        let started = Instant::now();
                        let chunk = AudioChunk {
                            format,
                            samples: input.to_vec(),
                            sequence,
                        };
                        let converted = engine
                            .process(&chunk, conversion)
                            .map_err(|error| AudioError::InvalidConfig(error.to_string()))?;
                        if converted.samples.len() != output.len() {
                            return Err(AudioError::BlockSizeMismatch {
                                expected: output.len(),
                                actual: converted.samples.len(),
                            });
                        }
                        output.copy_from_slice(&converted.samples);
                        worker_telemetry
                            .input_frames
                            .fetch_add(chunk.frames() as u64, Ordering::Relaxed);
                        worker_telemetry
                            .output_frames
                            .fetch_add(converted.frames() as u64, Ordering::Relaxed);
                        worker_telemetry.processing_micros.fetch_add(
                            started.elapsed().as_micros() as u64,
                            Ordering::Relaxed,
                        );
                        sequence = sequence.wrapping_add(1);
                        Ok(())
                    });
                    match result {
                        Ok(())
                        | Err(
                            AudioError::Timeout | AudioError::QueueEmpty | AudioError::QueueFull,
                        ) => {}
                        Err(error) => {
                            worker_telemetry.failures.fetch_add(1, Ordering::Relaxed);
                            worker_report(error.to_string());
                            worker_stop.store(true, Ordering::Release);
                            break;
                        }
                    }
                    }
                }));
                if result.is_err() {
                    worker_telemetry.failures.fetch_add(1, Ordering::Relaxed);
                    worker_report("audio worker panicked".into());
                    worker_stop.store(true, Ordering::Release);
                }
                worker_alive_flag.store(false, Ordering::Release);
            })
            .map_err(|error| error.to_string())?;

        Ok(Self {
            stream: Some(stream),
            stop,
            worker_alive,
            join: Some(join),
            telemetry,
            input_device: input_name,
            output_device: output_name,
        })
    }

    pub fn snapshot(&self) -> AudioTelemetry {
        let stats = &self.telemetry;
        let (underflow_samples, queue_overflows, latency_p95_micros, stream_state) = self
            .stream
            .as_ref()
            .map(|stream| {
                let latency = stream.latency();
                (
                    latency.map(|value| value.underflow_samples).unwrap_or(0),
                    latency.map(|value| value.queue_overflows).unwrap_or(0),
                    latency
                        .and_then(|value| value.p95)
                        .map(|value| value.as_micros() as u64),
                    format!("{:?}", stream.status().lifecycle),
                )
            })
            .unwrap_or_else(|| (0, 0, None, "Stopped".into()));
        let worker_alive = self.worker_alive.load(Ordering::Acquire);
        let state = self
            .stream
            .as_ref()
            .map(|stream| map_stream_state(stream.status().lifecycle))
            .map(|state| {
                if !worker_alive && state == LifecycleState::Running {
                    LifecycleState::Failed
                } else {
                    state
                }
            })
            .unwrap_or(LifecycleState::Stopped);
        AudioTelemetry {
            active: self.stream.is_some() && worker_alive,
            state,
            input_device: Some(self.input_device.clone()),
            output_device: Some(self.output_device.clone()),
            input_frames: stats.input_frames.load(Ordering::Relaxed),
            output_frames: stats.output_frames.load(Ordering::Relaxed),
            processing_micros: stats.processing_micros.load(Ordering::Relaxed),
            failures: stats.failures.load(Ordering::Relaxed),
            underflow_samples,
            queue_overflows,
            latency_p95_micros,
            stream_state,
        }
    }
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
        if let Some(join) = self.join.take() {
            let _ = join.join();
        }
        self.stream.take();
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
