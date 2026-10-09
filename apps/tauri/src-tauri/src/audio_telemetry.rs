//! Bounded UI measurements sampled by the control plane, never by a callback.
//! Cumulative counters stay in Rust. Multiple windows read a cached interval
//! instead of clearing counters or manufacturing a new sample on every read.

use std::time::{Duration, Instant};

use serde::Serialize;
use vocal_audio::{AudioLatencyDirectionSnapshot, AudioLatencySnapshot};
use vocal_types::AudioFormat;

const SAMPLE_INTERVAL: Duration = Duration::from_millis(500);

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct WorkerCounters {
    pub input_frames: u64,
    pub output_frames: u64,
    pub processing_micros: u64,
    pub processed_blocks: u64,
    pub processed_audio_micros: u64,
    pub failures: u64,
}

#[derive(Clone, Debug, Default, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DirectionTelemetry {
    pub queue_drops_per_second: f64,
    pub stale_drops_per_second: f64,
    pub queue_peak_blocks: u32,
    pub callback_period_ms: Option<f64>,
    pub callback_jitter_p95_ms: Option<f64>,
    pub endpoint_padding_p95_ms: Option<f64>,
    pub stream_latency_ms: Option<f64>,
    pub clock_drift_ppm: Option<f64>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct AudioSample {
    /// Wraps safely; elapsed_ms disambiguates it for frontend deduplication.
    pub sequence: u32,
    pub elapsed_ms: f64,
    pub window_ms: f64,
    pub input_frames_per_second: f64,
    /// Worker-produced native render frames, not frames already played.
    pub output_frames_per_second: f64,
    pub processing_ms: Option<f64>,
    pub block_budget_ms: Option<f64>,
    pub processing_load_percent: Option<f64>,
    pub worker_failures_per_second: f64,
    /// Interleaved missing samples converted with the physical render format.
    pub underflow_ms_per_second: f64,
    /// Latest timestamped audio age; None if no new measurement arrived.
    pub audio_age_ms: Option<f64>,
    /// Percentiles cover this session, not just this sampling interval.
    pub audio_age_p50_ms: Option<f64>,
    pub audio_age_p95_ms: Option<f64>,
    pub audio_age_p99_ms: Option<f64>,
    pub capture: DirectionTelemetry,
    pub render: DirectionTelemetry,
}

pub(crate) struct AudioSampler {
    started: Instant,
    sampled_at: Instant,
    previous: WorkerCounters,
    previous_latency: AudioLatencySnapshot,
    cached: Option<AudioSample>,
    sequence: u32,
}

impl AudioSampler {
    pub fn new(started: Instant) -> Self {
        Self {
            started,
            sampled_at: started,
            previous: WorkerCounters::default(),
            previous_latency: AudioLatencySnapshot::default(),
            cached: None,
            sequence: 0,
        }
    }

    pub fn sample(
        &mut self,
        now: Instant,
        counters: WorkerCounters,
        latency: Option<AudioLatencySnapshot>,
        render_format: AudioFormat,
    ) -> Option<AudioSample> {
        let interval = now.saturating_duration_since(self.sampled_at);
        if interval < SAMPLE_INTERVAL {
            return self.cached.clone();
        }
        let seconds = interval.as_secs_f64();
        let previous = self.previous;
        let blocks = counters
            .processed_blocks
            .wrapping_sub(previous.processed_blocks);
        let processing = counters
            .processing_micros
            .wrapping_sub(previous.processing_micros);
        let audio = counters
            .processed_audio_micros
            .wrapping_sub(previous.processed_audio_micros);
        let latency = latency.unwrap_or_default();
        let missing = latency
            .underflow_samples
            .saturating_sub(self.previous_latency.underflow_samples);
        self.sequence = self.sequence.wrapping_add(1);
        let sample = AudioSample {
            sequence: self.sequence,
            elapsed_ms: now.saturating_duration_since(self.started).as_secs_f64() * 1000.0,
            window_ms: seconds * 1000.0,
            input_frames_per_second: counters.input_frames.wrapping_sub(previous.input_frames)
                as f64
                / seconds,
            output_frames_per_second: counters.output_frames.wrapping_sub(previous.output_frames)
                as f64
                / seconds,
            processing_ms: (blocks > 0).then(|| processing as f64 / blocks as f64 / 1000.0),
            block_budget_ms: (blocks > 0).then(|| audio as f64 / blocks as f64 / 1000.0),
            processing_load_percent: (audio > 0).then(|| processing as f64 / audio as f64 * 100.0),
            worker_failures_per_second: counters.failures.wrapping_sub(previous.failures) as f64
                / seconds,
            underflow_ms_per_second: missing as f64 * 1000.0
                / f64::from(render_format.channels.max(1))
                / f64::from(render_format.sample_rate.max(1))
                / seconds,
            audio_age_ms: (latency.samples > self.previous_latency.samples)
                .then(|| milliseconds(latency.last))
                .flatten(),
            audio_age_p50_ms: milliseconds(latency.p50),
            audio_age_p95_ms: milliseconds(latency.p95),
            audio_age_p99_ms: milliseconds(latency.p99),
            capture: direction(latency.capture, self.previous_latency.capture, seconds),
            render: direction(latency.render, self.previous_latency.render, seconds),
        };
        self.previous = counters;
        self.previous_latency = latency;
        self.sampled_at = now;
        self.cached = Some(sample.clone());
        Some(sample)
    }
}

fn milliseconds(value: Option<Duration>) -> Option<f64> {
    value.map(|value| value.as_secs_f64() * 1000.0)
}

fn direction(
    current: AudioLatencyDirectionSnapshot,
    previous: AudioLatencyDirectionSnapshot,
    seconds: f64,
) -> DirectionTelemetry {
    DirectionTelemetry {
        queue_drops_per_second: current
            .queue_overflows
            .saturating_sub(previous.queue_overflows) as f64
            / seconds,
        stale_drops_per_second: current.stale_drops.saturating_sub(previous.stale_drops) as f64
            / seconds,
        queue_peak_blocks: current.max_queue_blocks.min(u32::MAX as usize) as u32,
        callback_period_ms: milliseconds(current.callback_period_mean),
        callback_jitter_p95_ms: milliseconds(current.callback_jitter_p95),
        endpoint_padding_p95_ms: milliseconds(current.endpoint_p95),
        stream_latency_ms: milliseconds(current.stream_latency),
        clock_drift_ppm: current.clock_drift_ppm.map(|value| value as f64),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn stereo() -> AudioFormat {
        AudioFormat {
            sample_rate: 48000,
            channels: 2,
        }
    }

    #[test]
    fn samples_rates_block_budget_and_both_drop_directions() {
        let now = Instant::now();
        let mut sampler = AudioSampler::new(now);
        let counters = WorkerCounters {
            input_frames: 16000,
            output_frames: 48000,
            processing_micros: 200000,
            processed_blocks: 100,
            processed_audio_micros: 1000000,
            failures: 1,
        };
        let latency = AudioLatencySnapshot {
            samples: 1,
            last: Some(Duration::from_millis(20)),
            underflow_samples: 960,
            capture: AudioLatencyDirectionSnapshot {
                queue_overflows: 2,
                stale_drops: 3,
                ..Default::default()
            },
            render: AudioLatencyDirectionSnapshot {
                queue_overflows: 4,
                ..Default::default()
            },
            ..Default::default()
        };
        let sample = sampler
            .sample(
                now + Duration::from_secs(1),
                counters,
                Some(latency),
                stereo(),
            )
            .unwrap();
        assert_eq!(sample.input_frames_per_second, 16000.0);
        assert_eq!(sample.output_frames_per_second, 48000.0);
        assert_eq!(sample.processing_ms, Some(2.0));
        assert_eq!(sample.block_budget_ms, Some(10.0));
        assert_eq!(sample.processing_load_percent, Some(20.0));
        assert_eq!(sample.underflow_ms_per_second, 10.0);
        assert_eq!(
            sample.capture.queue_drops_per_second + sample.render.queue_drops_per_second,
            6.0
        );
        assert_eq!(sample.capture.stale_drops_per_second, 3.0);
        assert_eq!(sample.audio_age_ms, Some(20.0));
        assert_eq!(sample.worker_failures_per_second, 1.0);
        // Another window sees the same interval, without resetting shared counters.
        let duplicate = sampler
            .sample(
                now + Duration::from_millis(1100),
                counters,
                Some(latency),
                stereo(),
            )
            .unwrap();
        assert_eq!(duplicate.sequence, sample.sequence);
        assert_eq!(duplicate.window_ms, sample.window_ms);
        let idle = sampler
            .sample(
                now + Duration::from_secs(2),
                counters,
                Some(latency),
                stereo(),
            )
            .unwrap();
        assert_eq!(idle.input_frames_per_second, 0.0);
        assert_eq!(idle.processing_ms, None);
        assert_eq!(idle.audio_age_ms, None);
        assert_eq!(idle.underflow_ms_per_second, 0.0);
    }

    #[test]
    fn worker_counter_wrap_and_saturated_profile_do_not_create_spikes() {
        let now = Instant::now();
        let mut sampler = AudioSampler::new(now);
        sampler.previous.input_frames = u64::MAX - 4;
        sampler.previous_latency.underflow_samples = u64::MAX;
        sampler.sequence = u32::MAX;
        let sample = sampler
            .sample(
                now + Duration::from_secs(1),
                WorkerCounters {
                    input_frames: 5,
                    ..Default::default()
                },
                Some(AudioLatencySnapshot {
                    underflow_samples: u64::MAX,
                    ..Default::default()
                }),
                stereo(),
            )
            .unwrap();
        assert_eq!(sample.input_frames_per_second, 10.0);
        assert_eq!(sample.underflow_ms_per_second, 0.0);
        assert_eq!(sample.sequence, 0);
        assert!(
            sampler
                .sample(now, WorkerCounters::default(), None, stereo())
                .is_some()
        );
    }

    #[test]
    fn first_interval_is_unavailable_and_json_has_no_cumulative_counters() {
        let now = Instant::now();
        let mut sampler = AudioSampler::new(now);
        assert!(
            sampler
                .sample(now, WorkerCounters::default(), None, stereo())
                .is_none()
        );
        let sample = sampler
            .sample(
                now + Duration::from_secs(1),
                WorkerCounters::default(),
                None,
                stereo(),
            )
            .unwrap();
        let json = serde_json::to_value(sample).unwrap();
        assert!(json.get("inputFrames").is_none());
        assert!(json.get("underflowSamples").is_none());
        assert!(json["processingMs"].is_null());
    }
}
