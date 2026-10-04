use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::time::{Duration, Instant};

/// Which endpoint produced a profile sample.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AudioDirection {
    Capture,
    Render,
}

impl AudioDirection {
    const fn index(self) -> usize {
        match self {
            Self::Capture => 0,
            Self::Render => 1,
        }
    }
}

static PROFILE_EPOCH: OnceLock<Instant> = OnceLock::new();

fn profile_now_nanos() -> u64 {
    PROFILE_EPOCH
        .get_or_init(Instant::now)
        .elapsed()
        .as_nanos()
        .min(u64::MAX as u128) as u64
}

/// Monotonic timestamp used only when an adapter opts into latency profiling.
/// Zero means that a slot is not timestamped, so the normal queue path keeps
/// the same memory and instruction footprint as before.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AudioTimestamp(u64);

impl AudioTimestamp {
    pub const ZERO: Self = Self(0);

    pub const fn is_set(self) -> bool {
        self.0 != 0
    }
}

/// Process-local monotonic clock for capture-to-render measurements.
#[derive(Clone, Debug)]
pub struct AudioClock {
    origin: Instant,
    #[cfg(windows)]
    qpc: Option<QpcOrigin>,
}

#[cfg(windows)]
#[derive(Clone, Copy, Debug)]
struct QpcOrigin {
    ticks: i64,
    frequency: i64,
}

impl Default for AudioClock {
    fn default() -> Self {
        Self::new()
    }
}

impl AudioClock {
    pub fn new() -> Self {
        Self {
            origin: Instant::now(),
            #[cfg(windows)]
            qpc: query_qpc_origin(),
        }
    }

    pub fn now(&self) -> AudioTimestamp {
        let nanos = self
            .qpc_now_nanos()
            .unwrap_or_else(|| self.origin.elapsed().as_nanos().min(u64::MAX as u128));
        AudioTimestamp(nanos.max(1).min(u64::MAX as u128) as u64)
    }

    /// Convert the QPC timestamp returned by `IAudioCaptureClient::GetBuffer`
    /// to this process-local monotonic clock. WASAPI reports that value in
    /// 100 ns units. A packet with an invalid timestamp falls back to the
    /// host clock and is accounted for by the caller.
    #[cfg(windows)]
    pub fn from_wasapi_qpc(&self, qpc_100ns: u64) -> Option<AudioTimestamp> {
        let origin = self.qpc?;
        let qpc_100ns = i128::from(qpc_100ns);
        let origin_100ns = i128::from(origin.ticks)
            .checked_mul(10_000_000)?
            .checked_div(i128::from(origin.frequency))?;
        let elapsed_100ns = qpc_100ns.checked_sub(origin_100ns)?;
        if elapsed_100ns < 0 {
            return None;
        }
        let nanos = elapsed_100ns.checked_mul(100)?;
        u64::try_from(nanos)
            .ok()
            .map(|nanos| AudioTimestamp(nanos.max(1)))
    }

    #[cfg(not(windows))]
    pub fn from_wasapi_qpc(&self, _qpc_100ns: u64) -> Option<AudioTimestamp> {
        None
    }

    #[cfg(windows)]
    fn qpc_now_nanos(&self) -> Option<u128> {
        let origin = self.qpc?;
        let ticks = query_qpc_ticks()?;
        let elapsed = ticks.checked_sub(origin.ticks)?;
        if elapsed < 0 {
            return None;
        }
        u128::try_from(elapsed)
            .ok()?
            .checked_mul(1_000_000_000)
            .and_then(|nanos| nanos.checked_div(u128::try_from(origin.frequency).ok()?))
    }

    #[cfg(not(windows))]
    fn qpc_now_nanos(&self) -> Option<u128> {
        None
    }

    pub fn elapsed(&self, timestamp: AudioTimestamp) -> Option<Duration> {
        if !timestamp.is_set() {
            return None;
        }
        let now = self.now().0;
        now.checked_sub(timestamp.0).map(Duration::from_nanos)
    }
}

/// Lock-free aggregate for opt-in output age measurements.
#[derive(Debug)]
pub struct AudioLatencyStats {
    samples: AtomicU64,
    total_nanos: AtomicU64,
    last_nanos: AtomicU64,
    max_nanos: AtomicU64,
    underflow_samples: AtomicU64,
    timestamped_packets: AtomicU64,
    timestamp_errors: AtomicU64,
    discontinuities: AtomicU64,
    max_queue_blocks: AtomicU64,
    queue_overflows: AtomicU64,
    stale_drops: AtomicU64,
    max_endpoint_frames: AtomicU64,
    max_endpoint_nanos: AtomicU64,
    max_endpoint_capacity_frames: AtomicU64,
    max_endpoint_capacity_nanos: AtomicU64,
    endpoint_samples: AtomicU64,
    endpoint_buckets: [AtomicU64; 64],
    latency_buckets: [AtomicU64; 64],
    callback_periods: AtomicU64,
    callback_period_total_nanos: AtomicU64,
    callback_jitter_max_nanos: AtomicU64,
    callback_jitter_buckets: [AtomicU64; 64],
    direction_callback_periods: [AtomicU64; 2],
    direction_callback_total_nanos: [AtomicU64; 2],
    direction_callback_jitter_max_nanos: [AtomicU64; 2],
    direction_callback_jitter_buckets: [[AtomicU64; 64]; 2],
    direction_max_queue_blocks: [AtomicU64; 2],
    direction_queue_overflows: [AtomicU64; 2],
    direction_stale_drops: [AtomicU64; 2],
    direction_max_endpoint_frames: [AtomicU64; 2],
    direction_max_endpoint_nanos: [AtomicU64; 2],
    direction_endpoint_capacity_frames: [AtomicU64; 2],
    direction_endpoint_capacity_nanos: [AtomicU64; 2],
    direction_endpoint_samples: [AtomicU64; 2],
    direction_endpoint_buckets: [[AtomicU64; 64]; 2],
    direction_stream_latency_nanos: [AtomicU64; 2],
    direction_stream_latency_valid: [AtomicBool; 2],
    direction_wake_to_buffer_samples: [AtomicU64; 2],
    direction_wake_to_buffer_total_nanos: [AtomicU64; 2],
    direction_wake_to_buffer_max_nanos: [AtomicU64; 2],
    direction_wake_to_buffer_buckets: [[AtomicU64; 64]; 2],
    direction_max_assembler_frames: AtomicU64,
    direction_max_drainer_frames: AtomicU64,
    direction_direct_blocks: [AtomicU64; 2],
    direction_staged_blocks: [AtomicU64; 2],
    queue_last_nanos: [AtomicU64; 2],
    queue_last_depth: [AtomicU64; 2],
    direction_queue_slope_milli_per_sec: [AtomicI64; 2],
    clock_samples: [AtomicU64; 2],
    clock_position: [AtomicU64; 2],
    clock_qpc: [AtomicU64; 2],
    clock_anchor_position: [AtomicU64; 2],
    clock_anchor_qpc: [AtomicU64; 2],
    clock_drift_ppm: [AtomicI64; 2],
    clock_drift_valid: [AtomicBool; 2],
}

impl Default for AudioLatencyStats {
    fn default() -> Self {
        // Initialize the profiling epoch before callbacks are started so the
        // first real-time queue sample never performs OnceLock initialization.
        let _ = PROFILE_EPOCH.get_or_init(Instant::now);
        Self {
            samples: AtomicU64::new(0),
            total_nanos: AtomicU64::new(0),
            last_nanos: AtomicU64::new(0),
            max_nanos: AtomicU64::new(0),
            underflow_samples: AtomicU64::new(0),
            timestamped_packets: AtomicU64::new(0),
            timestamp_errors: AtomicU64::new(0),
            discontinuities: AtomicU64::new(0),
            max_queue_blocks: AtomicU64::new(0),
            queue_overflows: AtomicU64::new(0),
            stale_drops: AtomicU64::new(0),
            max_endpoint_frames: AtomicU64::new(0),
            max_endpoint_nanos: AtomicU64::new(0),
            max_endpoint_capacity_frames: AtomicU64::new(0),
            max_endpoint_capacity_nanos: AtomicU64::new(0),
            endpoint_samples: AtomicU64::new(0),
            endpoint_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            latency_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            callback_periods: AtomicU64::new(0),
            callback_period_total_nanos: AtomicU64::new(0),
            callback_jitter_max_nanos: AtomicU64::new(0),
            callback_jitter_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
            direction_callback_periods: std::array::from_fn(|_| AtomicU64::new(0)),
            direction_callback_total_nanos: std::array::from_fn(|_| AtomicU64::new(0)),
            direction_callback_jitter_max_nanos: std::array::from_fn(|_| AtomicU64::new(0)),
            direction_callback_jitter_buckets: std::array::from_fn(|_| {
                std::array::from_fn(|_| AtomicU64::new(0))
            }),
            direction_max_queue_blocks: std::array::from_fn(|_| AtomicU64::new(0)),
            direction_queue_overflows: std::array::from_fn(|_| AtomicU64::new(0)),
            direction_stale_drops: std::array::from_fn(|_| AtomicU64::new(0)),
            direction_max_endpoint_frames: std::array::from_fn(|_| AtomicU64::new(0)),
            direction_max_endpoint_nanos: std::array::from_fn(|_| AtomicU64::new(0)),
            direction_endpoint_capacity_frames: std::array::from_fn(|_| AtomicU64::new(0)),
            direction_endpoint_capacity_nanos: std::array::from_fn(|_| AtomicU64::new(0)),
            direction_endpoint_samples: std::array::from_fn(|_| AtomicU64::new(0)),
            direction_endpoint_buckets: std::array::from_fn(|_| {
                std::array::from_fn(|_| AtomicU64::new(0))
            }),
            direction_stream_latency_nanos: std::array::from_fn(|_| AtomicU64::new(0)),
            direction_stream_latency_valid: std::array::from_fn(|_| AtomicBool::new(false)),
            direction_wake_to_buffer_samples: std::array::from_fn(|_| AtomicU64::new(0)),
            direction_wake_to_buffer_total_nanos: std::array::from_fn(|_| AtomicU64::new(0)),
            direction_wake_to_buffer_max_nanos: std::array::from_fn(|_| AtomicU64::new(0)),
            direction_wake_to_buffer_buckets: std::array::from_fn(|_| {
                std::array::from_fn(|_| AtomicU64::new(0))
            }),
            direction_max_assembler_frames: AtomicU64::new(0),
            direction_max_drainer_frames: AtomicU64::new(0),
            direction_direct_blocks: std::array::from_fn(|_| AtomicU64::new(0)),
            direction_staged_blocks: std::array::from_fn(|_| AtomicU64::new(0)),
            queue_last_nanos: std::array::from_fn(|_| AtomicU64::new(0)),
            queue_last_depth: std::array::from_fn(|_| AtomicU64::new(0)),
            direction_queue_slope_milli_per_sec: std::array::from_fn(|_| AtomicI64::new(0)),
            clock_samples: std::array::from_fn(|_| AtomicU64::new(0)),
            clock_position: std::array::from_fn(|_| AtomicU64::new(0)),
            clock_qpc: std::array::from_fn(|_| AtomicU64::new(0)),
            clock_anchor_position: std::array::from_fn(|_| AtomicU64::new(0)),
            clock_anchor_qpc: std::array::from_fn(|_| AtomicU64::new(0)),
            clock_drift_ppm: std::array::from_fn(|_| AtomicI64::new(0)),
            clock_drift_valid: std::array::from_fn(|_| AtomicBool::new(false)),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AudioLatencyDirectionSnapshot {
    pub callback_periods: u64,
    pub callback_period_mean: Option<Duration>,
    pub callback_jitter_max: Option<Duration>,
    pub callback_jitter_p50: Option<Duration>,
    pub callback_jitter_p95: Option<Duration>,
    pub callback_jitter_p99: Option<Duration>,
    pub callback_jitter_p999: Option<Duration>,
    pub max_queue_blocks: usize,
    pub queue_overflows: u64,
    pub stale_drops: u64,
    /// Queue depth slope in milli-blocks/second; positive means growth.
    pub queue_slope_milli_per_sec: i64,
    pub max_endpoint_frames: usize,
    pub max_endpoint_latency: Option<Duration>,
    /// Static endpoint buffer capacity reported by WASAPI.
    pub endpoint_capacity_frames: usize,
    pub endpoint_capacity_latency: Option<Duration>,
    pub endpoint_p50: Option<Duration>,
    pub endpoint_p95: Option<Duration>,
    pub endpoint_p99: Option<Duration>,
    pub endpoint_p999: Option<Duration>,
    pub max_assembler_frames: usize,
    pub max_drainer_frames: usize,
    pub direct_blocks: u64,
    pub staged_blocks: u64,
    pub direct_ratio_percent: Option<u32>,
    /// Static stream latency reported by `IAudioClient::GetStreamLatency`.
    pub stream_latency: Option<Duration>,
    /// Time from an audio event wake until the callback acquired its endpoint
    /// buffer. This is separate from the callback period and device padding.
    pub wake_to_buffer_samples: u64,
    pub wake_to_buffer_mean: Option<Duration>,
    pub wake_to_buffer_max: Option<Duration>,
    pub wake_to_buffer_p95: Option<Duration>,
    pub wake_to_buffer_p99: Option<Duration>,
    pub clock_samples: u64,
    pub clock_position_frames: u64,
    pub clock_qpc_100ns: u64,
    /// Estimated endpoint clock error relative to QPC, in parts per million.
    pub clock_drift_ppm: Option<i64>,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AudioLatencySnapshot {
    pub samples: u64,
    pub last: Option<Duration>,
    pub max: Option<Duration>,
    pub mean: Option<Duration>,
    pub underflow_samples: u64,
    pub timestamped_packets: u64,
    pub timestamp_errors: u64,
    pub discontinuities: u64,
    pub max_queue_blocks: usize,
    pub queue_overflows: u64,
    pub stale_drops: u64,
    pub max_endpoint_frames: usize,
    pub max_endpoint_latency: Option<Duration>,
    pub endpoint_capacity_frames: usize,
    pub endpoint_capacity_latency: Option<Duration>,
    pub endpoint_p50: Option<Duration>,
    pub endpoint_p95: Option<Duration>,
    pub endpoint_p99: Option<Duration>,
    pub endpoint_p999: Option<Duration>,
    pub p50: Option<Duration>,
    pub p95: Option<Duration>,
    pub p99: Option<Duration>,
    pub p999: Option<Duration>,
    pub callback_periods: u64,
    pub callback_period_mean: Option<Duration>,
    pub callback_jitter_max: Option<Duration>,
    pub callback_jitter_p50: Option<Duration>,
    pub callback_jitter_p95: Option<Duration>,
    pub callback_jitter_p99: Option<Duration>,
    pub callback_jitter_p999: Option<Duration>,
    pub capture: AudioLatencyDirectionSnapshot,
    pub render: AudioLatencyDirectionSnapshot,
}

impl AudioLatencyStats {
    pub fn record(&self, latency: Duration) {
        let nanos = latency.as_nanos().min(u64::MAX as u128) as u64;
        saturating_add(&self.samples, 1);
        saturating_add(&self.total_nanos, nanos);
        self.last_nanos.store(nanos, Ordering::Relaxed);
        self.latency_buckets[latency_bucket(nanos)].fetch_add(1, Ordering::Relaxed);
        let mut max = self.max_nanos.load(Ordering::Relaxed);
        while nanos > max {
            match self.max_nanos.compare_exchange_weak(
                max,
                nanos,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(current) => max = current,
            }
        }
    }

    pub fn record_underflow(&self, samples: usize) {
        saturating_add(
            &self.underflow_samples,
            samples.min(u64::MAX as usize) as u64,
        );
    }

    pub fn record_device_timestamp(&self, valid: bool) {
        if valid {
            saturating_add(&self.timestamped_packets, 1);
        } else {
            saturating_add(&self.timestamp_errors, 1);
        }
    }

    pub fn record_discontinuity(&self) {
        saturating_add(&self.discontinuities, 1);
    }

    pub fn record_queue_depth(&self, blocks: usize) {
        update_max(&self.max_queue_blocks, blocks.min(u64::MAX as usize) as u64);
    }

    pub fn record_queue_state(
        &self,
        queued_blocks: usize,
        high_water_blocks: usize,
        dropped_full: u64,
        dropped_stale: u64,
    ) {
        self.record_queue_depth(queued_blocks.max(high_water_blocks));
        update_max(&self.queue_overflows, dropped_full);
        update_max(&self.stale_drops, dropped_stale);
    }

    /// Record queue telemetry for one endpoint while retaining the aggregate
    /// counters used by the original profile API.
    pub fn record_queue_state_for(
        &self,
        direction: AudioDirection,
        queued_blocks: usize,
        high_water_blocks: usize,
        dropped_full: u64,
        dropped_stale: u64,
    ) {
        self.record_queue_state(
            queued_blocks,
            high_water_blocks,
            dropped_full,
            dropped_stale,
        );
        let index = direction.index();
        update_max(
            &self.direction_max_queue_blocks[index],
            queued_blocks.max(high_water_blocks).min(u64::MAX as usize) as u64,
        );
        update_max(&self.direction_queue_overflows[index], dropped_full);
        update_max(&self.direction_stale_drops[index], dropped_stale);

        let now = profile_now_nanos();
        let previous_time = self.queue_last_nanos[index].swap(now, Ordering::Relaxed);
        let previous_depth = self.queue_last_depth[index].swap(
            queued_blocks.min(u64::MAX as usize) as u64,
            Ordering::Relaxed,
        );
        if previous_time != 0 {
            let elapsed = now.saturating_sub(previous_time);
            // A short interval is dominated by callback jitter. A 50 ms
            // minimum gives a useful trend while still reacting within a few
            // audio periods.
            if elapsed >= 50_000_000 {
                let delta = queued_blocks as i128 - previous_depth as i128;
                let slope = delta
                    .saturating_mul(1_000_000_000_000_i128)
                    .checked_div(elapsed as i128)
                    .unwrap_or(0)
                    .clamp(i128::from(i64::MIN), i128::from(i64::MAX))
                    as i64;
                self.direction_queue_slope_milli_per_sec[index].store(slope, Ordering::Relaxed);
            }
        }
    }

    pub fn record_endpoint(&self, queued_frames: usize, sample_rate: u32) {
        update_max(
            &self.max_endpoint_frames,
            queued_frames.min(u64::MAX as usize) as u64,
        );
        if sample_rate != 0 {
            let nanos = ((queued_frames as u128) * 1_000_000_000)
                .div_ceil(u128::from(sample_rate))
                .min(u128::from(u64::MAX)) as u64;
            update_max(&self.max_endpoint_nanos, nanos);
        }
        if sample_rate != 0 {
            saturating_add(&self.endpoint_samples, 1);
            self.endpoint_buckets[latency_bucket(
                ((queued_frames as u128) * 1_000_000_000)
                    .div_ceil(u128::from(sample_rate))
                    .min(u128::from(u64::MAX)) as u64,
            )]
            .fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Record endpoint capacity without treating the static device buffer as
    /// a runtime padding sample.
    pub fn record_endpoint_capacity(&self, capacity_frames: usize, sample_rate: u32) {
        update_max(
            &self.max_endpoint_capacity_frames,
            capacity_frames.min(u64::MAX as usize) as u64,
        );
        if sample_rate != 0 {
            let nanos = ((capacity_frames as u128) * 1_000_000_000)
                .div_ceil(u128::from(sample_rate))
                .min(u128::from(u64::MAX)) as u64;
            update_max(&self.max_endpoint_capacity_nanos, nanos);
        }
    }

    pub fn record_endpoint_for(
        &self,
        direction: AudioDirection,
        queued_frames: usize,
        sample_rate: u32,
    ) {
        self.record_endpoint(queued_frames, sample_rate);
        let index = direction.index();
        update_max(
            &self.direction_max_endpoint_frames[index],
            queued_frames.min(u64::MAX as usize) as u64,
        );
        if sample_rate != 0 {
            let nanos = ((queued_frames as u128) * 1_000_000_000)
                .div_ceil(u128::from(sample_rate))
                .min(u128::from(u64::MAX)) as u64;
            update_max(&self.direction_max_endpoint_nanos[index], nanos);
        }
        if sample_rate != 0 {
            saturating_add(&self.direction_endpoint_samples[index], 1);
            self.direction_endpoint_buckets[index][latency_bucket(
                ((queued_frames as u128) * 1_000_000_000)
                    .div_ceil(u128::from(sample_rate))
                    .min(u128::from(u64::MAX)) as u64,
            )]
            .fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn record_endpoint_capacity_for(
        &self,
        direction: AudioDirection,
        capacity_frames: usize,
        sample_rate: u32,
    ) {
        self.record_endpoint_capacity(capacity_frames, sample_rate);
        let index = direction.index();
        update_max(
            &self.direction_endpoint_capacity_frames[index],
            capacity_frames.min(u64::MAX as usize) as u64,
        );
        if sample_rate != 0 {
            let nanos = ((capacity_frames as u128) * 1_000_000_000)
                .div_ceil(u128::from(sample_rate))
                .min(u128::from(u64::MAX)) as u64;
            update_max(&self.direction_endpoint_capacity_nanos[index], nanos);
        }
    }

    /// Record the static stream latency reported by WASAPI in 100 ns units.
    pub fn record_stream_latency_for(&self, direction: AudioDirection, latency_hns: i64) {
        if latency_hns <= 0 {
            return;
        }
        let nanos = u64::try_from(latency_hns)
            .ok()
            .and_then(|value| value.checked_mul(100));
        let Some(nanos) = nanos else {
            return;
        };
        let index = direction.index();
        self.direction_stream_latency_nanos[index].store(nanos, Ordering::Relaxed);
        self.direction_stream_latency_valid[index].store(true, Ordering::Release);
    }

    pub fn record_wake_to_buffer_for(&self, direction: AudioDirection, latency: Duration) {
        let index = direction.index();
        let nanos = latency.as_nanos().min(u64::MAX as u128) as u64;
        saturating_add(&self.direction_wake_to_buffer_samples[index], 1);
        saturating_add(&self.direction_wake_to_buffer_total_nanos[index], nanos);
        update_max(&self.direction_wake_to_buffer_max_nanos[index], nanos);
        self.direction_wake_to_buffer_buckets[index][latency_bucket(nanos)]
            .fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_assembler_buffered(&self, frames: usize) {
        update_max(
            &self.direction_max_assembler_frames,
            frames.min(u64::MAX as usize) as u64,
        );
    }

    pub fn record_drainer_pending(&self, frames: usize) {
        update_max(
            &self.direction_max_drainer_frames,
            frames.min(u64::MAX as usize) as u64,
        );
    }

    pub fn record_block_path(&self, direction: AudioDirection, direct: bool) {
        let index = direction.index();
        let counter = if direct {
            &self.direction_direct_blocks[index]
        } else {
            &self.direction_staged_blocks[index]
        };
        saturating_add(counter, 1);
    }

    /// Record one event-to-event interval from a device callback thread.
    /// This is opt-in with the rest of the profile and uses a separate
    /// histogram from capture-to-render age so scheduling jitter is not hidden
    /// by queue residency.
    pub fn record_callback_period(&self, actual: Duration, expected: Duration) {
        self.record_callback_period_for(AudioDirection::Capture, actual, expected);
    }

    pub fn record_callback_period_for(
        &self,
        direction: AudioDirection,
        actual: Duration,
        expected: Duration,
    ) {
        let actual_nanos = actual.as_nanos().min(u64::MAX as u128) as u64;
        let expected_nanos = expected.as_nanos().min(u64::MAX as u128) as u64;
        let jitter_nanos = actual_nanos.abs_diff(expected_nanos);
        saturating_add(&self.callback_periods, 1);
        saturating_add(&self.callback_period_total_nanos, actual_nanos);
        update_max(&self.callback_jitter_max_nanos, jitter_nanos);
        self.callback_jitter_buckets[latency_bucket(jitter_nanos)].fetch_add(1, Ordering::Relaxed);
        let index = direction.index();
        saturating_add(&self.direction_callback_periods[index], 1);
        saturating_add(&self.direction_callback_total_nanos[index], actual_nanos);
        update_max(
            &self.direction_callback_jitter_max_nanos[index],
            jitter_nanos,
        );
        self.direction_callback_jitter_buckets[index][latency_bucket(jitter_nanos)]
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Record a device clock sample. `qpc_100ns` is the QPC value returned by
    /// WASAPI in 100 ns units. The rate estimate is independent of callback
    /// wakeups and exposes long-run capture/render drift.
    pub fn record_device_clock(
        &self,
        direction: AudioDirection,
        position_frames: u64,
        qpc_100ns: u64,
        sample_rate: u32,
    ) {
        if qpc_100ns == 0 || sample_rate == 0 {
            return;
        }
        let index = direction.index();
        self.clock_position[index].store(position_frames, Ordering::Relaxed);
        self.clock_qpc[index].store(qpc_100ns, Ordering::Relaxed);
        saturating_add(&self.clock_samples[index], 1);
        let anchor_qpc = self.clock_anchor_qpc[index].load(Ordering::Relaxed);
        if anchor_qpc == 0 {
            self.clock_anchor_position[index].store(position_frames, Ordering::Relaxed);
            self.clock_anchor_qpc[index].store(qpc_100ns, Ordering::Relaxed);
            return;
        }
        let anchor_position = self.clock_anchor_position[index].load(Ordering::Relaxed);
        // Estimate over at least one second. Per-period integer frame deltas
        // are too quantized to be a useful ppm signal at a 10 ms period.
        if qpc_100ns > anchor_qpc
            && qpc_100ns - anchor_qpc >= 10_000_000
            && position_frames >= anchor_position
        {
            let frames = u128::from(position_frames - anchor_position);
            let elapsed_100ns = u128::from(qpc_100ns - anchor_qpc);
            let expected_frames = elapsed_100ns
                .saturating_mul(u128::from(sample_rate))
                .checked_div(10_000_000)
                .unwrap_or(0);
            if expected_frames > 0 {
                let ppm = (i128::try_from(frames).unwrap_or(i128::MAX)
                    - i128::try_from(expected_frames).unwrap_or(i128::MAX))
                .saturating_mul(1_000_000)
                .checked_div(i128::try_from(expected_frames).unwrap_or(i128::MAX))
                .unwrap_or(0)
                .clamp(i128::from(i64::MIN), i128::from(i64::MAX)) as i64;
                self.clock_drift_ppm[index].store(ppm, Ordering::Relaxed);
                self.clock_drift_valid[index].store(true, Ordering::Relaxed);
            }
            self.clock_anchor_position[index].store(position_frames, Ordering::Relaxed);
            self.clock_anchor_qpc[index].store(qpc_100ns, Ordering::Relaxed);
        }
    }

    pub fn snapshot(&self) -> AudioLatencySnapshot {
        let samples = self.samples.load(Ordering::Relaxed);
        let to_duration = |nanos| (samples > 0).then(|| Duration::from_nanos(nanos));
        AudioLatencySnapshot {
            samples,
            last: to_duration(self.last_nanos.load(Ordering::Relaxed)),
            max: to_duration(self.max_nanos.load(Ordering::Relaxed)),
            mean: (samples > 0).then(|| {
                Duration::from_nanos(
                    self.total_nanos
                        .load(Ordering::Relaxed)
                        .saturating_div(samples),
                )
            }),
            underflow_samples: self.underflow_samples.load(Ordering::Relaxed),
            timestamped_packets: self.timestamped_packets.load(Ordering::Relaxed),
            timestamp_errors: self.timestamp_errors.load(Ordering::Relaxed),
            discontinuities: self.discontinuities.load(Ordering::Relaxed),
            max_queue_blocks: self.max_queue_blocks.load(Ordering::Relaxed) as usize,
            queue_overflows: self.queue_overflows.load(Ordering::Relaxed),
            stale_drops: self.stale_drops.load(Ordering::Relaxed),
            max_endpoint_frames: self.max_endpoint_frames.load(Ordering::Relaxed) as usize,
            max_endpoint_latency: {
                let nanos = self.max_endpoint_nanos.load(Ordering::Relaxed);
                (nanos > 0).then(|| Duration::from_nanos(nanos))
            },
            endpoint_capacity_frames: self.max_endpoint_capacity_frames.load(Ordering::Relaxed)
                as usize,
            endpoint_capacity_latency: nonzero_duration(
                self.max_endpoint_capacity_nanos.load(Ordering::Relaxed),
            ),
            endpoint_p50: self.endpoint_percentile(50, 100),
            endpoint_p95: self.endpoint_percentile(95, 100),
            endpoint_p99: self.endpoint_percentile(99, 100),
            endpoint_p999: self.endpoint_percentile(999, 1000),
            p50: self.percentile(samples, 50, 100),
            p95: self.percentile(samples, 95, 100),
            p99: self.percentile(samples, 99, 100),
            p999: self.percentile(samples, 999, 1000),
            callback_periods: self.callback_periods.load(Ordering::Relaxed),
            callback_period_mean: {
                let periods = self.callback_periods.load(Ordering::Relaxed);
                (periods > 0).then(|| {
                    Duration::from_nanos(
                        self.callback_period_total_nanos
                            .load(Ordering::Relaxed)
                            .saturating_div(periods),
                    )
                })
            },
            callback_jitter_max: {
                let nanos = self.callback_jitter_max_nanos.load(Ordering::Relaxed);
                (nanos > 0).then(|| Duration::from_nanos(nanos))
            },
            callback_jitter_p50: self.callback_percentile(50, 100),
            callback_jitter_p95: self.callback_percentile(95, 100),
            callback_jitter_p99: self.callback_percentile(99, 100),
            callback_jitter_p999: self.callback_percentile(999, 1000),
            capture: self.direction_snapshot(AudioDirection::Capture),
            render: self.direction_snapshot(AudioDirection::Render),
        }
    }

    fn direction_snapshot(&self, direction: AudioDirection) -> AudioLatencyDirectionSnapshot {
        let index = direction.index();
        let periods = self.direction_callback_periods[index].load(Ordering::Relaxed);
        AudioLatencyDirectionSnapshot {
            callback_periods: periods,
            callback_period_mean: (periods > 0).then(|| {
                Duration::from_nanos(
                    self.direction_callback_total_nanos[index]
                        .load(Ordering::Relaxed)
                        .saturating_div(periods),
                )
            }),
            callback_jitter_max: nonzero_duration(
                self.direction_callback_jitter_max_nanos[index].load(Ordering::Relaxed),
            ),
            callback_jitter_p50: self.direction_callback_percentile(index, 50, 100),
            callback_jitter_p95: self.direction_callback_percentile(index, 95, 100),
            callback_jitter_p99: self.direction_callback_percentile(index, 99, 100),
            callback_jitter_p999: self.direction_callback_percentile(index, 999, 1000),
            max_queue_blocks: self.direction_max_queue_blocks[index].load(Ordering::Relaxed)
                as usize,
            queue_overflows: self.direction_queue_overflows[index].load(Ordering::Relaxed),
            stale_drops: self.direction_stale_drops[index].load(Ordering::Relaxed),
            queue_slope_milli_per_sec: self.direction_queue_slope_milli_per_sec[index]
                .load(Ordering::Relaxed),
            max_endpoint_frames: self.direction_max_endpoint_frames[index].load(Ordering::Relaxed)
                as usize,
            max_endpoint_latency: nonzero_duration(
                self.direction_max_endpoint_nanos[index].load(Ordering::Relaxed),
            ),
            endpoint_capacity_frames: self.direction_endpoint_capacity_frames[index]
                .load(Ordering::Relaxed) as usize,
            endpoint_capacity_latency: nonzero_duration(
                self.direction_endpoint_capacity_nanos[index].load(Ordering::Relaxed),
            ),
            endpoint_p50: self.direction_endpoint_percentile(index, 50, 100),
            endpoint_p95: self.direction_endpoint_percentile(index, 95, 100),
            endpoint_p99: self.direction_endpoint_percentile(index, 99, 100),
            endpoint_p999: self.direction_endpoint_percentile(index, 999, 1000),
            max_assembler_frames: (direction == AudioDirection::Capture)
                .then(|| self.direction_max_assembler_frames.load(Ordering::Relaxed) as usize)
                .unwrap_or(0),
            max_drainer_frames: (direction == AudioDirection::Render)
                .then(|| self.direction_max_drainer_frames.load(Ordering::Relaxed) as usize)
                .unwrap_or(0),
            direct_blocks: self.direction_direct_blocks[index].load(Ordering::Relaxed),
            staged_blocks: self.direction_staged_blocks[index].load(Ordering::Relaxed),
            direct_ratio_percent: {
                let direct = self.direction_direct_blocks[index].load(Ordering::Relaxed);
                let staged = self.direction_staged_blocks[index].load(Ordering::Relaxed);
                let total = direct.saturating_add(staged);
                (total > 0)
                    .then(|| direct.saturating_mul(100).checked_div(total).unwrap_or(0) as u32)
            },
            stream_latency: self.direction_stream_latency_valid[index]
                .load(Ordering::Acquire)
                .then(|| {
                    Duration::from_nanos(
                        self.direction_stream_latency_nanos[index].load(Ordering::Relaxed),
                    )
                }),
            wake_to_buffer_samples: self.direction_wake_to_buffer_samples[index]
                .load(Ordering::Relaxed),
            wake_to_buffer_mean: {
                let samples = self.direction_wake_to_buffer_samples[index].load(Ordering::Relaxed);
                (samples > 0).then(|| {
                    Duration::from_nanos(
                        self.direction_wake_to_buffer_total_nanos[index]
                            .load(Ordering::Relaxed)
                            .saturating_div(samples),
                    )
                })
            },
            wake_to_buffer_max: nonzero_duration(
                self.direction_wake_to_buffer_max_nanos[index].load(Ordering::Relaxed),
            ),
            wake_to_buffer_p95: percentile_buckets(
                &self.direction_wake_to_buffer_buckets[index],
                self.direction_wake_to_buffer_samples[index].load(Ordering::Relaxed),
                95,
                100,
            ),
            wake_to_buffer_p99: percentile_buckets(
                &self.direction_wake_to_buffer_buckets[index],
                self.direction_wake_to_buffer_samples[index].load(Ordering::Relaxed),
                99,
                100,
            ),
            clock_samples: self.clock_samples[index].load(Ordering::Relaxed),
            clock_position_frames: self.clock_position[index].load(Ordering::Relaxed),
            clock_qpc_100ns: self.clock_qpc[index].load(Ordering::Relaxed),
            clock_drift_ppm: self.clock_drift_valid[index]
                .load(Ordering::Relaxed)
                .then(|| self.clock_drift_ppm[index].load(Ordering::Relaxed)),
        }
    }

    fn direction_callback_percentile(
        &self,
        index: usize,
        numerator: u64,
        denominator: u64,
    ) -> Option<Duration> {
        let samples = self.direction_callback_periods[index].load(Ordering::Relaxed);
        percentile_buckets(
            &self.direction_callback_jitter_buckets[index],
            samples,
            numerator,
            denominator,
        )
    }

    fn endpoint_percentile(&self, numerator: u64, denominator: u64) -> Option<Duration> {
        let samples = self.endpoint_samples.load(Ordering::Relaxed);
        percentile_buckets(&self.endpoint_buckets, samples, numerator, denominator)
    }

    fn direction_endpoint_percentile(
        &self,
        index: usize,
        numerator: u64,
        denominator: u64,
    ) -> Option<Duration> {
        let samples = self.direction_endpoint_samples[index].load(Ordering::Relaxed);
        percentile_buckets(
            &self.direction_endpoint_buckets[index],
            samples,
            numerator,
            denominator,
        )
    }

    fn percentile(&self, samples: u64, numerator: u64, denominator: u64) -> Option<Duration> {
        if samples == 0 {
            return None;
        }
        let target = samples
            .saturating_mul(numerator)
            .saturating_add(denominator - 1)
            / denominator;
        let mut cumulative = 0_u64;
        for (index, bucket) in self.latency_buckets.iter().enumerate() {
            cumulative = cumulative.saturating_add(bucket.load(Ordering::Relaxed));
            if cumulative >= target {
                return Some(Duration::from_nanos(bucket_upper_nanos(index)));
            }
        }
        Some(Duration::from_nanos(u64::MAX))
    }

    fn callback_percentile(&self, numerator: u64, denominator: u64) -> Option<Duration> {
        let samples = self.callback_periods.load(Ordering::Relaxed);
        if samples == 0 {
            return None;
        }
        let target = samples
            .saturating_mul(numerator)
            .saturating_add(denominator - 1)
            / denominator;
        let mut cumulative = 0_u64;
        for (index, bucket) in self.callback_jitter_buckets.iter().enumerate() {
            cumulative = cumulative.saturating_add(bucket.load(Ordering::Relaxed));
            if cumulative >= target {
                return Some(Duration::from_nanos(bucket_upper_nanos(index)));
            }
        }
        Some(Duration::from_nanos(u64::MAX))
    }
}

fn nonzero_duration(nanos: u64) -> Option<Duration> {
    (nanos > 0).then(|| Duration::from_nanos(nanos))
}

fn percentile_buckets(
    buckets: &[AtomicU64; 64],
    samples: u64,
    numerator: u64,
    denominator: u64,
) -> Option<Duration> {
    if samples == 0 {
        return None;
    }
    let target = samples
        .saturating_mul(numerator)
        .saturating_add(denominator - 1)
        / denominator;
    let mut cumulative = 0_u64;
    for (index, bucket) in buckets.iter().enumerate() {
        cumulative = cumulative.saturating_add(bucket.load(Ordering::Relaxed));
        if cumulative >= target {
            return Some(Duration::from_nanos(bucket_upper_nanos(index)));
        }
    }
    Some(Duration::from_nanos(u64::MAX))
}

fn latency_bucket(nanos: u64) -> usize {
    if nanos == 0 {
        0
    } else {
        (63 - nanos.leading_zeros() as usize).min(63)
    }
}

fn bucket_upper_nanos(index: usize) -> u64 {
    if index >= 63 {
        u64::MAX
    } else {
        (1_u64 << (index + 1)).saturating_sub(1)
    }
}

fn update_max(target: &AtomicU64, value: u64) {
    let mut current = target.load(Ordering::Relaxed);
    while value > current {
        match target.compare_exchange_weak(current, value, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(next) => current = next,
        }
    }
}

fn saturating_add(target: &AtomicU64, value: u64) {
    let mut current = target.load(Ordering::Relaxed);
    loop {
        let next = current.saturating_add(value);
        match target.compare_exchange_weak(current, next, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
}

#[cfg(windows)]
fn query_qpc_ticks() -> Option<i64> {
    let mut value = 0_i64;
    let ok =
        unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut value) };
    (ok != 0).then_some(value)
}

#[cfg(windows)]
fn query_qpc_origin() -> Option<QpcOrigin> {
    let mut ticks = 0_i64;
    let mut frequency = 0_i64;
    let counter_ok =
        unsafe { windows_sys::Win32::System::Performance::QueryPerformanceCounter(&mut ticks) };
    let frequency_ok = unsafe {
        windows_sys::Win32::System::Performance::QueryPerformanceFrequency(&mut frequency)
    };
    (counter_ok != 0 && frequency_ok != 0 && frequency > 0)
        .then_some(QpcOrigin { ticks, frequency })
}
