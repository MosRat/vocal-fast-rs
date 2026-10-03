use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

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
    max_endpoint_frames: AtomicU64,
    max_endpoint_nanos: AtomicU64,
    latency_buckets: [AtomicU64; 64],
}

impl Default for AudioLatencyStats {
    fn default() -> Self {
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
            max_endpoint_frames: AtomicU64::new(0),
            max_endpoint_nanos: AtomicU64::new(0),
            latency_buckets: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }
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
    pub max_endpoint_frames: usize,
    pub max_endpoint_latency: Option<Duration>,
    pub p50: Option<Duration>,
    pub p95: Option<Duration>,
    pub p99: Option<Duration>,
    pub p999: Option<Duration>,
}

impl AudioLatencyStats {
    pub fn record(&self, latency: Duration) {
        let nanos = latency.as_nanos().min(u64::MAX as u128) as u64;
        self.samples.fetch_add(1, Ordering::Relaxed);
        self.total_nanos.fetch_add(nanos, Ordering::Relaxed);
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
        self.underflow_samples
            .fetch_add(samples.min(u64::MAX as usize) as u64, Ordering::Relaxed);
    }

    pub fn record_device_timestamp(&self, valid: bool) {
        if valid {
            self.timestamped_packets.fetch_add(1, Ordering::Relaxed);
        } else {
            self.timestamp_errors.fetch_add(1, Ordering::Relaxed);
        }
    }

    pub fn record_discontinuity(&self) {
        self.discontinuities.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_queue_depth(&self, blocks: usize) {
        update_max(&self.max_queue_blocks, blocks.min(u64::MAX as usize) as u64);
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
            max_endpoint_frames: self.max_endpoint_frames.load(Ordering::Relaxed) as usize,
            max_endpoint_latency: {
                let nanos = self.max_endpoint_nanos.load(Ordering::Relaxed);
                (nanos > 0).then(|| Duration::from_nanos(nanos))
            },
            p50: self.percentile(samples, 50, 100),
            p95: self.percentile(samples, 95, 100),
            p99: self.percentile(samples, 99, 100),
            p999: self.percentile(samples, 999, 1000),
        }
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
