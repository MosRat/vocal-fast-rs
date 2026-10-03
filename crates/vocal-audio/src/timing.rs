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
        }
    }

    pub fn now(&self) -> AudioTimestamp {
        let nanos = self.origin.elapsed().as_nanos().min(u64::MAX as u128) as u64;
        AudioTimestamp(nanos.max(1))
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
#[derive(Debug, Default)]
pub struct AudioLatencyStats {
    samples: AtomicU64,
    last_nanos: AtomicU64,
    max_nanos: AtomicU64,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AudioLatencySnapshot {
    pub samples: u64,
    pub last: Option<Duration>,
    pub max: Option<Duration>,
}

impl AudioLatencyStats {
    pub fn record(&self, latency: Duration) {
        let nanos = latency.as_nanos().min(u64::MAX as u128) as u64;
        self.samples.fetch_add(1, Ordering::Relaxed);
        self.last_nanos.store(nanos, Ordering::Relaxed);
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

    pub fn snapshot(&self) -> AudioLatencySnapshot {
        let samples = self.samples.load(Ordering::Relaxed);
        let to_duration = |nanos| (samples > 0).then(|| Duration::from_nanos(nanos));
        AudioLatencySnapshot {
            samples,
            last: to_duration(self.last_nanos.load(Ordering::Relaxed)),
            max: to_duration(self.max_nanos.load(Ordering::Relaxed)),
        }
    }
}
