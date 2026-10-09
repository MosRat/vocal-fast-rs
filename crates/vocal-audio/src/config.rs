use std::time::Duration;

use vocal_types::AudioFormat;

use super::error::AudioError;

/// Fixed-size block configuration shared by the device and worker sides.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamConfig {
    pub format: AudioFormat,
    /// Number of interleaved frames in one inference block.
    pub block_frames: usize,
    /// Number of pre-allocated blocks in each queue.
    pub queue_blocks: usize,
}

/// Independent queue capacities for the capture and render sides of a
/// duplex stream. Capture normally benefits from one extra period of slack,
/// while render should stay shallow so an underperforming worker cannot turn
/// into an audible backlog.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamQueueConfig {
    pub input_blocks: usize,
    pub output_blocks: usize,
    /// Render starts only after this many complete blocks have been published.
    /// Zero preserves immediate playback. Priming silence is not an underflow.
    pub render_prefill_blocks: usize,
}

impl StreamQueueConfig {
    pub const fn new(input_blocks: usize, output_blocks: usize) -> Self {
        Self {
            input_blocks,
            output_blocks,
            render_prefill_blocks: 0,
        }
    }

    pub const fn symmetric(blocks: usize) -> Self {
        Self::new(blocks, blocks)
    }

    pub const fn with_render_prefill(mut self, blocks: usize) -> Self {
        self.render_prefill_blocks = blocks;
        self
    }

    pub fn validate(self) -> Result<Self, AudioError> {
        if self.input_blocks == 0 || self.output_blocks == 0 {
            return Err(AudioError::InvalidConfig(
                "input_blocks and output_blocks must be non-zero".into(),
            ));
        }
        if self.render_prefill_blocks > self.output_blocks {
            return Err(AudioError::InvalidConfig(
                "render prefill exceeds output queue capacity".into(),
            ));
        }
        if self.input_blocks > isize::MAX as usize || self.output_blocks > isize::MAX as usize {
            return Err(AudioError::InvalidConfig(
                "queue capacity is too large for a bounded ring".into(),
            ));
        }
        Ok(self)
    }
}

/// Complete queue and stream configuration used to construct the IO core.
/// Keeping this pair together prevents a backend from validating the stream
/// and queue capacities in different orders.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AudioIoConfig {
    pub stream: StreamConfig,
    pub queues: StreamQueueConfig,
}

impl AudioIoConfig {
    pub const fn new(stream: StreamConfig, queues: StreamQueueConfig) -> Self {
        Self { stream, queues }
    }

    pub fn validate(self) -> Result<Self, AudioError> {
        Ok(Self {
            stream: self.stream.validate()?,
            queues: self.queues.validate()?,
        })
    }
}

impl Default for AudioIoConfig {
    fn default() -> Self {
        let stream = StreamConfig::default();
        Self::new(stream, StreamQueueConfig::default())
    }
}

/// Input policy for the standard worker scheduler.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkerInputPolicy {
    /// Preserve every block in FIFO order.
    Fifo,
    /// Drop all but the newest queued block before processing.
    Latest,
}

/// Blocking policy for one model step. Waiting is only performed by the
/// worker thread; device callbacks continue to use non-blocking methods.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkerSchedule {
    pub input_policy: WorkerInputPolicy,
    pub timeout: Option<Duration>,
}

impl Default for WorkerSchedule {
    fn default() -> Self {
        Self {
            input_policy: WorkerInputPolicy::Fifo,
            timeout: None,
        }
    }
}

impl Default for StreamQueueConfig {
    fn default() -> Self {
        Self::new(3, 2)
    }
}

/// Device slots stay small; capacities cover complete model batches and
/// render primes beyond one batch to absorb variation in inference duration.
#[derive(Clone, Copy, Debug)]
pub struct RealtimeBufferConfig {
    pub batch_frames: usize,
    pub queues: StreamQueueConfig,
}

impl RealtimeBufferConfig {
    pub fn for_batch(stream: StreamConfig, batch_frames: usize) -> Result<Self, AudioError> {
        stream.validate()?;
        if batch_frames == 0 {
            return Ok(Self {
                batch_frames: 0,
                queues: StreamQueueConfig::new(
                    stream.queue_blocks,
                    stream.queue_blocks.clamp(1, 2),
                ),
            });
        }
        let batch_frames = batch_frames.max(stream.block_frames);
        let blocks = batch_frames.div_ceil(stream.block_frames);
        let input = blocks.checked_mul(2).and_then(|v| v.checked_add(2));
        let output = blocks.checked_mul(3).and_then(|v| v.checked_add(2));
        let (Some(input), Some(output)) = (input, output) else {
            return Err(AudioError::InvalidConfig(
                "batch queue capacity overflows usize".into(),
            ));
        };
        let prefill = blocks + blocks.div_ceil(2);
        let queues = StreamQueueConfig::new(input.max(stream.queue_blocks), output)
            .with_render_prefill(prefill)
            .validate()?;
        Ok(Self {
            batch_frames,
            queues,
        })
    }
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            format: AudioFormat::MONO_48K,
            // 10 ms at 48 kHz. A real-time model can choose a larger block
            // when its receptive field requires it.
            block_frames: 480,
            // Three 10 ms blocks give the worker enough scheduling slack
            // without allowing a stalled model to accumulate 80 ms of stale
            // input or output on each side of the duplex path.
            queue_blocks: 3,
        }
    }
}

impl StreamConfig {
    /// Keep a duration budget while selecting an endpoint's actual format.
    /// Fractional frame ratios are retained by the streaming converter; this
    /// rounding only defines ring slot capacity and does not change playback speed.
    pub fn for_endpoint(self, format: AudioFormat) -> Result<Self, AudioError> {
        self.validate()?;
        format
            .validate()
            .map_err(|error| AudioError::InvalidConfig(error.to_string()))?;
        let frames = (self.block_frames as u128 * u128::from(format.sample_rate))
            .div_ceil(u128::from(self.format.sample_rate));
        let block_frames = usize::try_from(frames)
            .map_err(|_| AudioError::InvalidConfig("endpoint block size overflows usize".into()))?;
        Self {
            format,
            block_frames,
            ..self
        }
        .validate()
    }

    pub const fn with_queue_blocks(mut self, queue_blocks: usize) -> Self {
        self.queue_blocks = queue_blocks;
        self
    }

    pub fn validate(self) -> Result<Self, AudioError> {
        if self.format.sample_rate == 0
            || self.format.channels == 0
            || self.block_frames == 0
            || self.queue_blocks == 0
        {
            return Err(AudioError::InvalidConfig(
                "sample rate, channels, block_frames and queue_blocks must be non-zero".into(),
            ));
        }
        let block_samples = self
            .block_frames
            .checked_mul(self.format.channels as usize)
            .ok_or_else(|| AudioError::InvalidConfig("block size overflows usize".into()))?;
        if block_samples == 0 {
            return Err(AudioError::InvalidConfig(
                "block contains no samples".into(),
            ));
        }
        block_samples
            .checked_mul(std::mem::size_of::<f32>())
            .ok_or_else(|| AudioError::InvalidConfig("block byte size overflows usize".into()))?;
        self.queue_blocks
            .checked_mul(block_samples)
            .and_then(|samples| samples.checked_mul(std::mem::size_of::<f32>()))
            .ok_or_else(|| {
                AudioError::InvalidConfig("queue allocation size overflows usize".into())
            })?;
        if self.queue_blocks > isize::MAX as usize {
            return Err(AudioError::InvalidConfig(
                "queue_blocks is too large for a bounded ring".into(),
            ));
        }
        Ok(self)
    }

    pub const fn block_samples(self) -> usize {
        self.block_frames * self.format.channels as usize
    }

    /// Duration represented by one inference block, rounded up to a whole
    /// microsecond for telemetry and latency budgeting.
    pub fn block_duration(self) -> Duration {
        let micros = (self.block_frames as u128 * 1_000_000)
            .div_ceil(self.format.sample_rate as u128)
            .min(u64::MAX as u128) as u64;
        Duration::from_micros(micros)
    }
}

pub(crate) fn allocate_samples(samples: usize) -> Result<Vec<f32>, AudioError> {
    let mut buffer = Vec::new();
    buffer
        .try_reserve_exact(samples)
        .map_err(|error| AudioError::AllocationFailed(error.to_string()))?;
    buffer.resize(samples, 0.0);
    Ok(buffer)
}
