use super::config::{AudioError, StreamConfig, StreamQueueConfig, allocate_samples};
use super::ring::{
    AudioConsumer, AudioIoStats, AudioProducer, AudioReadLease, AudioRing, AudioWriteLease,
};
use super::timing::AudioTimestamp;
use std::time::Duration;

/// The callback side and the worker side of a duplex stream.
pub struct StreamIo {
    config: StreamConfig,
    input: AudioRing,
    output: AudioRing,
}

impl StreamIo {
    pub fn new(config: StreamConfig) -> Result<Self, AudioError> {
        Self::new_with_queue_config(config, StreamQueueConfig::symmetric(config.queue_blocks))
    }

    /// Construct a duplex stream with separate capture and render ring
    /// capacities. `StreamConfig::queue_blocks` remains the symmetric default
    /// used by [`Self::new`].
    pub fn new_with_queue_config(
        config: StreamConfig,
        queues: StreamQueueConfig,
    ) -> Result<Self, AudioError> {
        let config = config.validate()?;
        let queues = queues.validate()?;
        Ok(Self {
            config,
            input: AudioRing::new(config.with_queue_blocks(queues.input_blocks))?,
            output: AudioRing::new(config.with_queue_blocks(queues.output_blocks))?,
        })
    }

    pub fn config(&self) -> StreamConfig {
        self.config
    }

    pub fn queue_config(&self) -> StreamQueueConfig {
        StreamQueueConfig::new(
            self.input.config().queue_blocks,
            self.output.config().queue_blocks,
        )
    }

    pub fn split(self) -> (DeviceIo, WorkerIo) {
        let (input_producer, input_consumer) = self.input.split();
        let (output_producer, output_consumer) = self.output.split();
        (
            DeviceIo {
                input: input_producer,
                output: output_consumer,
            },
            WorkerIo {
                input: input_consumer,
                output: output_producer,
            },
        )
    }
}

pub struct DeviceIo {
    pub input: AudioProducer,
    pub output: AudioConsumer,
}

impl DeviceIo {
    pub fn stats(&self) -> AudioIoStats {
        AudioIoStats {
            input: self.input.stats(),
            output: self.output.stats(),
        }
    }
}

pub struct WorkerIo {
    pub input: AudioConsumer,
    pub output: AudioProducer,
}

impl WorkerIo {
    pub fn stats(&self) -> AudioIoStats {
        AudioIoStats {
            input: self.input.stats(),
            output: self.output.stats(),
        }
    }
}

impl WorkerIo {
    /// Borrow the next input block without copying it out of the queue.
    pub fn try_input(&mut self) -> Result<AudioReadLease<'_>, AudioError> {
        self.input.try_acquire()
    }

    /// Block until an input block is available, then borrow it in place.
    ///
    /// The producer side only publishes an atomic generation and wakes the
    /// waiter; no callback thread takes a mutex or waits for the worker.
    pub fn wait_input(
        &mut self,
        timeout: Option<Duration>,
    ) -> Result<AudioReadLease<'_>, AudioError> {
        if !self.input.wait_for_data(timeout) {
            return Err(AudioError::QueueEmpty);
        }
        self.input.try_acquire()
    }

    /// Wait for input and discard already queued blocks before acquiring the
    /// newest one. This bounds age when the model worker falls behind for a
    /// short interval and makes the drop direction visible in ring stats.
    pub fn wait_latest_input(
        &mut self,
        timeout: Option<Duration>,
    ) -> Result<AudioReadLease<'_>, AudioError> {
        if !self.input.wait_for_data(timeout) {
            return Err(AudioError::QueueEmpty);
        }
        self.input.discard_stale();
        self.input.try_acquire()
    }

    pub fn discard_stale_input(&mut self) -> usize {
        self.input.discard_stale()
    }

    /// Reserve the next output block for the inference backend to fill.
    pub fn try_output(&mut self) -> Result<AudioWriteLease<'_>, AudioError> {
        self.output.try_acquire()
    }

    /// Acquire one input and one output slot together. The pair keeps the two
    /// SPSC queues borrowed independently, allowing an inference step to read
    /// its input while writing its output without an intermediate buffer.
    pub fn try_input_output(
        &mut self,
    ) -> Result<(AudioReadLease<'_>, AudioWriteLease<'_>), AudioError> {
        let Self { input, output } = self;
        let input = input.try_acquire()?;
        let output = output.try_acquire()?;
        Ok((input, output))
    }
}

/// Assemble arbitrary device callback sizes into fixed inference blocks. The
/// staging buffer is allocated once and is reused for the lifetime of a
/// stream.
pub struct BlockAssembler {
    config: StreamConfig,
    staging: Vec<f32>,
    filled: usize,
    dropped_blocks: u64,
    staging_timestamp: AudioTimestamp,
}

impl BlockAssembler {
    pub fn new(config: StreamConfig) -> Result<Self, AudioError> {
        let config = config.validate()?;
        Ok(Self {
            config,
            staging: allocate_samples(config.block_samples())?,
            filled: 0,
            dropped_blocks: 0,
            staging_timestamp: AudioTimestamp::ZERO,
        })
    }

    pub fn push(
        &mut self,
        input: &[f32],
        producer: &mut AudioProducer,
    ) -> Result<usize, AudioError> {
        self.push_at(input, producer, AudioTimestamp::ZERO)
    }

    pub fn push_at(
        &mut self,
        input: &[f32],
        producer: &mut AudioProducer,
        timestamp: AudioTimestamp,
    ) -> Result<usize, AudioError> {
        if !input
            .len()
            .is_multiple_of(self.config.format.channels as usize)
        {
            return Err(AudioError::PartialFrame);
        }
        let mut offset = 0;
        let mut pushed = 0;
        while offset < input.len() {
            if self.filled == 0 && input.len() - offset >= self.staging.len() {
                let end = offset + self.staging.len();
                match producer.try_push_at(&input[offset..end], timestamp) {
                    Ok(_) => {}
                    Err(AudioError::QueueFull) => {
                        self.dropped_blocks = self.dropped_blocks.saturating_add(1);
                        return Err(AudioError::QueueFull);
                    }
                    Err(error) => return Err(error),
                }
                pushed += 1;
                offset = end;
                continue;
            }
            if self.filled == 0 {
                self.staging_timestamp = timestamp;
            }
            let take = (self.staging.len() - self.filled).min(input.len() - offset);
            self.staging[self.filled..self.filled + take]
                .copy_from_slice(&input[offset..offset + take]);
            self.filled += take;
            offset += take;
            if self.filled == self.staging.len() {
                match producer.try_push_at(&self.staging, self.staging_timestamp) {
                    Ok(_) => {}
                    Err(AudioError::QueueFull) => {
                        self.dropped_blocks = self.dropped_blocks.saturating_add(1);
                        self.filled = 0;
                        self.staging_timestamp = AudioTimestamp::ZERO;
                        return Err(AudioError::QueueFull);
                    }
                    Err(error) => return Err(error),
                }
                self.filled = 0;
                self.staging_timestamp = AudioTimestamp::ZERO;
                pushed += 1;
            }
        }
        Ok(pushed)
    }

    pub fn buffered_samples(&self) -> usize {
        self.filled
    }

    pub fn dropped_blocks(&self) -> u64 {
        self.dropped_blocks
    }

    pub fn reset(&mut self) {
        self.filled = 0;
        self.staging_timestamp = AudioTimestamp::ZERO;
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DrainReport {
    pub blocks_consumed: usize,
    pub underflow_samples: usize,
    /// Timestamp of the oldest queued block contributing to this output
    /// buffer, or [`AudioTimestamp::ZERO`] when the buffer is silence.
    pub source_timestamp: AudioTimestamp,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DrainStats {
    pub blocks_consumed: u64,
    pub underflow_samples: u64,
}

/// Drain fixed blocks into arbitrary device callback sizes. Underflow is
/// rendered as silence and reported to the caller for telemetry.
pub struct BlockDrainer {
    config: StreamConfig,
    current: Vec<f32>,
    offset: usize,
    current_timestamp: AudioTimestamp,
    stats: DrainStats,
}

impl BlockDrainer {
    pub fn new(config: StreamConfig) -> Result<Self, AudioError> {
        let config = config.validate()?;
        Ok(Self {
            config,
            current: allocate_samples(config.block_samples())?,
            offset: config.block_samples(),
            current_timestamp: AudioTimestamp::ZERO,
            stats: DrainStats::default(),
        })
    }

    pub fn drain(
        &mut self,
        output: &mut [f32],
        consumer: &mut AudioConsumer,
    ) -> Result<DrainReport, AudioError> {
        if !output
            .len()
            .is_multiple_of(self.config.format.channels as usize)
        {
            return Err(AudioError::PartialFrame);
        }
        let mut report = DrainReport::default();
        if self.offset < self.current.len() {
            report.source_timestamp = self.current_timestamp;
        }
        // The common WASAPI/cpal path uses a fixed engine period equal to one
        // inference block. Read directly into the device buffer and avoid the
        // extra `current` staging copy in that case.
        if self.offset == self.current.len() && output.len() == self.current.len() {
            match consumer.try_copy_into_with_timestamp(output) {
                Ok((_, timestamp)) => {
                    report.blocks_consumed = 1;
                    report.source_timestamp = timestamp;
                    self.record(report);
                    return Ok(report);
                }
                Err(AudioError::QueueEmpty) => {
                    output.fill(0.0);
                    report.underflow_samples = output.len();
                    self.record(report);
                    return Ok(report);
                }
                Err(error) => return Err(error),
            }
        }
        let mut written = 0;
        while written < output.len() {
            if self.offset == self.current.len() {
                match consumer.try_copy_into_with_timestamp(&mut self.current) {
                    Ok((_, timestamp)) => {
                        self.offset = 0;
                        self.current_timestamp = timestamp;
                        if !report.source_timestamp.is_set() {
                            report.source_timestamp = timestamp;
                        }
                        report.blocks_consumed += 1;
                    }
                    Err(AudioError::QueueEmpty) => {
                        output[written..].fill(0.0);
                        report.underflow_samples += output.len() - written;
                        break;
                    }
                    Err(error) => return Err(error),
                }
            }
            let take = (self.current.len() - self.offset).min(output.len() - written);
            output[written..written + take]
                .copy_from_slice(&self.current[self.offset..self.offset + take]);
            self.offset += take;
            written += take;
        }
        self.record(report);
        Ok(report)
    }

    fn record(&mut self, report: DrainReport) {
        self.stats.blocks_consumed = self
            .stats
            .blocks_consumed
            .saturating_add(report.blocks_consumed as u64);
        self.stats.underflow_samples = self
            .stats
            .underflow_samples
            .saturating_add(report.underflow_samples as u64);
    }

    pub fn stats(&self) -> DrainStats {
        self.stats
    }

    /// Whether the next device block can be read directly from the ring.
    /// This is the zero-copy fast path used by the native WASAPI adapter when
    /// its engine period equals the model block size.
    pub fn can_drain_direct(&self, output_samples: usize) -> bool {
        self.offset == self.current.len() && output_samples == self.current.len()
    }

    #[cfg(all(feature = "wasapi-backend", target_os = "windows"))]
    pub(crate) fn record_direct_block(&mut self) {
        self.stats.blocks_consumed = self.stats.blocks_consumed.saturating_add(1);
    }

    pub fn reset(&mut self) {
        self.offset = self.current.len();
        self.current_timestamp = AudioTimestamp::ZERO;
    }
}

/// Device adapters can use these aliases to make the callback contract
/// explicit: callbacks operate on interleaved f32 samples.
pub type F32InputCallback<'a> = dyn FnMut(&[f32]) + Send + 'a;
pub type F32OutputCallback<'a> = dyn FnMut(&mut [f32]) + Send + 'a;
