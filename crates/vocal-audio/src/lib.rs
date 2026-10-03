//! Real-time audio I/O primitives.
//!
//! The device callback and the inference worker have different timing
//! requirements. A callback must not allocate or wait for a lock, while the
//! worker needs an owned block that it can keep for the duration of inference.
//! This crate therefore uses a pre-allocated single-producer/single-consumer
//! ring. The producer writes directly into a slot and the consumer can either
//! borrow a slot for the duration of a callback or copy it into a worker-owned
//! buffer.

use std::cell::UnsafeCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use thiserror::Error;
use vocal_types::{AudioChunk, AudioFormat};

/// Fixed-size block configuration shared by the device and worker sides.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct StreamConfig {
    pub format: AudioFormat,
    /// Number of interleaved frames in one inference block.
    pub block_frames: usize,
    /// Number of pre-allocated blocks in each queue.
    pub queue_blocks: usize,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            format: AudioFormat::MONO_48K,
            // 10 ms at 48 kHz. A real-time model can choose a larger block
            // when its receptive field requires it.
            block_frames: 480,
            queue_blocks: 8,
        }
    }
}

impl StreamConfig {
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
}

#[derive(Debug, Error, Clone, PartialEq, Eq)]
pub enum AudioError {
    #[error("audio queue is full")]
    QueueFull,
    #[error("audio queue is empty")]
    QueueEmpty,
    #[error("invalid audio configuration: {0}")]
    InvalidConfig(String),
    #[error("audio block has the wrong number of samples: expected {expected}, got {actual}")]
    BlockSizeMismatch { expected: usize, actual: usize },
    #[error("audio callback data is not a whole number of frames")]
    PartialFrame,
}

struct RingSlot {
    samples: UnsafeCell<Vec<f32>>,
    sequence: UnsafeCell<u64>,
}

// A slot is only accessed by the producer before it publishes the write index
// and by the consumer after it observes that index. The acquire/release pairs
// in RingInner are the ownership hand-off for these cells.
unsafe impl Send for RingSlot {}
unsafe impl Sync for RingSlot {}

struct RingInner {
    config: StreamConfig,
    slots: Box<[RingSlot]>,
    write: AtomicUsize,
    read: AtomicUsize,
    next_sequence: AtomicU64,
    pushed: AtomicU64,
    popped: AtomicU64,
    dropped_full: AtomicU64,
}

unsafe impl Send for RingInner {}
unsafe impl Sync for RingInner {}

impl RingInner {
    fn new(config: StreamConfig) -> Result<Arc<Self>, AudioError> {
        let config = config.validate()?;
        let slots = (0..config.queue_blocks)
            .map(|_| RingSlot {
                samples: UnsafeCell::new(vec![0.0; config.block_samples()]),
                sequence: UnsafeCell::new(0),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();
        Ok(Arc::new(Self {
            config,
            slots,
            write: AtomicUsize::new(0),
            read: AtomicUsize::new(0),
            next_sequence: AtomicU64::new(0),
            pushed: AtomicU64::new(0),
            popped: AtomicU64::new(0),
            dropped_full: AtomicU64::new(0),
        }))
    }

    fn try_write(&self, samples: &[f32]) -> Result<u64, AudioError> {
        if samples.len() != self.config.block_samples() {
            return Err(AudioError::BlockSizeMismatch {
                expected: self.config.block_samples(),
                actual: samples.len(),
            });
        }

        let write = self.write.load(Ordering::Relaxed);
        let read = self.read.load(Ordering::Acquire);
        if write.wrapping_sub(read) >= self.slots.len() {
            self.dropped_full.fetch_add(1, Ordering::Relaxed);
            return Err(AudioError::QueueFull);
        }

        let sequence = self.next_sequence.fetch_add(1, Ordering::Relaxed);
        let slot = &self.slots[write % self.slots.len()];
        // SAFETY: only the single producer can reach this slot while the
        // write index has not been published.
        unsafe {
            (*slot.samples.get()).copy_from_slice(samples);
            *slot.sequence.get() = sequence;
        }
        self.pushed.fetch_add(1, Ordering::Relaxed);
        self.write.store(write.wrapping_add(1), Ordering::Release);
        Ok(sequence)
    }

    fn try_read<R>(&self, read_fn: impl FnOnce(&[f32], u64) -> R) -> Result<R, AudioError> {
        let read = self.read.load(Ordering::Relaxed);
        let write = self.write.load(Ordering::Acquire);
        if read == write {
            return Err(AudioError::QueueEmpty);
        }

        let slot = &self.slots[read % self.slots.len()];
        // SAFETY: the acquire load observes the producer's release store. The
        // producer cannot reuse this slot until the read index is published
        // below.
        let result = unsafe {
            let samples = &*slot.samples.get();
            read_fn(samples, *slot.sequence.get())
        };
        self.popped.fetch_add(1, Ordering::Relaxed);
        self.read.store(read.wrapping_add(1), Ordering::Release);
        Ok(result)
    }

    fn discard_one(&self) -> Result<(), AudioError> {
        self.try_read(|_, _| ())
    }

    fn stats(&self) -> AudioRingStats {
        let write = self.write.load(Ordering::Acquire);
        let read = self.read.load(Ordering::Acquire);
        AudioRingStats {
            queued_blocks: write.wrapping_sub(read).min(self.slots.len()),
            pushed: self.pushed.load(Ordering::Relaxed),
            popped: self.popped.load(Ordering::Relaxed),
            dropped_full: self.dropped_full.load(Ordering::Relaxed),
        }
    }
}

/// Snapshot of queue activity. Counters are monotonic for the lifetime of a
/// ring; `queued_blocks` is the current approximate depth.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AudioRingStats {
    pub queued_blocks: usize,
    pub pushed: u64,
    pub popped: u64,
    pub dropped_full: u64,
}

/// A bounded, pre-allocated SPSC queue.
///
/// `AudioRing` keeps the original convenient push/pop API for deterministic
/// tests and file-based callers. A device integration should consume it with
/// [`AudioRing::split`] and retain exactly one producer and one consumer.
pub struct AudioRing {
    inner: Arc<RingInner>,
}

impl AudioRing {
    pub fn new(config: StreamConfig) -> Result<Self, AudioError> {
        Ok(Self {
            inner: RingInner::new(config)?,
        })
    }

    pub fn config(&self) -> StreamConfig {
        self.inner.config
    }

    pub fn push(&mut self, samples: &[f32]) -> Result<(), AudioError> {
        self.inner.try_write(samples).map(|_| ())
    }

    /// Copies an owned chunk into a pre-allocated slot. The copy happens on
    /// the producer side; no `Vec` is allocated by the queue.
    pub fn push_owned(&mut self, chunk: AudioChunk) -> Result<(), AudioError> {
        if chunk.format != self.config().format {
            return Err(AudioError::InvalidConfig(
                "audio chunk format does not match stream configuration".into(),
            ));
        }
        self.push(&chunk.samples)
    }

    pub fn pop(&mut self) -> Result<AudioChunk, AudioError> {
        let format = self.config().format;
        self.inner.try_read(|samples, sequence| AudioChunk {
            format,
            samples: samples.to_vec(),
            sequence,
        })
    }

    pub fn len(&self) -> usize {
        self.inner.stats().queued_blocks
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn stats(&self) -> AudioRingStats {
        self.inner.stats()
    }

    /// Split ownership for a device callback and a worker thread.
    pub fn split(self) -> (AudioProducer, AudioConsumer) {
        let config = self.inner.config;
        (
            AudioProducer {
                inner: Arc::clone(&self.inner),
                config,
            },
            AudioConsumer {
                inner: self.inner,
                config,
            },
        )
    }
}

/// The only writer handle a ring should have. All methods are non-blocking.
pub struct AudioProducer {
    inner: Arc<RingInner>,
    config: StreamConfig,
}

impl AudioProducer {
    pub fn config(&self) -> StreamConfig {
        self.config
    }

    pub fn try_push(&self, samples: &[f32]) -> Result<u64, AudioError> {
        self.inner.try_write(samples)
    }

    /// Fill a queue slot in place. The callback is only invoked when a slot is
    /// available, so a device callback can avoid a temporary audio buffer.
    pub fn try_write_with(&self, fill: impl FnOnce(&mut [f32])) -> Result<u64, AudioError> {
        let write = self.inner.write.load(Ordering::Relaxed);
        let read = self.inner.read.load(Ordering::Acquire);
        if write.wrapping_sub(read) >= self.inner.slots.len() {
            self.inner.dropped_full.fetch_add(1, Ordering::Relaxed);
            return Err(AudioError::QueueFull);
        }
        let sequence = self.inner.next_sequence.fetch_add(1, Ordering::Relaxed);
        let slot = &self.inner.slots[write % self.inner.slots.len()];
        // SAFETY: this producer owns the unpublished slot.
        unsafe {
            fill(&mut *slot.samples.get());
            *slot.sequence.get() = sequence;
        }
        self.inner.pushed.fetch_add(1, Ordering::Relaxed);
        self.inner
            .write
            .store(write.wrapping_add(1), Ordering::Release);
        Ok(sequence)
    }

    pub fn available_capacity(&self) -> usize {
        self.inner
            .slots
            .len()
            .saturating_sub(self.inner.stats().queued_blocks)
    }

    pub fn stats(&self) -> AudioRingStats {
        self.inner.stats()
    }
}

/// The only reader handle a ring should have. All methods are non-blocking.
pub struct AudioConsumer {
    inner: Arc<RingInner>,
    config: StreamConfig,
}

impl AudioConsumer {
    pub fn config(&self) -> StreamConfig {
        self.config
    }

    pub fn try_pop(&self) -> Result<AudioChunk, AudioError> {
        let format = self.config.format;
        self.inner.try_read(|samples, sequence| AudioChunk {
            format,
            samples: samples.to_vec(),
            sequence,
        })
    }

    pub fn try_copy_into(&self, destination: &mut [f32]) -> Result<u64, AudioError> {
        if destination.len() != self.config.block_samples() {
            return Err(AudioError::BlockSizeMismatch {
                expected: self.config.block_samples(),
                actual: destination.len(),
            });
        }
        self.inner.try_read(|samples, sequence| {
            destination.copy_from_slice(samples);
            sequence
        })
    }

    /// Consume all but the newest queued block and copy that block into the
    /// destination. This is useful for a monitor path where old audio is worse
    /// than a single discontinuity.
    pub fn try_copy_latest_into(&self, destination: &mut [f32]) -> Result<u64, AudioError> {
        if destination.len() != self.config.block_samples() {
            return Err(AudioError::BlockSizeMismatch {
                expected: self.config.block_samples(),
                actual: destination.len(),
            });
        }
        while self.inner.stats().queued_blocks > 1 {
            self.inner.discard_one()?;
        }
        self.try_copy_into(destination)
    }

    pub fn try_read<R>(&self, read: impl FnOnce(&[f32], u64) -> R) -> Result<R, AudioError> {
        self.inner.try_read(read)
    }

    pub fn len(&self) -> usize {
        self.inner.stats().queued_blocks
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn stats(&self) -> AudioRingStats {
        self.inner.stats()
    }
}

/// The callback side and the worker side of a duplex stream.
pub struct StreamIo {
    config: StreamConfig,
    input: AudioRing,
    output: AudioRing,
}

impl StreamIo {
    pub fn new(config: StreamConfig) -> Result<Self, AudioError> {
        Ok(Self {
            config,
            input: AudioRing::new(config)?,
            output: AudioRing::new(config)?,
        })
    }

    pub fn config(&self) -> StreamConfig {
        self.config
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

pub struct WorkerIo {
    pub input: AudioConsumer,
    pub output: AudioProducer,
}

/// Assemble arbitrary device callback sizes into fixed inference blocks. The
/// staging buffer is allocated once and is reused for the lifetime of a
/// stream.
pub struct BlockAssembler {
    config: StreamConfig,
    staging: Vec<f32>,
    filled: usize,
    dropped_blocks: u64,
}

impl BlockAssembler {
    pub fn new(config: StreamConfig) -> Result<Self, AudioError> {
        let config = config.validate()?;
        Ok(Self {
            config,
            staging: vec![0.0; config.block_samples()],
            filled: 0,
            dropped_blocks: 0,
        })
    }

    pub fn push(&mut self, input: &[f32], producer: &AudioProducer) -> Result<usize, AudioError> {
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
                producer.try_push(&input[offset..end])?;
                pushed += 1;
                offset = end;
                continue;
            }
            let take = (self.staging.len() - self.filled).min(input.len() - offset);
            self.staging[self.filled..self.filled + take]
                .copy_from_slice(&input[offset..offset + take]);
            self.filled += take;
            offset += take;
            if self.filled == self.staging.len() {
                if producer.try_push(&self.staging).is_err() {
                    self.dropped_blocks = self.dropped_blocks.saturating_add(1);
                    self.filled = 0;
                    return Err(AudioError::QueueFull);
                }
                self.filled = 0;
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
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct DrainReport {
    pub blocks_consumed: usize,
    pub underflow_samples: usize,
}

/// Drain fixed blocks into arbitrary device callback sizes. Underflow is
/// rendered as silence and reported to the caller for telemetry.
pub struct BlockDrainer {
    config: StreamConfig,
    current: Vec<f32>,
    offset: usize,
}

impl BlockDrainer {
    pub fn new(config: StreamConfig) -> Result<Self, AudioError> {
        let config = config.validate()?;
        Ok(Self {
            config,
            current: vec![0.0; config.block_samples()],
            offset: config.block_samples(),
        })
    }

    pub fn drain(
        &mut self,
        output: &mut [f32],
        consumer: &AudioConsumer,
    ) -> Result<DrainReport, AudioError> {
        if !output
            .len()
            .is_multiple_of(self.config.format.channels as usize)
        {
            return Err(AudioError::PartialFrame);
        }
        let mut report = DrainReport::default();
        let mut written = 0;
        while written < output.len() {
            if self.offset == self.current.len() {
                match consumer.try_copy_into(&mut self.current) {
                    Ok(_) => {
                        self.offset = 0;
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
        Ok(report)
    }

    pub fn reset(&mut self) {
        self.offset = self.current.len();
    }
}

/// Device adapters can use these aliases to make the callback contract
/// explicit: callbacks operate on interleaved f32 samples.
pub type F32InputCallback<'a> = dyn FnMut(&[f32]) + Send + 'a;
pub type F32OutputCallback<'a> = dyn FnMut(&mut [f32]) + Send + 'a;

#[cfg(feature = "cpal-backend")]
pub mod cpal_backend;

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> StreamConfig {
        StreamConfig {
            format: AudioFormat::MONO_16K,
            block_frames: 2,
            queue_blocks: 2,
        }
    }

    #[test]
    fn ring_preserves_sequence_and_bounds() {
        let mut ring = AudioRing::new(StreamConfig {
            queue_blocks: 1,
            ..config()
        })
        .unwrap();
        ring.push(&[0.0, 0.1]).unwrap();
        assert_eq!(
            ring.push(&[0.2, 0.3]).unwrap_err().to_string(),
            "audio queue is full"
        );
        assert_eq!(ring.pop().unwrap().sequence, 0);
        assert!(ring.is_empty());
    }

    #[test]
    fn split_ring_writes_without_allocating_queue_slots() {
        let (producer, consumer) = AudioRing::new(config()).unwrap().split();
        producer
            .try_write_with(|slot| slot.copy_from_slice(&[1.0, 2.0]))
            .unwrap();
        let mut block = [0.0; 2];
        let sequence = consumer.try_copy_into(&mut block).unwrap();
        assert_eq!(sequence, 0);
        assert_eq!(block, [1.0, 2.0]);
    }

    #[test]
    fn assembler_and_drainer_handle_non_matching_callback_sizes() {
        let (producer, consumer) = AudioRing::new(config()).unwrap().split();
        let mut assembler = BlockAssembler::new(config()).unwrap();
        assert_eq!(assembler.push(&[1.0], &producer).unwrap(), 0);
        assert_eq!(assembler.push(&[2.0, 3.0, 4.0], &producer).unwrap(), 2);

        let mut drainer = BlockDrainer::new(config()).unwrap();
        let mut output = [0.0; 3];
        let report = drainer.drain(&mut output, &consumer).unwrap();
        assert_eq!(output, [1.0, 2.0, 3.0]);
        assert_eq!(report.blocks_consumed, 2);
        assert_eq!(report.underflow_samples, 0);
    }

    #[test]
    fn latest_copy_drops_stale_blocks() {
        let (producer, consumer) = AudioRing::new(config()).unwrap().split();
        producer.try_push(&[1.0, 1.0]).unwrap();
        producer.try_push(&[2.0, 2.0]).unwrap();
        let mut latest = [0.0; 2];
        consumer.try_copy_latest_into(&mut latest).unwrap();
        assert_eq!(latest, [2.0, 2.0]);
        assert!(consumer.is_empty());
    }
}
