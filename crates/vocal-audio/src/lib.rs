//! Real-time audio I/O primitives.
//!
//! The device callback and the inference worker have different timing
//! requirements. A callback must not allocate or wait for a lock, while the
//! worker needs a block that it can keep for the duration of inference. This
//! crate therefore uses a pre-allocated single-producer/single-consumer ring.
//! The producer and consumer expose leases so the hot path can borrow a slot
//! directly; copy-based helpers are available only at format boundaries.

use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};

use thiserror::Error;
use vocal_types::{AudioChunk, AudioFormat};

#[repr(align(64))]
struct CachePadded<T>(T);

impl<T> Deref for CachePadded<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for CachePadded<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

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
    write: CachePadded<AtomicUsize>,
    read: CachePadded<AtomicUsize>,
    next_sequence: AtomicU64,
    pushed: CachePadded<AtomicU64>,
    popped: CachePadded<AtomicU64>,
    dropped_full: CachePadded<AtomicU64>,
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
            write: CachePadded(AtomicUsize::new(0)),
            read: CachePadded(AtomicUsize::new(0)),
            next_sequence: AtomicU64::new(0),
            pushed: CachePadded(AtomicU64::new(0)),
            popped: CachePadded(AtomicU64::new(0)),
            dropped_full: CachePadded(AtomicU64::new(0)),
        }))
    }

    fn try_write(&self, samples: &[f32]) -> Result<u64, AudioError> {
        if samples.len() != self.config.block_samples() {
            return Err(AudioError::BlockSizeMismatch {
                expected: self.config.block_samples(),
                actual: samples.len(),
            });
        }

        let write = self.write.0.load(Ordering::Relaxed);
        let read = self.read.0.load(Ordering::Acquire);
        if write.wrapping_sub(read) >= self.slots.len() {
            self.dropped_full.0.fetch_add(1, Ordering::Relaxed);
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
        self.pushed.0.fetch_add(1, Ordering::Relaxed);
        self.write.0.store(write.wrapping_add(1), Ordering::Release);
        Ok(sequence)
    }

    fn try_read<R>(&self, read_fn: impl FnOnce(&[f32], u64) -> R) -> Result<R, AudioError> {
        let read = self.read.0.load(Ordering::Relaxed);
        let write = self.write.0.load(Ordering::Acquire);
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
        self.popped.0.fetch_add(1, Ordering::Relaxed);
        self.read.0.store(read.wrapping_add(1), Ordering::Release);
        Ok(result)
    }

    fn stats(&self) -> AudioRingStats {
        let write = self.write.0.load(Ordering::Acquire);
        let read = self.read.0.load(Ordering::Acquire);
        AudioRingStats {
            queued_blocks: write.wrapping_sub(read).min(self.slots.len()),
            pushed: self.pushed.0.load(Ordering::Relaxed),
            popped: self.popped.0.load(Ordering::Relaxed),
            dropped_full: self.dropped_full.0.load(Ordering::Relaxed),
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
        let write = self.inner.write.0.load(Ordering::Relaxed);
        let read = self.inner.read.0.load(Ordering::Relaxed);
        let next_sequence = self.inner.next_sequence.load(Ordering::Relaxed);
        (
            AudioProducer {
                inner: Arc::clone(&self.inner),
                config,
                write,
                cached_read: read,
                next_sequence,
            },
            AudioConsumer {
                inner: self.inner,
                config,
                read,
                cached_write: write,
            },
        )
    }
}

/// The only writer handle a ring should have. All methods are non-blocking.
pub struct AudioProducer {
    inner: Arc<RingInner>,
    config: StreamConfig,
    write: usize,
    cached_read: usize,
    next_sequence: u64,
}

impl AudioProducer {
    pub fn config(&self) -> StreamConfig {
        self.config
    }

    fn reserve(&mut self) -> Result<(usize, u64), AudioError> {
        let write = self.write;
        if write.wrapping_sub(self.cached_read) >= self.inner.slots.len() {
            self.cached_read = self.inner.read.0.load(Ordering::Acquire);
            if write.wrapping_sub(self.cached_read) >= self.inner.slots.len() {
                self.inner.dropped_full.0.fetch_add(1, Ordering::Relaxed);
                return Err(AudioError::QueueFull);
            }
        }
        Ok((write, self.next_sequence))
    }

    /// Reserve one slot. The returned lease exposes the pre-allocated sample
    /// storage and publishes it only when [`AudioWriteLease::commit`] is called.
    pub fn try_acquire(&mut self) -> Result<AudioWriteLease<'_>, AudioError> {
        let (index, sequence) = self.reserve()?;
        Ok(AudioWriteLease {
            producer: self,
            index,
            sequence,
        })
    }

    pub fn try_push(&mut self, samples: &[f32]) -> Result<u64, AudioError> {
        if samples.len() != self.config.block_samples() {
            return Err(AudioError::BlockSizeMismatch {
                expected: self.config.block_samples(),
                actual: samples.len(),
            });
        }
        let mut lease = self.try_acquire()?;
        lease.samples_mut().copy_from_slice(samples);
        Ok(lease.commit())
    }

    /// Fill a queue slot in place. The callback is only invoked when a slot is
    /// available, so a device callback can avoid a temporary audio buffer.
    pub fn try_write_with(&mut self, fill: impl FnOnce(&mut [f32])) -> Result<u64, AudioError> {
        let mut lease = self.try_acquire()?;
        fill(lease.samples_mut());
        Ok(lease.commit())
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

/// A producer-side zero-copy reservation. Dropping an uncommitted lease
/// aborts the reservation and leaves the queue index unchanged.
#[must_use = "fill the slot and call commit(), or drop it to abort"]
pub struct AudioWriteLease<'a> {
    producer: &'a mut AudioProducer,
    index: usize,
    sequence: u64,
}

impl AudioWriteLease<'_> {
    pub fn samples_mut(&mut self) -> &mut [f32] {
        let slot = &self.producer.inner.slots[self.index % self.producer.inner.slots.len()];
        // SAFETY: the producer owns this unpublished slot for the lifetime of
        // the lease; no consumer can observe it until commit.
        unsafe { &mut *slot.samples.get() }
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn commit(self) -> u64 {
        let next = self.index.wrapping_add(1);
        let sequence = self.sequence;
        let inner = &self.producer.inner;
        let slot = &inner.slots[self.index % inner.slots.len()];
        // SAFETY: the lease owns the unpublished slot and the sequence is
        // written before the release publication of the write cursor.
        unsafe {
            *slot.sequence.get() = sequence;
        }
        inner.pushed.0.fetch_add(1, Ordering::Relaxed);
        inner.write.0.store(next, Ordering::Release);
        self.producer.write = next;
        self.producer.next_sequence = sequence.wrapping_add(1);
        sequence
    }
}

impl Drop for AudioWriteLease<'_> {
    fn drop(&mut self) {
        // An aborted reservation is intentionally not published. The next
        // producer reservation reuses and fully overwrites this slot.
    }
}

/// The only reader handle a ring should have. All methods are non-blocking.
pub struct AudioConsumer {
    inner: Arc<RingInner>,
    config: StreamConfig,
    read: usize,
    cached_write: usize,
}

impl AudioConsumer {
    pub fn config(&self) -> StreamConfig {
        self.config
    }

    pub fn try_acquire(&mut self) -> Result<AudioReadLease<'_>, AudioError> {
        let read = self.read;
        if read == self.cached_write {
            self.cached_write = self.inner.write.0.load(Ordering::Acquire);
            if read == self.cached_write {
                return Err(AudioError::QueueEmpty);
            }
        }
        let slot_index = read % self.inner.slots.len();
        // SAFETY: the acquire load above makes the published slot visible.
        let sequence = unsafe { *self.inner.slots[slot_index].sequence.get() };
        Ok(AudioReadLease {
            consumer: self,
            index: read,
            sequence,
        })
    }

    pub fn try_pop(&mut self) -> Result<AudioChunk, AudioError> {
        let format = self.config.format;
        let lease = self.try_acquire()?;
        let chunk = AudioChunk {
            format,
            samples: lease.samples().to_vec(),
            sequence: lease.sequence(),
        };
        drop(lease);
        Ok(chunk)
    }

    pub fn try_copy_into(&mut self, destination: &mut [f32]) -> Result<u64, AudioError> {
        if destination.len() != self.config.block_samples() {
            return Err(AudioError::BlockSizeMismatch {
                expected: self.config.block_samples(),
                actual: destination.len(),
            });
        }
        let lease = self.try_acquire()?;
        destination.copy_from_slice(lease.samples());
        let sequence = lease.sequence();
        drop(lease);
        Ok(sequence)
    }

    /// Consume all but the newest queued block and copy that block into the
    /// destination. This is useful for a monitor path where old audio is worse
    /// than a single discontinuity.
    pub fn try_copy_latest_into(&mut self, destination: &mut [f32]) -> Result<u64, AudioError> {
        if destination.len() != self.config.block_samples() {
            return Err(AudioError::BlockSizeMismatch {
                expected: self.config.block_samples(),
                actual: destination.len(),
            });
        }
        while self.inner.stats().queued_blocks > 1 {
            drop(self.try_acquire()?);
        }
        self.try_copy_into(destination)
    }

    pub fn try_read<R>(&mut self, read: impl FnOnce(&[f32], u64) -> R) -> Result<R, AudioError> {
        let lease = self.try_acquire()?;
        let result = read(lease.samples(), lease.sequence());
        drop(lease);
        Ok(result)
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

/// A consumer-side zero-copy view. The slot is returned to the producer when
/// this value is dropped.
#[must_use = "keep the lease alive while reading the slot"]
pub struct AudioReadLease<'a> {
    consumer: &'a mut AudioConsumer,
    index: usize,
    sequence: u64,
}

impl AudioReadLease<'_> {
    pub fn samples(&self) -> &[f32] {
        let slot = &self.consumer.inner.slots[self.index % self.consumer.inner.slots.len()];
        // SAFETY: the consumer owns this published slot until the lease drops.
        unsafe { &*slot.samples.get() }
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }
}

impl Drop for AudioReadLease<'_> {
    fn drop(&mut self) {
        let next = self.index.wrapping_add(1);
        self.consumer.inner.popped.0.fetch_add(1, Ordering::Relaxed);
        self.consumer.inner.read.0.store(next, Ordering::Release);
        self.consumer.read = next;
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

impl WorkerIo {
    /// Borrow the next input block without copying it out of the queue.
    pub fn try_input(&mut self) -> Result<AudioReadLease<'_>, AudioError> {
        self.input.try_acquire()
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

    pub fn push(
        &mut self,
        input: &[f32],
        producer: &mut AudioProducer,
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
        consumer: &mut AudioConsumer,
    ) -> Result<DrainReport, AudioError> {
        if !output
            .len()
            .is_multiple_of(self.config.format.channels as usize)
        {
            return Err(AudioError::PartialFrame);
        }
        let mut report = DrainReport::default();
        // The common WASAPI/cpal path uses a fixed engine period equal to one
        // inference block. Read directly into the device buffer and avoid the
        // extra `current` staging copy in that case.
        if self.offset == self.current.len() && output.len() == self.current.len() {
            match consumer.try_copy_into(output) {
                Ok(_) => {
                    report.blocks_consumed = 1;
                    return Ok(report);
                }
                Err(AudioError::QueueEmpty) => {
                    output.fill(0.0);
                    report.underflow_samples = output.len();
                    return Ok(report);
                }
                Err(error) => return Err(error),
            }
        }
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

#[cfg(all(feature = "wasapi-backend", target_os = "windows"))]
pub mod wasapi_backend;

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
        let (mut producer, mut consumer) = AudioRing::new(config()).unwrap().split();
        producer
            .try_write_with(|slot| slot.copy_from_slice(&[1.0, 2.0]))
            .unwrap();
        let mut block = [0.0; 2];
        let sequence = consumer.try_copy_into(&mut block).unwrap();
        assert_eq!(sequence, 0);
        assert_eq!(block, [1.0, 2.0]);
    }

    #[test]
    fn leases_expose_slots_without_queue_copies() {
        let (mut producer, mut consumer) = AudioRing::new(config()).unwrap().split();
        let mut write = producer.try_acquire().unwrap();
        write.samples_mut().copy_from_slice(&[3.0, 4.0]);
        assert_eq!(write.commit(), 0);

        let read = consumer.try_acquire().unwrap();
        assert_eq!(read.sequence(), 0);
        assert_eq!(read.samples(), &[3.0, 4.0]);
        drop(read);
        assert_eq!(producer.available_capacity(), config().queue_blocks);
    }

    #[test]
    fn dropped_write_lease_does_not_publish_partial_audio() {
        let (mut producer, mut consumer) = AudioRing::new(config()).unwrap().split();
        {
            let mut lease = producer.try_acquire().unwrap();
            lease.samples_mut().fill(99.0);
        }
        assert!(consumer.try_acquire().is_err());

        producer.try_push(&[5.0, 6.0]).unwrap();
        assert_eq!(consumer.try_pop().unwrap().samples, vec![5.0, 6.0]);
    }

    #[test]
    fn worker_can_hold_input_and_output_leases_together() {
        let io = StreamIo::new(config()).unwrap();
        let (mut device, mut worker) = io.split();
        device.input.try_push(&[7.0, 8.0]).unwrap();

        let (input, mut output) = worker.try_input_output().unwrap();
        output.samples_mut().copy_from_slice(input.samples());
        output.commit();
        drop(input);

        assert_eq!(device.output.try_pop().unwrap().samples, vec![7.0, 8.0]);
    }

    #[test]
    fn assembler_and_drainer_handle_non_matching_callback_sizes() {
        let (mut producer, mut consumer) = AudioRing::new(config()).unwrap().split();
        let mut assembler = BlockAssembler::new(config()).unwrap();
        assert_eq!(assembler.push(&[1.0], &mut producer).unwrap(), 0);
        assert_eq!(assembler.push(&[2.0, 3.0, 4.0], &mut producer).unwrap(), 2);

        let mut drainer = BlockDrainer::new(config()).unwrap();
        let mut output = [0.0; 3];
        let report = drainer.drain(&mut output, &mut consumer).unwrap();
        assert_eq!(output, [1.0, 2.0, 3.0]);
        assert_eq!(report.blocks_consumed, 2);
        assert_eq!(report.underflow_samples, 0);
    }

    #[test]
    fn latest_copy_drops_stale_blocks() {
        let (mut producer, mut consumer) = AudioRing::new(config()).unwrap().split();
        producer.try_push(&[1.0, 1.0]).unwrap();
        producer.try_push(&[2.0, 2.0]).unwrap();
        let mut latest = [0.0; 2];
        consumer.try_copy_latest_into(&mut latest).unwrap();
        assert_eq!(latest, [2.0, 2.0]);
        assert!(consumer.is_empty());
    }
}
