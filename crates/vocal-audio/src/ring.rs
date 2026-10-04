use std::cell::UnsafeCell;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::Duration;

use vocal_types::AudioChunk;

use super::config::StreamConfig;
use super::config::allocate_samples;
use super::error::AudioError;
use super::timing::AudioTimestamp;
use super::wake::AudioWake;

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

struct RingSlot {
    sequence: UnsafeCell<u64>,
    timestamp: UnsafeCell<AudioTimestamp>,
}

// A slot is only accessed by the producer before it publishes the write index
// and by the consumer after it observes that index. The acquire/release pairs
// in RingInner are the ownership hand-off for these cells.
unsafe impl Send for RingSlot {}
unsafe impl Sync for RingSlot {}

struct RingInner {
    config: StreamConfig,
    /// One allocation per ring keeps slot samples contiguous. The producer
    /// and consumer still own disjoint ranges through the SPSC cursors.
    arena: UnsafeCell<Vec<f32>>,
    slots: Box<[RingSlot]>,
    wake: AudioWake,
    space_wake: AudioWake,
    write: CachePadded<AtomicUsize>,
    read: CachePadded<AtomicUsize>,
    next_sequence: AtomicU64,
    pushed: CachePadded<AtomicU64>,
    popped: CachePadded<AtomicU64>,
    dropped_full: CachePadded<AtomicU64>,
    dropped_stale: CachePadded<AtomicU64>,
    high_water: CachePadded<AtomicUsize>,
}

unsafe impl Send for RingInner {}
unsafe impl Sync for RingInner {}

impl RingInner {
    fn new(config: StreamConfig) -> Result<Arc<Self>, AudioError> {
        let config = config.validate()?;
        let arena = allocate_samples(
            config
                .queue_blocks
                .checked_mul(config.block_samples())
                .ok_or_else(|| {
                    AudioError::InvalidConfig("ring arena size overflows usize".into())
                })?,
        )?;
        let mut slots = Vec::new();
        slots
            .try_reserve_exact(config.queue_blocks)
            .map_err(|error| AudioError::AllocationFailed(error.to_string()))?;
        for _ in 0..config.queue_blocks {
            slots.push(RingSlot {
                sequence: UnsafeCell::new(0),
                timestamp: UnsafeCell::new(AudioTimestamp::ZERO),
            });
        }
        let slots = slots.into_boxed_slice();
        Ok(Arc::new(Self {
            config,
            arena: UnsafeCell::new(arena),
            slots,
            wake: AudioWake::new(),
            space_wake: AudioWake::new(),
            write: CachePadded(AtomicUsize::new(0)),
            read: CachePadded(AtomicUsize::new(0)),
            next_sequence: AtomicU64::new(0),
            pushed: CachePadded(AtomicU64::new(0)),
            popped: CachePadded(AtomicU64::new(0)),
            dropped_full: CachePadded(AtomicU64::new(0)),
            dropped_stale: CachePadded(AtomicU64::new(0)),
            high_water: CachePadded(AtomicUsize::new(0)),
        }))
    }

    unsafe fn samples(&self, index: usize) -> &[f32] {
        let start = (index % self.slots.len()) * self.config.block_samples();
        // SAFETY: the caller has established SPSC ownership of this slot.
        unsafe {
            std::slice::from_raw_parts(
                (*self.arena.get()).as_ptr().add(start),
                self.config.block_samples(),
            )
        }
    }

    unsafe fn samples_mut(&self, index: usize) -> &mut [f32] {
        let start = (index % self.slots.len()) * self.config.block_samples();
        // SAFETY: the caller has established exclusive SPSC ownership of this
        // slot and no published cursor points at it during the mutable borrow.
        unsafe {
            std::slice::from_raw_parts_mut(
                (*self.arena.get()).as_mut_ptr().add(start),
                self.config.block_samples(),
            )
        }
    }

    fn try_write(&self, samples: &[f32], timestamp: AudioTimestamp) -> Result<u64, AudioError> {
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
        // SAFETY: only the single producer can reach this slot while the
        // write index has not been published.
        unsafe {
            self.samples_mut(write).copy_from_slice(samples);
            let slot = &self.slots[write % self.slots.len()];
            *slot.sequence.get() = sequence;
            *slot.timestamp.get() = timestamp;
        }
        self.pushed.0.fetch_add(1, Ordering::Relaxed);
        let next = write.wrapping_add(1);
        self.update_high_water(next);
        self.write.0.store(next, Ordering::Release);
        self.wake.notify();
        Ok(sequence)
    }

    fn update_high_water(&self, write: usize) {
        let read = self.read.0.load(Ordering::Acquire);
        let depth = write.wrapping_sub(read).min(self.slots.len());
        let mut current = self.high_water.0.load(Ordering::Relaxed);
        while depth > current {
            match self.high_water.0.compare_exchange_weak(
                current,
                depth,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(next) => current = next,
            }
        }
    }

    fn try_read<R>(&self, read_fn: impl FnOnce(&[f32], u64) -> R) -> Result<R, AudioError> {
        let read = self.read.0.load(Ordering::Relaxed);
        let write = self.write.0.load(Ordering::Acquire);
        if read == write {
            return Err(AudioError::QueueEmpty);
        }

        // SAFETY: the acquire load observes the producer's release store. The
        // producer cannot reuse this slot until the read index is published
        // below.
        let result = unsafe {
            let samples = self.samples(read);
            let slot = &self.slots[read % self.slots.len()];
            read_fn(samples, *slot.sequence.get())
        };
        self.popped.0.fetch_add(1, Ordering::Relaxed);
        self.read.0.store(read.wrapping_add(1), Ordering::Release);
        if write.wrapping_sub(read) >= self.slots.len() {
            self.space_wake.notify();
        }
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
            dropped_stale: self.dropped_stale.0.load(Ordering::Relaxed),
            high_water_blocks: self.high_water.0.load(Ordering::Relaxed),
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
    pub dropped_stale: u64,
    pub high_water_blocks: usize,
}

impl AudioRingStats {
    pub fn queued_frames(self, block_frames: usize) -> Option<usize> {
        self.queued_blocks.checked_mul(block_frames)
    }

    pub fn high_water_frames(self, block_frames: usize) -> Option<usize> {
        self.high_water_blocks.checked_mul(block_frames)
    }
}

/// A read-only snapshot of both queues at an endpoint boundary.
///
/// `estimated_queue_latency` covers audio currently waiting in the two rings.
/// It intentionally excludes device safety buffers and model execution time;
/// those require backend timestamps or a hardware loopback measurement.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AudioIoStats {
    pub input: AudioRingStats,
    pub output: AudioRingStats,
}

impl AudioIoStats {
    pub fn queued_frames(self, config: StreamConfig) -> Option<usize> {
        self.input
            .queued_frames(config.block_frames)?
            .checked_add(self.output.queued_frames(config.block_frames)?)
    }

    pub fn estimated_queue_latency(self, config: StreamConfig) -> Duration {
        let blocks = self
            .input
            .queued_blocks
            .saturating_add(self.output.queued_blocks);
        let micros = config
            .block_duration()
            .as_micros()
            .saturating_mul(blocks as u128)
            .min(u64::MAX as u128) as u64;
        Duration::from_micros(micros)
    }
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
        self.push_at(samples, AudioTimestamp::ZERO).map(|_| ())
    }

    pub fn push_at(
        &mut self,
        samples: &[f32],
        timestamp: AudioTimestamp,
    ) -> Result<u64, AudioError> {
        self.inner.try_write(samples, timestamp)
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
        self.try_acquire_at(AudioTimestamp::ZERO)
    }

    pub fn try_acquire_at(
        &mut self,
        timestamp: AudioTimestamp,
    ) -> Result<AudioWriteLease<'_>, AudioError> {
        let (index, sequence) = self.reserve()?;
        Ok(AudioWriteLease {
            producer: self,
            index,
            sequence,
            timestamp,
        })
    }

    pub fn try_push(&mut self, samples: &[f32]) -> Result<u64, AudioError> {
        self.try_push_at(samples, AudioTimestamp::ZERO)
    }

    pub fn try_push_at(
        &mut self,
        samples: &[f32],
        timestamp: AudioTimestamp,
    ) -> Result<u64, AudioError> {
        if samples.len() != self.config.block_samples() {
            return Err(AudioError::BlockSizeMismatch {
                expected: self.config.block_samples(),
                actual: samples.len(),
            });
        }
        let mut lease = self.try_acquire_at(timestamp)?;
        lease.samples_mut().copy_from_slice(samples);
        Ok(lease.commit())
    }

    /// Fill a queue slot in place. The callback is only invoked when a slot is
    /// available, so a device callback can avoid a temporary audio buffer.
    pub fn try_write_with(&mut self, fill: impl FnOnce(&mut [f32])) -> Result<u64, AudioError> {
        self.try_write_with_at(fill, AudioTimestamp::ZERO)
    }

    pub fn try_write_with_at(
        &mut self,
        fill: impl FnOnce(&mut [f32]),
        timestamp: AudioTimestamp,
    ) -> Result<u64, AudioError> {
        let mut lease = self.try_acquire_at(timestamp)?;
        fill(lease.samples_mut());
        Ok(lease.commit())
    }

    pub fn available_capacity(&self) -> usize {
        self.inner
            .slots
            .len()
            .saturating_sub(self.inner.stats().queued_blocks)
    }

    /// Wait until at least one unpublished slot is available. This is meant
    /// for a worker producing output; device callbacks should continue using
    /// the non-blocking acquire methods.
    pub fn wait_for_capacity(&self, timeout: Option<Duration>) -> bool {
        if self.available_capacity() > 0 {
            return true;
        }
        let observed = self.inner.space_wake.generation();
        if self.available_capacity() > 0 {
            return true;
        }
        self.inner.space_wake.wait(observed, timeout) && self.available_capacity() > 0
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
    timestamp: AudioTimestamp,
}

impl AudioWriteLease<'_> {
    pub fn samples_mut(&mut self) -> &mut [f32] {
        // SAFETY: the producer owns this unpublished slot for the lifetime of
        // the lease; no consumer can observe it until commit.
        unsafe { self.producer.inner.samples_mut(self.index) }
    }

    pub fn samples(&self) -> &[f32] {
        // SAFETY: the producer owns this unpublished slot for the lifetime of
        // the lease. This read-only view is used only to recover an unusual
        // short packet before the lease is dropped.
        unsafe { self.producer.inner.samples(self.index) }
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn timestamp(&self) -> AudioTimestamp {
        self.timestamp
    }

    pub fn set_timestamp(&mut self, timestamp: AudioTimestamp) {
        self.timestamp = timestamp;
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
            *slot.timestamp.get() = self.timestamp;
        }
        inner.pushed.0.fetch_add(1, Ordering::Relaxed);
        inner.update_high_water(next);
        inner.write.0.store(next, Ordering::Release);
        inner.wake.notify();
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

    /// Wait for input to become available without taking a mutex in the
    /// producer callback. The queue is checked before and after the observed
    /// generation to avoid a lost wake-up.
    pub fn wait_for_data(&self, timeout: Option<Duration>) -> bool {
        if !self.is_empty() {
            return true;
        }
        let observed = self.inner.wake.generation();
        if !self.is_empty() {
            return true;
        }
        self.inner.wake.wait(observed, timeout) && !self.is_empty()
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
        let timestamp = unsafe { *self.inner.slots[slot_index].timestamp.get() };
        Ok(AudioReadLease {
            consumer: self,
            index: read,
            sequence,
            timestamp,
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
        self.try_copy_into_with_timestamp(destination)
            .map(|(sequence, _)| sequence)
    }

    pub fn try_copy_into_with_timestamp(
        &mut self,
        destination: &mut [f32],
    ) -> Result<(u64, AudioTimestamp), AudioError> {
        if destination.len() != self.config.block_samples() {
            return Err(AudioError::BlockSizeMismatch {
                expected: self.config.block_samples(),
                actual: destination.len(),
            });
        }
        let lease = self.try_acquire()?;
        destination.copy_from_slice(lease.samples());
        let sequence = lease.sequence();
        let timestamp = lease.timestamp();
        drop(lease);
        Ok((sequence, timestamp))
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
        self.discard_stale();
        self.try_copy_into(destination)
    }

    /// Drop all but the newest published block. This is a consumer-side
    /// policy: it never lets the producer overwrite a slot that a worker may
    /// still be reading, while allowing a real-time worker to recover from a
    /// temporary scheduling stall without replaying stale audio.
    pub fn discard_stale(&mut self) -> usize {
        let mut discarded = 0;
        while self.inner.stats().queued_blocks > 1 {
            if self.try_acquire().is_err() {
                break;
            }
            self.inner.dropped_stale.0.fetch_add(1, Ordering::Relaxed);
            discarded += 1;
        }
        discarded
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
    timestamp: AudioTimestamp,
}

impl AudioReadLease<'_> {
    pub fn samples(&self) -> &[f32] {
        // SAFETY: the consumer owns this published slot until the lease drops.
        unsafe { self.consumer.inner.samples(self.index) }
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    pub fn timestamp(&self) -> AudioTimestamp {
        self.timestamp
    }
}

impl Drop for AudioReadLease<'_> {
    fn drop(&mut self) {
        let next = self.index.wrapping_add(1);
        self.consumer.inner.popped.0.fetch_add(1, Ordering::Relaxed);
        self.consumer.inner.read.0.store(next, Ordering::Release);
        let write = self.consumer.inner.write.0.load(Ordering::Acquire);
        if write.wrapping_sub(self.index) >= self.consumer.inner.slots.len() {
            self.consumer.inner.space_wake.notify();
        }
        self.consumer.read = next;
    }
}
