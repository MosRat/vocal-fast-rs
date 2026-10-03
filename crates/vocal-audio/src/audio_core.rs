//! Shared real-time audio core.

#[path = "config.rs"]
mod config;
#[path = "ring.rs"]
mod ring;
#[path = "stream.rs"]
mod stream;
#[path = "timing.rs"]
mod timing;

pub use config::*;
pub use ring::*;
pub use stream::*;
pub use timing::*;

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use vocal_types::AudioFormat;

    fn config() -> StreamConfig {
        StreamConfig {
            format: AudioFormat::MONO_16K,
            block_frames: 2,
            queue_blocks: 2,
        }
    }

    #[test]
    fn config_reports_block_duration_and_rejects_allocation_overflow() {
        assert_eq!(config().block_duration(), Duration::from_micros(125));
        let invalid = StreamConfig {
            block_frames: usize::MAX,
            ..config()
        };
        assert!(matches!(
            invalid.validate(),
            Err(AudioError::InvalidConfig(_))
        ));
    }

    #[test]
    fn queue_stats_expose_a_latency_budget() {
        let io = StreamIo::new(config()).unwrap();
        let (mut device, worker) = io.split();
        device.input.try_push(&[1.0, 2.0]).unwrap();

        let stats = device.stats();
        assert_eq!(stats.queued_frames(config()), Some(2));
        assert_eq!(
            stats.estimated_queue_latency(config()),
            Duration::from_micros(125)
        );
        assert_eq!(worker.stats().input.queued_blocks, 1);
    }

    #[test]
    fn opt_in_timestamps_survive_the_ring_boundary() {
        let clock = AudioClock::new();
        let timestamp = clock.now();
        let (mut producer, mut consumer) = AudioRing::new(config()).unwrap().split();
        producer.try_push_at(&[1.0, 2.0], timestamp).unwrap();

        let lease = consumer.try_acquire().unwrap();
        assert_eq!(lease.timestamp(), timestamp);
        assert!(clock.elapsed(lease.timestamp()).is_some());
    }

    #[test]
    fn latency_stats_keep_last_and_maximum_age() {
        let stats = AudioLatencyStats::default();
        assert_eq!(stats.snapshot().samples, 0);
        stats.record(Duration::from_millis(3));
        stats.record(Duration::from_millis(7));
        let snapshot = stats.snapshot();
        assert_eq!(snapshot.samples, 2);
        assert_eq!(snapshot.last, Some(Duration::from_millis(7)));
        assert_eq!(snapshot.max, Some(Duration::from_millis(7)));
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
        let clock = AudioClock::new();
        let captured_at = clock.now();
        device.input.try_push_at(&[7.0, 8.0], captured_at).unwrap();

        let (input, mut output) = worker.try_input_output().unwrap();
        output.samples_mut().copy_from_slice(input.samples());
        output.set_timestamp(input.timestamp());
        output.commit();
        drop(input);

        let output_lease = device.output.try_acquire().unwrap();
        assert_eq!(output_lease.timestamp(), captured_at);
        assert_eq!(output_lease.samples(), &[7.0, 8.0]);
    }

    #[test]
    fn drainer_reports_source_timestamp_for_profiled_audio() {
        let clock = AudioClock::new();
        let timestamp = clock.now();
        let (mut producer, mut consumer) = AudioRing::new(config()).unwrap().split();
        producer.try_push_at(&[1.0, 2.0], timestamp).unwrap();

        let mut drainer = BlockDrainer::new(config()).unwrap();
        let mut output = [0.0; 2];
        let report = drainer.drain(&mut output, &mut consumer).unwrap();
        assert_eq!(report.source_timestamp, timestamp);
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
