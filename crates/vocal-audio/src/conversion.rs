//! Worker-owned, bounded PCM conversion. Endpoint callbacks only exchange
//! native-format blocks through SPSC rings; filter state never crosses threads.

use super::{
    AudioError, AudioTimestamp, StreamConfig, WorkerInputPolicy, WorkerIo, WorkerSchedule,
    allocate_samples,
};
use std::f64::consts::PI;
use vocal_types::AudioFormat;

const TAPS: usize = 64;
const PHASES: usize = 1024;

/// Stateful polyphase Blackman-windowed sinc resampler for interleaved f32.
/// A causal FIR delay avoids depending on future callbacks. Filter length
/// scales with decimation so high source rates retain an anti-aliasing stopband. The
/// fractional cursor and filter history survive arbitrary block boundaries.
/// Equal-rate conversion bypasses the filter and has no added delay.
/// Equal channel counts preserve channel order. Other layouts use a normalized
/// mono mix replicated into each destination channel (voice routing, not spatial
/// surround remixing; no speaker positions are inferred from a channel count).
pub struct StreamingFormatConverter {
    input: AudioFormat,
    output: AudioFormat,
    max_input_frames: usize,
    history: Vec<f32>,
    kernel: Vec<f32>,
    taps: usize,
    position_numerator: u64,
}

impl StreamingFormatConverter {
    pub fn new(
        input: AudioFormat,
        output: AudioFormat,
        max_input_frames: usize,
    ) -> Result<Self, AudioError> {
        for format in [input, output] {
            format
                .validate()
                .map_err(|e| AudioError::InvalidConfig(e.to_string()))?;
            if !(8_000..=384_000).contains(&format.sample_rate) || format.channels > 64 {
                return Err(AudioError::InvalidConfig(
                    "conversion supports 8–384 kHz and 1–64 channels".into(),
                ));
            }
        }
        if max_input_frames == 0 || max_input_frames > 1_048_576 {
            return Err(AudioError::InvalidConfig(
                "invalid converter block capacity".into(),
            ));
        }
        let channels = usize::from(output.channels);
        let taps = TAPS * input.sample_rate.div_ceil(output.sample_rate).max(1) as usize;
        let mut history = allocate_samples((max_input_frames + taps) * channels)?;
        history.truncate(taps * channels);
        let mut kernel = Vec::new();
        if input.sample_rate != output.sample_rate {
            kernel = allocate_samples(PHASES * taps)?;
            let cutoff =
                (f64::from(output.sample_rate) / f64::from(input.sample_rate)).min(1.0) * 0.94;
            for phase in 0..PHASES {
                let fraction = phase as f64 / PHASES as f64;
                let row = &mut kernel[phase * taps..(phase + 1) * taps];
                let mut sum = 0.0;
                for (tap, value) in row.iter_mut().enumerate() {
                    let x = tap as f64 - (taps / 2 - 1) as f64 - fraction;
                    let sinc = if x.abs() < 1e-12 {
                        cutoff
                    } else {
                        (PI * cutoff * x).sin() / (PI * x)
                    };
                    let window = 0.42
                        + 0.5 * (PI * x / (taps / 2) as f64).cos()
                        + 0.08 * (2.0 * PI * x / (taps / 2) as f64).cos();
                    *value = (sinc * window) as f32;
                    sum += f64::from(*value);
                }
                for value in row {
                    *value /= sum as f32;
                }
            }
        }
        Ok(Self {
            input,
            output,
            max_input_frames,
            history,
            kernel,
            taps,
            position_numerator: taps as u64 * u64::from(output.sample_rate),
        })
    }

    pub fn output_capacity_samples(&self) -> usize {
        ((self.max_input_frames as u64 * u64::from(self.output.sample_rate))
            .div_ceil(u64::from(self.input.sample_rate)) as usize
            + 2)
            * usize::from(self.output.channels)
    }

    pub fn delay_frames(&self) -> usize {
        if self.kernel.is_empty() {
            0
        } else {
            self.taps / 2
        }
    }

    pub fn reset(&mut self) {
        self.history
            .resize(self.taps * usize::from(self.output.channels), 0.0);
        self.history.fill(0.0);
        self.position_numerator = self.taps as u64 * u64::from(self.output.sample_rate);
    }

    /// Reuses caller storage. Capacity must be prepared before starting a worker.
    pub fn process(&mut self, samples: &[f32], output: &mut Vec<f32>) -> Result<(), AudioError> {
        let input_channels = usize::from(self.input.channels);
        let channels = usize::from(self.output.channels);
        if !samples.len().is_multiple_of(input_channels) {
            return Err(AudioError::PartialFrame);
        }
        if samples.len() / input_channels > self.max_input_frames {
            return Err(AudioError::InvalidConfig(
                "converter input exceeds its prepared capacity".into(),
            ));
        }
        if output.capacity() < self.output_capacity_samples() {
            return Err(AudioError::InvalidConfig(
                "converter output scratch was not preallocated".into(),
            ));
        }
        output.clear();
        if self.kernel.is_empty() {
            map_channels(samples, input_channels, channels, output);
            return Ok(());
        }
        map_channels(samples, input_channels, channels, &mut self.history);
        let frames = self.history.len() / channels;
        let denominator = u64::from(self.output.sample_rate);
        while self.position_numerator < frames as u64 * denominator {
            let center = (self.position_numerator / denominator) as usize - self.taps / 2;
            let start = center - (self.taps / 2 - 1);
            let phase =
                ((self.position_numerator % denominator) * PHASES as u64 / denominator) as usize;
            let weights = &self.kernel[phase * self.taps..(phase + 1) * self.taps];
            for channel in 0..channels {
                let mut value = 0.0;
                for (tap, weight) in weights.iter().enumerate() {
                    value += self.history[(start + tap) * channels + channel] * weight;
                }
                output.push(value);
            }
            self.position_numerator += u64::from(self.input.sample_rate);
        }
        let remove = (self.position_numerator / denominator) as usize;
        let remove = remove
            .saturating_sub(self.taps)
            .min(frames.saturating_sub(self.taps));
        self.history.copy_within(remove * channels.., 0);
        self.history.truncate((frames - remove) * channels);
        self.position_numerator -= remove as u64 * denominator;
        Ok(())
    }
}

fn map_channels(
    input: &[f32],
    input_channels: usize,
    output_channels: usize,
    output: &mut Vec<f32>,
) {
    if input_channels == output_channels {
        output.extend_from_slice(input);
    } else {
        for frame in input.chunks_exact(input_channels) {
            let mono = frame.iter().sum::<f32>() / input_channels as f32;
            output.extend(std::iter::repeat_n(mono, output_channels));
        }
    }
}

/// Standard blocking scheduler for differently sized capture/render slots.
/// Mutable converter and assembler state belongs exclusively to the worker.
pub struct DuplexWorkerAdapter {
    capture: StreamConfig,
    render: StreamConfig,
    processing: AudioFormat,
    input_converter: StreamingFormatConverter,
    output_converter: StreamingFormatConverter,
    logical_output_converter: Option<StreamingFormatConverter>,
    logical_output_scratch: Vec<f32>,
    logical_output_format: AudioFormat,
    input_scratch: Vec<f32>,
    output_scratch: Vec<f32>,
    batch_samples: usize,
    input_batch: Vec<f32>,
    input_timestamp: AudioTimestamp,
    pending_render: Vec<f32>,
    render_offset: usize,
    render_timestamp: AudioTimestamp,
    discontinuity: bool,
    last_sequence: Option<u64>,
}

impl DuplexWorkerAdapter {
    pub fn new(worker: &WorkerIo, processing: AudioFormat) -> Result<Self, AudioError> {
        Self::with_formats(worker, processing, processing)
    }

    /// Logical runtime formats are independent from both physical endpoints.
    pub fn with_formats(
        worker: &WorkerIo,
        processing: AudioFormat,
        logical_output_format: AudioFormat,
    ) -> Result<Self, AudioError> {
        Self::with_batch_frames(worker, processing, logical_output_format, 0)
    }

    /// Accumulate native device periods into complete model batches, then
    /// publish the whole result to render. A full output queue retains its
    /// unwritten tail across timeouts; it never silently discards model output.
    pub fn with_batch_frames(
        worker: &WorkerIo,
        processing: AudioFormat,
        logical_output_format: AudioFormat,
        batch_frames: usize,
    ) -> Result<Self, AudioError> {
        let capture = worker.input.config();
        let render = worker.output.config();
        let input_converter =
            StreamingFormatConverter::new(capture.format, processing, capture.block_frames)?;
        let internal_frames =
            input_converter.output_capacity_samples() / usize::from(processing.channels);
        let process_frames = if batch_frames == 0 {
            internal_frames
        } else {
            batch_frames
        };
        let logical_output_converter = if processing != logical_output_format {
            Some(StreamingFormatConverter::new(
                processing,
                logical_output_format,
                process_frames,
            )?)
        } else {
            None
        };
        let logical_output_scratch = match &logical_output_converter {
            Some(converter) => allocate_samples(converter.output_capacity_samples())?,
            None => Vec::new(),
        };
        let output_frames = logical_output_converter
            .as_ref()
            .map_or(process_frames, |converter| {
                converter.output_capacity_samples() / usize::from(logical_output_format.channels)
            });
        let output_converter =
            StreamingFormatConverter::new(logical_output_format, render.format, output_frames)?;
        let input_scratch = allocate_samples(input_converter.output_capacity_samples())?;
        let output_scratch = allocate_samples(output_converter.output_capacity_samples())?;
        let batch_samples = batch_frames
            .checked_mul(usize::from(processing.channels))
            .ok_or_else(|| AudioError::InvalidConfig("batch size overflows usize".into()))?;
        let mut input_batch = allocate_samples(
            batch_samples
                .checked_add(input_converter.output_capacity_samples())
                .ok_or_else(|| {
                    AudioError::InvalidConfig("batch capacity overflows usize".into())
                })?,
        )?;
        input_batch.clear();
        let mut pending_render = allocate_samples(
            output_converter
                .output_capacity_samples()
                .checked_add(render.block_samples())
                .ok_or_else(|| {
                    AudioError::InvalidConfig("render capacity overflows usize".into())
                })?,
        )?;
        pending_render.clear();
        Ok(Self {
            capture,
            render,
            processing,
            input_converter,
            output_converter,
            input_scratch,
            output_scratch,
            logical_output_converter,
            logical_output_scratch,
            logical_output_format,
            batch_samples,
            input_batch,
            input_timestamp: AudioTimestamp::ZERO,
            pending_render,
            render_offset: 0,
            render_timestamp: AudioTimestamp::ZERO,
            discontinuity: false,
            last_sequence: None,
        })
    }

    pub fn conversion_delay_micros(&self) -> u64 {
        self.input_converter.delay_frames() as u64 * 1_000_000
            / u64::from(self.capture.format.sample_rate)
            + self
                .logical_output_converter
                .as_ref()
                .map_or(0, |converter| {
                    converter.delay_frames() as u64 * 1_000_000
                        / u64::from(self.processing.sample_rate)
                })
            + self.output_converter.delay_frames() as u64 * 1_000_000
                / u64::from(self.logical_output_format.sample_rate)
    }

    /// The model may allocate its result according to its existing contract;
    /// device callbacks and both IO converters do not allocate per block.
    pub fn process_once(
        &mut self,
        worker: &mut WorkerIo,
        schedule: WorkerSchedule,
        process: impl FnOnce(AudioFormat, &[f32]) -> Result<Vec<f32>, AudioError>,
    ) -> Result<(usize, usize), AudioError> {
        self.process_batch_once(worker, schedule, |format, samples, _| {
            process(format, samples)
        })
    }

    /// `discontinuity` is delivered with the first complete batch after lost
    /// capture time. The model must reset its history before processing it.
    pub fn process_batch_once(
        &mut self,
        worker: &mut WorkerIo,
        schedule: WorkerSchedule,
        process: impl FnOnce(AudioFormat, &[f32], bool) -> Result<Vec<f32>, AudioError>,
    ) -> Result<(usize, usize), AudioError> {
        let started = std::time::Instant::now();
        let published = self.publish_pending(worker, schedule.timeout)?;
        if published > 0
            || self.pending_render.len() - self.render_offset >= self.render.block_samples()
        {
            return Ok((0, published));
        }
        if !worker.output.wait_for_capacity(schedule.timeout) {
            return Err(AudioError::Timeout);
        }
        let remaining = schedule
            .timeout
            .map(|timeout| timeout.saturating_sub(started.elapsed()));
        let input_frames = if self.batch_samples == 0 || self.input_batch.len() < self.batch_samples
        {
            self.receive_input(worker, schedule.input_policy, remaining)?
        } else {
            0
        };
        let target = if self.batch_samples == 0 {
            self.input_batch.len()
        } else {
            self.batch_samples
        };
        if target == 0 || self.input_batch.len() < target {
            return Ok((input_frames, published));
        }
        let converted = process(
            self.processing,
            &self.input_batch[..target],
            self.discontinuity,
        )?;
        self.discontinuity = false;
        if converted.len() != target {
            return Err(AudioError::BlockSizeMismatch {
                expected: target,
                actual: converted.len(),
            });
        }
        let batch_timestamp = self.input_timestamp;
        self.input_batch.copy_within(target.., 0);
        self.input_batch.truncate(self.input_batch.len() - target);
        self.input_timestamp = self
            .input_timestamp
            .later_by(std::time::Duration::from_secs_f64(
                (target / usize::from(self.processing.channels)) as f64
                    / f64::from(self.processing.sample_rate),
            ));
        if let Some(converter) = &mut self.logical_output_converter {
            converter.process(&converted, &mut self.logical_output_scratch)?;
            self.output_converter
                .process(&self.logical_output_scratch, &mut self.output_scratch)?;
        } else {
            self.output_converter
                .process(&converted, &mut self.output_scratch)?;
        }
        if self.render_offset > 0 {
            self.pending_render.copy_within(self.render_offset.., 0);
            self.pending_render
                .truncate(self.pending_render.len() - self.render_offset);
            self.render_offset = 0;
        }
        if self.pending_render.is_empty() {
            self.render_timestamp = batch_timestamp;
        }
        self.pending_render.extend_from_slice(&self.output_scratch);
        let remaining = schedule
            .timeout
            .map(|timeout| timeout.saturating_sub(started.elapsed()));
        let emitted = self.publish_pending(worker, remaining)?;
        Ok((input_frames, published + emitted))
    }

    fn receive_input(
        &mut self,
        worker: &mut WorkerIo,
        policy: WorkerInputPolicy,
        timeout: Option<std::time::Duration>,
    ) -> Result<usize, AudioError> {
        if !worker.input.wait_for_data(timeout) {
            return Err(AudioError::Timeout);
        }
        if policy == WorkerInputPolicy::Latest {
            worker.input.discard_stale();
        }
        let input = worker.input.try_acquire()?;
        let sequence = input.sequence();
        let timestamp = input
            .timestamp()
            .earlier_by(std::time::Duration::from_micros(
                self.conversion_delay_micros(),
            ));
        if self
            .last_sequence
            .is_some_and(|previous| sequence != previous.wrapping_add(1))
        {
            self.input_converter.reset();
            self.output_converter.reset();
            if let Some(converter) = &mut self.logical_output_converter {
                converter.reset();
            }
            self.input_batch.clear();
            self.input_timestamp = AudioTimestamp::ZERO;
            self.pending_render.clear();
            self.render_offset = 0;
            self.render_timestamp = AudioTimestamp::ZERO;
            self.discontinuity = true;
        }
        self.last_sequence = Some(sequence);
        self.input_converter
            .process(input.samples(), &mut self.input_scratch)?;
        let input_frames = input.samples().len() / usize::from(self.capture.format.channels);
        drop(input);
        if self.input_scratch.is_empty() {
            return Ok(input_frames);
        }
        if self.input_batch.is_empty() {
            self.input_timestamp = timestamp;
        }
        self.input_batch.extend_from_slice(&self.input_scratch);
        Ok(input_frames)
    }

    fn publish_pending(
        &mut self,
        worker: &mut WorkerIo,
        timeout: Option<std::time::Duration>,
    ) -> Result<usize, AudioError> {
        let started = std::time::Instant::now();
        let slot_samples = self.render.block_samples();
        let mut frames = 0;
        while self.pending_render.len() - self.render_offset >= slot_samples {
            let remaining = timeout.map(|timeout| timeout.saturating_sub(started.elapsed()));
            if !worker.output.wait_for_capacity(remaining) {
                return Ok(frames);
            }
            let mut lease = worker.output.try_acquire_at(self.render_timestamp)?;
            lease.samples_mut().copy_from_slice(
                &self.pending_render[self.render_offset..self.render_offset + slot_samples],
            );
            lease.commit();
            self.render_offset += slot_samples;
            frames += self.render.block_frames;
            self.render_timestamp = self.render_timestamp.later_by(self.render.block_duration());
        }
        Ok(frames)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn format(rate: u32, channels: u16) -> AudioFormat {
        AudioFormat {
            sample_rate: rate,
            channels,
        }
    }

    #[test]
    fn model_output_tail_survives_full_queue_and_timeout() {
        use super::super::{StreamIo, StreamQueueConfig};
        let stream = StreamConfig {
            format: format(16000, 1),
            block_frames: 160,
            queue_blocks: 3,
        };
        let (mut device, mut worker) =
            StreamIo::new_with_queue_config(stream, StreamQueueConfig::new(3, 2))
                .unwrap()
                .split();
        let mut adapter =
            DuplexWorkerAdapter::with_batch_frames(&worker, stream.format, stream.format, 480)
                .unwrap();
        let schedule = WorkerSchedule {
            timeout: Some(std::time::Duration::ZERO),
            ..Default::default()
        };
        let mut calls = 0;
        for value in [1., 2., 3.] {
            device.input.try_push(&vec![value; 160]).unwrap();
            adapter
                .process_once(&mut worker, schedule, |_, samples| {
                    calls += 1;
                    Ok(samples.to_vec())
                })
                .unwrap();
        }
        assert_eq!(calls, 1);
        assert_eq!(device.output.try_pop().unwrap().samples, vec![1.; 160]);
        assert_eq!(device.output.try_pop().unwrap().samples, vec![2.; 160]);
        let emitted = adapter
            .process_once(&mut worker, schedule, |_, _| {
                panic!("retained output must flush before new input")
            })
            .unwrap();
        assert_eq!(emitted, (0, 160));
        assert_eq!(device.output.try_pop().unwrap().samples, vec![3.; 160]);
        assert_eq!(worker.stats().output.dropped_full, 0);
    }

    #[test]
    fn discontinuity_discards_partial_batch_and_notifies_model_once() {
        use super::super::{StreamIo, StreamQueueConfig};
        let stream = StreamConfig {
            format: format(16000, 1),
            block_frames: 160,
            queue_blocks: 4,
        };
        let (mut device, mut worker) =
            StreamIo::new_with_queue_config(stream, StreamQueueConfig::new(4, 8))
                .unwrap()
                .split();
        let mut adapter =
            DuplexWorkerAdapter::with_batch_frames(&worker, stream.format, stream.format, 480)
                .unwrap();
        let mut observed = Vec::new();
        for index in 0..8 {
            if index == 2 {
                device.input.mark_discontinuity();
            }
            let value = if index < 2 { 1. } else { 2. };
            device.input.try_push(&vec![value; 160]).unwrap();
            adapter
                .process_batch_once(&mut worker, WorkerSchedule::default(), |_, samples, gap| {
                    observed.push(gap);
                    assert_eq!(samples, vec![2.; 480]);
                    Ok(samples.to_vec())
                })
                .unwrap();
        }
        assert_eq!(observed, [true, false]);
    }
    fn run(input_rate: u32, output_rate: u32, samples: &[f32], chunks: usize) -> Vec<f32> {
        let mut converter =
            StreamingFormatConverter::new(format(input_rate, 1), format(output_rate, 1), chunks)
                .unwrap();
        let mut scratch = Vec::with_capacity(converter.output_capacity_samples());
        let mut result = Vec::new();
        for chunk in samples.chunks(chunks) {
            converter.process(chunk, &mut scratch).unwrap();
            result.extend_from_slice(&scratch);
        }
        result
    }
    #[test]
    fn channel_routing_is_normalized_and_same_layout_preserves_channels() {
        for (source, destination, expected) in [
            (2, 1, vec![0.5, -0.25]),
            (
                2,
                4,
                vec![0.5; 4].into_iter().chain(vec![-0.25; 4]).collect(),
            ),
            (2, 2, vec![1., 0., -1., 0.5]),
        ] {
            let mut converter =
                StreamingFormatConverter::new(format(16000, source), format(16000, destination), 2)
                    .unwrap();
            let mut output = Vec::with_capacity(converter.output_capacity_samples());
            converter.process(&[1., 0., -1., 0.5], &mut output).unwrap();
            assert_eq!(output, expected);
        }
    }
    #[test]
    fn fractional_phase_and_history_are_independent_of_callback_boundaries() {
        let input: Vec<_> = (0..48000).map(|i| (i as f32 * 0.137).sin()).collect();
        for rates in [
            (16000, 48000),
            (48000, 16000),
            (48000, 44100),
            (44100, 48000),
        ] {
            let expected = run(rates.0, rates.1, &input, input.len());
            let actual = run(rates.0, rates.1, &input, 137);
            assert_eq!(actual.len(), expected.len());
            assert!(
                actual
                    .iter()
                    .zip(expected)
                    .all(|(a, b)| (a - b).abs() < 0.002)
            );
            assert!(
                (actual.len() as f64 - input.len() as f64 * rates.1 as f64 / rates.0 as f64).abs()
                    <= 1.
            );
        }
    }
    #[test]
    fn downsampling_rejects_out_of_band_signal() {
        let input: Vec<_> = (0..48000)
            .map(|i| (2. * std::f32::consts::PI * 12000. * i as f32 / 48000.).sin())
            .collect();
        let output = run(48000, 16000, &input, 480);
        let rms =
            (output[100..].iter().map(|x| x * x).sum::<f32>() / (output.len() - 100) as f32).sqrt();
        assert!(rms < 0.005, "aliased RMS {rms}");
    }
    #[test]
    fn reset_clears_history_and_process_reuses_preallocated_storage() {
        let mut converter =
            StreamingFormatConverter::new(format(16000, 2), format(48000, 1), 160).unwrap();
        let mut output = Vec::with_capacity(converter.output_capacity_samples());
        let capacity = converter.history.capacity();
        converter.process(&[1.; 320], &mut output).unwrap();
        converter.reset();
        converter.process(&[0.; 320], &mut output).unwrap();
        assert!(output.iter().all(|x| *x == 0.));
        assert_eq!(converter.history.capacity(), capacity);
        assert!(matches!(
            converter.process(&[0.], &mut output),
            Err(AudioError::PartialFrame)
        ));
    }

    #[test]
    fn duplex_worker_routes_different_rates_and_channels_without_duration_drift() {
        use super::super::{StreamIo, StreamQueueConfig};
        let reference = StreamConfig {
            format: format(48000, 1),
            block_frames: 512,
            queue_blocks: 3,
        };
        for (input, output) in [
            (format(16000, 2), format(48000, 1)),
            (format(48000, 1), format(16000, 2)),
            (format(44100, 2), format(48000, 6)),
        ] {
            let capture = reference.for_endpoint(input).unwrap();
            let render = reference.for_endpoint(output).unwrap();
            let io = StreamIo::new_duplex(capture, render, StreamQueueConfig::new(3, 3)).unwrap();
            let (mut device, mut worker) = io.split();
            let mut adapter =
                DuplexWorkerAdapter::with_formats(&worker, format(48000, 1), format(44100, 2))
                    .unwrap();
            let samples = vec![0.5; capture.block_samples()];
            let mut total_frames = 0;
            let mut steady = Vec::new();
            for block in 0..200 {
                device.input.try_push(&samples).unwrap();
                adapter
                    .process_once(&mut worker, WorkerSchedule::default(), |_, samples| {
                        Ok(samples.to_vec())
                    })
                    .unwrap();
                while let Ok(lease) = device.output.try_acquire() {
                    total_frames += lease.samples().len() / usize::from(output.channels);
                    if block > 5 {
                        steady.extend_from_slice(lease.samples());
                    }
                }
            }
            let expected = 200. * capture.block_frames as f64 * output.sample_rate as f64
                / input.sample_rate as f64;
            assert!(
                (total_frames as f64 - expected).abs() <= render.block_frames as f64 + 8.,
                "{input:?} -> {output:?}: {total_frames} != {expected}"
            );
            assert!(steady.iter().all(|value| (value - 0.5).abs() < 0.002));
            assert_eq!(worker.stats().output.dropped_full, 0);
        }
    }

    #[test]
    fn full_render_queue_does_not_consume_capture() {
        use super::super::{StreamIo, StreamQueueConfig};
        let capture = StreamConfig {
            format: format(16000, 2),
            block_frames: 160,
            queue_blocks: 1,
        };
        let render = capture.for_endpoint(format(48000, 1)).unwrap();
        let (mut device, mut worker) =
            StreamIo::new_duplex(capture, render, StreamQueueConfig::symmetric(1))
                .unwrap()
                .split();
        worker
            .output
            .try_push(&vec![0.; render.block_samples()])
            .unwrap();
        device
            .input
            .try_push(&vec![0.; capture.block_samples()])
            .unwrap();
        let mut adapter = DuplexWorkerAdapter::new(&worker, format(48000, 1)).unwrap();
        assert!(matches!(
            adapter.process_once(
                &mut worker,
                WorkerSchedule {
                    timeout: Some(std::time::Duration::ZERO),
                    ..WorkerSchedule::default()
                },
                |_, _| panic!("must not process a backpressured stream")
            ),
            Err(AudioError::Timeout)
        ));
        assert_eq!(worker.stats().input.queued_blocks, 1);
    }
}
