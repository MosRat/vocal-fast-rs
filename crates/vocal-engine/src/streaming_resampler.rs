//! Stateful anti-aliased input conversion; phase and history span callbacks.
use vocal_audio::StreamingFormatConverter;
use vocal_types::AudioFormat;

use crate::EngineError;

pub(crate) struct ModelInputResampler {
    converter: StreamingFormatConverter,
    scratch: Vec<f32>,
    max_frames: usize,
}

impl ModelInputResampler {
    pub(crate) fn new(sample_rate: u32) -> Result<Self, EngineError> {
        // Process large caller blocks in bounded sub-blocks without resetting
        // the filter or rounding each sub-block's fractional frame count.
        let max_frames = sample_rate.div_ceil(4) as usize;
        let converter = StreamingFormatConverter::new(
            AudioFormat {
                sample_rate,
                channels: 1,
            },
            AudioFormat::MONO_16K,
            max_frames,
        )
        .map_err(|error| EngineError::InvalidInput(error.to_string()))?;
        let scratch = Vec::with_capacity(converter.output_capacity_samples());
        Ok(Self {
            converter,
            scratch,
            max_frames,
        })
    }

    pub(crate) fn process(&mut self, mono: &[f32]) -> Result<Vec<f32>, EngineError> {
        let mut output = Vec::new();
        for chunk in mono.chunks(self.max_frames) {
            self.converter
                .process(chunk, &mut self.scratch)
                .map_err(|error| EngineError::InvalidInput(error.to_string()))?;
            output.extend_from_slice(&self.scratch);
        }
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filters_aliases_and_preserves_phase_across_callbacks() {
        let wave = (0..48_000)
            .map(|i| (std::f32::consts::TAU * 12_000. * i as f32 / 48_000.).sin())
            .collect::<Vec<_>>();
        let mut small = ModelInputResampler::new(48_000).unwrap();
        let mut blocked = Vec::new();
        for chunk in wave.chunks(479) {
            blocked.extend(small.process(chunk).unwrap());
        }
        let whole = ModelInputResampler::new(48_000)
            .unwrap()
            .process(&wave)
            .unwrap();
        assert_eq!(blocked, whole);
        assert_eq!(whole.len(), 16_000);
        let rms =
            (whole[100..].iter().map(|s| s * s).sum::<f32>() / (whole.len() - 100) as f32).sqrt();
        assert!(rms < 0.001, "aliased 12 kHz energy: {rms}");
    }
}
