//! PCM representation conversion only. Sample-rate and channel conversion stay
//! on the worker; integer endpoint callbacks reuse bounded scratch chunks.
use crate::cpal_backend::CpalError;
use cpal::traits::DeviceTrait;
use cpal::{Sample, SampleFormat};

pub(crate) fn build_input(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    format: SampleFormat,
    mut callback: impl FnMut(&[f32], &cpal::InputCallbackInfo, usize, usize) + Send + 'static,
    error: impl FnMut(cpal::Error) + Send + 'static,
) -> Result<cpal::Stream, CpalError> {
    macro_rules! build {
        ($sample:ty) => {{
            let mut scratch = crate::allocate_samples(4096 * usize::from(config.channels))?;
            device
                .build_input_stream(
                    config,
                    move |input: &[$sample], info| {
                        let frames = input.len() / usize::from(config.channels);
                        for (index, chunk) in input.chunks(scratch.len()).enumerate() {
                            for (output, sample) in scratch.iter_mut().zip(chunk) {
                                *output = sample.to_sample::<f32>();
                            }
                            callback(&scratch[..chunk.len()], info, frames, index * 4096);
                        }
                    },
                    error,
                    None,
                )
                .map_err(CpalError::BuildInput)
        }};
    }
    match format {
        SampleFormat::F32 => device
            .build_input_stream(
                config,
                move |data: &[f32], info| {
                    callback(data, info, data.len() / usize::from(config.channels), 0)
                },
                error,
                None,
            )
            .map_err(CpalError::BuildInput),
        SampleFormat::F64 => build!(f64),
        SampleFormat::I8 => build!(i8),
        SampleFormat::I16 => build!(i16),
        SampleFormat::I24 => build!(cpal::I24),
        SampleFormat::I32 => build!(i32),
        SampleFormat::I64 => build!(i64),
        SampleFormat::U8 => build!(u8),
        SampleFormat::U16 => build!(u16),
        SampleFormat::U24 => build!(cpal::U24),
        SampleFormat::U32 => build!(u32),
        SampleFormat::U64 => build!(u64),
        _ => Err(CpalError::UnsupportedSampleFormat {
            input: format,
            output: format,
        }),
    }
}

pub(crate) fn build_output(
    device: &cpal::Device,
    config: cpal::StreamConfig,
    format: SampleFormat,
    mut callback: impl FnMut(&mut [f32], &cpal::OutputCallbackInfo, usize, usize) + Send + 'static,
    error: impl FnMut(cpal::Error) + Send + 'static,
) -> Result<cpal::Stream, CpalError> {
    macro_rules! build {
        ($sample:ty) => {{
            let mut scratch = crate::allocate_samples(4096 * usize::from(config.channels))?;
            device
                .build_output_stream(
                    config,
                    move |output: &mut [$sample], info| {
                        let frames = output.len() / usize::from(config.channels);
                        for (index, chunk) in output.chunks_mut(scratch.len()).enumerate() {
                            callback(&mut scratch[..chunk.len()], info, frames, index * 4096);
                            for (sample, value) in chunk.iter_mut().zip(&scratch) {
                                *sample = <$sample>::from_sample(value.clamp(-1., 1.));
                            }
                        }
                    },
                    error,
                    None,
                )
                .map_err(CpalError::BuildOutput)
        }};
    }
    match format {
        SampleFormat::F32 => device
            .build_output_stream(
                config,
                move |data: &mut [f32], info| {
                    let frames = data.len() / usize::from(config.channels);
                    callback(data, info, frames, 0);
                },
                error,
                None,
            )
            .map_err(CpalError::BuildOutput),
        SampleFormat::F64 => build!(f64),
        SampleFormat::I8 => build!(i8),
        SampleFormat::I16 => build!(i16),
        SampleFormat::I24 => build!(cpal::I24),
        SampleFormat::I32 => build!(i32),
        SampleFormat::I64 => build!(i64),
        SampleFormat::U8 => build!(u8),
        SampleFormat::U16 => build!(u16),
        SampleFormat::U24 => build!(cpal::U24),
        SampleFormat::U32 => build!(u32),
        SampleFormat::U64 => build!(u64),
        _ => Err(CpalError::UnsupportedSampleFormat {
            input: format,
            output: format,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cpal::traits::{HostTrait, StreamTrait};
    #[test]
    #[ignore = "requires a live audio render endpoint"]
    fn live_cpal_native_render_callback_uses_actual_device_format() {
        let device = crate::cpal_backend::recommended_host()
            .default_output_device()
            .expect("render device");
        let supported = device.default_output_config().unwrap();
        let config = cpal::StreamConfig {
            sample_rate: supported.sample_rate(),
            channels: supported.channels(),
            buffer_size: cpal::BufferSize::Default,
        };
        let (tx, rx) = std::sync::mpsc::sync_channel(4);
        let channels = usize::from(config.channels);
        let stream = build_output(
            &device,
            config,
            supported.sample_format(),
            move |samples, _, _, _| {
                samples.fill(0.);
                let _ = tx.try_send(samples.len());
            },
            |error| panic!("render callback error: {error}"),
        )
        .unwrap();
        stream.play().unwrap();
        for _ in 0..3 {
            let samples = rx.recv_timeout(std::time::Duration::from_secs(2)).unwrap();
            assert!(samples > 0 && samples.is_multiple_of(channels));
        }
        println!(
            "CPAL render={} Hz/{} ch/{:?}; 3 native callbacks received",
            supported.sample_rate(),
            supported.channels(),
            supported.sample_format()
        );
    }
}
