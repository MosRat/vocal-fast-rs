//! Stable device controls keep matching compatible with captured CUDA graphs.
use crate::{CudaBuffer, CudaStream, NativeBuildError, ffi};
use std::ptr;
use vocal_types::{AutoPitchConfig, pitch_calibration_frames};

pub(crate) struct DevicePitchMatcher {
    pub scaled: CudaBuffer,
    scratch: CudaBuffer,
    controls: CudaBuffer,
    state: CudaBuffer,
    last: (f32, AutoPitchConfig),
}

#[cfg(test)]
mod tests {
    use super::*;
    use vocal_types::PitchMatcher;

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn graph_replay_tracks_controls_without_modifying_raw_or_repeating_shift() {
        let stream = CudaStream::new().unwrap();
        for n in [6, 7, 281, 282, 2201, 8193] {
            let mut matcher = DevicePitchMatcher::new(n).unwrap();
            let raw = CudaBuffer::new(n * 4).unwrap();
            raw.upload_f32(&vec![0.; n]).unwrap();
            matcher.set(0., AutoPitchConfig::default()).unwrap();
            let capture = stream.begin_capture().unwrap();
            matcher.enqueue(&raw, 0, n, true, &stream).unwrap();
            let graph = capture.finish().unwrap();
            let mut cpu = PitchMatcher::default();
            for (freq, manual, policy) in [
                (0., 0., AutoPitchConfig::default()),
                (140., 0., AutoPitchConfig::default()),
                (280., 2., AutoPitchConfig::default()),
                (140., 0., AutoPitchConfig::disabled()),
                (
                    140.,
                    -2.,
                    AutoPitchConfig {
                        target_hz: 155.,
                        ..Default::default()
                    },
                ),
            ] {
                let mut values = vec![freq; n];
                values[0] = 0.;
                values[n / 2] = 0.;
                values[n - 1] = 0.;
                raw.upload_f32(&values).unwrap();
                matcher.set(manual, policy).unwrap();
                graph.launch(&stream).unwrap();
                stream.synchronize().unwrap();
                let expected = cpu.resolve(&values, policy, manual, true);
                let mut actual = vec![0.; n];
                matcher.scaled.download_f32(&mut actual).unwrap();
                for (&a, &b) in actual.iter().zip(&values) {
                    assert!(
                        (a - b * 2f32.powf(expected / 12.)).abs() < 0.001,
                        "n={n} {a} {expected}"
                    );
                }
                raw.download_f32(&mut actual).unwrap();
                assert_eq!(actual, values);
            }
            matcher.reset().unwrap();
            cpu.reset();
            matcher.set(0., AutoPitchConfig::default()).unwrap();
            let values = (0..n)
                .map(|i| {
                    if i % 3 == 0 {
                        0.
                    } else {
                        100. + (i % 100) as f32 * 2.
                    }
                })
                .collect::<Vec<_>>();
            raw.upload_f32(&values).unwrap();
            matcher.enqueue(&raw, 0, n, false, &stream).unwrap();
            stream.synchronize().unwrap();
            let mut actual = vec![0.; n];
            matcher.scaled.download_f32(&mut actual).unwrap();
            let expected = cpu.resolve(&values, AutoPitchConfig::default(), 0., false);
            for (&a, &b) in actual.iter().zip(&values) {
                assert!(
                    (a - b * 2f32.powf(expected / 12.)).abs() < 0.001,
                    "offline n={n}"
                );
            }
            if n == 281 {
                let begin = crate::timing::CudaEvent::new().unwrap();
                let end = crate::timing::CudaEvent::new().unwrap();
                matcher.reset().unwrap();
                begin.record(&stream).unwrap();
                matcher.enqueue(&raw, 0, n, true, &stream).unwrap();
                end.record(&stream).unwrap();
                stream.synchronize().unwrap();
                let calibration_ms = begin.elapsed(&end).unwrap();
                begin.record(&stream).unwrap();
                for _ in 0..100 {
                    graph.launch(&stream).unwrap();
                }
                end.record(&stream).unwrap();
                stream.synchronize().unwrap();
                eprintln!(
                    "CUDA matcher n={n}: calibration_ms={calibration_ms}, locked_replay_ms={}",
                    begin.elapsed(&end).unwrap() / 100.
                );
            }
        }
    }
}

impl DevicePitchMatcher {
    pub fn new(max_frames: usize) -> Result<Self, NativeBuildError> {
        let bytes = max_frames
            .checked_mul(4)
            .filter(|&n| n > 0)
            .ok_or_else(|| NativeBuildError::Argument("pitch buffer capacity overflow".into()))?;
        let controls = CudaBuffer::new(12)?;
        // Low-level pipeline diagnostics preserve raw F0 by default. Engine
        // adapters explicitly install the shared, enabled-by-default policy.
        let policy = AutoPitchConfig::disabled();
        controls.upload_f32(&[0., policy.target_hz, 0.])?;
        let state = CudaBuffer::new(8)?;
        state.zero()?;
        Ok(Self {
            scaled: CudaBuffer::new(bytes)?,
            scratch: CudaBuffer::new(bytes)?,
            controls,
            state,
            last: (0., policy),
        })
    }
    pub fn reset(&self) -> Result<(), NativeBuildError> {
        self.state.zero()
    }
    pub fn set(&mut self, manual: f32, policy: AutoPitchConfig) -> Result<(), NativeBuildError> {
        policy
            .validate()
            .map_err(|e| NativeBuildError::Argument(e.to_string()))?;
        if !manual.is_finite() || !(-96.0..=96.0).contains(&manual) {
            return Err(NativeBuildError::Argument(
                "manual pitch must be finite and in [-96,96]".into(),
            ));
        }
        if self.last != (manual, policy) {
            if self.last.1 != policy {
                self.reset()?;
            }
            self.controls
                .upload_f32(&[f32::from(policy.enabled), policy.target_hz, manual])?;
            self.last = (manual, policy);
        }
        Ok(())
    }
    pub fn enqueue(
        &self,
        raw: &CudaBuffer,
        offset_frames: usize,
        frames: usize,
        streaming: bool,
        stream: &CudaStream,
    ) -> Result<(), NativeBuildError> {
        if frames == 0
            || frames > i32::MAX as usize
            || frames > self.scaled.bytes / 4
            || offset_frames
                .checked_add(frames)
                .is_none_or(|n| n > raw.bytes / 4)
        {
            return Err(NativeBuildError::Argument(
                "invalid pitch matching buffer geometry".into(),
            ));
        }
        let mut error = ptr::null_mut();
        let code = unsafe {
            ffi::vocal_pitch_match(
                raw.device_ptr_offset(offset_frames * 4).cast(),
                self.scaled.device_ptr().cast(),
                self.controls.device_ptr().cast(),
                self.state.device_ptr().cast(),
                self.scratch.device_ptr().cast(),
                frames as i32,
                if streaming {
                    pitch_calibration_frames(frames) as i32
                } else {
                    2
                },
                i32::from(streaming),
                stream.native_handle(),
                &mut error,
            )
        };
        if code != 0 {
            return Err(NativeBuildError::Api(ffi::native_message(
                error,
                "pitch matching failed",
            )));
        }
        Ok(())
    }
}
