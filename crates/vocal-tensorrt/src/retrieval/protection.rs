//! Preserve raw content on device for RVC's unvoiced feature protection.
use std::ptr;

use crate::{CudaBuffer, CudaStream, NativeBuildError, ffi};

pub(crate) struct RetrievalProtection {
    raw: CudaBuffer,
    width: usize,
    amount: CudaBuffer,
    last_amount: f32,
}

impl RetrievalProtection {
    pub fn new(max_hubert_frames: usize, width: usize) -> Result<Self, NativeBuildError> {
        let bytes = max_hubert_frames
            .checked_mul(width)
            .and_then(|n| n.checked_mul(4))
            .filter(|n| *n > 0)
            .ok_or_else(|| NativeBuildError::Argument("protection allocation overflow".into()))?;
        let amount = CudaBuffer::new(4)?;
        amount.upload_f32(&[0.33])?;
        Ok(Self {
            raw: CudaBuffer::new(bytes)?,
            width,
            amount,
            last_amount: 0.33,
        })
    }

    pub fn set_amount(&mut self, value: f32) -> Result<(), NativeBuildError> {
        if !value.is_finite() || !(0.0..=1.0).contains(&value) {
            return Err(NativeBuildError::Argument(
                "protect must be finite and in [0,1]".into(),
            ));
        }
        // The graph reads a stable device scalar. Avoid a synchronous upload
        // on every audio block; update the remembered value only on success.
        if value != self.last_amount {
            self.amount.upload_f32(&[value])?;
            self.last_amount = value;
        }
        Ok(())
    }

    pub fn save_raw(
        &self,
        features: &CudaBuffer,
        frames: usize,
        stream: &CudaStream,
    ) -> Result<(), NativeBuildError> {
        self.raw
            .device_copy_from(features, 0, 0, frames * self.width * 4, stream)
    }

    pub fn apply(
        &self,
        features: &CudaBuffer,
        pitchf: &CudaBuffer,
        ratio: &CudaBuffer,
        frames: usize,
        hubert_frames: usize,
        stream: &CudaStream,
    ) -> Result<(), NativeBuildError> {
        let elements = frames.checked_mul(self.width);
        if frames == 0
            || hubert_frames == 0
            || elements.is_none_or(|n| n > i32::MAX as usize || n * 4 > features.bytes)
            || hubert_frames
                .checked_mul(self.width)
                .and_then(|n| n.checked_mul(4))
                .is_none_or(|n| n > self.raw.bytes)
            || frames * 4 > pitchf.bytes
            || ratio.bytes < 4
        {
            return Err(NativeBuildError::Argument(
                "invalid protection buffer geometry".into(),
            ));
        }
        let mut error = ptr::null_mut();
        let code = unsafe {
            ffi::vocal_retrieval_protect(
                features.device_ptr().cast(),
                self.raw.device_ptr().cast(),
                pitchf.device_ptr().cast(),
                ratio.device_ptr().cast(),
                self.amount.device_ptr().cast(),
                frames as i32,
                hubert_frames as i32,
                self.width as i32,
                stream.native_handle(),
                &mut error,
            )
        };
        if code != 0 {
            return Err(NativeBuildError::Api(ffi::native_message(
                error,
                "retrieval protection failed",
            )));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires CUDA GPU"]
    fn voiced_unvoiced_boundary_and_appended_frame_match_reference() {
        let stream = CudaStream::new().unwrap();
        let raw = CudaBuffer::new(6 * 4).unwrap();
        raw.upload_f32(&[1., 2., 3., 4., 5., 6.]).unwrap();
        let features = CudaBuffer::new(15 * 4).unwrap();
        features.upload_f32(&[10.; 15]).unwrap();
        let f0 = CudaBuffer::new(5 * 4).unwrap();
        f0.upload_f32(&[0., 100., 0., 100., 0.]).unwrap();
        let ratio = CudaBuffer::new(4).unwrap();
        ratio.upload_f32(&[0.5]).unwrap();
        let mut protection = RetrievalProtection::new(2, 3).unwrap();
        protection.save_raw(&raw, 2, &stream).unwrap();
        protection
            .apply(&features, &f0, &ratio, 5, 2, &stream)
            .unwrap();
        stream.synchronize().unwrap();
        let mut actual = vec![0.; 15];
        features.download_f32(&mut actual).unwrap();
        for (frame, row) in actual.chunks_exact(3).enumerate() {
            for (channel, &value) in row.iter().enumerate() {
                let expected = if frame % 2 == 1 {
                    10.
                } else {
                    10. * 0.33 + ((frame / 2).min(1) * 3 + channel + 1) as f32 * 0.67
                };
                assert!((value - expected).abs() < 1e-6);
            }
        }
        ratio.upload_f32(&[0.]).unwrap();
        features.upload_f32(&[10.; 15]).unwrap();
        let capture = stream.begin_capture().unwrap();
        protection.save_raw(&raw, 2, &stream).unwrap();
        protection
            .apply(&features, &f0, &ratio, 5, 2, &stream)
            .unwrap();
        let graph = capture.finish().unwrap();
        graph.launch(&stream).unwrap();
        stream.synchronize().unwrap();
        features.download_f32(&mut actual).unwrap();
        assert_eq!(actual, vec![10.; 15]);
        ratio.upload_f32(&[0.5]).unwrap();
        graph.launch(&stream).unwrap();
        stream.synchronize().unwrap();
        features.download_f32(&mut actual).unwrap();
        assert!((actual[0] - 3.97).abs() < 1e-6);
        assert_eq!(&actual[3..6], &[10.; 3]);
        // Amount is device data, so disabling/re-enabling protection does not
        // require recapturing the graph or replacing any buffer address.
        features.upload_f32(&[10.; 15]).unwrap();
        protection.set_amount(0.5).unwrap();
        graph.launch(&stream).unwrap();
        stream.synchronize().unwrap();
        features.download_f32(&mut actual).unwrap();
        assert_eq!(actual, vec![10.; 15]);
        assert!(protection.set_amount(f32::NAN).is_err());
        protection.set_amount(0.).unwrap();
        graph.launch(&stream).unwrap();
        stream.synchronize().unwrap();
        features.download_f32(&mut actual).unwrap();
        assert_eq!(&actual[..3], &[1., 2., 3.]);
        for amount in [0.33, 0.5, 1.0, 0.0, 0.33] {
            features.upload_f32(&[10.; 15]).unwrap();
            protection.set_amount(amount).unwrap();
            graph.launch(&stream).unwrap();
            stream.synchronize().unwrap();
            features.download_f32(&mut actual).unwrap();
            let expected = if amount >= 0.5 {
                10.
            } else {
                10. * amount + 1. - amount
            };
            assert!((actual[0] - expected).abs() < 1e-6, "protect={amount}");
            assert_eq!(&actual[3..6], &[10.; 3]);
        }
    }
}
