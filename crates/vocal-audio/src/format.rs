//! Optional format conversion boundary for devices that do not expose the
//! stream's native interleaved f32 format.
//!
//! The native WASAPI path deliberately does not call this trait from an audio
//! callback. A caller that needs sample-rate or channel conversion owns one
//! adapter per direction and supplies reusable scratch buffers around the
//! [`StreamIo`] boundary. Returning a sample count allows an asynchronous
//! resampler to produce a different number of samples without allocating in
//! the callback.

use vocal_types::AudioFormat;

use super::error::AudioError;

/// External conversion hook for non-native device formats.
///
/// `input_format` describes samples entering the adapter from a device and
/// `output_format` describes the f32 stream presented to the worker (the
/// reverse applies to [`convert_output`](Self::convert_output)). The returned
/// value is a number of interleaved samples written to `output`, not frames.
/// Implementations must not retain either slice after the call.
pub trait AudioFormatAdapter: Send {
    fn input_format(&self) -> AudioFormat;

    fn output_format(&self) -> AudioFormat;

    fn convert_input(&mut self, input: &[f32], output: &mut [f32]) -> Result<usize, AudioError>;

    fn convert_output(&mut self, input: &[f32], output: &mut [f32]) -> Result<usize, AudioError>;

    /// Reset filter history after a device discontinuity.
    fn reset(&mut self) {}
}
