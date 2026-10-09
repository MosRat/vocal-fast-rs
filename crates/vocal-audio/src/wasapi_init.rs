//! WASAPI negotiation and typed startup failures. Runs on each endpoint's
//! owning COM thread; never falls back by retrying an occupied device.

use std::{fmt, ptr};
use windows::{
    Win32::{
        Media::Audio::{
            AUDCLNT_E_DEVICE_IN_USE, AUDCLNT_E_DEVICE_INVALIDATED, AUDCLNT_E_ENGINE_FORMAT_LOCKED,
            AUDCLNT_E_ENGINE_PERIODICITY_LOCKED, AUDCLNT_E_INVALID_DEVICE_PERIOD,
            AUDCLNT_E_SERVICE_NOT_RUNNING, AUDCLNT_E_UNSUPPORTED_FORMAT,
            AUDCLNT_STREAMFLAGS_EVENTCALLBACK, EDataFlow, IAudioClient, IAudioClient3, IAudioClock,
            IMMDeviceEnumerator, MMDeviceEnumerator, WAVEFORMATEX, WAVEFORMATEXTENSIBLE, eCapture,
        },
        Media::KernelStreaming::WAVE_FORMAT_EXTENSIBLE,
        Media::Multimedia::{KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, WAVE_FORMAT_IEEE_FLOAT},
        System::Com::{CLSCTX_ALL, CoCreateInstance},
        System::Threading::CreateEventW,
    },
    core::{Error, Interface},
};

use super::{CoTaskMemFormat, EventHandle, choose_period};
use crate::{AudioDirection, StreamConfig};

#[derive(Clone, Debug)]
pub struct WasapiInitError {
    pub operation: String,
    pub hresult: Option<i32>,
    pub direction: Option<AudioDirection>,
    pub device_id: Option<String>,
    pub message: String,
}

impl WasapiInitError {
    pub fn windows(operation: impl Into<String>, error: Error) -> Self {
        Self {
            operation: operation.into(),
            hresult: Some(error.code().0),
            direction: None,
            device_id: None,
            message: error.to_string(),
        }
    }

    pub fn with_endpoint(mut self, direction: AudioDirection, device_id: Option<&str>) -> Self {
        self.direction = Some(direction);
        self.device_id = device_id.map(str::to_owned);
        self
    }

    /// CPAL uses the same Windows endpoint. These failures cannot be cured by
    /// changing the wrapper; leave the device alone and report the cause.
    pub fn allows_fallback(&self) -> bool {
        !matches!(self.hresult, Some(code) if code == AUDCLNT_E_DEVICE_IN_USE.0
            || code == AUDCLNT_E_DEVICE_INVALIDATED.0
            || code == AUDCLNT_E_SERVICE_NOT_RUNNING.0 || code == 0x80070005u32 as i32
            || code == 0x80070490u32 as i32)
    }
}

impl From<String> for WasapiInitError {
    fn from(message: String) -> Self {
        Self {
            operation: "WASAPI startup".into(),
            hresult: None,
            direction: None,
            device_id: None,
            message,
        }
    }
}

impl fmt::Display for WasapiInitError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(direction) = self.direction {
            write!(f, "{direction:?}: ")?;
        }
        write!(f, "{}: {}", self.operation, self.message)?;
        if let Some(code) = self.hresult {
            write!(f, " [HRESULT 0x{:08X}]", code as u32)?;
        }
        if let Some(id) = &self.device_id {
            write!(f, " [endpoint {id}]")?;
        }
        Ok(())
    }
}
impl std::error::Error for WasapiInitError {}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SharedAttempt {
    Preferred(u32),
    Current,
    Default(u32),
}

/// At most two attempts, on fresh COM clients. Busy/access/service errors are
/// terminal; only a period/format negotiation conflict gets another attempt.
fn negotiate_shared<T>(
    preferred: u32,
    default: u32,
    mut attempt: impl FnMut(SharedAttempt) -> Result<T, WasapiInitError>,
) -> Result<T, WasapiInitError> {
    match attempt(SharedAttempt::Preferred(preferred)) {
        Err(error) if matches!(error.hresult, Some(code) if code == AUDCLNT_E_ENGINE_PERIODICITY_LOCKED.0 || code == AUDCLNT_E_ENGINE_FORMAT_LOCKED.0) => {
            attempt(SharedAttempt::Current)
        }
        Err(error)
            if error.hresult == Some(AUDCLNT_E_INVALID_DEVICE_PERIOD.0)
                && default != preferred
                && default > 0 =>
        {
            attempt(SharedAttempt::Default(default))
        }
        result => result,
    }
}

pub(super) struct InitializedClient {
    pub client: IAudioClient,
    pub clock: Option<IAudioClock>,
    pub event: EventHandle,
    pub stream_latency_hns: Option<i64>,
    pub block_align: usize,
    pub endpoint_frames: usize,
    pub period_frames: usize,
}

unsafe fn matches_format(format_ptr: *const WAVEFORMATEX, config: StreamConfig) -> bool {
    let format = unsafe { ptr::read_unaligned(format_ptr) };
    let float = match format.wFormatTag as u32 {
        WAVE_FORMAT_IEEE_FLOAT => true,
        WAVE_FORMAT_EXTENSIBLE if format.cbSize >= 22 => {
            let extended = format_ptr.cast::<WAVEFORMATEXTENSIBLE>();
            unsafe {
                ptr::read_unaligned(ptr::addr_of!((*extended).SubFormat))
                    == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT
            }
        }
        _ => false,
    };
    float
        && format.wBitsPerSample == 32
        && format.nSamplesPerSec == config.format.sample_rate
        && format.nChannels == config.format.channels
        && usize::from(format.nBlockAlign) == usize::from(format.nChannels) * 4
}

pub(super) fn setup_client(
    config: StreamConfig,
    flow: EDataFlow,
    device_id: Option<&str>,
) -> Result<InitializedClient, WasapiInitError> {
    let direction = if flow == eCapture {
        AudioDirection::Capture
    } else {
        AudioDirection::Render
    };
    setup_client_inner(config, flow, device_id)
        .map_err(|error| error.with_endpoint(direction, device_id))
}

fn setup_client_inner(
    config: StreamConfig,
    flow: EDataFlow,
    device_id: Option<&str>,
) -> Result<InitializedClient, WasapiInitError> {
    let enumerator: IMMDeviceEnumerator =
        unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL) }
            .map_err(|e| WasapiInitError::windows("Create MMDeviceEnumerator", e))?;
    let device = crate::wasapi_devices::select_endpoint(&enumerator, flow, device_id)?;
    let probe: IAudioClient3 = unsafe { device.Activate(CLSCTX_ALL, None) }
        .map_err(|e| WasapiInitError::windows("Activate IAudioClient3", e))?;
    let mix = CoTaskMemFormat(
        unsafe { probe.GetMixFormat() }.map_err(|e| WasapiInitError::windows("GetMixFormat", e))?,
    );
    if mix.0.is_null() || !unsafe { matches_format(mix.0, config) } {
        return Err(WasapiInitError::windows(
            "Validate native mix format",
            Error::from_hresult(AUDCLNT_E_UNSUPPORTED_FORMAT),
        ));
    }
    let (mut default, mut fundamental, mut min, mut max) = (0, 0, 0, 0);
    unsafe {
        probe.GetSharedModeEnginePeriod(mix.0, &mut default, &mut fundamental, &mut min, &mut max)
    }
    .map_err(|e| WasapiInitError::windows("GetSharedModeEnginePeriod", e))?;
    let preferred = choose_period(config.block_frames, fundamental, min, max)
            .ok_or_else(|| WasapiInitError::from(format!("Invalid engine period range: default={default}, fundamental={fundamental}, min={min}, max={max}")))?;
    drop(probe);
    let (client, period) = negotiate_shared(preferred, default, |attempt| {
        let client3: IAudioClient3 = unsafe { device.Activate(CLSCTX_ALL, None) }
            .map_err(|e| WasapiInitError::windows("Activate shared audio client", e))?;
        let mut period = match attempt {
            SharedAttempt::Preferred(p) | SharedAttempt::Default(p) => p,
            SharedAttempt::Current => 0,
        };
        let current = if attempt == SharedAttempt::Current {
            let mut format_ptr = ptr::null_mut();
            // Keep ownership even if a failing driver returned a buffer.
            let result =
                unsafe { client3.GetCurrentSharedModeEnginePeriod(&mut format_ptr, &mut period) };
            let format = CoTaskMemFormat(format_ptr);
            result.map_err(|e| WasapiInitError::windows("GetCurrentSharedModeEnginePeriod", e))?;
            if format.0.is_null() || period == 0 || !unsafe { matches_format(format.0, config) } {
                return Err(WasapiInitError::windows(
                    "Validate current engine format",
                    Error::from_hresult(AUDCLNT_E_UNSUPPORTED_FORMAT),
                ));
            }
            Some(format)
        } else {
            None
        };
        let format_ptr = current.as_ref().map_or(mix.0, |format| format.0);
        unsafe {
            client3.InitializeSharedAudioStream(
                AUDCLNT_STREAMFLAGS_EVENTCALLBACK,
                period,
                format_ptr,
                None,
            )
        }
        .map_err(|e| {
            WasapiInitError::windows(
                format!("InitializeSharedAudioStream(period={period} frames, {attempt:?})"),
                e,
            )
        })?;
        let client: IAudioClient = client3
            .cast()
            .map_err(|e| WasapiInitError::windows("Query IAudioClient", e))?;
        Ok((client, period))
    })?;
    if period != preferred {
        tracing::warn!(target: "vocal.audio", ?flow, preferred, period, "using negotiated shared audio period");
    }
    let clock = unsafe { client.GetService::<IAudioClock>().ok() };
    let event = EventHandle(
        unsafe { CreateEventW(None, false, false, None) }
            .map_err(|e| WasapiInitError::windows("Create audio event", e))?,
    );
    unsafe { client.SetEventHandle(event.0) }
        .map_err(|e| WasapiInitError::windows("SetEventHandle", e))?;
    let endpoint_frames = unsafe { client.GetBufferSize() }
        .map_err(|e| WasapiInitError::windows("GetBufferSize", e))?
        as usize;
    let stream_latency_hns = unsafe { client.GetStreamLatency().ok() };
    Ok(InitializedClient {
        client,
        clock,
        event,
        stream_latency_hns,
        block_align: config.format.channels as usize * 4,
        endpoint_frames,
        period_frames: period as usize,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    fn failure(code: windows::core::HRESULT) -> WasapiInitError {
        WasapiInitError::windows("Initialize", Error::from_hresult(code))
    }

    #[test]
    fn occupied_device_is_terminal_and_does_not_retry_or_fallback() {
        let mut attempts = Vec::new();
        let result: Result<(), _> = negotiate_shared(160, 480, |attempt| {
            attempts.push(attempt);
            Err(failure(AUDCLNT_E_DEVICE_IN_USE))
        });
        assert_eq!(attempts, [SharedAttempt::Preferred(160)]);
        let error = result.unwrap_err();
        assert_eq!(error.hresult, Some(-2004287478));
        assert!(!error.allows_fallback());
    }
    #[test]
    fn period_conflict_matches_current_engine_once() {
        let mut attempts = Vec::new();
        let result = negotiate_shared(160, 480, |attempt| {
            attempts.push(attempt);
            if attempt == SharedAttempt::Current {
                Ok(240)
            } else {
                Err(failure(AUDCLNT_E_ENGINE_PERIODICITY_LOCKED))
            }
        });
        assert_eq!(result.unwrap(), 240);
        assert_eq!(
            attempts,
            [SharedAttempt::Preferred(160), SharedAttempt::Current]
        );
    }
    #[test]
    fn invalid_period_uses_default_once_and_preserves_second_failure() {
        let mut attempts = Vec::new();
        let result: Result<(), _> = negotiate_shared(160, 480, |attempt| {
            attempts.push(attempt);
            Err(failure(if attempts.len() == 1 {
                AUDCLNT_E_INVALID_DEVICE_PERIOD
            } else {
                AUDCLNT_E_DEVICE_IN_USE
            }))
        });
        assert_eq!(
            attempts,
            [SharedAttempt::Preferred(160), SharedAttempt::Default(480)]
        );
        assert_eq!(result.unwrap_err().hresult, Some(AUDCLNT_E_DEVICE_IN_USE.0));
    }
}
