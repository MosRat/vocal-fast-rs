//! Actionable startup reports retain device and OS context. A failed device
//! open does not invalidate the resident inference engine.
use serde::Serialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum AudioStartFailure {
    DeviceBusy,
    DeviceUnavailable,
    PermissionDenied,
    ServiceUnavailable,
    UnsupportedFormat,
    Initialization,
}

impl AudioStartFailure {
    pub fn code(self) -> &'static str {
        match self {
            Self::DeviceBusy => "audio_device_busy",
            Self::DeviceUnavailable => "audio_device_unavailable",
            Self::PermissionDenied => "audio_permission_denied",
            Self::ServiceUnavailable => "audio_service_unavailable",
            Self::UnsupportedFormat => "audio_format_unsupported",
            Self::Initialization => "audio_start_failed",
        }
    }
}

#[derive(Clone, Debug, Serialize, thiserror::Error)]
#[serde(rename_all = "camelCase")]
#[error("{message}")]
pub struct AudioStartError {
    pub kind: AudioStartFailure,
    pub message: String,
    pub backend: String,
    pub direction: Option<String>,
    pub device_id: Option<String>,
    pub device_name: Option<String>,
    pub hresult: Option<String>,
    pub technical_details: Vec<String>,
    pub suggestions: Vec<String>,
}

impl AudioStartError {
    fn new(
        kind: AudioStartFailure,
        backend: &str,
        direction: Option<&str>,
        detail: String,
    ) -> Self {
        let endpoint = match direction {
            Some("capture") => "input device",
            Some("render") => "output device",
            _ => "audio device",
        };
        let (message, suggestions) = match kind {
            AudioStartFailure::DeviceBusy => (
                format!("The {endpoint} is already in use, usually by an exclusive audio session."),
                vec![
                    "Stop recording or exclusive playback in other applications, then retry Start.",
                    "Choose another device in Audio routing. If needed, turn off exclusive mode for this device in Windows Sound settings.",
                ],
            ),
            AudioStartFailure::DeviceUnavailable => (
                format!("The {endpoint} is unavailable."),
                vec![
                    "Reconnect or enable the device, refresh Audio routing, then select an available device.",
                ],
            ),
            AudioStartFailure::PermissionDenied => (
                format!("Windows denied access to the {endpoint}."),
                vec![
                    "Allow desktop apps to access the microphone in Windows Privacy & security settings, then retry Start.",
                ],
            ),
            AudioStartFailure::ServiceUnavailable => (
                "The Windows audio service is unavailable.".into(),
                vec!["Start the Windows Audio service, then refresh devices and retry Start."],
            ),
            AudioStartFailure::UnsupportedFormat => (
                format!("The {endpoint} could not provide a supported audio format."),
                vec!["Choose another device or check its format in Windows Sound settings."],
            ),
            AudioStartFailure::Initialization => (
                "Unable to initialize the audio session.".into(),
                vec![
                    "Refresh Audio routing and check the selected input and output devices before retrying Start.",
                ],
            ),
        };
        Self {
            kind,
            message,
            backend: backend.into(),
            direction: direction.map(str::to_owned),
            device_id: None,
            device_name: None,
            hresult: None,
            technical_details: vec![detail],
            suggestions: suggestions.into_iter().map(str::to_owned).collect(),
        }
    }

    pub fn allows_fallback(&self) -> bool {
        matches!(
            self.kind,
            AudioStartFailure::UnsupportedFormat | AudioStartFailure::Initialization
        )
    }

    pub fn unavailable(direction: &str, device_id: Option<&str>, detail: String) -> Self {
        let mut error = Self::new(
            AudioStartFailure::DeviceUnavailable,
            "device discovery",
            Some(direction),
            detail,
        );
        error.device_id = device_id.map(str::to_owned);
        error
    }

    #[cfg(windows)]
    pub fn native(error: vocal_audio::wasapi_backend::WasapiBackendError) -> Self {
        use vocal_audio::wasapi_backend::{NativeWasapiError, WasapiBackendError};
        let detail = error.to_string();
        let WasapiBackendError::Native(NativeWasapiError::Startup(failure)) = error else {
            return Self::from(detail);
        };
        let kind = match failure.hresult.map(|code| code as u32) {
            Some(0x8889000A) => AudioStartFailure::DeviceBusy,
            Some(0x88890004 | 0x80070490) => AudioStartFailure::DeviceUnavailable,
            Some(0x80070005) => AudioStartFailure::PermissionDenied,
            Some(0x88890010) => AudioStartFailure::ServiceUnavailable,
            Some(0x88890008) => AudioStartFailure::UnsupportedFormat,
            _ => AudioStartFailure::Initialization,
        };
        let direction = failure.direction.map(|direction| match direction {
            vocal_audio::AudioDirection::Capture => "capture",
            vocal_audio::AudioDirection::Render => "render",
        });
        let mut report = Self::new(kind, "nativeWasapi", direction, detail);
        report.device_id = failure.device_id;
        report.hresult = failure.hresult.map(|code| format!("0x{:08X}", code as u32));
        report
    }

    pub fn cpal(error: vocal_audio::cpal_backend::CpalError) -> Self {
        use vocal_audio::cpal_backend::CpalError;
        let (direction, cause) = match &error {
            CpalError::BuildInput(e) | CpalError::PlayInput(e) | CpalError::InputConfig(e) => {
                (Some("capture"), Some(e))
            }
            CpalError::BuildOutput(e) | CpalError::PlayOutput(e) | CpalError::OutputConfig(e) => {
                (Some("render"), Some(e))
            }
            CpalError::DeviceFormatMismatch {
                direction: "input", ..
            } => (Some("capture"), None),
            CpalError::DeviceFormatMismatch {
                direction: "output",
                ..
            } => (Some("render"), None),
            _ => (None, None),
        };
        let kind = match cause.map(cpal::Error::kind) {
            Some(cpal::ErrorKind::DeviceBusy) => AudioStartFailure::DeviceBusy,
            Some(cpal::ErrorKind::DeviceNotAvailable) => AudioStartFailure::DeviceUnavailable,
            Some(cpal::ErrorKind::PermissionDenied) => AudioStartFailure::PermissionDenied,
            Some(cpal::ErrorKind::HostUnavailable) => AudioStartFailure::ServiceUnavailable,
            Some(cpal::ErrorKind::UnsupportedConfig) => AudioStartFailure::UnsupportedFormat,
            _ if matches!(
                error,
                CpalError::UnsupportedSampleFormat { .. } | CpalError::DeviceFormatMismatch { .. }
            ) =>
            {
                AudioStartFailure::UnsupportedFormat
            }
            _ => AudioStartFailure::Initialization,
        };
        Self::new(kind, "cpal", direction, error.to_string())
    }
}

impl From<String> for AudioStartError {
    fn from(detail: String) -> Self {
        Self::new(AudioStartFailure::Initialization, "audio", None, detail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn occupied_native_device_has_a_clear_report_and_does_not_fallback() {
        let cause = vocal_audio::wasapi_backend::WasapiInitError {
            operation: "InitializeSharedAudioStream(period=160)".into(),
            hresult: Some(-2004287478),
            direction: Some(vocal_audio::AudioDirection::Capture),
            device_id: Some("mic".into()),
            message: "device in use".into(),
        };
        let error =
            AudioStartError::native(vocal_audio::wasapi_backend::WasapiBackendError::Native(
                vocal_audio::wasapi_backend::NativeWasapiError::Startup(cause),
            ));
        assert_eq!(error.kind, AudioStartFailure::DeviceBusy);
        assert_eq!(error.hresult.as_deref(), Some("0x8889000A"));
        assert_eq!(error.direction.as_deref(), Some("capture"));
        assert!(!error.allows_fallback());
        assert!(error.message.contains("input device"));
        assert!(!error.suggestions.is_empty());
    }
    #[test]
    fn cpal_busy_and_permission_errors_remain_typed() {
        for (source, expected) in [
            (cpal::ErrorKind::DeviceBusy, AudioStartFailure::DeviceBusy),
            (
                cpal::ErrorKind::PermissionDenied,
                AudioStartFailure::PermissionDenied,
            ),
        ] {
            let error = AudioStartError::cpal(vocal_audio::cpal_backend::CpalError::BuildInput(
                cpal::Error::new(source),
            ));
            assert_eq!(error.kind, expected);
            assert_eq!(error.direction.as_deref(), Some("capture"));
        }
    }

    #[test]
    fn missing_default_endpoint_is_unavailable_without_fallback() {
        let cause = vocal_audio::wasapi_backend::WasapiInitError {
            operation: "GetDefaultAudioEndpoint".into(),
            hresult: Some(0x80070490u32 as i32),
            direction: Some(vocal_audio::AudioDirection::Capture),
            device_id: None,
            message: "element not found".into(),
        };
        assert!(!cause.allows_fallback());
        let error =
            AudioStartError::native(vocal_audio::wasapi_backend::WasapiBackendError::Native(
                vocal_audio::wasapi_backend::NativeWasapiError::Startup(cause),
            ));
        assert_eq!(error.kind, AudioStartFailure::DeviceUnavailable);
        assert_eq!(error.direction.as_deref(), Some("capture"));
        assert!(!error.allows_fallback());
    }

    #[test]
    fn cpal_config_errors_retain_direction_and_classification() {
        let error = AudioStartError::cpal(vocal_audio::cpal_backend::CpalError::OutputConfig(
            cpal::Error::new(cpal::ErrorKind::UnsupportedConfig),
        ));
        assert_eq!(error.kind, AudioStartFailure::UnsupportedFormat);
        assert_eq!(error.direction.as_deref(), Some("render"));
        assert!(error.allows_fallback());
    }
}
