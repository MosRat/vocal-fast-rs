//! Control-plane endpoint discovery. Only owned IDs and PCM metadata leave
//! this COM thread; capture/render clients are created on their owning threads.
use crate::{AudioDirection, StreamConfig, wasapi_native::WasapiInitError};
use std::ptr;
use vocal_types::AudioFormat;
use windows::Win32::Media::Audio::{
    EDataFlow, IAudioClient, IMMDevice, IMMDeviceEnumerator, IMMEndpoint, MMDeviceEnumerator,
    WAVEFORMATEX, WAVEFORMATEXTENSIBLE, eCapture, eConsole, eRender,
};
use windows::Win32::Media::KernelStreaming::WAVE_FORMAT_EXTENSIBLE;
use windows::Win32::Media::Multimedia::{KSDATAFORMAT_SUBTYPE_IEEE_FLOAT, WAVE_FORMAT_IEEE_FLOAT};
use windows::Win32::System::Com::{
    CLSCTX_ALL, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoTaskMemFree,
    CoUninitialize,
};
use windows::core::{Error, HRESULT, Interface, PCWSTR};

fn direction(flow: EDataFlow) -> AudioDirection {
    if flow == eCapture {
        AudioDirection::Capture
    } else {
        AudioDirection::Render
    }
}

pub(super) struct EndpointPair {
    pub capture: StreamConfig,
    pub render: StreamConfig,
    pub capture_id: String,
    pub render_id: String,
}

pub(super) struct FormatMemory(pub *mut WAVEFORMATEX);
impl Drop for FormatMemory {
    fn drop(&mut self) {
        unsafe { CoTaskMemFree(Some(self.0.cast())) };
    }
}

pub(super) fn select_endpoint(
    enumerator: &IMMDeviceEnumerator,
    flow: EDataFlow,
    id: Option<&str>,
) -> Result<IMMDevice, WasapiInitError> {
    let operation = if id.is_some() {
        "GetDevice"
    } else {
        "GetDefaultAudioEndpoint"
    };
    let device = match id {
        Some(id) => {
            let wide: Vec<_> = id.encode_utf16().chain(Some(0)).collect();
            unsafe { enumerator.GetDevice(PCWSTR(wide.as_ptr())) }
        }
        None => unsafe { enumerator.GetDefaultAudioEndpoint(flow, eConsole) },
    }
    .map_err(|e| WasapiInitError::windows(operation, e).with_endpoint(direction(flow), id))?;
    let endpoint: IMMEndpoint = device.cast().map_err(|e| {
        WasapiInitError::windows("Query IMMEndpoint", e).with_endpoint(direction(flow), id)
    })?;
    if unsafe { endpoint.GetDataFlow() }.map_err(|e| {
        WasapiInitError::windows("GetDataFlow", e).with_endpoint(direction(flow), id)
    })? != flow
    {
        return Err(WasapiInitError::windows(
            "Selected endpoint has the wrong capture/render direction",
            Error::from_hresult(HRESULT(0x80070490u32 as i32)),
        )
        .with_endpoint(direction(flow), id));
    }
    Ok(device)
}

fn probe_one(
    enumerator: &IMMDeviceEnumerator,
    flow: EDataFlow,
    id: Option<&str>,
    reference: StreamConfig,
) -> Result<(StreamConfig, String), WasapiInitError> {
    let device = select_endpoint(enumerator, flow, id)?;
    let native_id = unsafe { device.GetId() }
        .map_err(|e| WasapiInitError::windows("GetId", e).with_endpoint(direction(flow), id))?;
    let result = unsafe { native_id.to_string() }.map_err(|e| {
        WasapiInitError::windows("Decode endpoint ID", e.into()).with_endpoint(direction(flow), id)
    });
    unsafe { CoTaskMemFree(Some(native_id.0.cast())) };
    let native_id = result?;
    let contextualize =
        |error: WasapiInitError| error.with_endpoint(direction(flow), Some(&native_id));
    let client: IAudioClient = unsafe { device.Activate(CLSCTX_ALL, None) }
        .map_err(|e| contextualize(WasapiInitError::windows("Activate format probe", e)))?;
    let memory = FormatMemory(
        unsafe { client.GetMixFormat() }
            .map_err(|e| contextualize(WasapiInitError::windows("GetMixFormat", e)))?,
    );
    if memory.0.is_null() {
        return Err(contextualize(
            "GetMixFormat returned no format".to_owned().into(),
        ));
    }
    let format = unsafe { ptr::read_unaligned(memory.0) };
    let float = if format.wFormatTag as u32 == WAVE_FORMAT_EXTENSIBLE {
        let extended = memory.0.cast::<WAVEFORMATEXTENSIBLE>();
        format.cbSize >= 22
            && unsafe { ptr::read_unaligned(ptr::addr_of!((*extended).SubFormat)) }
                == KSDATAFORMAT_SUBTYPE_IEEE_FLOAT
    } else {
        format.wFormatTag as u32 == WAVE_FORMAT_IEEE_FLOAT
    };
    if !float
        || format.wBitsPerSample != 32
        || usize::from(format.nBlockAlign) != usize::from(format.nChannels) * 4
    {
        let bits = format.wBitsPerSample;
        let tag = format.wFormatTag;
        return Err(contextualize(WasapiInitError::windows(
            format!("Native endpoint mix format requires f32; got {bits} bits, tag {tag}"),
            Error::from_hresult(windows::Win32::Media::Audio::AUDCLNT_E_UNSUPPORTED_FORMAT),
        )));
    }
    let config = reference
        .for_endpoint(AudioFormat {
            sample_rate: format.nSamplesPerSec,
            channels: format.nChannels,
        })
        .map_err(|e| contextualize(e.to_string().into()))?;
    Ok((config, native_id))
}

pub(super) fn probe(
    reference: StreamConfig,
    capture_id: Option<String>,
    render_id: Option<String>,
) -> Result<EndpointPair, WasapiInitError> {
    std::thread::Builder::new()
        .name("vocal-endpoint-probe".into())
        .spawn(move || {
            unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) }
                .ok()
                .map_err(|e| WasapiInitError::windows("Initialize endpoint probe COM", e))?;
            let result = (|| {
                let enumerator: IMMDeviceEnumerator =
                    unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL) }
                        .map_err(|e| WasapiInitError::windows("Create MMDeviceEnumerator", e))?;
                let (capture, capture_id) =
                    probe_one(&enumerator, eCapture, capture_id.as_deref(), reference)?;
                let (render, render_id) =
                    probe_one(&enumerator, eRender, render_id.as_deref(), reference)?;
                Ok(EndpointPair {
                    capture,
                    render,
                    capture_id,
                    render_id,
                })
            })();
            unsafe { CoUninitialize() };
            result
        })
        .map_err(|e| WasapiInitError::from(format!("cannot spawn endpoint probe: {e}")))?
        .join()
        .map_err(|_| WasapiInitError::from("endpoint probe thread panicked".to_owned()))?
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[ignore = "requires a live Windows render endpoint"]
    fn live_render_format_and_explicit_id_are_independent_of_worker_format() {
        unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) }
            .ok()
            .unwrap();
        let result = (|| {
            let enumerator: IMMDeviceEnumerator =
                unsafe { CoCreateInstance(&MMDeviceEnumerator, None, CLSCTX_ALL) }.unwrap();
            let reference = StreamConfig {
                format: AudioFormat::MONO_16K,
                block_frames: 160,
                queue_blocks: 2,
            };
            let (native, id) = probe_one(&enumerator, eRender, None, reference)?;
            let (explicit, _) = probe_one(&enumerator, eRender, Some(&id), reference)?;
            assert_eq!(native, explicit);
            assert!(probe_one(&enumerator, eCapture, Some(&id), reference).is_err());
            println!(
                "worker=16000 Hz/1 ch; render={} Hz/{} ch/{} frames; explicit endpoint ID verified",
                native.format.sample_rate, native.format.channels, native.block_frames
            );
            Ok::<_, WasapiInitError>(())
        })();
        unsafe { CoUninitialize() };
        result.unwrap();
    }
}
