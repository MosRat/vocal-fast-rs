//! Control-plane discovery. Never called from an audio callback or model block.
//! Hardware support and model-specific plan readiness are separate concerns.

use serde::Serialize;

use crate::{RuntimeBackend, RuntimeResult};

#[cfg(windows)]
mod windows;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BackendAvailability {
    pub backend: RuntimeBackend,
    pub available: bool,
    pub reason: Option<String>,
    pub device: Option<String>,
    pub description: String,
    pub devices: Vec<ProcessingDevice>,
    pub recommended_device_id: Option<String>,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProcessingDevice {
    pub id: String,
    pub name: String,
    pub vendor_id: u32,
    pub dedicated_memory_bytes: u64,
    pub shared_memory_bytes: u64,
    pub available: bool,
    pub reason: Option<String>,
    pub kind: ProcessingDeviceKind,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum ProcessingDeviceKind {
    Gpu,
    Virtual,
    Remote,
    Software,
}

#[cfg(any(feature = "directml", test))]
pub(crate) fn directml_device_kind(id: &str, software: bool) -> ProcessingDeviceKind {
    let id = id.to_ascii_uppercase();
    if software {
        ProcessingDeviceKind::Software
    } else if id.contains("REMOTEDISPLAYENUM") {
        ProcessingDeviceKind::Remote
    } else if id.contains("ROOT#DISPLAY") {
        ProcessingDeviceKind::Virtual
    } else {
        ProcessingDeviceKind::Gpu
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProcessingBackendCatalog {
    pub recommended: RuntimeBackend,
    pub backends: Vec<BackendAvailability>,
    pub cpu: CpuCapabilities,
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct CpuCapabilities {
    pub logical_processors: usize,
    pub recommended_threads: usize,
    pub operator_version: Option<String>,
}

impl Default for CpuCapabilities {
    fn default() -> Self {
        Self {
            logical_processors: std::thread::available_parallelism().map_or(1, |n| n.get()),
            recommended_threads: vocal_engine::CpuExecutionConfig::recommended_threads(),
            operator_version: None,
        }
    }
}

impl ProcessingBackendCatalog {
    pub fn ensure_available(&self, backend: RuntimeBackend) -> RuntimeResult<()> {
        let entry = self.backends.iter().find(|entry| entry.backend == backend);
        if entry.is_some_and(|entry| entry.available) {
            return Ok(());
        }
        Err(vocal_engine::EngineError::BackendUnavailable(
            entry
                .and_then(|entry| entry.reason.clone())
                .unwrap_or_else(|| format!("{backend:?} is not supported by this application")),
        )
        .into())
    }
}

/// The engine currently uses CUDA device zero. Probe that same device, rather
/// than recommending a second GPU the engine would never actually select.
pub fn detect_processing_backends() -> ProcessingBackendCatalog {
    let mut result = tensor_rt_catalog();
    #[cfg(feature = "directml")]
    let dml = vocal_directml::enumerate_adapters()
        .map(|adapters| {
            adapters
                .into_iter()
                .map(|a| {
                    let kind = directml_device_kind(&a.id, a.software);
                    ProcessingDevice {
                        id: a.id,
                        name: a.name,
                        vendor_id: a.vendor_id,
                        dedicated_memory_bytes: a.dedicated_memory_bytes,
                        shared_memory_bytes: a.shared_memory_bytes,
                        available: a.available,
                        reason: a.reason,
                        kind,
                    }
                })
                .collect()
        })
        .map_err(|e| e.to_string());
    #[cfg(not(feature = "directml"))]
    let dml: Result<Vec<ProcessingDevice>, String> =
        Err("This application was built without DirectML support".into());
    add_directml(&mut result, dml);
    result
}

/// Targeted preparation probe: no D3D12/DirectML devices are created when only
/// checking a TensorRT plan cache. Full discovery remains available for the UI.
pub fn detect_tensor_rt_backend() -> BackendAvailability {
    tensor_rt_catalog()
        .backends
        .into_iter()
        .find(|entry| entry.backend == RuntimeBackend::TensorRtRtx)
        .expect("catalog always contains the TensorRT entry")
}

fn tensor_rt_catalog() -> ProcessingBackendCatalog {
    #[cfg(windows)]
    let gpu = windows::probe;
    #[cfg(not(windows))]
    let gpu = || Err("GPU discovery is currently supported on Windows only".into());
    catalog(cfg!(feature = "tensorrt-rtx"), gpu)
}

fn add_directml(result: &mut ProcessingBackendCatalog, dml: Result<Vec<ProcessingDevice>, String>) {
    let (devices, reason) = match dml {
        Ok(devices) => {
            let reason = (!devices.iter().any(|d| d.available)).then(|| "No compatible DirectML GPU found. A D3D12 device with DirectML feature level 5.0 is required.".to_owned());
            (devices, reason)
        }
        Err(r) => (Vec::new(), Some(r)),
    };
    let recommended = devices
        .iter()
        .find(|d| d.available && d.kind == ProcessingDeviceKind::Gpu)
        .or_else(|| devices.iter().find(|d| d.available));
    let device = recommended.map(|d| d.name.clone());
    let recommended_device_id = recommended.map(|d| d.id.clone());
    // Prefer the mature RTX path when available; a real native DML backend is
    // the GPU fallback when this build/device cannot run TensorRT.
    if devices
        .iter()
        .any(|d| d.available && d.kind == ProcessingDeviceKind::Gpu)
        && result.recommended != RuntimeBackend::TensorRtRtx
    {
        result.recommended = RuntimeBackend::DirectMl;
    }
    result.backends.push(BackendAvailability {
        backend: RuntimeBackend::DirectMl,
        available: reason.is_none(),
        device,
        reason,
        description: "Native DirectML voice conversion from canonical VFRM weights".into(),
        devices,
        recommended_device_id,
    });
}

fn catalog(
    compiled: bool,
    probe: impl FnOnce() -> Result<String, String>,
) -> ProcessingBackendCatalog {
    let result = if compiled {
        probe()
    } else {
        Err("This application was built without TensorRT RTX support".into())
    };
    let recommended = if result.is_ok() {
        RuntimeBackend::TensorRtRtx
    } else {
        cpu_fallback()
    };
    let (device, reason) = match result {
        Ok(device) => (Some(device), None),
        Err(reason) => (None, Some(reason)),
    };
    ProcessingBackendCatalog {
        recommended,
        cpu: CpuCapabilities {
            logical_processors: std::thread::available_parallelism().map_or(1, |n| n.get()),
            recommended_threads: vocal_engine::CpuExecutionConfig::recommended_threads(),
            operator_version: cpu_operator_version(),
        },
        backends: vec![
            BackendAvailability {
                backend: RuntimeBackend::TensorRtRtx,
                available: reason.is_none(),
                reason,
                device,
                description: "GPU voice conversion; requires model-specific TensorRT plans".into(),
                devices: Vec::new(),
                recommended_device_id: None,
            },
            BackendAvailability {
                backend: RuntimeBackend::RvcCpu,
                available: cfg!(feature = "cpu-graph"),
                reason: (!cfg!(feature = "cpu-graph")).then(|| "This application was built without pretrained CPU inference support".into()),
                device: Some("CPU".into()),
                description: if cfg!(feature = "cpu-graph") {
                    "Native FP32 voice conversion with oneDNN; supports complete RVC v1 / 256-feature and v2 / 768-feature weights. No GPU or GPU driver required."
                } else {
                    "Install a package with CPU inference support"
                }
                .into(),
                devices: Vec::new(),
                recommended_device_id: None,
            },
            BackendAvailability {
                backend: RuntimeBackend::Passthrough,
                available: true,
                reason: None,
                device: Some("CPU".into()),
                description: "Audio monitoring without voice conversion".into(),
                devices: Vec::new(),
                recommended_device_id: None,
            },
        ],
    }
}

fn cpu_fallback() -> RuntimeBackend {
    if cfg!(feature = "cpu-graph") {
        RuntimeBackend::RvcCpu
    } else {
        RuntimeBackend::Passthrough
    }
}

fn cpu_operator_version() -> Option<String> {
    #[cfg(feature = "cpu-graph")]
    {
        Some(vocal_engine::cpu_operator_version())
    }
    #[cfg(not(feature = "cpu-graph"))]
    {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn missing_build_support_does_not_touch_gpu_libraries() {
        let result = catalog(false, || panic!("probe must not be called"));
        assert_eq!(result.recommended, cpu_fallback());
        assert!(
            result
                .ensure_available(RuntimeBackend::TensorRtRtx)
                .is_err()
        );
        assert_eq!(
            result.ensure_available(RuntimeBackend::RvcCpu).is_ok(),
            cfg!(feature = "cpu-graph")
        );
    }

    #[test]
    fn driver_device_and_runtime_failures_leave_safe_options_available() {
        for failure in [
            "driver missing",
            "no CUDA device",
            "unsupported GPU",
            "CUDA runtime mismatch",
        ] {
            let result = catalog(true, || Err(failure.into()));
            assert_eq!(result.recommended, cpu_fallback());
            assert_eq!(result.backends[0].reason.as_deref(), Some(failure));
            assert!(result.ensure_available(RuntimeBackend::Passthrough).is_ok());
            assert!(
                result
                    .ensure_available(RuntimeBackend::TensorRtRtx)
                    .is_err()
            );
        }
    }

    #[test]
    fn usable_gpu_is_recommended_even_before_model_plans_exist() {
        let result = catalog(true, || Ok("RTX test device".into()));
        assert_eq!(result.recommended, RuntimeBackend::TensorRtRtx);
        assert!(result.ensure_available(RuntimeBackend::TensorRtRtx).is_ok());
    }

    #[test]
    fn directml_is_the_gpu_fallback_and_preserves_rtx_preference() {
        let mut fallback = catalog(false, || unreachable!());
        add_directml(&mut fallback, Ok(vec![device("test-gpu", true)]));
        assert_eq!(fallback.recommended, RuntimeBackend::DirectMl);
        assert!(fallback.ensure_available(RuntimeBackend::DirectMl).is_ok());
        let mut rtx = catalog(true, || Ok("RTX adapter".into()));
        add_directml(&mut rtx, Ok(vec![device("test-gpu", true)]));
        assert_eq!(rtx.recommended, RuntimeBackend::TensorRtRtx);
        let mut unavailable = catalog(false, || unreachable!());
        add_directml(&mut unavailable, Err("unsupported DirectML adapter".into()));
        assert_eq!(unavailable.recommended, cpu_fallback());
        assert!(
            unavailable
                .ensure_available(RuntimeBackend::DirectMl)
                .is_err()
        );
    }

    fn device(id: &str, available: bool) -> ProcessingDevice {
        ProcessingDevice {
            id: id.into(),
            name: id.into(),
            vendor_id: 0,
            dedicated_memory_bytes: 0,
            shared_memory_bytes: 0,
            available,
            reason: (!available).then(|| "Unsupported GPU".into()),
            kind: ProcessingDeviceKind::Gpu,
        }
    }

    #[test]
    fn recommendation_skips_unsupported_first_adapter_and_preserves_devices() {
        let mut result = catalog(false, || unreachable!());
        add_directml(
            &mut result,
            Ok(vec![
                device("unsupported", false),
                device("second-gpu", true),
            ]),
        );
        let dml = result.backends.last().unwrap();
        assert_eq!(dml.recommended_device_id.as_deref(), Some("second-gpu"));
        assert_eq!(dml.devices.len(), 2);
        assert!(dml.available);
        let mut empty = catalog(false, || unreachable!());
        add_directml(&mut empty, Ok(Vec::new()));
        assert!(!empty.backends.last().unwrap().available);
        let mut virtual_first = catalog(false, || unreachable!());
        let mut remote = device("remote-session", true);
        remote.kind = ProcessingDeviceKind::Remote;
        add_directml(
            &mut virtual_first,
            Ok(vec![remote, device("physical-gpu", true)]),
        );
        assert_eq!(
            virtual_first
                .backends
                .last()
                .unwrap()
                .recommended_device_id
                .as_deref(),
            Some("physical-gpu")
        );
        assert_eq!(
            directml_device_kind("dxgi:root#display#0001", false),
            ProcessingDeviceKind::Virtual
        );
        assert_eq!(
            directml_device_kind("dxgi:swd#remotedisplayenum#session", false),
            ProcessingDeviceKind::Remote
        );
        assert_eq!(
            directml_device_kind("dxgi-luid:software", true),
            ProcessingDeviceKind::Software
        );
    }
}
