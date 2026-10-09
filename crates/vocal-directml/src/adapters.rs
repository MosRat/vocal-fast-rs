//! Device discovery belongs to the control plane; it never compiles a graph.
use serde::Serialize;

use crate::DirectMlError;

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DirectMlAdapter {
    pub id: String,
    pub name: String,
    pub adapter_index: u32,
    pub vendor_id: u32,
    pub dedicated_memory_bytes: u64,
    pub shared_memory_bytes: u64,
    pub available: bool,
    pub software: bool,
    pub reason: Option<String>,
}

/// Returns DXGI's high-performance order, including unsupported adapters so
/// controllers can explain why a listed device cannot be selected.
pub fn enumerate_adapters() -> Result<Vec<DirectMlAdapter>, DirectMlError> {
    #[cfg(windows)]
    {
        use crate::{ffi, runtime::check};
        use std::{
            ffi::{CStr, c_void},
            ptr,
        };

        struct List(*mut c_void);
        impl Drop for List {
            fn drop(&mut self) {
                unsafe { ffi::vdml_adapters_free(self.0) };
            }
        }
        let (mut raw, mut error) = (ptr::null_mut(), ptr::null_mut());
        check(
            unsafe { ffi::vdml_adapters_enumerate(&mut raw, &mut error) },
            error,
        )?;
        let list = List(raw);
        let count = unsafe { ffi::vdml_adapters_len(list.0) };
        let mut result = Vec::with_capacity(count);
        for index in 0..count {
            let mut desc = ffi::AdapterDesc::default();
            unsafe { ffi::vdml_adapters_get(list.0, index, &mut desc) };
            // The native list owns these terminated strings until List drops.
            let string = |p| unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
            let reason = string(desc.reason);
            result.push(DirectMlAdapter {
                id: string(desc.id),
                name: string(desc.name),
                adapter_index: desc.index,
                vendor_id: desc.vendor_id,
                dedicated_memory_bytes: desc.dedicated_memory,
                shared_memory_bytes: desc.shared_memory,
                available: desc.available != 0,
                software: desc.software != 0,
                reason: (!reason.is_empty()).then_some(reason),
            });
        }
        Ok(result)
    }
    #[cfg(not(windows))]
    Err(DirectMlError::Runtime("DirectML requires Windows".into()))
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use crate::DirectMlDevice;

    #[test]
    #[ignore = "requires a DirectML hardware adapter"]
    fn selected_identity_overrides_index_and_missing_devices_never_fall_back() {
        let adapters = enumerate_adapters().unwrap();
        let gpu = adapters
            .iter()
            .find(|a| a.available)
            .expect("compatible GPU");
        let selected = DirectMlDevice::with_adapter(u32::MAX, Some(&gpu.id), false).unwrap();
        assert_eq!(selected.adapter_name(), gpu.name);
        assert!(
            DirectMlDevice::with_adapter(gpu.adapter_index, Some("dxgi:missing-device"), false)
                .is_err()
        );
        if let Some(unsupported) = adapters.iter().find(|a| !a.available) {
            assert!(DirectMlDevice::with_adapter(0, Some(&unsupported.id), false).is_err());
        }
    }
}
