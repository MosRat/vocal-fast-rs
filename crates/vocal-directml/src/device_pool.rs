//! Control-plane reuse only. Weak entries never keep an unloaded GPU alive.
use super::{DeviceHandle, DirectMlDevice};
use crate::{DirectMlConfig, DirectMlError};
use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock, Weak},
};

type Key = (String, bool);
static DEVICES: OnceLock<Mutex<HashMap<Key, Weak<DeviceHandle>>>> = OnceLock::new();

pub(super) fn acquire(config: &DirectMlConfig) -> Result<DirectMlDevice, DirectMlError> {
    // Resolving an index here avoids reusing the wrong device if DXGI's order
    // changes. Persisted IDs are preferred throughout the GUI/runtime path.
    let id = if let Some(id) = &config.adapter_id {
        id.clone()
    } else {
        crate::enumerate_adapters()?
            .into_iter()
            .find(|a| a.adapter_index == config.adapter_index)
            .ok_or_else(|| DirectMlError::Config("DirectML adapter index is unavailable".into()))?
            .id
    };
    let key = (id.to_ascii_lowercase(), config.debug_layer);
    let mut cache = DEVICES
        .get_or_init(Default::default)
        .lock()
        .map_err(|_| DirectMlError::Runtime("DirectML device cache lock poisoned".into()))?;
    cache.retain(|_, device| device.strong_count() != 0);
    if let Some(device) = cache.get(&key).and_then(Weak::upgrade) {
        #[cfg(windows)]
        {
            let mut error = std::ptr::null_mut();
            let status = super::check(
                unsafe { crate::ffi::vdml_device_status(device.raw, &mut error) },
                error,
            );
            if status.is_ok() {
                tracing::debug!(adapter = %device.name, "reusing DirectML device");
                return Ok(DirectMlDevice(device, config.execution.clone()));
            }
            tracing::warn!(?status, "replacing removed DirectML device");
        }
    }
    let device = DirectMlDevice::with_adapter(config.adapter_index, Some(&id), config.debug_layer)?;
    cache.insert(key, std::sync::Arc::downgrade(&device.0));
    Ok(device)
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;

    #[test]
    #[ignore = "requires a DirectML hardware adapter"]
    fn reuse_shares_the_device_but_separates_debug_policy_and_uses_weak_ownership() {
        let config = DirectMlConfig::default();
        let first = acquire(&config).unwrap();
        let second = acquire(&config).unwrap();
        assert!(std::sync::Arc::ptr_eq(&first.0, &second.0));
        let weak = std::sync::Arc::downgrade(&first.0);
        let debug = acquire(&DirectMlConfig {
            debug_layer: true,
            ..config
        })
        .unwrap();
        assert!(!std::sync::Arc::ptr_eq(&first.0, &debug.0));
        drop((first, second));
        assert!(weak.upgrade().is_none());
    }
}
