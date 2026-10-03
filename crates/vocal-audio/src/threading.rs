//! Windows real-time thread policy.

use std::ptr;

/// RAII registration with Windows MMCSS. Registration is performed on the
/// target thread and reverted on that same thread when the guard is dropped.
pub struct MmcssGuard {
    handle: windows_sys::Win32::Foundation::HANDLE,
}

impl MmcssGuard {
    pub fn register(task: &str) -> Option<Self> {
        let mut name = task.encode_utf16().collect::<Vec<_>>();
        name.push(0);
        let mut index = 0_u32;
        let handle = unsafe {
            windows_sys::Win32::System::Threading::AvSetMmThreadCharacteristicsW(
                name.as_ptr(),
                &mut index,
            )
        };
        if handle.is_null() {
            return None;
        }
        // Pro Audio already receives a suitable MMCSS budget. Raising the
        // AVRT sub-priority is intentionally left to the caller because a
        // worker and a device callback may need different policies.
        Some(Self { handle })
    }

    pub fn set_priority(
        &self,
        priority: windows_sys::Win32::System::Threading::AVRT_PRIORITY,
    ) -> bool {
        unsafe {
            windows_sys::Win32::System::Threading::AvSetMmThreadPriority(self.handle, priority) != 0
        }
    }
}

impl Drop for MmcssGuard {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            unsafe {
                windows_sys::Win32::System::Threading::AvRevertMmThreadCharacteristics(self.handle);
            }
            self.handle = ptr::null_mut();
        }
    }
}
