use std::ffi::{CStr, c_char};

use windows_sys::Win32::{
    Foundation::{FreeLibrary, HMODULE},
    System::LibraryLoader::{GetProcAddress, LOAD_LIBRARY_SEARCH_SYSTEM32, LoadLibraryExW},
};

struct Driver(HMODULE);

impl Drop for Driver {
    fn drop(&mut self) {
        // All function pointers are used synchronously while this guard lives.
        unsafe { FreeLibrary(self.0) };
    }
}

fn check(status: i32, action: &str) -> Result<(), String> {
    if status == 0 {
        Ok(())
    } else {
        Err(format!(
            "{action} failed (CUDA status {status}); check the NVIDIA driver and device"
        ))
    }
}

pub(super) fn probe() -> Result<String, String> {
    let name: Vec<u16> = "nvcuda.dll\0".encode_utf16().collect();
    // Use the Windows driver, never a DLL from a model/library directory.
    let module = unsafe {
        LoadLibraryExW(
            name.as_ptr(),
            std::ptr::null_mut(),
            LOAD_LIBRARY_SEARCH_SYSTEM32,
        )
    };
    if module.is_null() {
        return Err("NVIDIA CUDA driver is not installed or cannot be loaded".into());
    }
    let driver = Driver(module);
    macro_rules! symbol {
        ($name:literal, $signature:ty) => {{
            let address = unsafe { GetProcAddress(driver.0, concat!($name, "\0").as_ptr()) }
                .ok_or_else(|| format!("NVIDIA driver does not provide {}", $name))?;
            // CUDA Driver API exports use the system calling convention.
            unsafe {
                std::mem::transmute::<unsafe extern "system" fn() -> isize, $signature>(address)
            }
        }};
    }
    let init = symbol!("cuInit", unsafe extern "system" fn(u32) -> i32);
    let get_device = symbol!(
        "cuDeviceGet",
        unsafe extern "system" fn(*mut i32, i32) -> i32
    );
    let get_name = symbol!(
        "cuDeviceGetName",
        unsafe extern "system" fn(*mut c_char, i32, i32) -> i32
    );
    let get_attribute = symbol!(
        "cuDeviceGetAttribute",
        unsafe extern "system" fn(*mut i32, i32, i32) -> i32
    );
    let get_version = symbol!(
        "cuDriverGetVersion",
        unsafe extern "system" fn(*mut i32) -> i32
    );
    let (mut device, mut major, mut minor, mut version) = (0, 0, 0, 0);
    let mut name = [0 as c_char; 256];
    unsafe {
        check(init(0), "CUDA driver initialization")?;
        check(get_device(&mut device, 0), "CUDA device discovery")?;
        check(
            get_name(name.as_mut_ptr(), name.len() as i32, device),
            "GPU name discovery",
        )?;
        check(
            get_attribute(&mut major, 75, device),
            "GPU compute capability discovery",
        )?;
        check(
            get_attribute(&mut minor, 76, device),
            "GPU compute capability discovery",
        )?;
        check(get_version(&mut version), "CUDA driver version discovery")?;
    }
    // These are the architectures supported by the current portable RTX plan
    // builder. Turing requires separate flags/precision policies we don't set.
    if !matches!((major, minor), (8, 6) | (8, 9) | (12, 0) | (12, 1)) {
        return Err(format!(
            "GPU SM {major}.{minor} is not supported by this application's TensorRT RTX plan builder"
        ));
    }
    #[cfg(feature = "tensorrt-rtx")]
    {
        let stream = vocal_tensorrt::CudaStream::new()
            .map_err(|error| format!("CUDA runtime cannot use this GPU: {error}"))?;
        stream
            .synchronize()
            .map_err(|error| format!("CUDA runtime probe failed: {error}"))?;
    }
    let name = unsafe { CStr::from_ptr(name.as_ptr()) }.to_string_lossy();
    Ok(format!(
        "{name} · SM {major}.{minor} · CUDA driver API {}.{}",
        version / 1000,
        (version % 1000) / 10
    ))
}
