//! Immutable cache reads and atomic publication, outside inference callbacks.
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    sync::atomic::{AtomicU64, Ordering},
};

pub(crate) fn map(path: &Path) -> std::io::Result<memmap2::Mmap> {
    let file = fs::File::open(path)?;
    // Plans/cache blobs are published by rename, never edited through an
    // existing mapping. Keep the mapping alive through the native call.
    unsafe { memmap2::MmapOptions::new().map(&file) }
}

pub(crate) fn publish(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    static SERIAL: AtomicU64 = AtomicU64::new(0);
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension(format!(
        "tmp-{}-{}",
        std::process::id(),
        SERIAL.fetch_add(1, Ordering::Relaxed)
    ));
    let result = (|| {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        #[cfg(windows)]
        {
            use std::os::windows::ffi::OsStrExt;
            #[link(name = "kernel32")]
            unsafe extern "system" {
                fn MoveFileExW(from: *const u16, to: *const u16, flags: u32) -> i32;
            }
            let from = temporary
                .as_os_str()
                .encode_wide()
                .chain(Some(0))
                .collect::<Vec<_>>();
            let to = path
                .as_os_str()
                .encode_wide()
                .chain(Some(0))
                .collect::<Vec<_>>();
            if unsafe { MoveFileExW(from.as_ptr(), to.as_ptr(), 1 | 8) } == 0 {
                return Err(std::io::Error::last_os_error());
            }
        }
        #[cfg(not(windows))]
        fs::rename(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

pub(crate) fn runtime_root() -> Option<PathBuf> {
    if let Some(root) = std::env::var_os("VOCAL_TRT_RUNTIME_CACHE_DIR") {
        return (!root.is_empty()).then(|| root.into());
    }
    Some(
        std::env::var_os("LOCALAPPDATA")
            .map(PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join("vocal-fast/cache/tensorrt/runtime"),
    )
}
