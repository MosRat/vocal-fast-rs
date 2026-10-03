//! Allocation-free producer/consumer notification primitives.
//!
//! The audio callback must never take a mutex or wait for the worker.  Windows
//! provides address-based waiting for exactly this hand-off: the producer
//! increments a generation and wakes one waiter, while the worker re-checks
//! the ring before sleeping.  Non-Windows builds retain a bounded polling
//! fallback so the core remains portable.

use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

#[derive(Debug)]
pub(crate) struct AudioWake {
    generation: AtomicU32,
}

impl AudioWake {
    pub(crate) const fn new() -> Self {
        Self {
            generation: AtomicU32::new(0),
        }
    }

    pub(crate) fn generation(&self) -> u32 {
        self.generation.load(Ordering::Acquire)
    }

    /// Publish a new generation and wake one waiting worker.
    pub(crate) fn notify(&self) {
        self.generation.fetch_add(1, Ordering::Release);
        #[cfg(windows)]
        unsafe {
            windows_sys::Win32::System::Threading::WakeByAddressSingle(
                (&self.generation as *const AtomicU32).cast(),
            );
        }
    }

    /// Wait until the generation differs from `observed`.
    ///
    /// The caller must check the queue before and after taking the observed
    /// value.  `WaitOnAddress` performs the final compare atomically with the
    /// sleep transition, so a producer notification cannot be lost between
    /// those checks.
    pub(crate) fn wait(&self, observed: u32, timeout: Option<Duration>) -> bool {
        if self.generation() != observed {
            return true;
        }

        #[cfg(windows)]
        {
            let timeout_ms = timeout.map_or(u32::MAX, |duration| {
                duration.as_millis().min(u128::from(u32::MAX - 1)) as u32
            });
            let compare = observed;
            unsafe {
                windows_sys::Win32::System::Threading::WaitOnAddress(
                    (&self.generation as *const AtomicU32).cast(),
                    (&compare as *const u32).cast(),
                    std::mem::size_of::<u32>(),
                    timeout_ms,
                );
            }
            self.generation() != observed
        }

        #[cfg(not(windows))]
        {
            // The portable core has no platform-independent address-wait
            // primitive. Keep the fallback bounded so callers never sleep
            // forever without a platform wake source.
            std::thread::sleep(timeout.unwrap_or(Duration::from_millis(1)));
            self.generation() != observed
        }
    }
}
