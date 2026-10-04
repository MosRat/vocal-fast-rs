//! Lock-free lifecycle state shared by backend control planes.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};

use super::error::{AudioFault, AudioFaultDirection, AudioFaultKind};

#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum StreamLifecycle {
    #[default]
    Created = 0,
    Starting = 1,
    Running = 2,
    Stopping = 3,
    Stopped = 4,
    Failed = 5,
}

impl StreamLifecycle {
    fn from_raw(value: u8) -> Self {
        match value {
            1 => Self::Starting,
            2 => Self::Running,
            3 => Self::Stopping,
            4 => Self::Stopped,
            5 => Self::Failed,
            _ => Self::Created,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct StreamStatus {
    pub lifecycle: StreamLifecycle,
    pub transitions: u64,
    pub faults: u64,
}

/// Atomic lifecycle model. It is deliberately independent from the audio
/// queues so callback code can publish faults without taking a lock.
#[derive(Debug, Default)]
pub struct StreamState {
    lifecycle: AtomicU8,
    transitions: AtomicU64,
    faults: AtomicU64,
    pending_faults: AtomicU64,
    first_fault: OnceLock<AudioFault>,
}

impl StreamState {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn lifecycle(&self) -> StreamLifecycle {
        StreamLifecycle::from_raw(self.lifecycle.load(Ordering::Acquire))
    }

    pub fn status(&self) -> StreamStatus {
        StreamStatus {
            lifecycle: self.lifecycle(),
            transitions: self.transitions.load(Ordering::Relaxed),
            faults: self.faults.load(Ordering::Relaxed),
        }
    }

    pub fn set(&self, lifecycle: StreamLifecycle) {
        let requested = lifecycle as u8;
        loop {
            let previous = self.lifecycle.load(Ordering::Acquire);
            if previous == StreamLifecycle::Failed as u8 && lifecycle != StreamLifecycle::Failed {
                return;
            }
            match self.lifecycle.compare_exchange(
                previous,
                requested,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    if previous != requested {
                        self.transitions.fetch_add(1, Ordering::Relaxed);
                    }
                    return;
                }
                Err(_) => continue,
            }
        }
    }

    pub fn mark_fault(&self) {
        self.faults.fetch_add(1, Ordering::Relaxed);
        self.pending_faults.fetch_add(1, Ordering::Relaxed);
        self.set(StreamLifecycle::Failed);
    }

    pub fn record_fault(
        &self,
        kind: AudioFaultKind,
        direction: Option<AudioFaultDirection>,
        message: String,
    ) -> bool {
        let first = self
            .first_fault
            .set(AudioFault::new(kind, direction, message))
            .is_ok();
        self.mark_fault();
        first
    }

    pub fn take_faults(&self) -> Vec<AudioFault> {
        let count = self.pending_faults.swap(0, Ordering::AcqRel);
        if count == 0 {
            return Vec::new();
        }
        let Some(first) = self.first_fault.get() else {
            return Vec::new();
        };
        let mut fault = first.clone();
        fault.occurrences = count;
        vec![fault]
    }

    pub fn begin_shutdown(&self) -> bool {
        if self.lifecycle() == StreamLifecycle::Failed {
            return false;
        }
        self.set(StreamLifecycle::Stopping);
        true
    }

    pub fn finish_shutdown(&self) {
        if self.lifecycle() != StreamLifecycle::Failed {
            self.set(StreamLifecycle::Stopped);
        }
    }
}
