use crate::{
    error::{GuiError, GuiResult},
    state::{UiMetrics, UiSnapshot},
};
use serde::Serialize;
use std::sync::atomic::{AtomicU64, Ordering};
use tauri::ipc::Channel;

/// Event envelope sent to the frontend.
///
/// `sequence` is monotonic for the lifetime of the application.  The UI can
/// detect a reconnect or a dropped event and request a full snapshot instead
/// of applying an out-of-order delta.  Keeping the sequence in the envelope
/// also leaves the payload types useful for command responses.
#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum BackendEvent {
    Snapshot { sequence: u64, payload: UiSnapshot },
    Metrics { sequence: u64, payload: UiMetrics },
    Error { sequence: u64, payload: ErrorReport },
}

impl BackendEvent {
    pub fn snapshot(payload: UiSnapshot) -> Self {
        Self::Snapshot {
            sequence: 0,
            payload,
        }
    }

    pub fn metrics(payload: UiMetrics) -> Self {
        Self::Metrics {
            sequence: 0,
            payload,
        }
    }

    pub fn error(payload: ErrorReport) -> Self {
        Self::Error {
            sequence: 0,
            payload,
        }
    }

    fn with_sequence(self, sequence: u64) -> Self {
        match self {
            Self::Snapshot { payload, .. } => Self::Snapshot { sequence, payload },
            Self::Metrics { payload, .. } => Self::Metrics { sequence, payload },
            Self::Error { payload, .. } => Self::Error { sequence, payload },
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ErrorReport {
    pub code: String,
    pub message: String,
    pub recoverable: bool,
}

impl ErrorReport {
    pub fn from_error(error: &GuiError) -> Self {
        Self {
            code: error.code().into(),
            message: error.to_string(),
            recoverable: !matches!(error, GuiError::Internal(_) | GuiError::Channel(_)),
        }
    }
}

#[derive(Default)]
pub struct EventSink {
    channel: std::sync::Mutex<Option<Channel<BackendEvent>>>,
    sequence: AtomicU64,
}

impl EventSink {
    pub fn attach(&self, channel: Channel<BackendEvent>) -> GuiResult<()> {
        let mut current = self
            .channel
            .lock()
            .map_err(|_| GuiError::Internal("event channel lock poisoned".into()))?;
        *current = Some(channel);
        Ok(())
    }

    pub fn send(&self, event: BackendEvent) -> GuiResult<()> {
        let channel = self
            .channel
            .lock()
            .map_err(|_| GuiError::Internal("event channel lock poisoned".into()))?
            .clone();
        if let Some(channel) = channel {
            let sequence = self.next_sequence();
            if let Err(error) = channel.send(event.with_sequence(sequence)) {
                // A closed webview channel must not poison future reconnects.
                // Clear it only after the send fails; a new `subscribe` call
                // can then install a fresh channel without restarting audio.
                if let Ok(mut current) = self.channel.lock() {
                    *current = None;
                }
                return Err(GuiError::Channel(error.to_string()));
            }
        }
        Ok(())
    }

    fn next_sequence(&self) -> u64 {
        let mut current = self.sequence.load(Ordering::Relaxed);
        loop {
            let next = current.saturating_add(1);
            match self.sequence.compare_exchange_weak(
                current,
                next,
                Ordering::Relaxed,
                Ordering::Relaxed,
            ) {
                Ok(_) => return next,
                Err(observed) => current = observed,
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_shape_contains_sequence_and_payload() {
        let value = serde_json::to_value(BackendEvent::metrics(UiMetrics::default()))
            .expect("event serialization");
        assert_eq!(value["type"], "metrics");
        assert_eq!(value["sequence"], 0);
        assert!(value["payload"].is_object());
    }
}
