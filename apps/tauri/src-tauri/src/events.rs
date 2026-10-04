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
            recoverable: !matches!(error, GuiError::Internal(_)),
        }
    }
}

#[derive(Default)]
pub struct EventSink {
    channels: std::sync::Mutex<Vec<(u64, Channel<BackendEvent>)>>,
    subscriber_id: AtomicU64,
    sequence: AtomicU64,
}

impl EventSink {
    pub fn attach(&self, channel: Channel<BackendEvent>) -> GuiResult<()> {
        let mut channels = self
            .channels
            .lock()
            .map_err(|_| GuiError::Internal("event channel lock poisoned".into()))?;
        let id = self.subscriber_id.fetch_add(1, Ordering::Relaxed).saturating_add(1);
        channels.push((id, channel));
        Ok(())
    }

    pub fn send(&self, event: BackendEvent) -> GuiResult<()> {
        let channels = self
            .channels
            .lock()
            .map_err(|_| GuiError::Internal("event channel lock poisoned".into()))?
            .clone();
        if channels.is_empty() {
            return Ok(());
        }

        let sequence = self.next_sequence();
        let event = event.with_sequence(sequence);
        let mut failed = Vec::new();
        for (id, channel) in channels {
            if let Err(error) = channel.send(event.clone()) {
                tracing::debug!(target: "vocal.gui", subscriber = id, error = %error, "removing closed event channel");
                failed.push(id);
            }
        }

        if !failed.is_empty() {
            let mut current = self
                .channels
                .lock()
                .map_err(|_| GuiError::Internal("event channel lock poisoned".into()))?;
            current.retain(|(id, _)| !failed.contains(id));
        }
        Ok(())
    }

    /// Publish a control-plane event without making the state mutation depend
    /// on the webview being alive. A Tauri window can disappear during a
    /// reload while the audio worker continues safely; the next subscriber
    /// receives a fresh snapshot from `subscribe`.
    pub fn publish(&self, event: BackendEvent) {
        if let Err(error) = self.send(event) {
            tracing::debug!(target: "vocal.gui", error = %error, "control event was not delivered");
        }
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
