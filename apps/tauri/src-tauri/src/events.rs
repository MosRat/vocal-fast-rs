use crate::{
    error::{GuiError, GuiResult},
    state::UiSnapshot,
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
    ModelImport {
        sequence: u64,
        payload: vocal_runtime::model_library::ModelImportStatus,
    },
    Boot {
        sequence: u64,
        payload: crate::snapshot::UiBootStatus,
    },
    Snapshot {
        sequence: u64,
        payload: UiSnapshot,
    },
    Error {
        sequence: u64,
        payload: ErrorReport,
    },
    Preparation {
        sequence: u64,
        payload: crate::preparation::PreparationStatus,
    },
}

impl BackendEvent {
    pub fn model_import(payload: vocal_runtime::model_library::ModelImportStatus) -> Self {
        Self::ModelImport {
            sequence: 0,
            payload,
        }
    }
    pub fn boot(payload: crate::snapshot::UiBootStatus) -> Self {
        Self::Boot {
            sequence: 0,
            payload,
        }
    }

    pub fn preparation(payload: crate::preparation::PreparationStatus) -> Self {
        Self::Preparation {
            sequence: 0,
            payload,
        }
    }

    pub fn snapshot(payload: UiSnapshot) -> Self {
        Self::Snapshot {
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
            Self::Boot { payload, .. } => Self::Boot { sequence, payload },
            Self::Snapshot { payload, .. } => Self::Snapshot { sequence, payload },
            Self::Error { payload, .. } => Self::Error { sequence, payload },
            Self::Preparation { payload, .. } => Self::Preparation { sequence, payload },
            Self::ModelImport { payload, .. } => Self::ModelImport { sequence, payload },
        }
    }
}

#[derive(Clone, Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ErrorReport {
    pub code: String,
    pub message: String,
    pub recoverable: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub details: Option<crate::audio_error::AudioStartError>,
}

impl ErrorReport {
    pub fn from_error(error: &GuiError) -> Self {
        Self {
            code: error.code().into(),
            message: error.to_string(),
            recoverable: !matches!(error, GuiError::Internal(_)),
            details: match error {
                GuiError::AudioStartup(error) => Some(error.clone()),
                _ => None,
            },
        }
    }
}

#[derive(Default)]
pub struct EventSink {
    channels: std::sync::Mutex<Vec<(String, u64, Channel<BackendEvent>)>>,
    subscriber_id: AtomicU64,
    sequence: AtomicU64,
}

impl EventSink {
    pub fn attach(&self, label: String, channel: Channel<BackendEvent>) -> GuiResult<()> {
        let mut channels = self
            .channels
            .lock()
            .map_err(|_| GuiError::Internal("event channel lock poisoned".into()))?;
        channels.retain(|(current, _, _)| current != &label);
        let id = self
            .subscriber_id
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1);
        channels.push((label, id, channel));
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
        for (label, id, channel) in channels {
            if let Err(error) = channel.send(event.clone()) {
                tracing::debug!(target: "vocal.gui", subscriber = %label, error = %error, "removing closed event channel");
                failed.push((label, id));
            }
        }

        if !failed.is_empty() {
            let mut current = self
                .channels
                .lock()
                .map_err(|_| GuiError::Internal("event channel lock poisoned".into()))?;
            current.retain(|(label, id, _)| {
                !failed
                    .iter()
                    .any(|(failed_label, failed_id)| failed_label == label && failed_id == id)
            });
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
        let value = serde_json::to_value(BackendEvent::boot(crate::snapshot::UiBootStatus {
            revision: 0,
            phase: crate::snapshot::BootPhase::Ready,
            message: None,
            elapsed_ms: 0,
            recoverable: false,
        }))
        .expect("event serialization");
        assert_eq!(value["type"], "boot");
        assert_eq!(value["sequence"], 0);
        assert!(value["payload"].is_object());
    }
}
