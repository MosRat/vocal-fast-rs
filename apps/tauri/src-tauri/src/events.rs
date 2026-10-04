use crate::{
    error::{GuiError, GuiResult},
    state::{UiMetrics, UiSnapshot},
};
use serde::Serialize;
use tauri::ipc::Channel;

#[derive(Clone, Debug, Serialize)]
#[serde(tag = "type", content = "payload", rename_all = "camelCase")]
pub enum BackendEvent {
    Snapshot(UiSnapshot),
    Metrics(UiMetrics),
    Error(ErrorReport),
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
            channel
                .send(event)
                .map_err(|error| GuiError::Channel(error.to_string()))?;
        }
        Ok(())
    }
}
