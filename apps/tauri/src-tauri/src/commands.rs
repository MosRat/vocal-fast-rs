use tauri::{State, ipc::Channel};
use vocal_types::AudioFormat;

use crate::{
    config::GuiConfig,
    error::GuiResult,
    events::BackendEvent,
    state::{AppState, UiSnapshot},
};

#[tauri::command]
pub fn list_audio_devices(
    state: State<'_, AppState>,
) -> GuiResult<crate::audio::AudioDeviceCatalog> {
    report(state.inner(), state.list_audio_devices())
}

#[tauri::command]
pub fn get_snapshot(state: State<'_, AppState>) -> GuiResult<UiSnapshot> {
    report(state.inner(), state.snapshot())
}

#[tauri::command]
pub fn subscribe(
    state: State<'_, AppState>,
    window_label: String,
    channel: Channel<BackendEvent>,
) -> GuiResult<UiSnapshot> {
    report(state.inner(), state.attach_channel(window_label, channel))
}

#[tauri::command]
pub fn update_config(
    state: State<'_, AppState>,
    config: GuiConfig,
    expected_revision: u64,
) -> GuiResult<UiSnapshot> {
    report(
        state.inner(),
        state.update_config(config, expected_revision),
    )
}

#[tauri::command]
pub async fn load_model(state: State<'_, AppState>, path: String) -> GuiResult<UiSnapshot> {
    let state = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || state.load_model(path))
        .await
        .map_err(|error| crate::error::GuiError::Internal(error.to_string()))?
}

#[tauri::command]
pub fn unload_model(state: State<'_, AppState>) -> GuiResult<UiSnapshot> {
    report(state.inner(), state.unload_model())
}

#[tauri::command]
pub fn start_session(state: State<'_, AppState>) -> GuiResult<UiSnapshot> {
    report(state.inner(), state.start())
}

#[tauri::command]
pub fn stop_session(state: State<'_, AppState>) -> GuiResult<UiSnapshot> {
    report(state.inner(), state.stop())
}

#[tauri::command]
pub fn clear_error(state: State<'_, AppState>) -> GuiResult<UiSnapshot> {
    report(state.inner(), state.clear_error())
}

#[tauri::command]
pub fn process_block(
    state: State<'_, AppState>,
    samples: Vec<f32>,
    sample_rate: u32,
    channels: u16,
    sequence: u64,
) -> GuiResult<UiSnapshot> {
    report(
        state.inner(),
        state.process(
            samples,
            AudioFormat {
                sample_rate,
                channels,
            },
            sequence,
        ),
    )
}

fn report<T>(state: &AppState, result: GuiResult<T>) -> GuiResult<T> {
    if let Err(error) = &result {
        state.report(error);
    }
    result
}
