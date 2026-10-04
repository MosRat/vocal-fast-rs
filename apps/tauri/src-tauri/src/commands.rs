use tauri::{State, WebviewWindow, ipc::Channel};
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

/// Open the native WebView inspector on an explicit user action.
///
/// DevTools are enabled in the Tauri feature set, but no browser shortcut or
/// context menu is exposed by the frontend. Keeping this behind a command
/// makes the escape hatch available to diagnostics and automation without
/// making it part of the normal window interaction model.
#[tauri::command]
pub fn open_devtools(window: WebviewWindow) -> GuiResult<()> {
    harden_webview_impl(&window)?;
    window.open_devtools();
    Ok(())
}

/// Disable browser-owned context menus and accelerator keys while keeping the
/// WebView inspector enabled for the explicit diagnostics command.
#[tauri::command]
pub fn harden_webview(window: WebviewWindow) -> GuiResult<()> {
    harden_webview_impl(&window)
}

fn harden_webview_impl(window: &WebviewWindow) -> GuiResult<()> {
    #[cfg(windows)]
    {
        use std::sync::{Arc, Mutex};
        use webview2_com::Microsoft::Web::WebView2::Win32::ICoreWebView2Settings3;
        use windows_core::Interface;

        let failure = Arc::new(Mutex::new(None::<String>));
        let failure_in_callback = failure.clone();
        window.with_webview(move |webview| {
            let result = (|| -> Result<(), String> {
                let core = unsafe { webview.controller().CoreWebView2() }
                    .map_err(|error| error.to_string())?;
                let settings = unsafe { core.Settings() }.map_err(|error| error.to_string())?;
                unsafe {
                    settings
                        .SetAreDefaultContextMenusEnabled(false)
                        .map_err(|error| error.to_string())?;
                    let settings3 = settings
                        .cast::<ICoreWebView2Settings3>()
                        .map_err(|error| error.to_string())?;
                    settings3
                        .SetAreBrowserAcceleratorKeysEnabled(false)
                        .map_err(|error| error.to_string())?;
                }
                Ok(())
            })();
            if let Err(error) = result {
                if let Ok(mut slot) = failure_in_callback.lock() {
                    *slot = Some(error);
                }
            }
        })?;
        if let Some(error) = failure.lock().ok().and_then(|mut slot| slot.take()) {
            return Err(crate::error::GuiError::Internal(format!(
                "failed to configure WebView2 interaction policy: {error}"
            )));
        }
    }
    Ok(())
}

fn report<T>(state: &AppState, result: GuiResult<T>) -> GuiResult<T> {
    if let Err(error) = &result {
        state.report(error);
    }
    result
}
