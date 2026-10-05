use tauri::{State, WebviewWindow, ipc::Channel};
use vocal_types::AudioFormat;

use crate::{
    config::GuiConfig,
    error::GuiResult,
    events::BackendEvent,
    state::{AppState, UiSnapshot},
};

#[tauri::command]
pub async fn list_audio_devices(
    state: State<'_, AppState>,
) -> GuiResult<crate::audio::AudioDeviceCatalog> {
    run_blocking(state.inner().clone(), |state| state.list_audio_devices()).await
}

#[tauri::command]
pub async fn list_models(
    state: State<'_, AppState>,
    path: String,
) -> GuiResult<crate::state::ModelCatalog> {
    run_blocking(state.inner().clone(), move |state| {
        state.scan_model_directory(path)
    })
    .await
}

/// Open the platform folder picker without blocking the Tauri command thread.
/// The Windows implementation uses the native common-controls dialog; other
/// targets return a structured error so the web preview can keep using the
/// path field directly.
#[tauri::command]
pub async fn pick_model_directory() -> GuiResult<Option<String>> {
    tauri::async_runtime::spawn_blocking(|| {
        #[cfg(windows)]
        {
            return Ok(rfd::FileDialog::new()
                .set_title("Choose model library")
                .pick_folder()
                .map(|path| path.to_string_lossy().into_owned()));
        }
        #[cfg(not(windows))]
        {
            Err(crate::error::GuiError::Internal(
                "native model directory picker is only available on Windows".into(),
            ))
        }
    })
    .await
    .map_err(|error| crate::error::GuiError::Internal(error.to_string()))?
}

#[tauri::command]
pub async fn get_snapshot(state: State<'_, AppState>) -> GuiResult<UiSnapshot> {
    run_blocking(state.inner().clone(), |state| state.snapshot()).await
}

#[tauri::command]
pub async fn subscribe(
    state: State<'_, AppState>,
    window_label: String,
    channel: Channel<BackendEvent>,
) -> GuiResult<UiSnapshot> {
    run_blocking(state.inner().clone(), move |state| {
        state.attach_channel(window_label, channel)
    })
    .await
}

#[tauri::command]
pub async fn update_config(
    state: State<'_, AppState>,
    config: GuiConfig,
    expected_revision: u64,
) -> GuiResult<UiSnapshot> {
    run_blocking(state.inner().clone(), move |state| {
        state.update_config(config, expected_revision)
    })
    .await
}

#[tauri::command]
pub async fn load_model(state: State<'_, AppState>, path: String) -> GuiResult<UiSnapshot> {
    let state = state.inner().clone();
    let reporter = state.clone();
    match tauri::async_runtime::spawn_blocking(move || state.load_model(path)).await {
        Ok(result) => result,
        Err(error) => {
            let error = crate::error::GuiError::Internal(error.to_string());
            reporter.report(&error);
            Err(error)
        }
    }
}

#[tauri::command]
pub async fn unload_model(state: State<'_, AppState>) -> GuiResult<UiSnapshot> {
    run_blocking(state.inner().clone(), |state| state.unload_model()).await
}

#[tauri::command]
pub async fn start_session(state: State<'_, AppState>) -> GuiResult<UiSnapshot> {
    run_blocking(state.inner().clone(), |state| state.start()).await
}

#[tauri::command]
pub async fn stop_session(state: State<'_, AppState>) -> GuiResult<UiSnapshot> {
    run_blocking(state.inner().clone(), |state| state.stop()).await
}

#[tauri::command]
pub async fn clear_error(state: State<'_, AppState>) -> GuiResult<UiSnapshot> {
    run_blocking(state.inner().clone(), |state| state.clear_error()).await
}

#[tauri::command]
pub async fn process_block(
    state: State<'_, AppState>,
    samples: Vec<f32>,
    sample_rate: u32,
    channels: u16,
    sequence: u64,
) -> GuiResult<UiSnapshot> {
    run_blocking(state.inner().clone(), move |state| {
        state.process(
            samples,
            AudioFormat {
                sample_rate,
                channels,
            },
            sequence,
        )
    })
    .await
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

/// Execute filesystem, device and lifecycle operations away from Tauri's
/// command executor. The helper owns the same AppState handle as the caller,
/// so every failure still enters the central error/event path exactly once.
async fn run_blocking<T, F>(state: AppState, operation: F) -> GuiResult<T>
where
    T: Send + 'static,
    F: FnOnce(&AppState) -> GuiResult<T> + Send + 'static,
{
    let reporter = state.clone();
    let result = match tauri::async_runtime::spawn_blocking(move || operation(&state)).await {
        Ok(result) => result,
        Err(error) => {
            let error = crate::error::GuiError::Internal(error.to_string());
            reporter.report(&error);
            return Err(error);
        }
    };
    if let Err(error) = &result {
        reporter.report(error);
    }
    result
}
