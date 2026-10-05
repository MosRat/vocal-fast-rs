mod audio;
mod commands;
mod config;
mod error;
mod events;
mod snapshot;
mod state;
mod tray;

use std::path::PathBuf;

use tauri::{Manager, WindowEvent};

use config::{ConfigStore, GuiConfig};
use snapshot::BootPhase;
use state::AppState;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let state = AppState::new(GuiConfig::default());
    tauri::Builder::default()
        .manage(state)
        .on_window_event(|window, event| {
            if window.label() != "main" {
                return;
            }
            if let WindowEvent::CloseRequested { api, .. } = event {
                api.prevent_close();
                let handle = window.app_handle().clone();
                tauri::async_runtime::spawn_blocking(move || {
                    let state = handle.state::<AppState>();
                    if let Err(error) = state.stop() {
                        state.report(&error);
                    }
                    if let Some(main) = handle.get_webview_window("main") {
                        let _ = main.hide();
                    }
                });
            }
        })
        .setup(|app| {
            if let Err(error) = tray::install(app) {
                // The tray is an optional shell affordance. A platform or
                // shell integration failure must not prevent the audio UI
                // from starting; the native close path still remains safe.
                tracing::warn!(target: "vocal.gui", %error, "system tray unavailable");
            }
            let path = app
                .path()
                .app_config_dir()
                .map_err(|error| Box::<dyn std::error::Error>::from(error.to_string()))?
                .join("config.json");
            let state = app.state::<AppState>().inner().clone();
            let handle = app.handle().clone();
            let _ = state.set_boot_phase(BootPhase::RestoringConfig, None);
            if let Err(error) = state.set_config_store(ConfigStore::new(PathBuf::from(path))) {
                tracing::warn!(target: "vocal.gui", error = %error, "failed to load persisted GUI config");
                state.fail_boot(&error);
                reveal_main(&handle);
                return Ok(());
            }

            match state.persisted_model_path() {
                Ok(Some(model_path))
                    if std::path::Path::new(&model_path).is_file()
                        || std::path::Path::new(&model_path).is_dir() =>
                {
                    let _ = state.set_boot_phase(
                        BootPhase::RestoringModel,
                        Some("Restoring the last voice model…".into()),
                    );
                    let worker_state = state.clone();
                    let worker_handle = handle.clone();
                    tauri::async_runtime::spawn_blocking(move || {
                        let result = worker_state.load_model(model_path);
                        match result {
                            Ok(_) => worker_state.mark_boot_ready(),
                            Err(error) => {
                                tracing::warn!(target: "vocal.gui", error = %error, "failed to restore persisted model");
                                worker_state.fail_boot(&error);
                            }
                        }
                        reveal_main(&worker_handle);
                    });

                    // A slow disk scan must not leave the user staring at an
                    // invisible main window forever. The restore worker keeps
                    // running and can still publish Ready afterwards.
                    let timeout_state = state.clone();
                    let timeout_handle = handle.clone();
                    tauri::async_runtime::spawn_blocking(move || {
                        std::thread::sleep(std::time::Duration::from_secs(30));
                        if matches!(
                            timeout_state.snapshot().ok().map(|snapshot| snapshot.boot.phase),
                            Some(BootPhase::RestoringConfig | BootPhase::RestoringModel)
                        ) {
                            timeout_state.mark_boot_timeout();
                            reveal_main(&timeout_handle);
                        }
                    });
                }
                Ok(Some(model_path)) => {
                    tracing::warn!(target: "vocal.gui", path = %model_path, "persisted model path is unavailable");
                    let _ = state.set_boot_phase(
                        BootPhase::Ready,
                        Some("The saved model path is unavailable; choose a model to continue.".into()),
                    );
                    reveal_main(&handle);
                }
                Ok(None) => {
                    state.mark_boot_ready();
                    reveal_main(&handle);
                }
                Err(error) => {
                    state.fail_boot(&error);
                    reveal_main(&handle);
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_snapshot,
            commands::subscribe,
            commands::list_audio_devices,
            commands::list_models,
            commands::pick_model_directory,
            commands::update_config,
            commands::load_model,
            commands::unload_model,
            commands::start_session,
            commands::stop_session,
            commands::clear_error,
            commands::process_block,
            commands::open_devtools,
            commands::harden_webview,
        ])
        .run(tauri::generate_context!())
        .expect("error while running vocal-fast Tauri application");
}

fn reveal_main<R: tauri::Runtime>(app: &tauri::AppHandle<R>) {
    if let Some(main) = app.get_webview_window("main") {
        let _ = main.show();
        let _ = main.set_focus();
    }
    if let Some(splash) = app.get_webview_window("splash") {
        let _ = splash.close();
    }
}
