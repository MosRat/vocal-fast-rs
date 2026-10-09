mod audio;
mod audio_error;
mod audio_telemetry;
mod commands;
mod config;
mod error;
mod events;
mod model_commands;
mod preparation;
mod snapshot;
mod startup;
mod state;
mod tensorrt;
mod tray;

use tauri::{Manager, WindowEvent};

use config::GuiConfig;
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
                // Hide immediately; joining audio/model threads happens away
                // from the UI thread, even while engine preparation is busy.
                let _ = window.hide();
                let handle = window.app_handle().clone();
                tauri::async_runtime::spawn_blocking(move || {
                    let state = handle.state::<AppState>();
                    if let Err(error) = state.stop() {
                        state.report(&error);
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
            let cache_path = app
                .path()
                .app_cache_dir()
                .ok()
                .map(|path| path.join("vocal-fast").join("tensorrt"));
            startup::launch(handle, state, path, cache_path);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_snapshot,
            model_commands::read_model_metadata,
            model_commands::edit_model_metadata,
            model_commands::list_remote_models,
            model_commands::get_model_import_status,
            model_commands::start_model_import,
            model_commands::cancel_model_import,
            model_commands::default_model_library,
            model_commands::pick_model_file,
            commands::get_boot_status,
            commands::get_startup_resources,
            commands::subscribe_boot,
            commands::restore_saved_model,
            commands::subscribe,
            commands::subscribe_preparation,
            commands::list_audio_devices,
            commands::list_processing_backends,
            commands::list_models,
            commands::inspect_model_artifact,
            commands::pick_model_directory,
            commands::update_config,
            commands::prepare_tensorrt_pipeline,
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
    if app
        .state::<AppState>()
        .boot_status()
        .is_ok_and(|status| status.phase == snapshot::BootPhase::ShuttingDown)
    {
        return;
    }
    if let Some(main) = app.get_webview_window("main") {
        let _ = main.show();
        let _ = main.set_focus();
    }
    if let Some(splash) = app.get_webview_window("splash") {
        let _ = splash.close();
    }
}
