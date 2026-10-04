mod audio;
mod commands;
mod config;
mod error;
mod events;
mod snapshot;
mod state;

use std::path::PathBuf;

use tauri::Manager;

use config::{ConfigStore, GuiConfig};
use state::AppState;

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let state = AppState::new(GuiConfig::default());
    tauri::Builder::default()
        .manage(state)
        .setup(|app| {
            let path = app
                .path()
                .app_config_dir()
                .map_err(|error| Box::<dyn std::error::Error>::from(error.to_string()))?
                .join("config.json");
            let state = app.state::<AppState>();
            if let Err(error) = state.set_config_store(ConfigStore::new(PathBuf::from(path))) {
                tracing::warn!(target: "vocal.gui", error = %error, "failed to load persisted GUI config");
            }
            if let Ok(Some(model_path)) = state.persisted_model_path() {
                if std::path::Path::new(&model_path).is_file() {
                    let state = state.inner().clone();
                    tauri::async_runtime::spawn_blocking(move || {
                        if let Err(error) = state.load_model(model_path) {
                            tracing::warn!(target: "vocal.gui", error = %error, "failed to restore persisted model");
                        }
                    });
                } else {
                    tracing::warn!(target: "vocal.gui", path = %model_path, "persisted model path is unavailable");
                }
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_snapshot,
            commands::subscribe,
            commands::list_audio_devices,
            commands::update_config,
            commands::load_model,
            commands::unload_model,
            commands::start_session,
            commands::stop_session,
            commands::clear_error,
            commands::process_block,
        ])
        .run(tauri::generate_context!())
        .expect("error while running vocal-fast Tauri application");
}
