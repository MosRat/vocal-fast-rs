mod commands;
mod config;
mod error;
mod events;
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
                tracing::warn!(target: "vocal_gui", error = %error, "failed to load persisted GUI config");
            }
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::get_snapshot,
            commands::subscribe,
            commands::update_config,
            commands::load_model,
            commands::start_session,
            commands::stop_session,
            commands::clear_error,
            commands::process_block,
        ])
        .run(tauri::generate_context!())
        .expect("error while running vocal-fast Tauri application");
}
