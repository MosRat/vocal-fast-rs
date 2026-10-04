//! Native Windows tray integration.
//!
//! The tray is deliberately a thin control surface over [`AppState`]. It
//! never owns an audio session, so close, stop, and quit all use the same
//! state machine and cannot leave a worker behind.

use tauri::{
    App, AppHandle, Emitter, Manager, Runtime,
    tray::{TrayIconBuilder, TrayIconEvent},
};

use crate::state::AppState;

pub const OPEN_ROUTE_EVENT: &str = "vocal-fast://open-route";

const SHOW_ID: &str = "tray-show";
const START_ID: &str = "tray-start";
const STOP_ID: &str = "tray-stop";
const SETTINGS_ID: &str = "tray-settings";
const DIAGNOSTICS_ID: &str = "tray-diagnostics";
const QUIT_ID: &str = "tray-quit";

pub fn install<R: Runtime>(app: &mut App<R>) -> tauri::Result<()> {
    let show = tauri::menu::MenuItem::with_id(app, SHOW_ID, "Show Studio", true, None::<&str>)?;
    let start = tauri::menu::MenuItem::with_id(app, START_ID, "Start session", true, None::<&str>)?;
    let stop = tauri::menu::MenuItem::with_id(app, STOP_ID, "Stop session", true, None::<&str>)?;
    let settings =
        tauri::menu::MenuItem::with_id(app, SETTINGS_ID, "Settings", true, None::<&str>)?;
    let diagnostics =
        tauri::menu::MenuItem::with_id(app, DIAGNOSTICS_ID, "Diagnostics", true, None::<&str>)?;
    let quit = tauri::menu::MenuItem::with_id(app, QUIT_ID, "Quit vocal-fast", true, None::<&str>)?;
    let separator = tauri::menu::PredefinedMenuItem::separator(app)?;
    let menu = tauri::menu::Menu::with_items(
        app,
        &[
            &show,
            &start,
            &stop,
            &separator,
            &settings,
            &diagnostics,
            &separator,
            &quit,
        ],
    )?;

    let mut builder = TrayIconBuilder::with_id("vocal-fast-tray")
        .menu(&menu)
        .tooltip("vocal-fast · realtime voice")
        .show_menu_on_left_click(false)
        .on_menu_event(handle_menu_event::<R>);
    if let Some(icon) = app.default_window_icon().cloned() {
        builder = builder.icon(icon);
    }
    builder
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::DoubleClick { .. } = event {
                show_main(tray.app_handle());
            }
        })
        .build(app)?;
    Ok(())
}

fn handle_menu_event<R: Runtime>(app: &AppHandle<R>, event: tauri::menu::MenuEvent) {
    match event.id().as_ref() {
        SHOW_ID => show_main(app),
        SETTINGS_ID => show_route(app, "settings"),
        DIAGNOSTICS_ID => show_route(app, "diagnostics"),
        START_ID => run_state_action(app, |state| state.start()),
        STOP_ID => run_state_action(app, |state| state.stop()),
        QUIT_ID => {
            let handle = app.clone();
            tauri::async_runtime::spawn_blocking(move || {
                let state = handle.state::<AppState>();
                if let Err(error) = state.shutdown() {
                    state.report(&error);
                }
                handle.exit(0);
            });
        }
        _ => {}
    }
}

fn run_state_action<R, F>(app: &AppHandle<R>, action: F)
where
    R: Runtime,
    F: FnOnce(&AppState) -> crate::error::GuiResult<crate::state::UiSnapshot> + Send + 'static,
{
    let state = app.state::<AppState>().inner().clone();
    tauri::async_runtime::spawn_blocking(move || {
        if let Err(error) = action(&state) {
            state.report(&error);
        }
    });
}

fn show_main<R: Runtime>(app: &AppHandle<R>) {
    if let Some(window) = app.get_webview_window("main") {
        let _ = window.show();
        let _ = window.set_focus();
    }
}

fn show_route<R: Runtime>(app: &AppHandle<R>, route: &str) {
    show_main(app);
    if let Err(error) = app.emit(OPEN_ROUTE_EVENT, route) {
        tracing::debug!(target: "vocal.gui", %error, "failed to route tray action");
    }
}
