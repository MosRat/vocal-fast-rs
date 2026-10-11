//! Filesystem/network work is isolated from the engine operation lock.
use crate::{
    error::{GuiError, GuiResult},
    state::AppState,
};
use tauri::State;
use vocal_runtime::model_library::*;

#[tauri::command]
pub async fn read_model_metadata(path: String) -> GuiResult<vocal_model::ModelMetadataView> {
    tauri::async_runtime::spawn_blocking(move || {
        vocal_model::read_model_metadata(path).map_err(GuiError::from)
    })
    .await
    .map_err(|e| GuiError::Internal(e.to_string()))?
}
#[tauri::command]
pub async fn edit_model_metadata(
    state: State<'_, AppState>,
    path: String,
    revision: String,
    edit: vocal_model::ModelMetadataEdit,
) -> GuiResult<vocal_model::ModelMetadataView> {
    let state = state.inner().clone();
    tauri::async_runtime::spawn_blocking(move || state.edit_model_metadata(path, revision, edit))
        .await
        .map_err(|e| GuiError::Internal(e.to_string()))?
}
#[tauri::command]
pub async fn list_remote_models(repository: RemoteRepository) -> GuiResult<RemoteModelCatalog> {
    tauri::async_runtime::spawn_blocking(move || {
        vocal_runtime::model_library::list_remote_models(repository).map_err(GuiError::ModelLibrary)
    })
    .await
    .map_err(|e| GuiError::Internal(e.to_string()))?
}
#[tauri::command]
pub fn get_model_import_status(state: State<'_, AppState>) -> ModelImportStatus {
    state.model_library.status()
}
#[tauri::command]
pub fn start_model_import(
    state: State<'_, AppState>,
    request: ModelImportRequest,
) -> GuiResult<ModelImportStatus> {
    state
        .model_library
        .start(request)
        .map_err(GuiError::ModelLibrary)
}
#[tauri::command]
pub fn cancel_model_import(
    state: State<'_, AppState>,
    job_id: u64,
) -> GuiResult<ModelImportStatus> {
    state
        .model_library
        .cancel(job_id)
        .map_err(GuiError::ModelLibrary)
}
#[tauri::command]
pub fn default_model_library() -> String {
    default_library_directory().display().to_string()
}

#[tauri::command]
pub fn get_model_dependency_status(
    state: State<'_, AppState>,
) -> ModelDependencyStatus {
    state.model_dependencies.status()
}

#[tauri::command]
pub async fn inspect_model_dependencies(
    state: State<'_, AppState>,
    request: ModelDependencyRequest,
) -> GuiResult<ModelDependencyStatus> {
    let manager = state.model_dependencies.clone();
    tauri::async_runtime::spawn_blocking(move || {
        manager.inspect(&request).map_err(GuiError::ModelLibrary)
    })
    .await
    .map_err(|error| GuiError::Internal(error.to_string()))?
}

#[tauri::command]
pub fn start_model_dependency_install(
    state: State<'_, AppState>,
    request: ModelDependencyRequest,
) -> GuiResult<ModelDependencyStatus> {
    state
        .model_dependencies
        .start(request)
        .map_err(GuiError::ModelLibrary)
}

#[tauri::command]
pub fn cancel_model_dependency_install(
    state: State<'_, AppState>,
    job_id: u64,
) -> GuiResult<ModelDependencyStatus> {
    state
        .model_dependencies
        .cancel(job_id)
        .map_err(GuiError::ModelLibrary)
}
#[tauri::command]
pub async fn pick_model_file(kind: String) -> GuiResult<Option<String>> {
    tauri::async_runtime::spawn_blocking(move || {
        let dialog = rfd::FileDialog::new();
        let dialog = match kind.as_str() {
            "index" => dialog
                .set_title("Choose retrieval index")
                .add_filter("FAISS index", &["index"]),
            "acoustic" => dialog
                .set_title("Choose canonical acoustic or complete VFRM")
                .add_filter("VFRM", &["vfr", "vfrm"]),
            _ => dialog
                .set_title("Import voice model")
                .add_filter("RVC model", &["pth", "vfr", "vfrm", "zip"]),
        };
        Ok(dialog.pick_file().map(|p| p.to_string_lossy().into_owned()))
    })
    .await
    .map_err(|e| GuiError::Internal(e.to_string()))?
}
