//! Dependency aware voice preparation.
//!
//! A voice VFRM contains the Synth weights and may optionally contain the two
//! acoustic models.  This module resolves those components without making the
//! engine backends know where shared files came from.  Missing shared models
//! are downloaded only after the caller explicitly opts in, then bound into a
//! derived complete VFRM in the local model cache.

use super::{
    LibraryError, LibraryResult, ModelProvider, RemoteFile, RemoteRepository, check_cancel, client,
    default_library_directory, download::download, provider,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::Read,
    path::{Path, PathBuf},
    sync::{Arc, Mutex, atomic::AtomicBool},
    time::Instant,
};

pub const SHARED_REPOSITORY: &str = "MosRat/vfrm-models";
const HUBERT_FILE: &str = "shared/hubert.vfrm";
const RMVPE_FILE: &str = "shared/rmvpe.vfrm";
const HUBERT_SIZE: u64 = 378_427_648;
const RMVPE_SIZE: u64 = 362_064_656;

#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub enum DependencyPhase {
    #[default]
    Idle,
    Inspecting,
    NeedsConsent,
    Queued,
    Listing,
    Downloading,
    Binding,
    Ready,
    Cancelling,
    Cancelled,
    Failed,
}

impl DependencyPhase {
    pub fn is_active(self) -> bool {
        matches!(
            self,
            Self::Inspecting
                | Self::Queued
                | Self::Listing
                | Self::Downloading
                | Self::Binding
                | Self::Cancelling
        )
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AcousticDependencyStatus {
    pub kind: String,
    pub present: bool,
    pub path: Option<PathBuf>,
    pub size_bytes: u64,
    pub expected_size_bytes: Option<u64>,
    pub source: Option<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelDependencyStatus {
    pub job_id: u64,
    pub revision: u64,
    pub phase: DependencyPhase,
    pub message: String,
    pub model_path: Option<PathBuf>,
    pub resolved_model_path: Option<PathBuf>,
    pub hubert: AcousticDependencyStatus,
    pub rmvpe: AcousticDependencyStatus,
    pub downloaded_bytes: u64,
    pub total_bytes: Option<u64>,
    pub elapsed_ms: u64,
    pub complete: bool,
    pub error: Option<LibraryError>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct ModelDependencyRequest {
    pub model_path: PathBuf,
    #[serde(default)]
    pub library_directory: Option<PathBuf>,
    #[serde(default)]
    pub cache_directory: Option<PathBuf>,
    #[serde(default)]
    pub allow_download: bool,
    #[serde(default)]
    pub hubert_source: Option<PathBuf>,
    #[serde(default)]
    pub rmvpe_source: Option<PathBuf>,
}

#[derive(Clone)]
pub struct DependencyManager {
    state: Arc<Mutex<DependencyState>>,
    listener: Arc<dyn Fn(ModelDependencyStatus) + Send + Sync>,
}

struct DependencyState {
    status: ModelDependencyStatus,
    cancel: Arc<AtomicBool>,
    started: Instant,
}

impl DependencyManager {
    pub fn new(listener: impl Fn(ModelDependencyStatus) + Send + Sync + 'static) -> Self {
        Self {
            state: Arc::new(Mutex::new(DependencyState {
                status: ModelDependencyStatus::default(),
                cancel: Arc::new(AtomicBool::new(false)),
                started: Instant::now(),
            })),
            listener: Arc::new(listener),
        }
    }

    pub fn status(&self) -> ModelDependencyStatus {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .status
            .clone()
    }

    pub fn inspect(
        &self,
        request: &ModelDependencyRequest,
    ) -> LibraryResult<ModelDependencyStatus> {
        let status = inspect_dependencies(request)?;
        self.publish(status.clone());
        Ok(status)
    }

    pub fn start(&self, request: ModelDependencyRequest) -> LibraryResult<ModelDependencyStatus> {
        if self.status().phase.is_active() {
            return Err(LibraryError::new(
                "busy",
                "A model dependency operation is already running",
            ));
        }
        let initial = inspect_dependencies(&request)?;
        if initial.phase == DependencyPhase::Ready {
            self.publish(initial.clone());
            return Ok(initial);
        }
        let has_local_sources = request.hubert_source.is_some() && request.rmvpe_source.is_some();
        if initial.phase == DependencyPhase::NeedsConsent
            && !request.allow_download
            && !has_local_sources
        {
            self.publish(initial.clone());
            return Ok(initial);
        }
        let (status, cancel) = {
            let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            if state.status.phase.is_active() {
                return Err(LibraryError::new(
                    "busy",
                    "A model dependency operation is already running",
                ));
            }
            state.cancel = Arc::new(AtomicBool::new(false));
            state.started = Instant::now();
            state.status = ModelDependencyStatus {
                job_id: state.status.job_id.saturating_add(1),
                revision: 1,
                phase: DependencyPhase::Queued,
                message: "Preparing shared acoustic models".into(),
                model_path: Some(request.model_path.clone()),
                hubert: initial.hubert.clone(),
                rmvpe: initial.rmvpe.clone(),
                ..Default::default()
            };
            (state.status.clone(), state.cancel.clone())
        };
        self.publish(status.clone());
        let service = self.clone();
        std::thread::Builder::new()
            .name("model-dependencies".into())
            .spawn(move || {
                let job_id = status.job_id;
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    execute(
                        &request,
                        &cancel,
                        &mut |phase, message, h, r, downloaded, total| {
                            service.update(job_id, |state| {
                                state.phase = phase;
                                state.message = message;
                                state.hubert = h;
                                state.rmvpe = r;
                                state.downloaded_bytes = downloaded;
                                state.total_bytes = total;
                            });
                            Ok(())
                        },
                    )
                }))
                .unwrap_or_else(|_| {
                    Err(LibraryError::new(
                        "internal",
                        "Model dependency preparation failed unexpectedly",
                    ))
                });
                service.update(job_id, |state| match result {
                    Ok((resolved, h, r)) => {
                        state.phase = DependencyPhase::Ready;
                        state.message = "Model dependencies are ready".into();
                        state.resolved_model_path = Some(resolved);
                        state.hubert = h;
                        state.rmvpe = r;
                        state.complete = true;
                        state.error = None;
                    }
                    Err(error) => {
                        state.phase = if error.code == "cancelled" {
                            DependencyPhase::Cancelled
                        } else {
                            DependencyPhase::Failed
                        };
                        state.message = error.message.clone();
                        state.error = Some(error);
                    }
                });
            })
            .map_err(|error| LibraryError::new("internal", error.to_string()))?;
        Ok(self.status())
    }

    pub fn cancel(&self, job_id: u64) -> LibraryResult<ModelDependencyStatus> {
        {
            let state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            if state.status.job_id != job_id {
                return Err(LibraryError::new(
                    "staleJob",
                    "This dependency operation is no longer current",
                ));
            }
            if !state.status.phase.is_active() {
                return Ok(state.status.clone());
            }
            state
                .cancel
                .store(true, std::sync::atomic::Ordering::Release);
        }
        self.update(job_id, |state| {
            state.phase = DependencyPhase::Cancelling;
            state.message = "Cancelling dependency preparation".into();
        });
        Ok(self.status())
    }

    fn publish(&self, status: ModelDependencyStatus) {
        (self.listener)(status);
    }

    fn update(&self, job_id: u64, change: impl FnOnce(&mut ModelDependencyStatus)) {
        let status = {
            let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            if state.status.job_id != job_id {
                return;
            }
            change(&mut state.status);
            state.status.revision = state.status.revision.saturating_add(1);
            state.status.elapsed_ms = state
                .started
                .elapsed()
                .as_millis()
                .min(u128::from(u64::MAX)) as u64;
            state.status.clone()
        };
        self.publish(status);
    }
}

pub fn inspect_dependencies(
    request: &ModelDependencyRequest,
) -> LibraryResult<ModelDependencyStatus> {
    let model = vocal_model::resolve_model_path(&request.model_path)?;
    let inspection = vocal_model::inspect_model_artifact(&model)?;
    if !inspection.components.synth.present {
        return Err(LibraryError::new(
            "invalidModel",
            "Selected artifact has no Synth weights",
        ));
    }
    let root = request
        .cache_directory
        .clone()
        .unwrap_or_else(default_library_directory);
    let shared = shared_status(&root);
    let complete = inspection.components.automatic_tensor_rt;
    let bound = complete
        .then(|| model.clone())
        .or_else(|| bound_model_path(&model, &shared, &root).filter(|path| valid_complete(path)));
    let phase = if bound.is_some() {
        DependencyPhase::Ready
    } else if shared.0.present && shared.1.present {
        DependencyPhase::Binding
    } else {
        DependencyPhase::NeedsConsent
    };
    let ready = bound.is_some();
    Ok(ModelDependencyStatus {
        phase,
        message: if bound.is_some() {
            "Model dependencies are ready".into()
        } else if shared.0.present && shared.1.present {
            "Shared acoustic models are available; bind them to this voice".into()
        } else {
            "HuBERT and RMVPE are required for this voice model".into()
        },
        model_path: Some(model),
        resolved_model_path: bound,
        hubert: shared.0,
        rmvpe: shared.1,
        complete: ready,
        ..Default::default()
    })
}

type Progress<'a> = &'a mut dyn FnMut(
    DependencyPhase,
    String,
    AcousticDependencyStatus,
    AcousticDependencyStatus,
    u64,
    Option<u64>,
) -> LibraryResult<()>;

fn execute(
    request: &ModelDependencyRequest,
    cancel: &AtomicBool,
    progress: Progress<'_>,
) -> LibraryResult<(PathBuf, AcousticDependencyStatus, AcousticDependencyStatus)> {
    let model = vocal_model::resolve_model_path(&request.model_path)?;
    let inspection = vocal_model::inspect_model_artifact(&model)?;
    if inspection.components.automatic_tensor_rt {
        let shared = shared_status(
            &request
                .cache_directory
                .clone()
                .unwrap_or_else(default_library_directory),
        );
        return Ok((model, shared.0, shared.1));
    }
    check_cancel(cancel)?;
    let root = request
        .cache_directory
        .clone()
        .unwrap_or_else(default_library_directory);
    let shared_root = root.join("shared");
    fs::create_dir_all(&shared_root)?;
    let mut h = request
        .hubert_source
        .clone()
        .map(validate_hubert)
        .transpose()?
        .or_else(|| local_shared(&shared_root, "hubert"));
    let mut r = request
        .rmvpe_source
        .clone()
        .map(validate_rmvpe)
        .transpose()?
        .or_else(|| local_shared(&shared_root, "rmvpe"));
    if h.is_none() || r.is_none() {
        progress(
            DependencyPhase::Listing,
            "Resolving the public shared model registry".into(),
            dependency_status("hubert", h.clone(), None),
            dependency_status("rmvpe", r.clone(), None),
            0,
            None,
        )?;
        let catalog = provider::list_cached_with_client(
            &client()?,
            RemoteRepository {
                provider: ModelProvider::ModelScope,
                repository: SHARED_REPOSITORY.into(),
                revision: Some("master".into()),
            },
            &root.join("catalog-cache"),
            std::time::Duration::from_secs(6 * 60 * 60),
        )?;
        let find = |path: &str| {
            catalog
                .files
                .iter()
                .find(|file| file.path == path)
                .cloned()
                .ok_or_else(|| {
                    LibraryError::new(
                        "notFound",
                        format!("Shared model is missing from registry: {path}"),
                    )
                })
        };
        if h.is_none() {
            let entry = find(HUBERT_FILE)?;
            let path = download_shared(
                &entry,
                &catalog.repository,
                &shared_root,
                cancel,
                progress,
                h.clone(),
                r.clone(),
            )?;
            h = Some(path);
        }
        if r.is_none() {
            let entry = find(RMVPE_FILE)?;
            let path = download_shared(
                &entry,
                &catalog.repository,
                &shared_root,
                cancel,
                progress,
                h.clone(),
                r.clone(),
            )?;
            r = Some(path);
        }
    }
    let h = h.ok_or_else(|| LibraryError::new("acoustic", "HuBERT source is unavailable"))?;
    let r = r.ok_or_else(|| LibraryError::new("acoustic", "RMVPE source is unavailable"))?;
    let hs = dependency_status("hubert", Some(h.clone()), None);
    let rs = dependency_status("rmvpe", Some(r.clone()), None);
    progress(
        DependencyPhase::Binding,
        "Binding shared acoustic models to the voice".into(),
        hs.clone(),
        rs.clone(),
        0,
        None,
    )?;
    check_cancel(cancel)?;
    let output = bound_model_path(&model, &(hs.clone(), rs.clone()), &root)
        .ok_or_else(|| LibraryError::new("cache", "Could not create a stable bound model path"))?;
    if !valid_complete(&output) {
        if output.exists() {
            fs::remove_file(&output)?;
        }
        let temp = output.with_extension(format!("tmp-{}.vfrm", std::process::id()));
        let _ = fs::remove_file(&temp);
        vocal_model::bundle_rvc_cache(&model, &h, &r, &temp)?;
        fs::rename(temp, &output)?;
    }
    Ok((output, hs, rs))
}

fn download_shared(
    entry: &RemoteFile,
    repository: &RemoteRepository,
    shared_root: &Path,
    cancel: &AtomicBool,
    progress: Progress<'_>,
    h: Option<PathBuf>,
    r: Option<PathBuf>,
) -> LibraryResult<PathBuf> {
    let cache = shared_root.join(".downloads");
    let downloaded = download(
        &client()?,
        repository,
        entry,
        &cache,
        cancel,
        &mut |bytes, total| {
            let _ = progress(
                DependencyPhase::Downloading,
                format!("Downloading {}", entry.path),
                dependency_status("hubert", h.clone(), None),
                dependency_status("rmvpe", r.clone(), None),
                bytes,
                total,
            );
        },
    )?;
    let kind = if entry.path == HUBERT_FILE {
        "hubert"
    } else {
        "rmvpe"
    };
    let output = shared_root.join(format!("{kind}.vfrm"));
    let existing_valid = if kind == "hubert" {
        validate_hubert(output.clone()).is_ok()
    } else {
        validate_rmvpe(output.clone()).is_ok()
    };
    if !existing_valid {
        if output.exists() {
            fs::remove_file(&output)?;
        }
        let temp = output.with_extension(format!("tmp-{}.vfrm", std::process::id()));
        let _ = fs::remove_file(&temp);
        fs::copy(&downloaded, &temp)?;
        File::open(&temp)?.sync_all()?;
        fs::rename(temp, &output)?;
    }
    if kind == "hubert" {
        validate_hubert(output.clone())?;
    } else {
        validate_rmvpe(output.clone())?;
    }
    Ok(output)
}

fn shared_status(root: &Path) -> (AcousticDependencyStatus, AcousticDependencyStatus) {
    let dir = root.join("shared");
    (
        dependency_status("hubert", local_shared(&dir, "hubert"), Some(HUBERT_SIZE)),
        dependency_status("rmvpe", local_shared(&dir, "rmvpe"), Some(RMVPE_SIZE)),
    )
}

fn local_shared(root: &Path, kind: &str) -> Option<PathBuf> {
    let path = root.join(format!("{kind}.vfrm"));
    let valid = if kind == "hubert" {
        validate_hubert(path.clone()).is_ok()
    } else {
        validate_rmvpe(path.clone()).is_ok()
    };
    valid.then_some(path)
}

fn dependency_status(
    kind: &str,
    path: Option<PathBuf>,
    expected: Option<u64>,
) -> AcousticDependencyStatus {
    let size = path
        .as_ref()
        .and_then(|path| fs::metadata(path).ok())
        .map_or(0, |metadata| metadata.len());
    AcousticDependencyStatus {
        kind: kind.into(),
        present: path.is_some(),
        path,
        size_bytes: size,
        expected_size_bytes: expected,
        source: Some(if kind == "hubert" {
            SHARED_REPOSITORY.into()
        } else {
            SHARED_REPOSITORY.into()
        }),
    }
}

fn validate_hubert(path: PathBuf) -> LibraryResult<PathBuf> {
    let inspection = vocal_model::inspect_model_artifact(&path)?;
    if !inspection.components.hubert.present {
        return Err(LibraryError::new(
            "acoustic",
            "Selected file is not a HuBERT VFRM",
        ));
    }
    Ok(path)
}

fn validate_rmvpe(path: PathBuf) -> LibraryResult<PathBuf> {
    let inspection = vocal_model::inspect_model_artifact(&path)?;
    if !inspection.components.rmvpe.present {
        return Err(LibraryError::new(
            "acoustic",
            "Selected file is not an RMVPE VFRM",
        ));
    }
    Ok(path)
}

fn bound_model_path(
    model: &Path,
    shared: &(AcousticDependencyStatus, AcousticDependencyStatus),
    root: &Path,
) -> Option<PathBuf> {
    let h = shared.0.path.as_ref()?;
    let r = shared.1.path.as_ref()?;
    let mut hash = Sha256::new();
    hash.update(file_hash(model).ok()?.finalize());
    hash.update(file_hash(h).ok()?.finalize());
    hash.update(file_hash(r).ok()?.finalize());
    let digest = hash.finalize();
    let bound = root.join("bound");
    let _ = fs::create_dir_all(&bound);
    Some(bound.join(format!("{:x}.vfrm", digest)))
}

fn valid_complete(path: &Path) -> bool {
    vocal_model::inspect_model_artifact(path)
        .is_ok_and(|inspection| inspection.components.automatic_tensor_rt)
}

fn file_hash(path: &Path) -> LibraryResult<Sha256> {
    let mut input = File::open(path)?;
    let mut hash = Sha256::new();
    let mut bytes = [0; 128 * 1024];
    loop {
        let count = input.read(&mut bytes)?;
        if count == 0 {
            break;
        }
        hash.update(&bytes[..count]);
    }
    Ok(hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_and_phase_contract_are_stable() {
        assert_eq!(SHARED_REPOSITORY, "MosRat/vfrm-models");
        assert!(!DependencyPhase::Ready.is_active());
        assert!(DependencyPhase::Downloading.is_active());
        let value = serde_json::to_value(DependencyPhase::NeedsConsent).unwrap();
        assert_eq!(value, "needsConsent");
    }
}
