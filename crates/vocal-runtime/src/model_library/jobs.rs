use super::*;
use std::{
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};
type Listener = Arc<dyn Fn(ModelImportStatus) + Send + Sync>;
struct State {
    status: ModelImportStatus,
    cancel: Arc<AtomicBool>,
    started: Instant,
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bounded_job_and_cancel_identity_are_enforced() {
        let library = ModelLibrary::new(|_| {});
        {
            let mut state = library.state.lock().unwrap();
            state.status.job_id = 10;
            state.status.phase = ImportPhase::Converting;
        }
        let request = ModelImportRequest {
            source: ImportSource::Local {
                model_path: "unused.pth".into(),
                index_path: None,
            },
            display_name: "Voice".into(),
            library_directory: None,
            acoustic_bundle: None,
            hubert_source: None,
            rmvpe_source: None,
        };
        assert_eq!(library.start(request).unwrap_err().code, "busy");
        assert_eq!(library.cancel(9).unwrap_err().code, "staleJob");
        assert_eq!(library.cancel(10).unwrap().phase, ImportPhase::Cancelling);
        assert!(library.state.lock().unwrap().cancel.load(Ordering::Acquire));
        library.state.lock().unwrap().status.phase = ImportPhase::Publishing;
        assert_eq!(library.cancel(10).unwrap_err().code, "publishing");
    }
}
#[derive(Clone)]
pub struct ModelLibrary {
    state: Arc<Mutex<State>>,
    listener: Listener,
}
impl ModelLibrary {
    pub fn new(listener: impl Fn(ModelImportStatus) + Send + Sync + 'static) -> Self {
        Self {
            state: Arc::new(Mutex::new(State {
                status: ModelImportStatus::default(),
                cancel: Arc::new(AtomicBool::new(false)),
                started: Instant::now(),
            })),
            listener: Arc::new(listener),
        }
    }
    pub fn status(&self) -> ModelImportStatus {
        self.state
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .status
            .clone()
    }
    pub fn start(&self, request: ModelImportRequest) -> LibraryResult<ModelImportStatus> {
        if request
            .library_directory
            .as_ref()
            .is_some_and(|p| !p.is_absolute())
        {
            return Err(LibraryError::new(
                "path",
                "Choose an absolute destination library path",
            ));
        }
        if request.display_name.trim().is_empty() || request.display_name.len() > 512 {
            return Err(LibraryError::new(
                "name",
                "Enter a model name (up to 512 UTF-8 bytes)",
            ));
        }
        let (status, cancel) = {
            let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if s.status.phase.is_active() {
                return Err(LibraryError::new(
                    "busy",
                    "A model import is already running",
                ));
            }
            s.started = Instant::now();
            s.cancel = Arc::new(AtomicBool::new(false));
            s.status = ModelImportStatus {
                job_id: s.status.job_id + 1,
                revision: 1,
                phase: ImportPhase::Queued,
                message: "Preparing model import".into(),
                library_directory: Some(
                    request
                        .library_directory
                        .clone()
                        .unwrap_or_else(default_library_directory),
                ),
                ..Default::default()
            };
            (s.status.clone(), s.cancel.clone())
        };
        (self.listener)(status.clone());
        let accepted = status.clone();
        let service = self.clone();
        if let Err(error) = std::thread::Builder::new()
            .name("model-library-import".into())
            .spawn(move || {
                let job_id = status.job_id;
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    convert::execute(
                        &request,
                        &cancel,
                        &mut |phase, message, file, downloaded, total| {
                            let status = {
                                let mut s = service.state.lock().unwrap_or_else(|e| e.into_inner());
                                check_cancel(&s.cancel)?;
                                s.status.phase = phase;
                                s.status.message = message;
                                s.status.file = file;
                                s.status.downloaded_bytes = downloaded;
                                s.status.total_bytes = total;
                                s.status.revision += 1;
                                s.status.elapsed_ms =
                                    s.started.elapsed().as_millis().min(u64::MAX as u128) as u64;
                                s.status.clone()
                            };
                            (service.listener)(status);
                            Ok(())
                        },
                    )
                }));
                let result = result.unwrap_or_else(|_| {
                    Err(LibraryError::new(
                        "internal",
                        "Model importer failed unexpectedly; the original files were preserved",
                    ))
                });
                service.update(job_id, |s| match result {
                    Ok((path, complete, warnings)) => {
                        s.phase = ImportPhase::Ready;
                        s.message = "Model added to your library".into();
                        s.output_path = Some(path);
                        s.complete = complete;
                        s.warnings = warnings;
                        s.error = None;
                    }
                    Err(e) => {
                        s.phase = if e.code == "cancelled" {
                            ImportPhase::Cancelled
                        } else {
                            ImportPhase::Failed
                        };
                        s.message = e.message.clone();
                        s.error = Some(e);
                    }
                });
                crate::memory::collect_idle();
            })
        {
            self.update(self.status().job_id, |s| {
                s.phase = ImportPhase::Failed;
                s.error = Some(LibraryError::new("internal", error.to_string()));
            });
            return Err(LibraryError::new("internal", error.to_string()));
        }
        Ok(accepted)
    }
    pub fn cancel(&self, job_id: u64) -> LibraryResult<ModelImportStatus> {
        {
            let s = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if s.status.job_id != job_id {
                return Err(LibraryError::new(
                    "staleJob",
                    "This import is no longer current",
                ));
            }
            if s.status.phase == ImportPhase::Publishing {
                return Err(LibraryError::new(
                    "publishing",
                    "The model is being committed; wait for completion",
                ));
            }
            if !s.status.phase.is_active() {
                return Ok(s.status.clone());
            }
            s.cancel.store(true, Ordering::Release);
        }
        self.update(job_id, |s| {
            if s.phase.is_active() {
                s.phase = ImportPhase::Cancelling;
                s.message = "Cancelling; finishing the current conversion step".into();
            }
        });
        Ok(self.status())
    }
    fn update(&self, job_id: u64, change: impl FnOnce(&mut ModelImportStatus)) {
        let status = {
            let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
            if s.status.job_id != job_id {
                return;
            }
            change(&mut s.status);
            s.status.revision += 1;
            s.status.elapsed_ms = s.started.elapsed().as_millis().min(u64::MAX as u128) as u64;
            s.status.clone()
        };
        (self.listener)(status);
    }
}
