use crate::{AudioFault, StreamState};

/// Control-plane backend error storage. Audio callbacks never need a mutex or
/// an ever-growing allocation list when a device starts returning the same
/// fatal error repeatedly.
#[derive(Debug)]
pub(crate) struct BackendErrorState {
    state: std::sync::Arc<StreamState>,
    component: &'static str,
}

impl BackendErrorState {
    pub(crate) fn new(state: std::sync::Arc<StreamState>, component: &'static str) -> Self {
        Self { state, component }
    }

    pub(crate) fn record(&self, error: String) {
        let first = self
            .state
            .record_fault(crate::AudioFaultKind::Unknown, None, error.clone());
        if first {
            crate::logging::fault(self.component, crate::AudioFaultKind::Unknown, None, &error);
        }
    }

    pub(crate) fn take(&self) -> Vec<String> {
        self.take_faults()
            .into_iter()
            .flat_map(|fault| {
                if fault.occurrences > 1 {
                    vec![
                        fault.message,
                        format!(
                            "{} backend errors occurred; first error shown above",
                            fault.occurrences
                        ),
                    ]
                } else {
                    vec![fault.message]
                }
            })
            .collect()
    }

    pub(crate) fn take_faults(&self) -> Vec<AudioFault> {
        self.state.take_faults()
    }
}
