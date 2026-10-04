use std::sync::OnceLock;
use std::sync::atomic::{AtomicU64, Ordering};

/// Control-plane backend error storage. Audio callbacks never need a mutex or
/// an ever-growing allocation list when a device starts returning the same
/// fatal error repeatedly.
#[derive(Debug, Default)]
pub(crate) struct BackendErrorState {
    first: OnceLock<String>,
    count: AtomicU64,
}

impl BackendErrorState {
    pub(crate) fn record(&self, error: String) {
        let _ = self.first.set(error);
        self.count.fetch_add(1, Ordering::Release);
    }

    pub(crate) fn take(&self) -> Vec<String> {
        let count = self.count.swap(0, Ordering::AcqRel);
        if count == 0 {
            return Vec::new();
        }
        let Some(first) = self.first.get() else {
            return Vec::new();
        };
        let mut errors = vec![first.clone()];
        if count > 1 {
            errors.push(format!(
                "{count} backend errors occurred; first error shown above"
            ));
        }
        errors
    }
}
