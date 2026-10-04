pub const LOG_TARGET: &str = "vocal.model";

pub(crate) fn load_started(operation: &'static str) {
    tracing::debug!(target: LOG_TARGET, operation, "model load started");
}

pub(crate) fn load_finished(operation: &'static str, tensors: usize) {
    tracing::debug!(target: LOG_TARGET, operation, tensors, "model load finished");
}

pub(crate) fn load_failed(operation: &'static str) {
    tracing::warn!(target: LOG_TARGET, operation, "model load failed");
}
