pub const LOG_TARGET: &str = "vocal_engine";

pub(crate) fn created(backend: &'static str) {
    tracing::debug!(target: LOG_TARGET, backend, "engine created");
}

pub(crate) fn failed(error: &dyn std::fmt::Display) {
    tracing::warn!(target: LOG_TARGET, error = %error, "engine operation failed");
}
