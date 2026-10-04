pub const LOG_TARGET: &str = "vocal_cli";

pub fn command_started(command: &str) {
    tracing::debug!(target: LOG_TARGET, command, "command started");
}

pub fn command_finished(command: &str) {
    tracing::debug!(target: LOG_TARGET, command, "command finished");
}

pub fn command_failed(command: &str, error: &dyn std::fmt::Display) {
    tracing::warn!(target: LOG_TARGET, command, error = %error, "command failed");
}
