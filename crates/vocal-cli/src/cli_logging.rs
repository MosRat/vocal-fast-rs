pub const LOG_TARGET: &str = "vocal_cli";

pub fn init() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(true)
        .try_init();
}

pub fn command_started(command: &str) {
    tracing::debug!(target: LOG_TARGET, command, "command started");
}

pub fn command_finished(command: &str) {
    tracing::debug!(target: LOG_TARGET, command, "command finished");
}

pub fn command_failed(command: &str, error: &dyn std::fmt::Display) {
    tracing::warn!(target: LOG_TARGET, command, error = %error, "command failed");
}
