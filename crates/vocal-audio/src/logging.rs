//! Centralized control-plane logging for audio backends.
//!
//! The helpers in this module are never called from the normal callback data
//! path. They keep log target and field names stable while allowing an
//! application to install any `tracing` subscriber it prefers.

#![allow(dead_code)]

use super::error::{AudioFaultDirection, AudioFaultKind};
use super::state::StreamStatus;

pub(crate) fn state(component: &'static str, status: StreamStatus) {
    tracing::info!(
        target: "vocal.audio",
        component,
        lifecycle = ?status.lifecycle,
        transitions = status.transitions,
        faults = status.faults,
        "audio stream state changed"
    );
}

pub(crate) fn started(component: &'static str) {
    tracing::info!(target: "vocal.audio", component, "audio backend started");
}

pub(crate) fn stopped(component: &'static str) {
    tracing::debug!(target: "vocal.audio", component, "audio backend stopped");
}

pub(crate) fn fault(
    component: &'static str,
    kind: AudioFaultKind,
    direction: Option<AudioFaultDirection>,
    message: &str,
) {
    tracing::error!(
        target: "vocal.audio",
        component,
        kind = ?kind,
        direction = ?direction,
        error = message,
        "audio backend fault"
    );
}
