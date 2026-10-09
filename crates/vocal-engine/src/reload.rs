//! Idle configuration updates are prepared before persistence and committed afterwards.

/// A fully prepared update borrowing its owning engine. Dropping it discards
/// the candidate; committing performs only infallible state replacement.
/// This lets controllers persist configuration without rollback I/O.
pub trait PreparedEngineReload {
    fn commit(self: Box<Self>);
}

pub(crate) fn prepared_reload<'a>(
    commit: impl FnOnce() + 'a,
) -> Box<dyn PreparedEngineReload + 'a> {
    Box::new(PreparedCommit(commit))
}

struct PreparedCommit<F>(F);

impl<F: FnOnce()> PreparedEngineReload for PreparedCommit<F> {
    fn commit(self: Box<Self>) {
        (self.0)();
    }
}
