#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum CliLifecycle {
    #[default]
    Parsed,
    Running,
    Completed,
    Failed,
}

#[derive(Clone, Debug, Default)]
pub struct CliStatus {
    pub state: CliLifecycle,
    pub command_count: u64,
}

impl CliStatus {
    pub fn start(&mut self) {
        self.state = CliLifecycle::Running;
        self.command_count = self.command_count.saturating_add(1);
    }

    pub fn complete(&mut self) {
        self.state = CliLifecycle::Completed;
    }

    pub fn fail(&mut self) {
        self.state = CliLifecycle::Failed;
    }
}
