#[derive(Clone, Copy, Debug)]
pub struct CliConfig {
    pub default_demo_blocks: usize,
    pub default_benchmark_iterations: usize,
}

impl Default for CliConfig {
    fn default() -> Self {
        Self {
            default_demo_blocks: 4,
            default_benchmark_iterations: 30,
        }
    }
}

impl CliConfig {
    pub fn benchmark_iterations(self, requested: Option<usize>) -> usize {
        requested
            .filter(|iterations| *iterations > 0)
            .unwrap_or(self.default_benchmark_iterations)
    }
}
