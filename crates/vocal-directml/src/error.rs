#[derive(Debug, thiserror::Error)]
pub enum DirectMlError {
    #[error(transparent)]
    Architecture(#[from] vocal_graph::GraphError),
    #[error("invalid DirectML configuration: {0}")]
    Config(String),
    #[error("DirectML graph: {0}")]
    Graph(String),
    #[error("DirectML runtime: {0}")]
    Runtime(String),
    #[error(transparent)]
    Model(#[from] vocal_model::ModelError),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}
