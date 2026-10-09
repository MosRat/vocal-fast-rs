//! Backend-independent, fixed-shape VFRM model graphs.
pub mod graph;
pub mod models;
mod precision;
pub use precision::{Precision, PrecisionConfig, RetrievalSearch};
#[derive(Debug, thiserror::Error)]
pub enum GraphError {
    #[error("model graph: {0}")]
    Graph(String),
    #[error("graph configuration: {0}")]
    Config(String),
    #[error(transparent)]
    Model(#[from] vocal_model::ModelError),
}
