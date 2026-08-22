//! Error and Result definitions for the neural network engine.

use thiserror::Error;

/// Result alias for neural network engine operations.
pub type Result<T> = std::result::Result<T, EngineError>;

/// Engine error enumeration representing all tensor, autograd, and layer errors.
#[derive(Error, Debug, Clone, PartialEq)]
pub enum EngineError {
    #[error("Shape mismatch error: expected shape {expected:?}, got {actual:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("Incompatible shapes for operation {op}: {shapes:?}")]
    IncompatibleShapes {
        op: &'static str,
        shapes: Vec<Vec<usize>>,
    },

    #[error("Dimension out of bounds: axis {axis} for tensor with {ndim} dimensions")]
    DimensionOutOfBounds { axis: usize, ndim: usize },

    #[error("Invalid broadcast shapes: cannot broadcast from {from:?} to {to:?}")]
    BroadcastError { from: Vec<usize>, to: Vec<usize> },

    #[error("Index out of bounds: index {index} out of range for dimension with size {size}")]
    IndexOutOfBounds { index: usize, size: usize },

    #[error("Invalid convolution parameters: {details}")]
    InvalidConvParams { details: String },

    #[error("Gradient computation error: {0}")]
    GradientError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("Invalid configuration or argument: {0}")]
    InvalidArgument(String),

    #[error("Tokenizer error: {0}")]
    TokenizerError(String),

    #[error("GPU compute error: {0}")]
    GpuError(String),

    #[error("Tensor is not contiguous in memory")]
    NonContiguousTensor,
}
