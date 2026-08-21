//! Dynamic reverse-mode automatic differentiation engine.

pub mod context;
pub mod node;
pub mod tensor;

pub use context::{is_grad_enabled, no_grad, set_grad_enabled, NoGradGuard};
pub use node::TensorInner;
pub use tensor::Tensor;
