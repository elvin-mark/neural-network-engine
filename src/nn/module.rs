//! Base Module trait for composable neural network architectures.

use crate::autograd::Tensor;
use crate::error::Result;

/// Common trait implemented by all neural network layers and containers.
pub trait Module: Send + Sync {
    /// Computes the forward pass of the module.
    fn forward(&self, input: &Tensor) -> Result<Tensor>;

    /// Returns a vector containing all learnable parameter tensors in this module.
    fn parameters(&self) -> Vec<Tensor>;

    /// Resets the gradients of all learnable parameters to zero.
    fn zero_grad(&self) {
        for param in self.parameters() {
            param.zero_grad();
        }
    }

    /// Sets the module and all submodules to training mode (enabling dropout, batch norm updates, etc.).
    fn train(&mut self) {}

    /// Sets the module and all submodules to evaluation mode (disabling dropout, using running stats in batch norm).
    fn eval(&mut self) {}
}
