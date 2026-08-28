//! Optimization algorithms (SGD, Adam, AdamW, RMSprop), gradient clipping, and learning rate schedulers.

pub mod adam;
pub mod amp;
pub mod clip;
pub mod rmsprop;
pub mod scheduler;
pub mod sgd;

use crate::autograd::Tensor;
use crate::error::Result;

pub use adam::Adam;
pub use amp::LossScaler;
pub use clip::{clip_grad_norm, clip_grad_value};
pub use rmsprop::RMSprop;
pub use scheduler::{
    CosineAnnealingLR, ExponentialLR, LRScheduler, LinearWarmupCosineLR, MultiStepLR, StepLR,
};
pub use sgd::SGD;

/// Common trait implemented by all first-order optimizers.
pub trait Optimizer {
    /// Performs a single optimization parameter update step.
    fn step(&mut self) -> Result<()>;

    /// Clears the gradients of all optimized parameters.
    fn zero_grad(&self);

    /// Returns the current learning rate.
    fn get_lr(&self) -> f32;

    /// Sets the learning rate for subsequent parameter updates.
    fn set_lr(&mut self, lr: f32);

    /// Returns a slice of the parameter tensors managed by this optimizer.
    fn params(&self) -> &[Tensor];
}

impl Optimizer for SGD {
    fn step(&mut self) -> Result<()> {
        self.step()
    }

    fn zero_grad(&self) {
        self.zero_grad();
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn params(&self) -> &[Tensor] {
        &self.params
    }
}

impl Optimizer for Adam {
    fn step(&mut self) -> Result<()> {
        self.step()
    }

    fn zero_grad(&self) {
        self.zero_grad();
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn params(&self) -> &[Tensor] {
        &self.params
    }
}

impl Optimizer for RMSprop {
    fn step(&mut self) -> Result<()> {
        self.step()
    }

    fn zero_grad(&self) {
        self.zero_grad();
    }

    fn get_lr(&self) -> f32 {
        self.lr
    }

    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn params(&self) -> &[Tensor] {
        &self.params
    }
}
