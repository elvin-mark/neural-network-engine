//! Automatic Mixed Precision (AMP) and Dynamic Loss Scaling.
//!
//! Prevents gradient underflow when executing half-precision calculations,
//! maintaining full-precision master weights for numerical stability.

use crate::autograd::Tensor;
use crate::error::Result;
use crate::optim::Optimizer;
use std::sync::atomic::{AtomicU32, Ordering};

/// Dynamic Loss Scaler for Mixed Precision Training.
///
/// Multiplies loss by `scale` during backward pass to push gradients above FP16 underflow threshold,
/// then un-scales gradients before optimizer step. Automatically adjusts `scale` when NaNs/Infs are detected.
pub struct LossScaler {
    scale: AtomicU32,
    growth_factor: f32,
    backoff_factor: f32,
    growth_interval: usize,
    steps_since_growth: AtomicU32,
    min_scale: f32,
    max_scale: f32,
}

impl LossScaler {
    /// Creates a new `LossScaler` with standard hyper-parameters.
    pub fn new(init_scale: f32) -> Self {
        Self {
            scale: AtomicU32::new(init_scale.to_bits()),
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            steps_since_growth: AtomicU32::new(0),
            min_scale: 1.0,
            max_scale: 65536.0,
        }
    }

    /// Returns current loss scale.
    pub fn current_scale(&self) -> f32 {
        f32::from_bits(self.scale.load(Ordering::Relaxed))
    }

    /// Scales the loss tensor before calling `loss.backward()`.
    pub fn scale(&self, loss: &Tensor) -> Result<Tensor> {
        let current = self.current_scale();
        loss.mul_scalar(current)
    }

    /// Unscales gradients across parameter tensors before optimizer step.
    /// Returns true if all gradients are finite (no NaNs or Infs).
    pub fn unscale_grads(&self, parameters: &[Tensor]) -> bool {
        let current = self.current_scale();
        let inv_scale = 1.0 / current;
        let mut all_finite = true;

        for param in parameters {
            if let Some(grad) = param.grad() {
                let grad_contig = grad.to_contiguous();
                let slice = grad_contig.as_slice();

                for &val in slice {
                    if !val.is_finite() {
                        all_finite = false;
                        break;
                    }
                }

                if all_finite {
                    let mut unscaled_data = Vec::with_capacity(slice.len());
                    for &val in slice {
                        unscaled_data.push(val * inv_scale);
                    }
                    param.set_grad(Some(crate::tensor::RawTensor::from_vec(
                        unscaled_data,
                        grad.shape().to_vec(),
                    )));
                } else {
                    break;
                }
            }
        }

        all_finite
    }

    /// Performs optimizer step using the optimizer's internal parameters if gradients are finite.
    /// Skips step and reduces scale if NaNs/Infs are encountered.
    pub fn step<O: Optimizer>(&self, optimizer: &mut O) -> Result<bool> {
        let finite = self.unscale_grads(optimizer.params());

        if finite {
            optimizer.step()?;
            let steps = self.steps_since_growth.fetch_add(1, Ordering::Relaxed) + 1;
            if steps >= self.growth_interval as u32 {
                let current = self.current_scale();
                let next = (current * self.growth_factor).min(self.max_scale);
                self.scale.store(next.to_bits(), Ordering::Relaxed);
                self.steps_since_growth.store(0, Ordering::Relaxed);
            }
            Ok(true)
        } else {
            // Found NaN/Inf: skip optimizer step and back off scale
            let current = self.current_scale();
            let next = (current * self.backoff_factor).max(self.min_scale);
            self.scale.store(next.to_bits(), Ordering::Relaxed);
            self.steps_since_growth.store(0, Ordering::Relaxed);

            // Zero out corrupt gradients
            optimizer.zero_grad();
            Ok(false)
        }
    }
}

impl Default for LossScaler {
    fn default() -> Self {
        Self::new(1024.0)
    }
}
