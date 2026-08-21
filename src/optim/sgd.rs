//! Stochastic Gradient Descent (SGD) optimizer with momentum, weight decay, and Nesterov acceleration.

use crate::autograd::Tensor;
use crate::error::Result;
use crate::tensor::RawTensor;

/// Stochastic Gradient Descent optimizer.
pub struct SGD {
    pub params: Vec<Tensor>,
    pub lr: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    pub nesterov: bool,
    velocities: Vec<Option<RawTensor>>,
}

impl SGD {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let n = params.len();
        Self {
            params,
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            velocities: vec![None; n],
        }
    }

    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    /// Performs a single optimization step.
    pub fn step(&mut self) -> Result<()> {
        for (i, param) in self.params.iter().enumerate() {
            let grad_opt = param.grad();
            if let Some(grad) = grad_opt {
                let data = param.data();
                let mut g = grad;

                // Weight decay (L2 penalty): g = g + weight_decay * data
                if self.weight_decay != 0.0 {
                    let wd = data.mul_scalar(self.weight_decay)?;
                    g = g.add(&wd)?;
                }

                // Momentum
                if self.momentum != 0.0 {
                    let v = match self.velocities[i].take() {
                        Some(prev_v) => {
                            let scaled_v = prev_v.mul_scalar(self.momentum)?;
                            scaled_v.add(&g)?
                        }
                        None => g.clone(),
                    };

                    g = if self.nesterov {
                        let scaled_v = v.mul_scalar(self.momentum)?;
                        g.add(&scaled_v)?
                    } else {
                        v.clone()
                    };

                    self.velocities[i] = Some(v);
                }

                // Update parameter data: data = data - lr * g
                let update = g.mul_scalar(self.lr)?;
                let new_data = data.sub(&update)?;
                param.set_data(new_data);
            }
        }
        Ok(())
    }

    /// Resets parameter gradients to zero.
    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
}
