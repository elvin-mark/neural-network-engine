//! Adam and AdamW optimizers with first and second moment estimations and decoupled weight decay.

use crate::autograd::Tensor;
use crate::error::Result;
use crate::tensor::RawTensor;

/// Adam / AdamW optimizer.
pub struct Adam {
    pub params: Vec<Tensor>,
    pub lr: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub decoupled_weight_decay: bool,
    t: usize,
    m: Vec<RawTensor>,
    v: Vec<RawTensor>,
}

impl Adam {
    /// Creates a standard Adam optimizer.
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let m = params
            .iter()
            .map(|p| RawTensor::zeros(&p.shape()))
            .collect();
        let v = params
            .iter()
            .map(|p| RawTensor::zeros(&p.shape()))
            .collect();
        Self {
            params,
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            decoupled_weight_decay: false,
            t: 0,
            m,
            v,
        }
    }

    /// Creates an AdamW optimizer with decoupled weight decay.
    pub fn adamw(params: Vec<Tensor>, lr: f32, weight_decay: f32) -> Self {
        let mut opt = Self::new(params, lr);
        opt.weight_decay = weight_decay;
        opt.decoupled_weight_decay = true;
        opt
    }

    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.beta1 = beta1;
        self.beta2 = beta2;
        self
    }

    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }

    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Performs a single optimization step.
    pub fn step(&mut self) -> Result<()> {
        self.t += 1;
        let bias_correction1 = 1.0 - self.beta1.powi(self.t as i32);
        let bias_correction2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, param) in self.params.iter().enumerate() {
            let grad_opt = param.grad();
            if let Some(grad) = grad_opt {
                let mut data = param.data();

                // Decoupled weight decay (AdamW)
                if self.decoupled_weight_decay && self.weight_decay != 0.0 {
                    let decay = data.mul_scalar(self.lr * self.weight_decay)?;
                    data = data.sub(&decay)?;
                }

                let mut g = grad;
                // Standard L2 weight decay (Adam)
                if !self.decoupled_weight_decay && self.weight_decay != 0.0 {
                    let wd = data.mul_scalar(self.weight_decay)?;
                    g = g.add(&wd)?;
                }

                // Update biased first moment: m = beta1 * m + (1 - beta1) * g
                let m_scaled = self.m[i].mul_scalar(self.beta1)?;
                let g_scaled = g.mul_scalar(1.0 - self.beta1)?;
                self.m[i] = m_scaled.add(&g_scaled)?;

                // Update biased second moment: v = beta2 * v + (1 - beta2) * g^2
                let v_scaled = self.v[i].mul_scalar(self.beta2)?;
                let g2 = g.mul(&g)?;
                let g2_scaled = g2.mul_scalar(1.0 - self.beta2)?;
                self.v[i] = v_scaled.add(&g2_scaled)?;

                // Bias-corrected estimates
                let m_hat = self.m[i].div_scalar(bias_correction1)?;
                let v_hat = self.v[i].div_scalar(bias_correction2)?;

                // Step: theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
                let sqrt_v = v_hat.sqrt()?;
                let denom = sqrt_v.add_scalar(self.eps)?;
                let step = m_hat.div(&denom)?;
                let update = step.mul_scalar(self.lr)?;

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
