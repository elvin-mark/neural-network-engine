//! RMSprop optimizer.

use crate::autograd::Tensor;
use crate::error::Result;
use crate::tensor::RawTensor;

/// RMSprop optimizer.
pub struct RMSprop {
    pub params: Vec<Tensor>,
    pub lr: f32,
    pub alpha: f32,
    pub eps: f32,
    pub weight_decay: f32,
    pub momentum: f32,
    v: Vec<RawTensor>,
    buf: Vec<Option<RawTensor>>,
}

impl RMSprop {
    pub fn new(params: Vec<Tensor>, lr: f32) -> Self {
        let n = params.len();
        let v = params
            .iter()
            .map(|p| RawTensor::zeros(&p.shape()))
            .collect();
        Self {
            params,
            lr,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            v,
            buf: vec![None; n],
        }
    }

    pub fn step(&mut self) -> Result<()> {
        for (i, param) in self.params.iter().enumerate() {
            let grad_opt = param.grad();
            if let Some(grad) = grad_opt {
                let data = param.data();
                let mut g = grad;

                if self.weight_decay != 0.0 {
                    let wd = data.mul_scalar(self.weight_decay)?;
                    g = g.add(&wd)?;
                }

                // v = alpha * v + (1 - alpha) * g^2
                let v_scaled = self.v[i].mul_scalar(self.alpha)?;
                let g2 = g.mul(&g)?;
                let g2_scaled = g2.mul_scalar(1.0 - self.alpha)?;
                self.v[i] = v_scaled.add(&g2_scaled)?;

                let avg = self.v[i].sqrt()?.add_scalar(self.eps)?;
                let mut step = g.div(&avg)?;

                if self.momentum != 0.0 {
                    let buf = match self.buf[i].take() {
                        Some(prev_b) => {
                            let b_scaled = prev_b.mul_scalar(self.momentum)?;
                            b_scaled.add(&step)?
                        }
                        None => step.clone(),
                    };
                    step = buf.clone();
                    self.buf[i] = Some(buf);
                }

                let update = step.mul_scalar(self.lr)?;
                let new_data = data.sub(&update)?;
                param.set_data(new_data);
            }
        }
        Ok(())
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
}
