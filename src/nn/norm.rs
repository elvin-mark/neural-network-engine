//! Normalization layers (LayerNorm and BatchNorm1d).

use crate::autograd::Tensor;
use crate::error::Result;
use crate::nn::module::Module;
use crate::tensor::RawTensor;

/// Layer Normalization over the last dimension: y = (x - mean) / sqrt(var + eps) * gamma + beta.
#[derive(Clone)]
pub struct LayerNorm {
    pub normalized_dim: usize,
    pub weight: Tensor, // gamma
    pub bias: Tensor,   // beta
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(normalized_dim: usize) -> Self {
        Self::with_eps(normalized_dim, 1e-5)
    }

    pub fn with_eps(normalized_dim: usize, eps: f32) -> Self {
        Self {
            normalized_dim,
            weight: Tensor::ones(&[normalized_dim], true),
            bias: Tensor::zeros(&[normalized_dim], true),
            eps,
        }
    }
}

impl Module for LayerNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let last_axis = input.ndim() - 1;
        let mean = input.mean(last_axis, true)?;
        let diff = input.sub(&mean)?;
        let diff_sq = diff.powf(2.0)?;
        let var = diff_sq.mean(last_axis, true)?;
        let var_eps = var.add(&Tensor::scalar(self.eps, false))?;
        let std = var_eps.powf(0.5)?;
        let norm = diff.div(&std)?;

        let scaled = norm.mul(&self.weight)?;
        scaled.add(&self.bias)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

/// Root Mean Square Normalization (RMSNorm as used in LLaMA / LLaMA 2):
/// y = x / sqrt(mean(x^2) + eps) * gamma
#[derive(Clone)]
pub struct RMSNorm {
    pub normalized_dim: usize,
    pub weight: Tensor, // gamma
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(normalized_dim: usize) -> Self {
        Self::with_eps(normalized_dim, 1e-6)
    }

    pub fn with_eps(normalized_dim: usize, eps: f32) -> Self {
        Self {
            normalized_dim,
            weight: Tensor::ones(&[normalized_dim], true),
            eps,
        }
    }
}

impl Module for RMSNorm {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let last_axis = input.ndim() - 1;
        let sq = input.powf(2.0)?;
        let mean_sq = sq.mean(last_axis, true)?;
        let mean_sq_eps = mean_sq.add_scalar(self.eps)?;
        let rrms = mean_sq_eps.powf(0.5)?;
        let norm = input.div(&rrms)?;
        norm.mul(&self.weight)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}

/// 1D Batch Normalization over a 2D batch [BatchSize, NumFeatures].
#[derive(Clone)]
pub struct BatchNorm1d {
    pub num_features: usize,
    pub weight: Tensor, // gamma
    pub bias: Tensor,   // beta
    pub running_mean: RawTensor,
    pub running_var: RawTensor,
    pub eps: f32,
    pub momentum: f32,
    pub is_training: bool,
}

impl BatchNorm1d {
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            weight: Tensor::ones(&[1, num_features], true),
            bias: Tensor::zeros(&[1, num_features], true),
            running_mean: RawTensor::zeros(&[1, num_features]),
            running_var: RawTensor::ones(&[1, num_features]),
            eps: 1e-5,
            momentum: 0.1,
            is_training: true,
        }
    }
}

impl Module for BatchNorm1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        if self.is_training {
            let mean = input.mean(0, true)?;
            let diff = input.sub(&mean)?;
            let diff_sq = diff.powf(2.0)?;
            let var = diff_sq.mean(0, true)?;

            let var_eps = var.add(&Tensor::scalar(self.eps, false))?;
            let std = var_eps.powf(0.5)?;
            let norm = diff.div(&std)?;

            let scaled = norm.mul(&self.weight)?;
            scaled.add(&self.bias)
        } else {
            let mean_tensor = Tensor::new(self.running_mean.clone(), false);
            let var_tensor = Tensor::new(self.running_var.clone(), false);

            let diff = input.sub(&mean_tensor)?;
            let var_eps = var_tensor.add(&Tensor::scalar(self.eps, false))?;
            let std = var_eps.powf(0.5)?;
            let norm = diff.div(&std)?;

            let scaled = norm.mul(&self.weight)?;
            scaled.add(&self.bias)
        }
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }

    fn train(&mut self) {
        self.is_training = true;
    }

    fn eval(&mut self) {
        self.is_training = false;
    }
}
