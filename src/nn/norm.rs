//! Normalization layers (LayerNorm and BatchNorm1d).

use crate::autograd::Tensor;
use crate::error::{EngineError, Result};
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
        let shape = input.shape();
        if shape.is_empty() {
            return Err(EngineError::InvalidArgument(
                "LayerNorm cannot be applied to a 0-D scalar tensor".to_string(),
            ));
        }
        let last_dim = *shape.last().unwrap();
        if last_dim != self.normalized_dim {
            return Err(EngineError::ShapeMismatch {
                expected: vec![self.normalized_dim],
                actual: vec![last_dim],
            });
        }

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
        let shape = input.shape();
        if shape.is_empty() {
            return Err(EngineError::InvalidArgument(
                "RMSNorm cannot be applied to a 0-D scalar tensor".to_string(),
            ));
        }
        let last_dim = *shape.last().unwrap();
        if last_dim != self.normalized_dim {
            return Err(EngineError::ShapeMismatch {
                expected: vec![self.normalized_dim],
                actual: vec![last_dim],
            });
        }

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

use std::sync::{Arc, RwLock};

/// 1D Batch Normalization over a 2D batch [BatchSize, NumFeatures].
#[derive(Clone)]
pub struct BatchNorm1d {
    pub num_features: usize,
    pub weight: Tensor, // gamma
    pub bias: Tensor,   // beta
    pub running_mean: Arc<RwLock<RawTensor>>,
    pub running_var: Arc<RwLock<RawTensor>>,
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
            running_mean: Arc::new(RwLock::new(RawTensor::zeros(&[1, num_features]))),
            running_var: Arc::new(RwLock::new(RawTensor::ones(&[1, num_features]))),
            eps: 1e-5,
            momentum: 0.1,
            is_training: true,
        }
    }

    /// Returns the current running mean tensor.
    pub fn running_mean(&self) -> RawTensor {
        self.running_mean.read().unwrap().clone()
    }

    /// Returns the current running variance tensor.
    pub fn running_var(&self) -> RawTensor {
        self.running_var.read().unwrap().clone()
    }
}

impl Module for BatchNorm1d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape();
        if shape.len() != 2 {
            return Err(EngineError::InvalidArgument(format!(
                "BatchNorm1d expects 2D tensor [BatchSize, NumFeatures], got rank {} with shape {:?}",
                shape.len(),
                shape
            )));
        }
        if shape[1] != self.num_features {
            return Err(EngineError::ShapeMismatch {
                expected: vec![self.num_features],
                actual: vec![shape[1]],
            });
        }

        if self.is_training {
            let mean = input.mean(0, true)?;
            let diff = input.sub(&mean)?;
            let diff_sq = diff.powf(2.0)?;
            let var = diff_sq.mean(0, true)?;

            // Update exponential moving average running statistics
            let m = self.momentum;
            let batch_mean = mean.data();
            let batch_var = var.data();
            {
                let mut rm = self.running_mean.write().unwrap();
                let scaled_rm = rm.mul_scalar(1.0 - m)?;
                let scaled_bm = batch_mean.mul_scalar(m)?;
                *rm = scaled_rm.add(&scaled_bm)?;
            }
            {
                let mut rv = self.running_var.write().unwrap();
                let scaled_rv = rv.mul_scalar(1.0 - m)?;
                let scaled_bv = batch_var.mul_scalar(m)?;
                *rv = scaled_rv.add(&scaled_bv)?;
            }

            let var_eps = var.add(&Tensor::scalar(self.eps, false))?;
            let std = var_eps.powf(0.5)?;
            let norm = diff.div(&std)?;

            let scaled = norm.mul(&self.weight)?;
            scaled.add(&self.bias)
        } else {
            let mean_raw = self.running_mean.read().unwrap().clone();
            let var_raw = self.running_var.read().unwrap().clone();
            let mean_tensor = Tensor::new(mean_raw, false);
            let var_tensor = Tensor::new(var_raw, false);

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

/// 2D Batch Normalization over a 4D spatial batch [BatchSize, NumChannels, Height, Width].
#[derive(Clone)]
pub struct BatchNorm2d {
    pub num_features: usize,
    pub weight: Tensor, // gamma [1, C, 1, 1]
    pub bias: Tensor,   // beta [1, C, 1, 1]
    pub running_mean: Arc<RwLock<RawTensor>>,
    pub running_var: Arc<RwLock<RawTensor>>,
    pub eps: f32,
    pub momentum: f32,
    pub is_training: bool,
}

impl BatchNorm2d {
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            weight: Tensor::ones(&[1, num_features, 1, 1], true),
            bias: Tensor::zeros(&[1, num_features, 1, 1], true),
            running_mean: Arc::new(RwLock::new(RawTensor::zeros(&[1, num_features, 1, 1]))),
            running_var: Arc::new(RwLock::new(RawTensor::ones(&[1, num_features, 1, 1]))),
            eps: 1e-5,
            momentum: 0.1,
            is_training: true,
        }
    }

    /// Returns the current running mean tensor.
    pub fn running_mean(&self) -> RawTensor {
        self.running_mean.read().unwrap().clone()
    }

    /// Returns the current running variance tensor.
    pub fn running_var(&self) -> RawTensor {
        self.running_var.read().unwrap().clone()
    }
}

impl Module for BatchNorm2d {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(EngineError::InvalidArgument(format!(
                "BatchNorm2d expects 4D tensor [BatchSize, Channels, Height, Width], got rank {} with shape {:?}",
                shape.len(),
                shape
            )));
        }
        if shape[1] != self.num_features {
            return Err(EngineError::ShapeMismatch {
                expected: vec![self.num_features],
                actual: vec![shape[1]],
            });
        }

        if self.is_training {
            let m_w = input.mean(3, true)?;
            let m_hw = m_w.mean(2, true)?;
            let mean = m_hw.mean(0, true)?; // [1, C, 1, 1]

            let diff = input.sub(&mean)?;
            let diff_sq = diff.powf(2.0)?;
            let v_w = diff_sq.mean(3, true)?;
            let v_hw = v_w.mean(2, true)?;
            let var = v_hw.mean(0, true)?; // [1, C, 1, 1]

            // Update exponential moving average running statistics
            let m = self.momentum;
            let batch_mean = mean.data();
            let batch_var = var.data();
            {
                let mut rm = self.running_mean.write().unwrap();
                let scaled_rm = rm.mul_scalar(1.0 - m)?;
                let scaled_bm = batch_mean.mul_scalar(m)?;
                *rm = scaled_rm.add(&scaled_bm)?;
            }
            {
                let mut rv = self.running_var.write().unwrap();
                let scaled_rv = rv.mul_scalar(1.0 - m)?;
                let scaled_bv = batch_var.mul_scalar(m)?;
                *rv = scaled_rv.add(&scaled_bv)?;
            }

            let var_eps = var.add(&Tensor::scalar(self.eps, false))?;
            let std = var_eps.powf(0.5)?;
            let norm = diff.div(&std)?;

            let scaled = norm.mul(&self.weight)?;
            scaled.add(&self.bias)
        } else {
            let mean_raw = self.running_mean.read().unwrap().clone();
            let var_raw = self.running_var.read().unwrap().clone();
            let mean_tensor = Tensor::new(mean_raw, false);
            let var_tensor = Tensor::new(var_raw, false);

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
