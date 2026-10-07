//! Normalization layers (LayerNorm and BatchNorm1d).

use crate::autograd::Tensor;
use crate::error::{EngineError, Result};
use crate::nn::module::Module;
use crate::tensor::RawTensor;
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

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

        // Fast-path: Fused LayerNorm when autograd graph tracking is not required
        if !crate::autograd::is_grad_enabled()
            || (!input.requires_grad()
                && !self.weight.requires_grad()
                && !self.bias.requires_grad())
        {
            let in_raw = input.data().to_contiguous();
            let in_slice = in_raw.as_slice();
            let gamma = self.weight.data().to_contiguous();
            let gamma_slice = gamma.as_slice();
            let beta = self.bias.data().to_contiguous();
            let beta_slice = beta.as_slice();

            let d = self.normalized_dim;
            let mut out_data = vec![0.0f32; in_slice.len()];

            out_data
                .par_chunks_mut(d)
                .enumerate()
                .for_each(|(row_idx, row_out)| {
                    let row_in = &in_slice[row_idx * d..(row_idx + 1) * d];
                    fused_layer_norm_row(row_in, gamma_slice, beta_slice, self.eps, row_out);
                });

            let out_raw = RawTensor::from_vec(out_data, shape);
            return Ok(Tensor::new(out_raw, false));
        }

        // Standard graph-building autograd path during training
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

        // Fast-path: Fused RMSNorm when autograd graph tracking is not required
        if !crate::autograd::is_grad_enabled()
            || (!input.requires_grad() && !self.weight.requires_grad())
        {
            let in_raw = input.data().to_contiguous();
            let in_slice = in_raw.as_slice();
            let gamma = self.weight.data().to_contiguous();
            let gamma_slice = gamma.as_slice();

            let d = self.normalized_dim;
            let mut out_data = vec![0.0f32; in_slice.len()];

            out_data
                .par_chunks_mut(d)
                .enumerate()
                .for_each(|(row_idx, row_out)| {
                    let row_in = &in_slice[row_idx * d..(row_idx + 1) * d];
                    fused_rms_norm_row(row_in, gamma_slice, self.eps, row_out);
                });

            let out_raw = RawTensor::from_vec(out_data, shape);
            return Ok(Tensor::new(out_raw, false));
        }

        // Standard graph-building autograd path during training
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

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn hsum256(v: __m256) -> f32 {
    let v_high = _mm256_extractf128_ps(v, 1);
    let v_low = _mm256_castps256_ps128(v);
    let sum128 = _mm_add_ps(v_low, v_high);
    let shuf = _mm_movehl_ps(sum128, sum128);
    let sum64 = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_shuffle_ps(sum64, sum64, 1);
    let sum32 = _mm_add_ss(sum64, shuf2);
    _mm_cvtss_f32(sum32)
}

fn fused_layer_norm_row(x: &[f32], gamma: &[f32], beta: &[f32], eps: f32, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                fused_layer_norm_row_avx2(x, gamma, beta, eps, out);
            }
            return;
        }
    }

    let d = x.len();
    let mean = x.iter().sum::<f32>() / (d as f32);
    let var = x
        .iter()
        .map(|&v| {
            let diff = v - mean;
            diff * diff
        })
        .sum::<f32>()
        / (d as f32);
    let inv_std = 1.0 / (var + eps).sqrt();

    for j in 0..d {
        out[j] = (x[j] - mean) * inv_std * gamma[j] + beta[j];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_layer_norm_row_avx2(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    eps: f32,
    out: &mut [f32],
) {
    let d = x.len();
    let d_f32 = d as f32;
    let x_ptr = x.as_ptr();

    // 1. Mean
    let mut acc_sum = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= d {
        acc_sum = _mm256_add_ps(acc_sum, _mm256_loadu_ps(x_ptr.add(i)));
        i += 8;
    }
    let mut sum = hsum256(acc_sum);
    while i < d {
        sum += *x_ptr.add(i);
        i += 1;
    }
    let mean = sum / d_f32;
    let v_mean = _mm256_set1_ps(mean);

    // 2. Variance
    let mut acc_var = _mm256_setzero_ps();
    i = 0;
    while i + 8 <= d {
        let diff = _mm256_sub_ps(_mm256_loadu_ps(x_ptr.add(i)), v_mean);
        acc_var = _mm256_fmadd_ps(diff, diff, acc_var);
        i += 8;
    }
    let mut var_sum = hsum256(acc_var);
    while i < d {
        let diff = *x_ptr.add(i) - mean;
        var_sum += diff * diff;
        i += 1;
    }
    let var = var_sum / d_f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    let v_inv_std = _mm256_set1_ps(inv_std);

    // 3. Affine transform
    let g_ptr = gamma.as_ptr();
    let b_ptr = beta.as_ptr();
    let o_ptr = out.as_mut_ptr();
    i = 0;
    while i + 8 <= d {
        let vx = _mm256_loadu_ps(x_ptr.add(i));
        let vg = _mm256_loadu_ps(g_ptr.add(i));
        let vb = _mm256_loadu_ps(b_ptr.add(i));
        let norm = _mm256_mul_ps(_mm256_sub_ps(vx, v_mean), v_inv_std);
        let vy = _mm256_fmadd_ps(norm, vg, vb);
        _mm256_storeu_ps(o_ptr.add(i), vy);
        i += 8;
    }
    while i < d {
        *o_ptr.add(i) = (*x_ptr.add(i) - mean) * inv_std * *g_ptr.add(i) + *b_ptr.add(i);
        i += 1;
    }
}

fn fused_rms_norm_row(x: &[f32], gamma: &[f32], eps: f32, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                fused_rms_norm_row_avx2(x, gamma, eps, out);
            }
            return;
        }
    }

    let d = x.len();
    let mean_sq = x.iter().map(|&v| v * v).sum::<f32>() / (d as f32);
    let rrms = 1.0 / (mean_sq + eps).sqrt();

    for j in 0..d {
        out[j] = x[j] * rrms * gamma[j];
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn fused_rms_norm_row_avx2(x: &[f32], gamma: &[f32], eps: f32, out: &mut [f32]) {
    let d = x.len();
    let d_f32 = d as f32;
    let x_ptr = x.as_ptr();

    // 1. Mean square
    let mut acc_sq = _mm256_setzero_ps();
    let mut i = 0;
    while i + 8 <= d {
        let vx = _mm256_loadu_ps(x_ptr.add(i));
        acc_sq = _mm256_fmadd_ps(vx, vx, acc_sq);
        i += 8;
    }
    let mut sum_sq = hsum256(acc_sq);
    while i < d {
        let v = *x_ptr.add(i);
        sum_sq += v * v;
        i += 1;
    }
    let mean_sq = sum_sq / d_f32;
    let rrms = 1.0 / (mean_sq + eps).sqrt();
    let v_rrms = _mm256_set1_ps(rrms);

    // 2. Scale
    let g_ptr = gamma.as_ptr();
    let o_ptr = out.as_mut_ptr();
    i = 0;
    while i + 8 <= d {
        let vx = _mm256_loadu_ps(x_ptr.add(i));
        let vg = _mm256_loadu_ps(g_ptr.add(i));
        let vy = _mm256_mul_ps(_mm256_mul_ps(vx, v_rrms), vg);
        _mm256_storeu_ps(o_ptr.add(i), vy);
        i += 8;
    }
    while i < d {
        *o_ptr.add(i) = *x_ptr.add(i) * rrms * *g_ptr.add(i);
        i += 1;
    }
}
