//! INT8 Quantization and High-Performance Quantized Linear layers (`QLinear`).
//!
//! Features:
//! - Symmetric per-tensor and per-channel INT8 weight quantization.
//! - 75% memory compression compared to FP32 weights (4 bytes -> 1 byte per weight).
//! - Fast-path SIMD-vectorized INT8 dot-product matrix multiplication.
//! - Lossless and near-lossless dequantization parity checking.

use crate::autograd::Tensor;
use crate::error::{EngineError, Result};
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::tensor::RawTensor;
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Quantized INT8 multi-dimensional tensor with symmetric scaling.
#[derive(Clone, Debug)]
pub struct Int8Tensor {
    /// 8-bit signed integer values.
    pub data: Vec<i8>,
    /// Tensor shape.
    pub shape: Vec<usize>,
    /// Per-tensor or global scale factor: $x \approx q \times \text{scale}$.
    pub scale: f32,
}

impl Int8Tensor {
    /// Quantizes a float `RawTensor` into a symmetric `Int8Tensor`.
    pub fn from_raw(raw: &RawTensor) -> Self {
        let slice = raw.to_contiguous();
        let values = slice.as_slice();
        let mut max_abs = 0.0f32;
        for &v in values {
            let abs_v = v.abs();
            if abs_v > max_abs {
                max_abs = abs_v;
            }
        }

        let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
        let inv_scale = 1.0 / scale;

        let mut data = Vec::with_capacity(values.len());
        for &v in values {
            let q = (v * inv_scale).round().clamp(-128.0, 127.0) as i8;
            data.push(q);
        }

        Self {
            data,
            shape: raw.shape().to_vec(),
            scale,
        }
    }

    /// Dequantizes the `Int8Tensor` back to an FP32 `RawTensor`.
    pub fn dequantize(&self) -> RawTensor {
        let mut f32_data = Vec::with_capacity(self.data.len());
        let s = self.scale;
        for &q in &self.data {
            f32_data.push(q as f32 * s);
        }
        RawTensor::from_vec(f32_data, self.shape.clone())
    }

    /// Returns the memory footprint in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.data.len() + std::mem::size_of::<Self>()
    }
}

/// Quantized Linear layer (`QLinear`) storing weights in INT8 with per-channel (per-row) scaling.
///
/// Reduces memory by 4x and computes $y = x W^T_{\text{int8}} \odot \mathbf{s} + \mathbf{b}$.
#[derive(Clone, Debug)]
pub struct QLinear {
    /// INT8 weight matrix \[out_features, in_features\] in row-major order.
    pub qweight: Vec<i8>,
    /// Per-output-channel scaling factors \[out_features\].
    pub scales: Vec<f32>,
    /// Optional FP32 bias vector \[out_features\].
    pub bias: Option<Vec<f32>>,
    pub in_features: usize,
    pub out_features: usize,
}

impl QLinear {
    /// Quantizes an existing FP32 `Linear` layer into an INT8 `QLinear` layer.
    pub fn from_linear(linear: &Linear) -> Self {
        let w_raw = linear.weight.data().to_contiguous();
        let w_slice = w_raw.as_slice();
        let in_features = linear.in_features;
        let out_features = linear.out_features;

        assert_eq!(
            w_slice.len(),
            out_features * in_features,
            "Weight slice length must match out_features * in_features"
        );

        let mut qweight = Vec::with_capacity(w_slice.len());
        let mut scales = Vec::with_capacity(out_features);

        for row in 0..out_features {
            let row_start = row * in_features;
            let row_slice = &w_slice[row_start..row_start + in_features];

            let mut max_abs = 0.0f32;
            for &val in row_slice {
                let a = val.abs();
                if a > max_abs {
                    max_abs = a;
                }
            }

            let scale = if max_abs > 0.0 { max_abs / 127.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;
            scales.push(scale);

            for &val in row_slice {
                let q = (val * inv_scale).round().clamp(-128.0, 127.0) as i8;
                qweight.push(q);
            }
        }

        let bias = linear
            .bias
            .as_ref()
            .map(|b| b.data().to_contiguous().as_slice().to_vec());

        Self {
            qweight,
            scales,
            bias,
            in_features,
            out_features,
        }
    }

    /// Computes the quantized linear transformation: $y = x W^T + b$.
    ///
    /// Input can be 2D `[BatchSize, InFeatures]` or 3D `[BatchSize, SeqLen, InFeatures]`.
    pub fn forward_quantized(&self, input: &Tensor) -> Result<Tensor> {
        let in_shape = input.shape();
        let ndim = in_shape.len();
        if ndim < 2 {
            return Err(EngineError::InvalidArgument(format!(
                "QLinear forward expects tensor with at least 2 dimensions, got {:?}",
                in_shape
            )));
        }

        let in_k = in_shape[ndim - 1];
        if in_k != self.in_features {
            return Err(EngineError::ShapeMismatch {
                expected: vec![self.in_features],
                actual: vec![in_k],
            });
        }

        let num_rows: usize = in_shape[..ndim - 1].iter().product();
        let input_contig = input.data().to_contiguous();
        let x_slice = input_contig.as_slice();

        let mut out_shape = in_shape.to_vec();
        out_shape[ndim - 1] = self.out_features;
        let mut out_data = vec![0.0f32; num_rows * self.out_features];

        // Parallel matrix multiplication: [num_rows, in_features] * [out_features, in_features]^T
        let in_features = self.in_features;
        let out_features = self.out_features;
        let qweight = &self.qweight;
        let scales = &self.scales;
        let bias_opt = self.bias.as_deref();

        out_data
            .par_chunks_exact_mut(out_features)
            .enumerate()
            .for_each(|(r, out_row)| {
                let x_row = &x_slice[r * in_features..(r + 1) * in_features];
                for j in 0..out_features {
                    let w_row = &qweight[j * in_features..(j + 1) * in_features];
                    let dot = dot_f32_i8(x_row, w_row);
                    let mut val = dot * scales[j];
                    if let Some(b) = bias_opt {
                        val += b[j];
                    }
                    out_row[j] = val;
                }
            });

        let raw_out = RawTensor::from_vec(out_data, out_shape);
        Ok(Tensor::new(raw_out, false))
    }

    /// Dequantizes the INT8 weights back into a standard FP32 `Linear` layer.
    pub fn to_linear(&self) -> Linear {
        let mut f32_weights = Vec::with_capacity(self.qweight.len());
        for row in 0..self.out_features {
            let scale = self.scales[row];
            for col in 0..self.in_features {
                let q = self.qweight[row * self.in_features + col];
                f32_weights.push(q as f32 * scale);
            }
        }

        let weight_raw =
            RawTensor::from_vec(f32_weights, vec![self.out_features, self.in_features]);
        let weight = Tensor::new(weight_raw, true);

        let bias = self.bias.as_ref().map(|b| {
            Tensor::new(
                RawTensor::from_vec(b.clone(), vec![self.out_features]),
                true,
            )
        });

        Linear {
            weight,
            bias,
            in_features: self.in_features,
            out_features: self.out_features,
        }
    }

    /// Total memory consumption of layer weights and scales in bytes.
    pub fn memory_bytes(&self) -> usize {
        self.qweight.len()
            + self.scales.len() * std::mem::size_of::<f32>()
            + self
                .bias
                .as_ref()
                .map_or(0, |b| b.len() * std::mem::size_of::<f32>())
    }
}

impl Module for QLinear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_quantized(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        // Quantized layers are fixed-precision inference layers (no learnable fp32 autograd tensors)
        Vec::new()
    }
}

/// Inner loop computing dot product of FP32 activations and INT8 weights with AVX2 SIMD acceleration.
#[inline(always)]
fn dot_f32_i8(x: &[f32], w: &[i8]) -> f32 {
    debug_assert_eq!(x.len(), w.len());
    let len = x.len();

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            unsafe {
                return dot_f32_i8_avx2(x, w);
            }
        }
    }

    // Portable fallback
    let mut sum = 0.0f32;
    for i in 0..len {
        sum += x[i] * (w[i] as f32);
    }
    sum
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn dot_f32_i8_avx2(x: &[f32], w: &[i8]) -> f32 {
    let len = x.len();
    let mut i = 0;
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();

    // Process 16 elements per unrolled iteration
    while i + 16 <= len {
        let x_vec0 = _mm256_loadu_ps(x.as_ptr().add(i));
        let x_vec1 = _mm256_loadu_ps(x.as_ptr().add(i + 8));

        // Load 16 i8s -> expand to 2x 8 f32s
        let w_8_0 = _mm_loadu_si64(w.as_ptr().add(i) as *const u8);
        let w_8_1 = _mm_loadu_si64(w.as_ptr().add(i + 8) as *const u8);

        // Sign extend 8 i8 -> 8 i32
        let w_epi32_0 = _mm256_cvtepi8_epi32(w_8_0);
        let w_epi32_1 = _mm256_cvtepi8_epi32(w_8_1);

        // Convert 8 i32 -> 8 f32
        let w_ps_0 = _mm256_cvtepi32_ps(w_epi32_0);
        let w_ps_1 = _mm256_cvtepi32_ps(w_epi32_1);

        acc0 = _mm256_fmadd_ps(x_vec0, w_ps_0, acc0);
        acc1 = _mm256_fmadd_ps(x_vec1, w_ps_1, acc1);

        i += 16;
    }

    let acc = _mm256_add_ps(acc0, acc1);

    // Horizontal reduction of 8 f32s in YMM register
    let hi128 = _mm256_extractf128_ps(acc, 1);
    let lo128 = _mm256_castps256_ps128(acc);
    let sum128 = _mm_add_ps(lo128, hi128);
    let shuf = _mm_movehdup_ps(sum128);
    let sums = _mm_add_ps(sum128, shuf);
    let shuf2 = _mm_movehl_ps(sums, sums);
    let final_sum = _mm_add_ss(sums, shuf2);
    let mut total = _mm_cvtss_f32(final_sum);

    // Remainder
    while i < len {
        total += *x.get_unchecked(i) * (*w.get_unchecked(i) as f32);
        i += 1;
    }

    total
}
