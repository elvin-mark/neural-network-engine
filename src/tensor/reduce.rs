//! Reduction operations (sum, mean, max, min, argmax, logsumexp, softmax).

use crate::error::{EngineError, Result};
use crate::tensor::shape::{flat_to_multi_index, multi_index_to_offset, numel};
use crate::tensor::RawTensor;
use rayon::prelude::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

impl RawTensor {
    /// Computes the sum of all elements in the tensor.
    pub fn sum_all(&self) -> f32 {
        if self.is_contiguous() {
            self.as_slice().par_iter().sum()
        } else {
            let total = self.numel();
            (0..total)
                .into_par_iter()
                .map(|i| self.get_by_flat_index(i))
                .sum()
        }
    }

    /// Computes the arithmetic mean of all elements in the tensor.
    pub fn mean_all(&self) -> f32 {
        let n = self.numel();
        if n == 0 {
            0.0
        } else {
            self.sum_all() / (n as f32)
        }
    }

    /// Computes the sum along a specified axis.
    #[allow(clippy::needless_range_loop)]
    pub fn sum(&self, axis: usize, keepdim: bool) -> Result<RawTensor> {
        let shape = self.shape();
        if axis >= shape.len() {
            return Err(EngineError::DimensionOutOfBounds {
                axis,
                ndim: shape.len(),
            });
        }

        let mut out_shape = shape.to_vec();
        let axis_size = shape[axis];
        if keepdim {
            out_shape[axis] = 1;
        } else {
            out_shape.remove(axis);
        }

        let out_numel = numel(&out_shape);
        let mut out_data = vec![0.0; out_numel];

        // Parallelize over output elements
        out_data
            .par_iter_mut()
            .enumerate()
            .for_each(|(out_idx, out_val)| {
                let mut out_multi = vec![0; out_shape.len()];
                flat_to_multi_index(out_idx, &out_shape, &mut out_multi);

                let mut in_multi = vec![0; shape.len()];
                let mut o_i = 0;
                for i in 0..shape.len() {
                    if i == axis {
                        if keepdim {
                            o_i += 1;
                        }
                    } else {
                        in_multi[i] = out_multi[o_i];
                        o_i += 1;
                    }
                }

                let mut acc = 0.0;
                for a in 0..axis_size {
                    in_multi[axis] = a;
                    let offset = multi_index_to_offset(&in_multi, self.strides(), self.offset());
                    acc += self.storage.get(offset);
                }
                *out_val = acc;
            });

        Ok(RawTensor::from_vec(out_data, out_shape))
    }

    /// Computes the mean along a specified axis.
    pub fn mean(&self, axis: usize, keepdim: bool) -> Result<RawTensor> {
        let axis_size =
            self.shape()
                .get(axis)
                .copied()
                .ok_or(EngineError::DimensionOutOfBounds {
                    axis,
                    ndim: self.ndim(),
                })?;
        let sum_tensor = self.sum(axis, keepdim)?;
        sum_tensor.div_scalar(axis_size as f32)
    }

    /// Computes the maximum value along a specified axis.
    #[allow(clippy::needless_range_loop)]
    pub fn max(&self, axis: usize, keepdim: bool) -> Result<RawTensor> {
        let shape = self.shape();
        if axis >= shape.len() {
            return Err(EngineError::DimensionOutOfBounds {
                axis,
                ndim: shape.len(),
            });
        }

        let mut out_shape = shape.to_vec();
        let axis_size = shape[axis];
        if keepdim {
            out_shape[axis] = 1;
        } else {
            out_shape.remove(axis);
        }

        let out_numel = numel(&out_shape);
        let mut out_data = vec![f32::NEG_INFINITY; out_numel];

        out_data
            .par_iter_mut()
            .enumerate()
            .for_each(|(out_idx, out_val)| {
                let mut out_multi = vec![0; out_shape.len()];
                flat_to_multi_index(out_idx, &out_shape, &mut out_multi);

                let mut in_multi = vec![0; shape.len()];
                let mut o_i = 0;
                for i in 0..shape.len() {
                    if i == axis {
                        if keepdim {
                            o_i += 1;
                        }
                    } else {
                        in_multi[i] = out_multi[o_i];
                        o_i += 1;
                    }
                }

                let mut max_val = f32::NEG_INFINITY;
                for a in 0..axis_size {
                    in_multi[axis] = a;
                    let offset = multi_index_to_offset(&in_multi, self.strides(), self.offset());
                    let v = self.storage.get(offset);
                    if v > max_val {
                        max_val = v;
                    }
                }
                *out_val = max_val;
            });

        Ok(RawTensor::from_vec(out_data, out_shape))
    }

    /// Computes the argmax along a specified axis.
    #[allow(clippy::needless_range_loop)]
    pub fn argmax(&self, axis: usize) -> Result<Vec<usize>> {
        let shape = self.shape();
        if axis >= shape.len() {
            return Err(EngineError::DimensionOutOfBounds {
                axis,
                ndim: shape.len(),
            });
        }

        let mut out_shape = shape.to_vec();
        let axis_size = shape[axis];
        out_shape.remove(axis);

        let out_numel = numel(&out_shape);
        let mut out_indices = vec![0; out_numel];

        out_indices
            .par_iter_mut()
            .enumerate()
            .for_each(|(out_idx, out_val)| {
                let mut out_multi = vec![0; out_shape.len()];
                flat_to_multi_index(out_idx, &out_shape, &mut out_multi);

                let mut in_multi = vec![0; shape.len()];
                let mut o_i = 0;
                for i in 0..shape.len() {
                    if i == axis {
                        // skip
                    } else {
                        in_multi[i] = out_multi[o_i];
                        o_i += 1;
                    }
                }

                let mut max_val = f32::NEG_INFINITY;
                let mut max_idx = 0;
                for a in 0..axis_size {
                    in_multi[axis] = a;
                    let offset = multi_index_to_offset(&in_multi, self.strides(), self.offset());
                    let v = self.storage.get(offset);
                    if v > max_val {
                        max_val = v;
                        max_idx = a;
                    }
                }
                *out_val = max_idx;
            });

        Ok(out_indices)
    }

    /// Computes numerically stable log-sum-exp along a specified axis: log(sum(exp(x - max(x)))) + max(x).
    pub fn logsumexp(&self, axis: usize, keepdim: bool) -> Result<RawTensor> {
        let max_val = self.max(axis, true)?;
        let sub = self.sub(&max_val)?;
        let exp_sub = sub.exp()?;
        let sum_exp = exp_sub.sum(axis, true)?;
        let log_sum = sum_exp.log()?;
        let res = log_sum.add(&max_val)?;

        if !keepdim {
            let mut final_shape = self.shape().to_vec();
            final_shape.remove(axis);
            res.reshape(&final_shape)
        } else {
            Ok(res)
        }
    }

    /// Computes numerically stable softmax along a specified axis.
    /// Fast-path: single-pass fused AVX2 kernel when reducing along the last dimension.
    pub fn softmax(&self, axis: usize) -> Result<RawTensor> {
        let ndim = self.ndim();
        if axis >= ndim {
            return Err(EngineError::DimensionOutOfBounds { axis, ndim });
        }

        // Fused fast-path for the last dimension (common in Transformers, Attention, Classifiers)
        if axis == ndim - 1 {
            let shape = self.shape().to_vec();
            let d = shape[axis];
            if d == 0 {
                return Ok(RawTensor::zeros(&shape));
            }

            let contig = self.to_contiguous();
            let in_slice = contig.as_slice();
            let mut out_data = vec![0.0f32; in_slice.len()];

            out_data
                .par_chunks_mut(d)
                .enumerate()
                .for_each(|(row_idx, row_out)| {
                    let row_in = &in_slice[row_idx * d..(row_idx + 1) * d];
                    let max_val = find_row_max(row_in);

                    let mut sum_exp = 0.0f32;
                    for j in 0..d {
                        let e = (row_in[j] - max_val).exp();
                        row_out[j] = e;
                        sum_exp += e;
                    }

                    let inv_sum = if sum_exp > 0.0 { 1.0 / sum_exp } else { 0.0 };
                    scale_row(row_out, inv_sum);
                });

            return Ok(RawTensor::from_vec(out_data, shape));
        }

        // Generic fallback for intermediate axes
        let max_val = self.max(axis, true)?;
        let shifted = self.sub(&max_val)?;
        let exp_shifted = shifted.exp()?;
        let sum_exp = exp_shifted.sum(axis, true)?;
        exp_shifted.div(&sum_exp)
    }

    /// Computes numerically stable log-softmax along a specified axis.
    pub fn log_softmax(&self, axis: usize) -> Result<RawTensor> {
        let lse = self.logsumexp(axis, true)?;
        self.sub(&lse)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn find_row_max_avx2(row: &[f32]) -> f32 {
    let mut max_vec = _mm256_set1_ps(f32::NEG_INFINITY);
    let mut i = 0;
    while i + 8 <= row.len() {
        let v = _mm256_loadu_ps(row.as_ptr().add(i));
        max_vec = _mm256_max_ps(max_vec, v);
        i += 8;
    }
    let v_high = _mm256_extractf128_ps(max_vec, 1);
    let v_low = _mm256_castps256_ps128(max_vec);
    let m128 = _mm_max_ps(v_low, v_high);
    let shuf = _mm_movehl_ps(m128, m128);
    let m64 = _mm_max_ps(m128, shuf);
    let shuf2 = _mm_shuffle_ps(m64, m64, 1);
    let m32 = _mm_max_ss(m64, shuf2);
    let mut max_scalar = _mm_cvtss_f32(m32);
    while i < row.len() {
        if row[i] > max_scalar {
            max_scalar = row[i];
        }
        i += 1;
    }
    max_scalar
}

fn find_row_max(row: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { find_row_max_avx2(row) };
        }
    }
    row.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scale_row_avx2(row: &mut [f32], factor: f32) {
    let vf = _mm256_set1_ps(factor);
    let mut i = 0;
    while i + 8 <= row.len() {
        let ptr = row.as_mut_ptr().add(i);
        let v = _mm256_loadu_ps(ptr);
        _mm256_storeu_ps(ptr, _mm256_mul_ps(v, vf));
        i += 8;
    }
    while i < row.len() {
        row[i] *= factor;
        i += 1;
    }
}

fn scale_row(row: &mut [f32], factor: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                scale_row_avx2(row, factor);
            }
            return;
        }
    }
    for val in row.iter_mut() {
        *val *= factor;
    }
}
