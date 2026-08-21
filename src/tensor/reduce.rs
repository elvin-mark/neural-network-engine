//! Reduction operations (sum, mean, max, min, argmax, logsumexp, softmax).

use crate::error::{EngineError, Result};
use crate::tensor::shape::{flat_to_multi_index, multi_index_to_offset, numel};
use crate::tensor::RawTensor;
use rayon::prelude::*;

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
    pub fn softmax(&self, axis: usize) -> Result<RawTensor> {
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
