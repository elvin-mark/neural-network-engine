//! Elementwise arithmetic, unary mathematical operations, activations, slicing, and concatenations.

use crate::error::{EngineError, Result};
use crate::tensor::shape::{
    broadcast_shapes, broadcast_strides, compute_c_contiguous_strides, flat_to_multi_index,
    multi_index_to_offset, numel,
};
use crate::tensor::RawTensor;
use rayon::prelude::*;

impl RawTensor {
    /// Applies a unary function elementwise across the tensor.
    pub fn unary_op<F>(&self, op: F) -> Self
    where
        F: Fn(f32) -> f32 + Sync + Send,
    {
        let total = self.numel();
        let mut out_data = vec![0.0; total];

        if self.is_contiguous() {
            let slice = self.as_slice();
            out_data.par_iter_mut().enumerate().for_each(|(i, out)| {
                *out = op(slice[i]);
            });
        } else {
            let shape = self.shape();
            let strides = self.strides();
            let offset = self.offset();

            out_data.par_iter_mut().enumerate().for_each(|(i, out)| {
                let mut multi = vec![0; shape.len()];
                flat_to_multi_index(i, shape, &mut multi);
                let off = multi_index_to_offset(&multi, strides, offset);
                *out = op(self.storage.get(off));
            });
        }

        Self::from_vec(out_data, self.shape().to_vec())
    }

    /// Performs broadcasted elementwise binary operation between self and other.
    pub fn binary_op<F>(&self, other: &RawTensor, op: F) -> Result<RawTensor>
    where
        F: Fn(f32, f32) -> f32 + Sync + Send,
    {
        // Fast path: identical shapes and both contiguous
        if self.shape() == other.shape() && self.is_contiguous() && other.is_contiguous() {
            let total = self.numel();
            let mut out_data = vec![0.0; total];
            let a_slice = self.as_slice();
            let b_slice = other.as_slice();

            out_data.par_iter_mut().enumerate().for_each(|(i, out)| {
                *out = op(a_slice[i], b_slice[i]);
            });

            return Ok(RawTensor::from_vec(out_data, self.shape().to_vec()));
        }

        // General broadcasted path
        let out_shape = broadcast_shapes(self.shape(), other.shape())?;
        let self_strides = broadcast_strides(self.shape(), self.strides(), &out_shape)?;
        let other_strides = broadcast_strides(other.shape(), other.strides(), &out_shape)?;

        let total = numel(&out_shape);
        let mut out_data = vec![0.0; total];

        let self_offset = self.offset();
        let other_offset = other.offset();

        out_data.par_iter_mut().enumerate().for_each(|(i, out)| {
            let mut multi = vec![0; out_shape.len()];
            flat_to_multi_index(i, &out_shape, &mut multi);

            let a_off = multi_index_to_offset(&multi, &self_strides, self_offset);
            let b_off = multi_index_to_offset(&multi, &other_strides, other_offset);

            let a_val = self.storage.get(a_off);
            let b_val = other.storage.get(b_off);

            *out = op(a_val, b_val);
        });

        Ok(RawTensor::from_vec(out_data, out_shape))
    }

    // --- Standard Arithmetic ---

    pub fn add(&self, other: &RawTensor) -> Result<RawTensor> {
        self.binary_op(other, |a, b| a + b)
    }

    pub fn sub(&self, other: &RawTensor) -> Result<RawTensor> {
        self.binary_op(other, |a, b| a - b)
    }

    pub fn mul(&self, other: &RawTensor) -> Result<RawTensor> {
        self.binary_op(other, |a, b| a * b)
    }

    pub fn div(&self, other: &RawTensor) -> Result<RawTensor> {
        self.binary_op(other, |a, b| a / b)
    }

    pub fn neg(&self) -> RawTensor {
        self.unary_op(|a| -a)
    }

    // --- Scalar Operations ---

    pub fn add_scalar(&self, val: f32) -> Result<RawTensor> {
        Ok(self.unary_op(|a| a + val))
    }

    pub fn sub_scalar(&self, val: f32) -> Result<RawTensor> {
        Ok(self.unary_op(|a| a - val))
    }

    pub fn mul_scalar(&self, val: f32) -> Result<RawTensor> {
        Ok(self.unary_op(|a| a * val))
    }

    pub fn div_scalar(&self, val: f32) -> Result<RawTensor> {
        Ok(self.unary_op(|a| a / val))
    }

    // --- Math & Activations ---

    pub fn exp(&self) -> Result<RawTensor> {
        Ok(self.unary_op(|a| a.exp()))
    }

    pub fn log(&self) -> Result<RawTensor> {
        Ok(self.unary_op(|a| a.ln()))
    }

    pub fn sqrt(&self) -> Result<RawTensor> {
        Ok(self.unary_op(|a| a.sqrt()))
    }

    pub fn powf(&self, exponent: f32) -> Result<RawTensor> {
        Ok(self.unary_op(|a| a.powf(exponent)))
    }

    pub fn abs(&self) -> Result<RawTensor> {
        Ok(self.unary_op(|a| a.abs()))
    }

    pub fn clamp(&self, min: f32, max: f32) -> Result<RawTensor> {
        Ok(self.unary_op(|a| a.clamp(min, max)))
    }

    pub fn tanh(&self) -> Result<RawTensor> {
        Ok(self.unary_op(|a| a.tanh()))
    }

    pub fn sigmoid(&self) -> Result<RawTensor> {
        Ok(self.unary_op(|a| 1.0 / (1.0 + (-a).exp())))
    }

    pub fn relu(&self) -> Result<RawTensor> {
        Ok(self.unary_op(|a| if a > 0.0 { a } else { 0.0 }))
    }

    pub fn gelu(&self) -> Result<RawTensor> {
        // Fast approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        const SQRT_2_OVER_PI: f32 = 0.797_884_6;
        Ok(self
            .unary_op(|x| 0.5 * x * (1.0 + ((SQRT_2_OVER_PI * (x + 0.044715 * x * x * x)).tanh()))))
    }

    pub fn leaky_relu(&self, negative_slope: f32) -> Result<RawTensor> {
        Ok(self.unary_op(|a| if a > 0.0 { a } else { a * negative_slope }))
    }

    // --- Slicing & Concatenation ---

    /// Slices the tensor along an axis from start to end (end-exclusive).
    pub fn slice(&self, axis: usize, start: usize, end: usize) -> Result<RawTensor> {
        let shape = self.shape();
        if axis >= shape.len() {
            return Err(EngineError::DimensionOutOfBounds {
                axis,
                ndim: shape.len(),
            });
        }

        let dim = shape[axis];
        if start > dim || end > dim || start > end {
            return Err(EngineError::IndexOutOfBounds {
                index: end,
                size: dim,
            });
        }

        let mut new_shape = shape.to_vec();
        new_shape[axis] = end - start;

        let new_offset = self.offset() + start * self.strides()[axis];

        Ok(RawTensor {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: self.strides().to_vec(),
            offset: new_offset,
        })
    }

    /// Concatenates a sequence of tensors along a specified axis.
    pub fn cat(tensors: &[&RawTensor], axis: usize) -> Result<RawTensor> {
        if tensors.is_empty() {
            return Err(EngineError::InvalidArgument(
                "Cannot concatenate empty list of tensors".to_string(),
            ));
        }

        let first = tensors[0];
        let ndim = first.ndim();

        if axis >= ndim {
            return Err(EngineError::DimensionOutOfBounds { axis, ndim });
        }

        let mut total_axis_dim = 0;
        for t in tensors {
            if t.ndim() != ndim {
                return Err(EngineError::IncompatibleShapes {
                    op: "cat (different number of dimensions)",
                    shapes: tensors.iter().map(|x| x.shape().to_vec()).collect(),
                });
            }
            for (i, (&d1, &d2)) in first.shape().iter().zip(t.shape().iter()).enumerate() {
                if i != axis && d1 != d2 {
                    return Err(EngineError::IncompatibleShapes {
                        op: "cat (dimension mismatch on non-concatenated axis)",
                        shapes: tensors.iter().map(|x| x.shape().to_vec()).collect(),
                    });
                }
            }
            total_axis_dim += t.shape()[axis];
        }

        let mut out_shape = first.shape().to_vec();
        out_shape[axis] = total_axis_dim;
        let out_numel = numel(&out_shape);

        let mut out_data = vec![0.0; out_numel];
        let out_strides = compute_c_contiguous_strides(&out_shape);

        let mut current_offset = 0;
        for t in tensors {
            let t_shape = t.shape();
            let t_axis_len = t_shape[axis];
            let t_numel = t.numel();

            for idx in 0..t_numel {
                let mut multi = vec![0; ndim];
                flat_to_multi_index(idx, t_shape, &mut multi);
                let val = t.get(&multi);

                multi[axis] += current_offset;
                let out_off = multi_index_to_offset(&multi, &out_strides, 0);
                out_data[out_off] = val;
            }

            current_offset += t_axis_len;
        }

        Ok(RawTensor::from_vec(out_data, out_shape))
    }

    /// Selects indexed rows along dimension 0 (useful for Embedding layer).
    pub fn embedding_lookup(&self, indices: &[usize]) -> Result<RawTensor> {
        let shape = self.shape();
        if shape.len() != 2 {
            return Err(EngineError::IncompatibleShapes {
                op: "embedding_lookup (weight must be 2D)",
                shapes: vec![shape.to_vec()],
            });
        }

        let (num_embeddings, embedding_dim) = (shape[0], shape[1]);
        let num_indices = indices.len();
        let mut out_data = vec![0.0; num_indices * embedding_dim];

        let weight_contig = self.to_contiguous();
        let w_slice = weight_contig.as_slice();

        for (i, &idx) in indices.iter().enumerate() {
            if idx >= num_embeddings {
                return Err(EngineError::IndexOutOfBounds {
                    index: idx,
                    size: num_embeddings,
                });
            }
            let w_start = idx * embedding_dim;
            let out_start = i * embedding_dim;
            out_data[out_start..out_start + embedding_dim]
                .copy_from_slice(&w_slice[w_start..w_start + embedding_dim]);
        }

        Ok(RawTensor::from_vec(
            out_data,
            vec![num_indices, embedding_dim],
        ))
    }
}
