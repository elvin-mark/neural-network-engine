//! Multi-dimensional strided tensor runtime in pure Rust.

pub mod conv;
pub mod display;
pub mod matmul;
pub mod ops;
pub mod reduce;
pub mod shape;
pub mod storage;

use crate::error::{EngineError, Result};
use rand_distr::{Distribution, Normal, Uniform};
use shape::{compute_c_contiguous_strides, is_contiguous, multi_index_to_offset, numel};
use std::fmt;
use storage::Storage;

/// N-dimensional strided tensor holding 32-bit floating point data.
#[derive(Clone, PartialEq)]
pub struct RawTensor {
    pub(crate) storage: Storage,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
    pub(crate) offset: usize,
}

impl RawTensor {
    /// Creates a new tensor filled with zeros.
    pub fn zeros(shape: &[usize]) -> Self {
        let size = numel(shape);
        let strides = compute_c_contiguous_strides(shape);
        Self {
            storage: Storage::zeros(size),
            shape: shape.to_vec(),
            strides,
            offset: 0,
        }
    }

    /// Creates a new tensor filled with ones.
    pub fn ones(shape: &[usize]) -> Self {
        Self::full(shape, 1.0)
    }

    /// Creates a new tensor filled with a constant value.
    pub fn full(shape: &[usize], val: f32) -> Self {
        let size = numel(shape);
        let strides = compute_c_contiguous_strides(shape);
        Self {
            storage: Storage::filled(size, val),
            shape: shape.to_vec(),
            strides,
            offset: 0,
        }
    }

    /// Creates a tensor from a vector of floats and a shape.
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let expected = numel(&shape);
        assert_eq!(
            data.len(),
            expected,
            "Data length {} does not match shape {:?} (expected {})",
            data.len(),
            shape,
            expected
        );
        let strides = compute_c_contiguous_strides(&shape);
        Self {
            storage: Storage::from_vec(data),
            shape,
            strides,
            offset: 0,
        }
    }

    /// Creates a tensor from a slice of floats and a shape.
    pub fn from_slice(data: &[f32], shape: &[usize]) -> Self {
        let expected = numel(shape);
        assert_eq!(
            data.len(),
            expected,
            "Data length {} does not match shape {:?} (expected {})",
            data.len(),
            shape,
            expected
        );
        let strides = compute_c_contiguous_strides(shape);
        Self {
            storage: Storage::from_slice(data),
            shape: shape.to_vec(),
            strides,
            offset: 0,
        }
    }

    /// Creates a scalar (0-dimensional) tensor.
    pub fn scalar(val: f32) -> Self {
        Self {
            storage: Storage::from_vec(vec![val]),
            shape: vec![],
            strides: vec![],
            offset: 0,
        }
    }

    /// Creates a tensor with values drawn from a Gaussian normal distribution.
    pub fn randn(shape: &[usize], mean: f32, std: f32) -> Self {
        let mut rng = rand::thread_rng();
        let normal = Normal::new(mean, std).expect("Invalid normal distribution parameters");
        let count = numel(shape);
        let data: Vec<f32> = (0..count).map(|_| normal.sample(&mut rng)).collect();
        Self::from_vec(data, shape.to_vec())
    }

    /// Creates a tensor with values drawn from a uniform distribution [low, high).
    pub fn uniform(shape: &[usize], low: f32, high: f32) -> Self {
        let mut rng = rand::thread_rng();
        let unif = Uniform::new(low, high);
        let count = numel(shape);
        let data: Vec<f32> = (0..count).map(|_| unif.sample(&mut rng)).collect();
        Self::from_vec(data, shape.to_vec())
    }

    /// Kaiming (He) uniform initialization for neural network weights.
    pub fn kaiming_uniform(shape: &[usize], fan_in: usize) -> Self {
        let bound = (6.0 / fan_in.max(1) as f32).sqrt();
        Self::uniform(shape, -bound, bound)
    }

    /// Kaiming (He) normal initialization for neural network weights.
    pub fn kaiming_normal(shape: &[usize], fan_in: usize) -> Self {
        let std = (2.0 / fan_in.max(1) as f32).sqrt();
        Self::randn(shape, 0.0, std)
    }

    /// Xavier (Glorot) uniform initialization for neural network weights.
    pub fn xavier_uniform(shape: &[usize], fan_in: usize, fan_out: usize) -> Self {
        let bound = (6.0 / (fan_in + fan_out).max(1) as f32).sqrt();
        Self::uniform(shape, -bound, bound)
    }

    /// Xavier (Glorot) normal initialization for neural network weights.
    pub fn xavier_normal(shape: &[usize], fan_in: usize, fan_out: usize) -> Self {
        let std = (2.0 / (fan_in + fan_out).max(1) as f32).sqrt();
        Self::randn(shape, 0.0, std)
    }

    // --- Accessors ---

    #[inline(always)]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    #[inline(always)]
    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    #[inline(always)]
    pub fn offset(&self) -> usize {
        self.offset
    }

    #[inline(always)]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    #[inline(always)]
    pub fn numel(&self) -> usize {
        numel(&self.shape)
    }

    #[inline(always)]
    pub fn is_contiguous(&self) -> bool {
        self.offset == 0 && is_contiguous(&self.shape, &self.strides)
    }

    /// Ensures the tensor data is contiguous in memory, copying if necessary.
    pub fn to_contiguous(&self) -> RawTensor {
        if self.is_contiguous() {
            return self.clone();
        }

        let total = self.numel();
        let mut new_data = Vec::with_capacity(total);

        for i in 0..total {
            new_data.push(self.get_by_flat_index(i));
        }

        Self::from_vec(new_data, self.shape.clone())
    }

    /// Returns a contiguous slice view of the tensor. Panics if non-contiguous.
    #[inline(always)]
    pub fn as_slice(&self) -> &[f32] {
        assert!(
            self.is_contiguous(),
            "Cannot take contiguous slice of non-contiguous tensor"
        );
        &self.storage.as_slice()[self.offset..self.offset + self.numel()]
    }

    /// Returns a mutable slice view of the tensor. Panics if non-contiguous.
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        assert!(
            self.is_contiguous(),
            "Cannot take contiguous mutable slice of non-contiguous tensor"
        );
        let off = self.offset;
        let len = self.numel();
        &mut self.storage.as_mut_slice()[off..off + len]
    }

    /// Returns the single scalar value of a 0-d or 1-element tensor.
    pub fn item(&self) -> f32 {
        assert_eq!(
            self.numel(),
            1,
            "item() only supported for single-element tensors"
        );
        self.storage.get(self.offset)
    }

    /// Fallibly gets an element by multi-dimensional coordinates with full per-axis bounds checking.
    pub fn try_get(&self, indices: &[usize]) -> Result<f32> {
        if indices.len() != self.shape.len() {
            return Err(EngineError::InvalidArgument(format!(
                "Indices length {} does not match tensor rank {}",
                indices.len(),
                self.shape.len()
            )));
        }
        for (&idx, &dim) in indices.iter().zip(self.shape.iter()) {
            if idx >= dim {
                return Err(EngineError::DimensionOutOfBounds {
                    axis: idx,
                    ndim: dim,
                });
            }
        }
        let off = multi_index_to_offset(indices, &self.strides, self.offset);
        Ok(self.storage.get(off))
    }

    /// Gets an element by multi-dimensional coordinates. Panics if indices are out of bounds.
    pub fn get(&self, indices: &[usize]) -> f32 {
        self.try_get(indices)
            .unwrap_or_else(|e| panic!("RawTensor::get failed: {}", e))
    }

    /// Fallibly sets an element by multi-dimensional coordinates with full per-axis bounds checking.
    pub fn try_set(&mut self, indices: &[usize], value: f32) -> Result<()> {
        if indices.len() != self.shape.len() {
            return Err(EngineError::InvalidArgument(format!(
                "Indices length {} does not match tensor rank {}",
                indices.len(),
                self.shape.len()
            )));
        }
        for (&idx, &dim) in indices.iter().zip(self.shape.iter()) {
            if idx >= dim {
                return Err(EngineError::DimensionOutOfBounds {
                    axis: idx,
                    ndim: dim,
                });
            }
        }
        let off = multi_index_to_offset(indices, &self.strides, self.offset);
        self.storage.set(off, value);
        Ok(())
    }

    /// Sets an element by multi-dimensional coordinates. Panics if indices are out of bounds.
    pub fn set(&mut self, indices: &[usize], value: f32) {
        self.try_set(indices, value)
            .unwrap_or_else(|e| panic!("RawTensor::set failed: {}", e));
    }

    /// Gets an element by logical flat index in [0, numel).
    pub fn get_by_flat_index(&self, flat: usize) -> f32 {
        let mut multi = vec![0; self.shape.len()];
        shape::flat_to_multi_index(flat, &self.shape, &mut multi);
        let off = multi_index_to_offset(&multi, &self.strides, self.offset);
        self.storage.get(off)
    }

    // --- Shape Manipulations ---

    /// Reshapes the tensor to a new shape with the same number of elements.
    pub fn reshape(&self, new_shape: &[usize]) -> Result<RawTensor> {
        let target_numel = numel(new_shape);
        if target_numel != self.numel() {
            return Err(EngineError::ShapeMismatch {
                expected: vec![self.numel()],
                actual: new_shape.to_vec(),
            });
        }

        let contig = self.to_contiguous();
        let strides = compute_c_contiguous_strides(new_shape);

        Ok(RawTensor {
            storage: contig.storage,
            shape: new_shape.to_vec(),
            strides,
            offset: 0,
        })
    }

    /// Transposes two dimensions of the tensor (zero-copy strided view).
    pub fn transpose(&self, dim0: usize, dim1: usize) -> Result<RawTensor> {
        let ndim = self.ndim();
        if dim0 >= ndim || dim1 >= ndim {
            return Err(EngineError::DimensionOutOfBounds {
                axis: dim0.max(dim1),
                ndim,
            });
        }

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();

        new_shape.swap(dim0, dim1);
        new_strides.swap(dim0, dim1);

        Ok(RawTensor {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    /// Permutes the dimensions of the tensor according to a permutation order.
    pub fn permute(&self, dims: &[usize]) -> Result<RawTensor> {
        let ndim = self.ndim();
        if dims.len() != ndim {
            return Err(EngineError::InvalidArgument(format!(
                "Permute dims length {} does not match tensor ndim {}",
                dims.len(),
                ndim
            )));
        }

        let mut seen = vec![false; ndim];
        for &d in dims {
            if d >= ndim {
                return Err(EngineError::DimensionOutOfBounds { axis: d, ndim });
            }
            if seen[d] {
                return Err(EngineError::InvalidArgument(format!(
                    "Duplicate axis {} in permute order {:?}",
                    d, dims
                )));
            }
            seen[d] = true;
        }

        let mut new_shape = vec![0; ndim];
        let mut new_strides = vec![0; ndim];

        for (i, &d) in dims.iter().enumerate() {
            new_shape[i] = self.shape[d];
            new_strides[i] = self.strides[d];
        }

        Ok(RawTensor {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    /// Removes dimensions of size 1.
    pub fn squeeze(&self, axis: Option<usize>) -> Result<RawTensor> {
        let mut new_shape = Vec::new();
        let mut new_strides = Vec::new();

        match axis {
            Some(a) => {
                if a >= self.ndim() {
                    return Err(EngineError::DimensionOutOfBounds {
                        axis: a,
                        ndim: self.ndim(),
                    });
                }
                for (i, (&d, &s)) in self.shape.iter().zip(self.strides.iter()).enumerate() {
                    if i != a || d != 1 {
                        new_shape.push(d);
                        new_strides.push(s);
                    }
                }
            }
            None => {
                for (&d, &s) in self.shape.iter().zip(self.strides.iter()) {
                    if d != 1 {
                        new_shape.push(d);
                        new_strides.push(s);
                    }
                }
            }
        }

        Ok(RawTensor {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    /// Inserts a new dimension of size 1 at the specified axis.
    pub fn unsqueeze(&self, axis: usize) -> Result<RawTensor> {
        if axis > self.ndim() {
            return Err(EngineError::DimensionOutOfBounds {
                axis,
                ndim: self.ndim() + 1,
            });
        }

        let mut new_shape = self.shape.clone();
        let mut new_strides = self.strides.clone();

        new_shape.insert(axis, 1);
        let next_stride = if axis < self.strides.len() {
            self.strides[axis] * self.shape[axis]
        } else {
            1
        };
        new_strides.insert(axis, next_stride);

        Ok(RawTensor {
            storage: self.storage.clone(),
            shape: new_shape,
            strides: new_strides,
            offset: self.offset,
        })
    }

    /// Flattens the tensor into a 1D contiguous tensor.
    pub fn flatten(&self) -> Result<RawTensor> {
        self.reshape(&[self.numel()])
    }
}

impl fmt::Debug for RawTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        display::format_tensor(self, f)
    }
}

impl fmt::Display for RawTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        display::format_tensor(self, f)
    }
}
