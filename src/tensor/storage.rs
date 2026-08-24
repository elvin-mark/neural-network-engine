//! Underlying memory buffer storage for tensors with automatic zero-allocation pool recycling.

use crate::tensor::pool::TensorPool;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

/// RAII wrapper for a `Vec<f32>` that automatically recycles its memory back into `TensorPool` on drop.
pub struct PooledVec {
    pub(crate) vec: Option<Vec<f32>>,
}

impl PooledVec {
    pub fn new(vec: Vec<f32>) -> Self {
        Self { vec: Some(vec) }
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[f32] {
        self.vec.as_ref().unwrap().as_slice()
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        self.vec.as_mut().unwrap().as_mut_slice()
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.vec.as_ref().unwrap().len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.vec.as_ref().unwrap().is_empty()
    }
}

impl Clone for PooledVec {
    fn clone(&self) -> Self {
        let src = self.as_slice();
        let mut v = TensorPool::acquire(src.len());
        v.clear();
        v.extend_from_slice(src);
        Self::new(v)
    }
}

impl PartialEq for PooledVec {
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl fmt::Debug for PooledVec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", self.as_slice())
    }
}

impl Deref for PooledVec {
    type Target = [f32];
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for PooledVec {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}

impl Drop for PooledVec {
    fn drop(&mut self) {
        if let Some(vec) = self.vec.take() {
            TensorPool::recycle(vec);
        }
    }
}

/// Reference-counted contiguous storage buffer holding float32 values with automatic buffer recycling.
#[derive(Clone, Debug, PartialEq)]
pub struct Storage {
    data: Arc<PooledVec>,
}

impl Storage {
    /// Creates a new storage buffer with the given size initialized to zeros.
    pub fn zeros(size: usize) -> Self {
        let mut v = TensorPool::acquire(size);
        v.resize(size, 0.0);
        v.fill(0.0);
        Self {
            data: Arc::new(PooledVec::new(v)),
        }
    }

    /// Creates a new storage buffer filled with a constant value.
    pub fn filled(size: usize, val: f32) -> Self {
        let mut v = TensorPool::acquire(size);
        v.resize(size, val);
        v.fill(val);
        Self {
            data: Arc::new(PooledVec::new(v)),
        }
    }

    /// Creates storage from an existing `Vec<f32>`.
    pub fn from_vec(vec: Vec<f32>) -> Self {
        Self {
            data: Arc::new(PooledVec::new(vec)),
        }
    }

    /// Creates storage from a slice of floats.
    pub fn from_slice(slice: &[f32]) -> Self {
        let mut v = TensorPool::acquire(slice.len());
        v.clear();
        v.extend_from_slice(slice);
        Self {
            data: Arc::new(PooledVec::new(v)),
        }
    }

    /// Returns the number of elements in the storage buffer.
    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Checks if the buffer is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Returns an immutable slice view of the buffer.
    #[inline(always)]
    pub fn as_slice(&self) -> &[f32] {
        self.data.as_slice()
    }

    /// Returns a mutable slice of the buffer. Clones the buffer if shared (copy-on-write).
    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        Arc::make_mut(&mut self.data).as_mut_slice()
    }

    /// Reads the element at the specified physical offset.
    #[inline(always)]
    pub fn get(&self, offset: usize) -> f32 {
        self.data[offset]
    }

    /// Sets the element at the specified physical offset.
    #[inline(always)]
    pub fn set(&mut self, offset: usize, value: f32) {
        Arc::make_mut(&mut self.data)[offset] = value;
    }
}
