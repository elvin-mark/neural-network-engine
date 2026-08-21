//! Underlying memory buffer storage for tensors.

use std::sync::Arc;

/// Reference-counted contiguous storage buffer holding float32 values.
#[derive(Clone, Debug, PartialEq)]
pub struct Storage {
    data: Arc<Vec<f32>>,
}

impl Storage {
    /// Creates a new storage buffer with the given size initialized to zeros.
    pub fn zeros(size: usize) -> Self {
        Self {
            data: Arc::new(vec![0.0; size]),
        }
    }

    /// Creates a new storage buffer filled with a constant value.
    pub fn filled(size: usize, val: f32) -> Self {
        Self {
            data: Arc::new(vec![val; size]),
        }
    }

    /// Creates storage from an existing `Vec<f32>`.
    pub fn from_vec(vec: Vec<f32>) -> Self {
        Self {
            data: Arc::new(vec),
        }
    }

    /// Creates storage from a slice of floats.
    pub fn from_slice(slice: &[f32]) -> Self {
        Self {
            data: Arc::new(slice.to_vec()),
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
        &self.data
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
