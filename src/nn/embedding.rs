//! Embedding lookup table layer.

use crate::autograd::Tensor;
use crate::error::Result;
use crate::nn::module::Module;

/// Lookup table that stores embeddings of a fixed dictionary and size.
#[derive(Clone)]
pub struct Embedding {
    pub weight: Tensor,
    pub num_embeddings: usize,
    pub embedding_dim: usize,
}

impl Embedding {
    /// Creates a new Embedding layer with standard normal initialization.
    pub fn new(num_embeddings: usize, embedding_dim: usize) -> Self {
        let weight = Tensor::randn(&[num_embeddings, embedding_dim], 0.0, 1.0, true);
        Self {
            weight,
            num_embeddings,
            embedding_dim,
        }
    }

    /// Looks up embeddings for a sequence of token indices.
    pub fn forward_indices(&self, indices: &[usize]) -> Result<Tensor> {
        self.weight.embedding(indices)
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Assume input contains integer indices as floats
        let data = input.data();
        let slice = data.as_slice();
        let indices: Vec<usize> = slice.iter().map(|&x| x as usize).collect();
        let out = self.forward_indices(&indices)?;

        // If input was multi-dimensional, reshape accordingly
        let mut target_shape = input.shape();
        target_shape.push(self.embedding_dim);
        out.reshape(&target_shape)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}
