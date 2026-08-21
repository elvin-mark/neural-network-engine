use crate::autograd::Tensor;
use crate::error::{EngineError, Result};
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

    /// Looks up embeddings for a sequence of token indices with bounds validation.
    pub fn forward_indices(&self, indices: &[usize]) -> Result<Tensor> {
        for (pos, &idx) in indices.iter().enumerate() {
            if idx >= self.num_embeddings {
                return Err(EngineError::InvalidArgument(format!(
                    "Token index {} at position {} is out of bounds for vocab size {}",
                    idx, pos, self.num_embeddings
                )));
            }
        }
        self.weight.embedding(indices)
    }
}

impl Module for Embedding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let contig = input.data().to_contiguous();
        let slice = contig.as_slice();
        let mut indices = Vec::with_capacity(slice.len());

        for (pos, &x) in slice.iter().enumerate() {
            if !x.is_finite() || x < 0.0 || x.fract() != 0.0 {
                return Err(EngineError::InvalidArgument(format!(
                    "Embedding forward expects non-negative integral token indices as floats, found {} at position {}",
                    x, pos
                )));
            }
            let idx = x as usize;
            if idx >= self.num_embeddings {
                return Err(EngineError::InvalidArgument(format!(
                    "Token index {} at position {} is out of bounds for vocab size {}",
                    idx, pos, self.num_embeddings
                )));
            }
            indices.push(idx);
        }

        let out = self.weight.embedding(&indices)?;

        // If input was multi-dimensional, reshape accordingly: [..., D]
        let mut target_shape = input.shape();
        target_shape.push(self.embedding_dim);
        out.reshape(&target_shape)
    }

    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone()]
    }
}
