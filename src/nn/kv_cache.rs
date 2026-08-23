//! Key-Value Cache (KV-Cache) for fast $O(N)$ autoregressive token generation.
//!
//! Reuses previously projected Key and Value attention states across decoding steps,
//! eliminating quadratic $O(N^2)$ recomputations during LLM generation.

use crate::autograd::Tensor;
use crate::error::{EngineError, Result};

/// Key-Value cache storing past attention projections across Transformer layers.
#[derive(Clone, Debug, Default)]
pub struct KVCache {
    /// Cached (Key, Value) tensors for each layer, shaped [Batch, NumHeads, CachedSeqLen, HeadDim].
    pub layers: Vec<Option<(Tensor, Tensor)>>,
}

impl KVCache {
    /// Creates a new `KVCache` with capacity for `num_layers`.
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: vec![None; num_layers],
        }
    }

    /// Resets / clears all cached Key-Value states across all layers.
    pub fn reset(&mut self) {
        for slot in &mut self.layers {
            *slot = None;
        }
    }

    /// Returns the number of layers managed by this cache.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Returns the total sequence length currently stored in the cache for the given layer.
    pub fn current_seq_len(&self, layer_idx: usize) -> usize {
        if layer_idx < self.layers.len() {
            if let Some((ref k, _)) = self.layers[layer_idx] {
                return k.shape()[2];
            }
        }
        0
    }

    /// Updates the KV-cache for a given layer by appending `k_new` and `v_new` along the sequence dimension (axis 2).
    ///
    /// Returns the updated full `(k_all, v_all)` tensors.
    pub fn update(
        &mut self,
        layer_idx: usize,
        k_new: &Tensor,
        v_new: &Tensor,
    ) -> Result<(Tensor, Tensor)> {
        if layer_idx >= self.layers.len() {
            return Err(EngineError::InvalidArgument(format!(
                "KVCache layer index {} out of range (total layers: {})",
                layer_idx,
                self.layers.len()
            )));
        }

        let (k_full, v_full) = match self.layers[layer_idx].take() {
            Some((k_prev, v_prev)) => {
                let k_cat = Tensor::cat(&[&k_prev, k_new], 2)?;
                let v_cat = Tensor::cat(&[&v_prev, v_new], 2)?;
                (k_cat, v_cat)
            }
            None => (k_new.clone(), v_new.clone()),
        };

        self.layers[layer_idx] = Some((k_full.clone(), v_full.clone()));
        Ok((k_full, v_full))
    }
}
