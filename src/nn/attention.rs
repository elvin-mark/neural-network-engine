//! Multi-Head Causal Self-Attention module.

use crate::autograd::Tensor;
use crate::error::{EngineError, Result};
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::tensor::RawTensor;

/// Multi-Head Attention module supporting optional causal autoregressive masking.
pub struct MultiHeadAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub num_heads: usize,
    pub d_model: usize,
    pub head_dim: usize,
    pub is_causal: bool,
}

impl MultiHeadAttention {
    pub fn new(d_model: usize, num_heads: usize, is_causal: bool) -> Self {
        assert_eq!(
            d_model % num_heads,
            0,
            "d_model ({}) must be divisible by num_heads ({})",
            d_model,
            num_heads
        );
        let head_dim = d_model / num_heads;

        Self {
            q_proj: Linear::new(d_model, d_model),
            k_proj: Linear::new(d_model, d_model),
            v_proj: Linear::new(d_model, d_model),
            out_proj: Linear::new(d_model, d_model),
            num_heads,
            d_model,
            head_dim,
            is_causal,
        }
    }

    /// Computes multi-head self-attention on input tensor of shape [BatchSize, SeqLen, DModel].
    pub fn forward_attention(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.shape();
        if shape.len() != 3 {
            return Err(EngineError::IncompatibleShapes {
                op: "MultiHeadAttention forward (expected 3D input [B, T, C])",
                shapes: vec![shape],
            });
        }

        let (b, t, c) = (shape[0], shape[1], shape[2]);
        let h = self.num_heads;
        let d = self.head_dim;

        // 1. Compute Q, K, V projections -> [B, T, C]
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        // 2. Reshape & transpose to [B, H, T, D]
        let q = q.reshape(&[b, t, h, d])?.transpose(1, 2)?;
        let k = k.reshape(&[b, t, h, d])?.transpose(1, 2)?;
        let v = v.reshape(&[b, t, h, d])?.transpose(1, 2)?;

        // 3. Attention scores = Q * K^T / sqrt(D) -> [B, H, T, T]
        let k_t = k.transpose(2, 3)?;
        let scores = q.matmul(&k_t)?;
        let scale = 1.0 / (d as f32).sqrt();
        let scale_tensor = Tensor::scalar(scale, false);
        let mut scaled_scores = scores.mul(&scale_tensor)?;

        // 4. Apply causal mask if enabled
        if self.is_causal && t > 1 {
            let mut mask_data = vec![0.0; t * t];
            for row in 0..t {
                for col in 0..t {
                    if col > row {
                        mask_data[row * t + col] = -1e4; // large negative value
                    }
                }
            }
            let mask_raw = RawTensor::from_vec(mask_data, vec![1, 1, t, t]);
            let mask = Tensor::new(mask_raw, false);
            scaled_scores = scaled_scores.add(&mask)?;
        }

        // 5. Softmax along last dimension -> Attention weights [B, H, T, T]
        let attn_weights = scaled_scores.softmax(3)?;

        // 6. Context = Weights * V -> [B, H, T, D]
        let context = attn_weights.matmul(&v)?;

        // 7. Transpose & reshape back to [B, T, C]
        let context = context.transpose(1, 2)?.reshape(&[b, t, c])?;

        // 8. Output linear projection
        self.out_proj.forward(&context)
    }
}

impl Module for MultiHeadAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_attention(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.out_proj.parameters());
        params
    }
}
