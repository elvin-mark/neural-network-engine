//! Fully connected dense (Linear) neural network layer.

use crate::autograd::Tensor;
use crate::error::Result;
use crate::nn::module::Module;

/// Fully connected linear layer: y = x * W^T + b.
#[derive(Clone)]
pub struct Linear {
    pub weight: Tensor,
    pub bias: Option<Tensor>,
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    /// Creates a new Linear layer with bias enabled and Kaiming uniform weight initialization.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self::with_bias(in_features, out_features, true)
    }

    /// Creates a new Linear layer with optional bias.
    pub fn with_bias(in_features: usize, out_features: usize, has_bias: bool) -> Self {
        let weight = Tensor::kaiming_uniform(&[out_features, in_features], in_features, true);
        let bias = if has_bias {
            let bound = 1.0 / (in_features as f32).sqrt();
            Some(Tensor::uniform(&[out_features], -bound, bound, true))
        } else {
            None
        };

        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }
}

impl Module for Linear {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let w_t = self.weight.transpose(0, 1)?;
        let mut out = input.matmul(&w_t)?;

        if let Some(ref b) = self.bias {
            // If input is batched (e.g. 2D [B, out_features]), broadcasting works automatically
            out = out.add(b)?;
        }

        Ok(out)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight.clone()];
        if let Some(ref b) = self.bias {
            params.push(b.clone());
        }
        params
    }
}
