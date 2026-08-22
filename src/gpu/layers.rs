//! GPU-accelerated Neural Network Layers (Linear, LayerNorm, RMSNorm) and `ToGpu` conversion traits.

use crate::error::Result;
use crate::gpu::context::GpuContext;
use crate::gpu::tensor::GpuTensor;
use crate::nn::linear::Linear;
use crate::nn::norm::{LayerNorm, RMSNorm};
use crate::tensor::RawTensor;
use std::sync::Arc;

/// Trait for transferring models, layers, or tensors from Host CPU memory into GPU VRAM.
pub trait ToGpu {
    type Target;
    fn to_gpu(&self, ctx: &Arc<GpuContext>) -> Result<Self::Target>;
}

impl ToGpu for RawTensor {
    type Target = GpuTensor;
    fn to_gpu(&self, ctx: &Arc<GpuContext>) -> Result<GpuTensor> {
        GpuTensor::from_raw(self, ctx)
    }
}

/// GPU-accelerated Dense Linear Layer.
pub struct GpuLinear {
    pub weight: GpuTensor,
    pub bias: Option<GpuTensor>,
}

impl GpuLinear {
    pub fn new(weight: GpuTensor, bias: Option<GpuTensor>) -> Self {
        Self { weight, bias }
    }

    /// Computes `Y = X * W + b` in GPU VRAM.
    pub fn forward(&self, input: &GpuTensor) -> Result<GpuTensor> {
        let mut out = input.matmul(&self.weight)?;
        if let Some(ref b) = self.bias {
            out = out.add(b)?;
        }
        Ok(out)
    }
}

impl ToGpu for Linear {
    type Target = GpuLinear;
    fn to_gpu(&self, ctx: &Arc<GpuContext>) -> Result<GpuLinear> {
        let weight_raw = self.weight.data().transpose(0, 1)?;
        let weight_gpu = weight_raw.to_gpu(ctx)?;
        let bias_gpu = if let Some(ref b) = self.bias {
            Some(b.data().to_gpu(ctx)?)
        } else {
            None
        };
        Ok(GpuLinear::new(weight_gpu, bias_gpu))
    }
}

/// GPU-accelerated Layer Normalization.
pub struct GpuLayerNorm {
    pub gamma: GpuTensor,
    pub beta: Option<GpuTensor>,
    pub eps: f32,
}

impl GpuLayerNorm {
    pub fn forward(&self, input: &GpuTensor) -> Result<GpuTensor> {
        input.layernorm(&self.gamma, self.beta.as_ref(), self.eps)
    }
}

impl ToGpu for LayerNorm {
    type Target = GpuLayerNorm;
    fn to_gpu(&self, ctx: &Arc<GpuContext>) -> Result<GpuLayerNorm> {
        let gamma_gpu = self.weight.data().to_gpu(ctx)?;
        let beta_gpu = Some(self.bias.data().to_gpu(ctx)?);
        Ok(GpuLayerNorm {
            gamma: gamma_gpu,
            beta: beta_gpu,
            eps: self.eps,
        })
    }
}

/// GPU-accelerated Root-Mean-Square Normalization (RMSNorm).
pub struct GpuRMSNorm {
    pub gamma: GpuTensor,
    pub eps: f32,
}

impl GpuRMSNorm {
    pub fn forward(&self, input: &GpuTensor) -> Result<GpuTensor> {
        input.rmsnorm(&self.gamma, self.eps)
    }
}

impl ToGpu for RMSNorm {
    type Target = GpuRMSNorm;
    fn to_gpu(&self, ctx: &Arc<GpuContext>) -> Result<GpuRMSNorm> {
        let gamma_gpu = self.weight.data().to_gpu(ctx)?;
        Ok(GpuRMSNorm {
            gamma: gamma_gpu,
            eps: self.eps,
        })
    }
}
