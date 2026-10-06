//! Vision Transformer (ViT) model for image classification.

use crate::autograd::Tensor;
use crate::error::{EngineError, Result};
use crate::nn::conv::Conv2d;
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::nn::norm::LayerNorm;
use crate::nn::transformer::TransformerBlock;

/// Configuration hyperparameters for Vision Transformer (ViT).
#[derive(Debug, Clone)]
pub struct ViTConfig {
    /// Height and width of input square image in pixels (e.g., 32 for CIFAR-10).
    pub image_size: usize,
    /// Spatial dimension of square patch in pixels (e.g., 4).
    pub patch_size: usize,
    /// Number of input image channels (e.g., 3 for RGB).
    pub in_channels: usize,
    /// Number of target classification categories (e.g., 10 for CIFAR-10).
    pub num_classes: usize,
    /// Latent embedding dimension ($d_{model}$).
    pub d_model: usize,
    /// Number of Transformer encoder layers.
    pub num_layers: usize,
    /// Number of self-attention heads.
    pub num_heads: usize,
    /// Hidden dimension of feedforward MLP.
    pub mlp_dim: usize,
}

impl ViTConfig {
    /// Pre-configured compact ViT for 32x32 CIFAR-10 classification.
    /// Uses 4x4 patches resulting in 64 spatial tokens ($8 \times 8$).
    pub fn cifar10() -> Self {
        Self {
            image_size: 32,
            patch_size: 4,
            in_channels: 3,
            num_classes: 10,
            d_model: 64,
            num_layers: 3,
            num_heads: 4,
            mlp_dim: 256,
        }
    }

    /// Pre-configured compact ViT for 32x32 CIFAR-100 classification (100 categories).
    /// Uses 4x4 patches resulting in 64 spatial tokens ($8 \times 8$).
    pub fn cifar100() -> Self {
        Self {
            image_size: 32,
            patch_size: 4,
            in_channels: 3,
            num_classes: 100,
            d_model: 64,
            num_layers: 3,
            num_heads: 4,
            mlp_dim: 256,
        }
    }

    /// Pre-configured compact ViT for 28x28 MNIST classification.
    /// Uses 4x4 patches resulting in 49 spatial tokens ($7 \times 7$).
    pub fn mnist() -> Self {
        Self {
            image_size: 28,
            patch_size: 4,
            in_channels: 1,
            num_classes: 10,
            d_model: 64,
            num_layers: 2,
            num_heads: 4,
            mlp_dim: 128,
        }
    }

    /// Computes the total number of patches $(H / P) \times (W / P)$.
    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size) * (self.image_size / self.patch_size)
    }
}

/// Vision Transformer (ViT) architecture for 2D image classification.
///
/// Steps:
/// 1. Patch Partitioning & Linear Projection via strided 2D Convolution: `[B, C, H, W] -> [B, D, H/P, W/P]`
/// 2. Flatten & Transpose to token sequence: `[B, NumPatches, D]`
/// 3. Add Learnable Positional Embeddings: `x + pos_embed`
/// 4. Stack of Pre-LayerNorm Bidirectional Transformer Encoder Layers
/// 5. Global Mean Pooling over patch sequence dimension
/// 6. Classification MLP Head: `LayerNorm(D) -> Linear(D, NumClasses)`
pub struct VisionTransformer {
    pub config: ViTConfig,
    pub patch_embed: Conv2d,
    pub pos_embed: Tensor,
    pub blocks: Vec<TransformerBlock>,
    pub norm: LayerNorm,
    pub head: Linear,
}

impl VisionTransformer {
    /// Creates a new VisionTransformer model initialized with Xavier/Kaiming weights.
    pub fn new(config: ViTConfig) -> Self {
        assert_eq!(
            config.image_size % config.patch_size,
            0,
            "image_size ({}) must be divisible by patch_size ({})",
            config.image_size,
            config.patch_size
        );
        let num_patches = config.num_patches();

        // 1. Patch projection via strided Conv2d: kernel_size = patch_size, stride = patch_size
        let patch_embed = Conv2d::with_options(
            config.in_channels,
            config.d_model,
            (config.patch_size, config.patch_size),
            (config.patch_size, config.patch_size),
            (0, 0),
            (1, 1),
            true,
        );

        // 2. Learnable positional embeddings [1, NumPatches, D]
        let pos_embed = Tensor::randn(&[1, num_patches, config.d_model], 0.0, 0.02, true);

        // 3. Transformer Encoder Stack (is_causal = false for bidirectional vision attention)
        let mut blocks = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            blocks.push(TransformerBlock::new(
                config.d_model,
                config.num_heads,
                false,
            ));
        }

        // 4. Final LayerNorm & Classification Head
        let norm = LayerNorm::new(config.d_model);
        let head = Linear::new(config.d_model, config.num_classes);

        Self {
            config,
            patch_embed,
            pos_embed,
            blocks,
            norm,
            head,
        }
    }
}

impl Module for VisionTransformer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let shape = input.shape();
        if shape.len() != 4 {
            return Err(EngineError::IncompatibleShapes {
                op: "VisionTransformer forward (expected 4D tensor [B, C, H, W])",
                shapes: vec![shape],
            });
        }

        let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        if c != self.config.in_channels
            || h != self.config.image_size
            || w != self.config.image_size
        {
            return Err(EngineError::ShapeMismatch {
                expected: vec![
                    b,
                    self.config.in_channels,
                    self.config.image_size,
                    self.config.image_size,
                ],
                actual: shape,
            });
        }

        // 1. Patch projection -> [B, D, H/P, W/P]
        let p = self.patch_embed.forward(input)?;
        let num_patches = self.config.num_patches();

        // 2. Permute [B, D, H', W'] -> [B, H', W', D] and reshape to [B, NumPatches, D]
        let tokens = p
            .permute(&[0, 2, 3, 1])?
            .reshape(&[b, num_patches, self.config.d_model])?;

        // 3. Add positional embeddings -> [B, NumPatches, D]
        let mut x = tokens.add(&self.pos_embed)?;

        // 4. Transformer Encoder Layers
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        // 5. Final LayerNorm
        let x = self.norm.forward(&x)?;

        // 6. Global Mean Pooling over spatial patches (axis 1) -> [B, D]
        let pooled = x.mean(1, false)?;

        // 7. Linear classification head -> [B, NumClasses]
        self.head.forward(&pooled)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.patch_embed.parameters());
        params.push(self.pos_embed.clone());
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.norm.parameters());
        params.extend(self.head.parameters());
        params
    }
}
