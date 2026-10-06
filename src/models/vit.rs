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
    /// Whether to prepend a learnable CLS token for classification (standard in ViT).
    pub use_cls_token: bool,
    /// LayerNorm epsilon.
    pub eps: f32,
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
            use_cls_token: false,
            eps: 1e-5,
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
            use_cls_token: false,
            eps: 1e-5,
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
            use_cls_token: false,
            eps: 1e-5,
        }
    }

    /// Standard Vision Transformer Tiny (`vit_tiny_patch16_224` / `timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k`).
    /// Uses 16x16 patches on 224x224 RGB images resulting in 196 image patch tokens + 1 CLS token ($14 \times 14 + 1 = 197$).
    pub fn vit_tiny_patch16_224() -> Self {
        Self {
            image_size: 224,
            patch_size: 16,
            in_channels: 3,
            num_classes: 1000,
            d_model: 192,
            num_layers: 12,
            num_heads: 3,
            mlp_dim: 768,
            use_cls_token: true,
            eps: 1e-6,
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
/// 3. Prepend Learnable Class Token: `[B, 1 + NumPatches, D]` (if `use_cls_token` is enabled)
/// 4. Add Learnable Positional Embeddings: `x + pos_embed`
/// 5. Stack of Pre-LayerNorm Bidirectional Transformer Encoder Layers
/// 6. Classification Head on CLS token or Global Mean Pooling: `LayerNorm(D) -> Linear(D, NumClasses)`
pub struct VisionTransformer {
    pub config: ViTConfig,
    pub patch_embed: Conv2d,
    pub cls_token: Option<Tensor>,
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
        let seq_len = if config.use_cls_token {
            num_patches + 1
        } else {
            num_patches
        };

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

        // 2. Class token if configured
        let cls_token = if config.use_cls_token {
            Some(Tensor::zeros(&[1, 1, config.d_model], true))
        } else {
            None
        };

        // 3. Learnable positional embeddings [1, SeqLen, D]
        let pos_embed = Tensor::randn(&[1, seq_len, config.d_model], 0.0, 0.02, true);

        // 4. Transformer Encoder Stack (is_causal = false for bidirectional vision attention)
        let mut blocks = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            blocks.push(TransformerBlock::with_options(
                config.d_model,
                config.num_heads,
                config.mlp_dim,
                false,
                config.eps,
            ));
        }

        // 5. Final LayerNorm & Classification Head
        let norm = LayerNorm::with_eps(config.d_model, config.eps);
        let head = Linear::new(config.d_model, config.num_classes);

        Self {
            config,
            patch_embed,
            cls_token,
            pos_embed,
            blocks,
            norm,
            head,
        }
    }

    /// Loads model weights from an in-memory dictionary of tensors.
    pub fn load_weights(
        &mut self,
        weights: &std::collections::HashMap<String, crate::tensor::RawTensor>,
    ) -> Result<()> {
        let set_tensor = |target: &mut Tensor, key: &str| -> Result<()> {
            if let Some(raw) = weights.get(key) {
                target.set_data(raw.clone());
                Ok(())
            } else {
                Err(EngineError::InvalidArgument(format!(
                    "Missing expected weight '{}' in weights map",
                    key
                )))
            }
        };

        let set_opt_tensor = |target: &mut Option<Tensor>, key: &str| -> Result<()> {
            if let Some(ref mut t) = target {
                set_tensor(t, key)
            } else {
                Ok(())
            }
        };

        // Patch embedding
        set_tensor(&mut self.patch_embed.weight, "patch_embed.proj.weight")?;
        set_opt_tensor(&mut self.patch_embed.bias, "patch_embed.proj.bias")?;

        // Optional class token
        if let Some(ref mut cls) = self.cls_token {
            set_tensor(cls, "cls_token")?;
        }

        // Position embedding
        set_tensor(&mut self.pos_embed, "pos_embed")?;

        // Transformer blocks
        for (i, block) in self.blocks.iter_mut().enumerate() {
            set_tensor(&mut block.ln1.weight, &format!("blocks.{}.norm1.weight", i))?;
            set_tensor(&mut block.ln1.bias, &format!("blocks.{}.norm1.bias", i))?;

            set_tensor(
                &mut block.attn.q_proj.weight,
                &format!("blocks.{}.attn.q_proj.weight", i),
            )?;
            set_opt_tensor(
                &mut block.attn.q_proj.bias,
                &format!("blocks.{}.attn.q_proj.bias", i),
            )?;
            set_tensor(
                &mut block.attn.k_proj.weight,
                &format!("blocks.{}.attn.k_proj.weight", i),
            )?;
            set_opt_tensor(
                &mut block.attn.k_proj.bias,
                &format!("blocks.{}.attn.k_proj.bias", i),
            )?;
            set_tensor(
                &mut block.attn.v_proj.weight,
                &format!("blocks.{}.attn.v_proj.weight", i),
            )?;
            set_opt_tensor(
                &mut block.attn.v_proj.bias,
                &format!("blocks.{}.attn.v_proj.bias", i),
            )?;
            set_tensor(
                &mut block.attn.out_proj.weight,
                &format!("blocks.{}.attn.out_proj.weight", i),
            )?;
            set_opt_tensor(
                &mut block.attn.out_proj.bias,
                &format!("blocks.{}.attn.out_proj.bias", i),
            )?;

            set_tensor(&mut block.ln2.weight, &format!("blocks.{}.norm2.weight", i))?;
            set_tensor(&mut block.ln2.bias, &format!("blocks.{}.norm2.bias", i))?;

            set_tensor(
                &mut block.mlp_fc1.weight,
                &format!("blocks.{}.mlp_fc1.weight", i),
            )?;
            set_opt_tensor(
                &mut block.mlp_fc1.bias,
                &format!("blocks.{}.mlp_fc1.bias", i),
            )?;
            set_tensor(
                &mut block.mlp_fc2.weight,
                &format!("blocks.{}.mlp_fc2.weight", i),
            )?;
            set_opt_tensor(
                &mut block.mlp_fc2.bias,
                &format!("blocks.{}.mlp_fc2.bias", i),
            )?;
        }

        // Post norm & head
        set_tensor(&mut self.norm.weight, "norm.weight")?;
        set_tensor(&mut self.norm.bias, "norm.bias")?;

        set_tensor(&mut self.head.weight, "head.weight")?;
        set_opt_tensor(&mut self.head.bias, "head.bias")?;

        Ok(())
    }

    /// Loads model weights directly from a SafeTensors file.
    pub fn load_safetensors<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<()> {
        let weights = crate::io::load_safetensors(path)?;
        self.load_weights(&weights)
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

        // 3. Prepend CLS token if present
        let mut x = if let Some(ref cls) = self.cls_token {
            let zeros = Tensor::zeros(&[b, 1, self.config.d_model], false);
            let cls_batched = cls.add(&zeros)?;
            Tensor::cat(&[&cls_batched, &tokens], 1)?
        } else {
            tokens
        };

        // 4. Add positional embeddings -> [B, SeqLen, D]
        x = x.add(&self.pos_embed)?;

        // 5. Transformer Encoder Layers
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        // 6. Final LayerNorm
        let x = self.norm.forward(&x)?;

        // 7. Representation for classification head
        let rep = if self.config.use_cls_token {
            // Extract CLS token at index 0 -> [B, D]
            x.slice(1, 0, 1)?.reshape(&[b, self.config.d_model])?
        } else {
            // Global Mean Pooling over spatial patches (axis 1) -> [B, D]
            x.mean(1, false)?
        };

        // 8. Linear classification head -> [B, NumClasses]
        self.head.forward(&rep)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.patch_embed.parameters());
        if let Some(ref cls) = self.cls_token {
            params.push(cls.clone());
        }
        params.push(self.pos_embed.clone());
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.norm.parameters());
        params.extend(self.head.parameters());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vit_cls_token_forward_and_backward() {
        let config = ViTConfig {
            image_size: 16,
            patch_size: 4,
            in_channels: 3,
            num_classes: 10,
            d_model: 16,
            num_layers: 2,
            num_heads: 2,
            mlp_dim: 32,
            use_cls_token: true,
            eps: 1e-6,
        };

        let vit = VisionTransformer::new(config);
        assert_eq!(vit.pos_embed.shape(), &[1, 17, 16]);
        assert!(vit.cls_token.is_some());

        let x = Tensor::randn(&[2, 3, 16, 16], 0.0, 1.0, true);
        let logits = vit.forward(&x).unwrap();
        assert_eq!(logits.shape(), &[2, 10]);

        let loss = logits.sum_all();
        loss.backward();

        assert!(x.grad().is_some());
        assert_eq!(x.grad().unwrap().shape(), &[2, 3, 16, 16]);
        assert!(vit.cls_token.as_ref().unwrap().grad().is_some());
        assert!(vit.pos_embed.grad().is_some());
        assert!(vit.head.weight.grad().is_some());
    }
}
