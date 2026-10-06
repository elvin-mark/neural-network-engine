//! ModernBERT architecture (Answer.AI / LightOn 2024).
//!
//! ModernBERT is a modernized bidirectional encoder that incorporates architectural improvements
//! from recent LLM advances:
//! - **Rotary Position Embeddings (RoPE)** replacing learned absolute 1D positional embeddings.
//! - **Pre-RMSNorm** residual stream (instead of post-LayerNorm), providing gradient stability.
//! - **Gated Feed-Forward Networks (GeGLU / SwiGLU)** replacing standard GELU FFNs.
//! - **Zero Biases** across linear projections for cleaner representations and faster GEMM execution.
//! - **Unpadded/Bidirectional Attention** without causal masking, natively scalable to 8k+ sequence lengths.

use crate::autograd::Tensor;
use crate::error::{EngineError, Result};
use crate::models::llama::RotaryEmbedding;
use crate::nn::embedding::Embedding;
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::nn::norm::RMSNorm;

/// Configuration hyperparameters for ModernBERT.
#[derive(Debug, Clone)]
pub struct ModernBertConfig {
    /// Target vocabulary size (e.g. 50368).
    pub vocab_size: usize,
    /// Hidden dimension of representations ($d_{model}$, e.g. 768).
    pub d_model: usize,
    /// Number of bidirectional encoder layers.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Intermediate hidden dimension of the Gated FFN (e.g. $8/3 \times d_{model}$).
    pub intermediate_size: usize,
    /// Maximum sequence length supported by RoPE cache (e.g. 8192).
    pub max_position_embeddings: usize,
    /// Base frequency for Rotary Position Embeddings (e.g. 10000.0 or 160000.0).
    pub rope_theta: f32,
    /// Epsilon value for RMSNorm.
    pub norm_eps: f32,
}

impl ModernBertConfig {
    /// Pre-configured compact ModernBERT model for fast local testing and unit tests.
    pub fn tiny() -> Self {
        Self {
            vocab_size: 256,
            d_model: 64,
            num_layers: 2,
            num_heads: 4,
            intermediate_size: 160,
            max_position_embeddings: 512,
            rope_theta: 10000.0,
            norm_eps: 1e-6,
        }
    }

    /// ModernBERT-base reference architecture configuration.
    pub fn base(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            d_model: 768,
            num_layers: 22,
            num_heads: 12,
            intermediate_size: 2048,
            max_position_embeddings: 8192,
            rope_theta: 10000.0,
            norm_eps: 1e-6,
        }
    }
}

/// GeGLU / Gated Linear Unit feed-forward network with zero bias.
///
/// Computes: $\text{down\_proj}(\text{gelu}(\text{gate\_proj}(x)) \odot \text{up\_proj}(x))$
#[derive(Clone)]
pub struct ModernBertMLP {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl ModernBertMLP {
    pub fn new(d_model: usize, intermediate_size: usize) -> Self {
        Self {
            gate_proj: Linear::without_bias(d_model, intermediate_size),
            up_proj: Linear::without_bias(d_model, intermediate_size),
            down_proj: Linear::without_bias(intermediate_size, d_model),
        }
    }
}

impl Module for ModernBertMLP {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.gelu()?;
        let up = self.up_proj.forward(x)?;
        let h = gate.mul(&up)?;
        self.down_proj.forward(&h)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.gate_proj.parameters());
        params.extend(self.up_proj.parameters());
        params.extend(self.down_proj.parameters());
        params
    }
}

/// Bidirectional multi-head self-attention with Rotary Position Embeddings (RoPE) and zero bias.
#[derive(Clone)]
pub struct ModernBertAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub out_proj: Linear,
    pub rope: RotaryEmbedding,
    pub num_heads: usize,
    pub d_model: usize,
    pub head_dim: usize,
}

impl ModernBertAttention {
    pub fn new(d_model: usize, num_heads: usize, max_seq_len: usize, rope_theta: f32) -> Self {
        assert_eq!(
            d_model % num_heads,
            0,
            "d_model ({}) must be divisible by num_heads ({})",
            d_model,
            num_heads
        );
        let head_dim = d_model / num_heads;

        Self {
            q_proj: Linear::without_bias(d_model, d_model),
            k_proj: Linear::without_bias(d_model, d_model),
            v_proj: Linear::without_bias(d_model, d_model),
            out_proj: Linear::without_bias(d_model, d_model),
            rope: RotaryEmbedding::new(head_dim, max_seq_len, rope_theta),
            num_heads,
            d_model,
            head_dim,
        }
    }

    /// Computes bidirectional multi-head self-attention with RoPE on input tensor `x` `[B, T, D]`.
    pub fn forward_attention(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.shape();
        if shape.len() != 3 {
            return Err(EngineError::InvalidArgument(format!(
                "ModernBertAttention expects 3D tensor [Batch, SeqLen, DModel], got shape {:?}",
                shape
            )));
        }
        let (b, t, d) = (shape[0], shape[1], shape[2]);
        if d != self.d_model {
            return Err(EngineError::ShapeMismatch {
                expected: vec![self.d_model],
                actual: vec![d],
            });
        }

        let h = self.num_heads;
        let hd = self.head_dim;

        // 1. Projections -> [B, T, H, HD] -> [B, H, T, HD]
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape(&[b, t, h, hd])?.transpose(1, 2)?;
        let k = k.reshape(&[b, t, h, hd])?.transpose(1, 2)?;
        let v = v.reshape(&[b, t, h, hd])?.transpose(1, 2)?;

        // 2. Apply RoPE to Queries and Keys
        let q = self.rope.apply(&q, 0)?;
        let k = self.rope.apply(&k, 0)?;

        // 3. Bidirectional Scaled Dot-Product Attention: (Q @ K^T) / sqrt(HD)
        let k_t = k.transpose(2, 3)?;
        let scores = q.matmul(&k_t)?;
        let scale = 1.0 / (hd as f32).sqrt();
        let scaled_scores = scores.mul_scalar(scale)?;

        // 4. Softmax over entire sequence (bidirectional, no causal mask)
        let weights = scaled_scores.softmax(3)?;

        // 5. Context aggregation -> [B, H, T, HD]
        let context = weights.matmul(&v)?;

        // 6. Merge heads -> [B, T, D] -> Output projection
        let merged = context.transpose(1, 2)?.reshape(&[b, t, h * hd])?;
        self.out_proj.forward(&merged)
    }
}

impl Module for ModernBertAttention {
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

/// A single ModernBERT Transformer Encoder Layer with Pre-RMSNorm residual connections.
///
/// Modern pre-norm design:
/// $$x = x + \text{Attention}(\text{RMSNorm}(x))$$
/// $$x = x + \text{MLP}(\text{RMSNorm}(x))$$
#[derive(Clone)]
pub struct ModernBertLayer {
    pub attn_norm: RMSNorm,
    pub attention: ModernBertAttention,
    pub mlp_norm: RMSNorm,
    pub mlp: ModernBertMLP,
}

impl ModernBertLayer {
    pub fn new(config: &ModernBertConfig) -> Self {
        Self {
            attn_norm: RMSNorm::with_eps(config.d_model, config.norm_eps),
            attention: ModernBertAttention::new(
                config.d_model,
                config.num_heads,
                config.max_position_embeddings,
                config.rope_theta,
            ),
            mlp_norm: RMSNorm::with_eps(config.d_model, config.norm_eps),
            mlp: ModernBertMLP::new(config.d_model, config.intermediate_size),
        }
    }

    pub fn forward_layer(&self, x: &Tensor) -> Result<Tensor> {
        // 1. Pre-Norm Attention Residual
        let norm_x = self.attn_norm.forward(x)?;
        let attn_out = self.attention.forward_attention(&norm_x)?;
        let x = x.add(&attn_out)?;

        // 2. Pre-Norm MLP Residual
        let norm_x = self.mlp_norm.forward(&x)?;
        let mlp_out = self.mlp.forward(&norm_x)?;
        x.add(&mlp_out)
    }
}

impl Module for ModernBertLayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_layer(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.attn_norm.parameters());
        params.extend(self.attention.parameters());
        params.extend(self.mlp_norm.parameters());
        params.extend(self.mlp.parameters());
        params
    }
}

/// Stack of ModernBERT Transformer Encoder Layers.
#[derive(Clone)]
pub struct ModernBertEncoder {
    pub layers: Vec<ModernBertLayer>,
    pub final_norm: RMSNorm,
}

impl ModernBertEncoder {
    pub fn new(config: &ModernBertConfig) -> Self {
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(ModernBertLayer::new(config));
        }
        let final_norm = RMSNorm::with_eps(config.d_model, config.norm_eps);
        Self { layers, final_norm }
    }

    pub fn forward_encoder(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let mut x = hidden_states.clone();
        for layer in &self.layers {
            x = layer.forward_layer(&x)?;
        }
        self.final_norm.forward(&x)
    }
}

impl Module for ModernBertEncoder {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_encoder(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params.extend(self.final_norm.parameters());
        params
    }
}

/// The core ModernBERT Model (Word Embeddings + Modern Encoder + Mean/CLS Pooling).
#[derive(Clone)]
pub struct ModernBertModel {
    pub embeddings: Embedding,
    pub encoder: ModernBertEncoder,
    pub config: ModernBertConfig,
}

impl ModernBertModel {
    pub fn new(config: ModernBertConfig) -> Self {
        let embeddings = Embedding::new(config.vocab_size, config.d_model);
        let encoder = ModernBertEncoder::new(&config);
        Self {
            embeddings,
            encoder,
            config,
        }
    }

    /// Forward pass returning `(sequence_output [B, T, d_model], pooled_output [B, d_model])`.
    pub fn forward_model(&self, input_ids: &Tensor) -> Result<(Tensor, Tensor)> {
        // 1. Word Embeddings only (RoPE in attention encodes positional geometry)
        let embeds = self.embeddings.forward(input_ids)?;

        // 2. ModernBERT Encoder with Pre-RMSNorm
        let seq_out = self.encoder.forward_encoder(&embeds)?;

        // 3. Pooled representation: Mean pooling across token representations along sequence dim
        let pooled_out = seq_out.mean(1, false)?;

        Ok((seq_out, pooled_out))
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

        set_tensor(&mut self.embeddings.weight, "embeddings.weight")?;

        for (i, layer) in self.encoder.layers.iter_mut().enumerate() {
            set_tensor(
                &mut layer.attn_norm.weight,
                &format!("encoder.layers.{}.attn_norm.weight", i),
            )?;

            set_tensor(
                &mut layer.attention.q_proj.weight,
                &format!("encoder.layers.{}.attention.q_proj.weight", i),
            )?;
            set_tensor(
                &mut layer.attention.k_proj.weight,
                &format!("encoder.layers.{}.attention.k_proj.weight", i),
            )?;
            set_tensor(
                &mut layer.attention.v_proj.weight,
                &format!("encoder.layers.{}.attention.v_proj.weight", i),
            )?;
            set_tensor(
                &mut layer.attention.out_proj.weight,
                &format!("encoder.layers.{}.attention.out_proj.weight", i),
            )?;

            set_tensor(
                &mut layer.mlp_norm.weight,
                &format!("encoder.layers.{}.mlp_norm.weight", i),
            )?;

            set_tensor(
                &mut layer.mlp.gate_proj.weight,
                &format!("encoder.layers.{}.mlp.gate_proj.weight", i),
            )?;
            set_tensor(
                &mut layer.mlp.up_proj.weight,
                &format!("encoder.layers.{}.mlp.up_proj.weight", i),
            )?;
            set_tensor(
                &mut layer.mlp.down_proj.weight,
                &format!("encoder.layers.{}.mlp.down_proj.weight", i),
            )?;
        }

        set_tensor(
            &mut self.encoder.final_norm.weight,
            "encoder.final_norm.weight",
        )?;

        Ok(())
    }

    /// Loads model weights directly from a SafeTensors file.
    pub fn load_safetensors<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<()> {
        let weights = crate::io::load_safetensors(path)?;
        self.load_weights(&weights)
    }
}

impl Module for ModernBertModel {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (seq_out, _) = self.forward_model(input)?;
        Ok(seq_out)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.embeddings.parameters());
        params.extend(self.encoder.parameters());
        params
    }
}

/// ModernBERT model with a sequence classification head (GLUE, sentiment, intent classification).
#[derive(Clone)]
pub struct ModernBertForSequenceClassification {
    pub model: ModernBertModel,
    pub classifier: Linear,
}

impl ModernBertForSequenceClassification {
    pub fn new(config: ModernBertConfig, num_classes: usize) -> Self {
        let d_model = config.d_model;
        let model = ModernBertModel::new(config);
        let classifier = Linear::new(d_model, num_classes);
        Self { model, classifier }
    }

    /// Forward pass returning logits `[B, num_classes]`.
    pub fn forward_classification(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_, pooled_out) = self.model.forward_model(input_ids)?;
        self.classifier.forward(&pooled_out)
    }
}

impl Module for ModernBertForSequenceClassification {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_classification(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.model.parameters());
        params.extend(self.classifier.parameters());
        params
    }
}

/// ModernBERT model specialized for generating dense semantic text embeddings.
#[derive(Clone)]
pub struct ModernBertForSequenceEmbedding {
    pub model: ModernBertModel,
}

impl ModernBertForSequenceEmbedding {
    pub fn new(config: ModernBertConfig) -> Self {
        Self {
            model: ModernBertModel::new(config),
        }
    }

    /// Computes normalized semantic text embeddings `[B, d_model]`.
    pub fn forward_embedding(&self, input_ids: &Tensor) -> Result<Tensor> {
        let (_, pooled) = self.model.forward_model(input_ids)?;
        // L2 normalize embeddings: v / sqrt(sum(v^2) + eps)
        let sq = pooled.powf(2.0)?;
        let sum_sq = sq.sum(1, true)?;
        let eps = Tensor::scalar(1e-8, false);
        let norm = sum_sq.add(&eps)?.powf(0.5)?;
        pooled.div(&norm)
    }
}

impl Module for ModernBertForSequenceEmbedding {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_embedding(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.model.parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::RawTensor;

    #[test]
    fn test_modern_bert_forward_and_backward() {
        let config = ModernBertConfig::tiny();
        let model = ModernBertModel::new(config);

        let input_raw = RawTensor::from_slice(
            &[
                1.0, 10.0, 25.0, 42.0, 50.0, 60.0, // Batch 0
                2.0, 8.0, 19.0, 33.0, 47.0, 55.0, // Batch 1
            ],
            &[2, 6],
        );
        let input_ids = Tensor::new(input_raw, false);

        let (seq_out, pooled_out) = model.forward_model(&input_ids).unwrap();
        assert_eq!(seq_out.shape(), &[2, 6, 64]);
        assert_eq!(pooled_out.shape(), &[2, 64]);

        // Autograd backward verification
        let loss = seq_out.sum_all().add(&pooled_out.sum_all()).unwrap();
        loss.backward();

        assert!(model.embeddings.weight.grad().is_some());
        assert!(model.encoder.layers[0]
            .attention
            .q_proj
            .weight
            .grad()
            .is_some());
        assert!(model.encoder.layers[0]
            .mlp
            .gate_proj
            .weight
            .grad()
            .is_some());
    }

    #[test]
    fn test_modern_bert_classification_head() {
        let config = ModernBertConfig::tiny();
        let num_classes = 3;
        let model = ModernBertForSequenceClassification::new(config, num_classes);

        let input_raw = RawTensor::from_slice(&[1.0, 5.0, 12.0, 18.0], &[1, 4]);
        let input_ids = Tensor::new(input_raw, false);

        let logits = model.forward_classification(&input_ids).unwrap();
        assert_eq!(logits.shape(), &[1, 3]);

        let loss = logits.sum_all();
        loss.backward();
        assert!(model.classifier.weight.grad().is_some());
    }

    #[test]
    fn test_modern_bert_embedding_l2_normalization() {
        let config = ModernBertConfig::tiny();
        let model = ModernBertForSequenceEmbedding::new(config);

        let input_raw = RawTensor::from_slice(&[1.0, 4.0, 9.0], &[1, 3]);
        let input_ids = Tensor::new(input_raw, false);

        let emb = model.forward_embedding(&input_ids).unwrap();
        assert_eq!(emb.shape(), &[1, 64]);

        // Check L2 norm is ~1.0
        let emb_data = emb.data();
        let slice = emb_data.as_slice();
        let norm_sq: f32 = slice.iter().map(|&x| x * x).sum();
        let norm = norm_sq.sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Expected L2 norm ~1.0, got {}",
            norm
        );
    }
}
