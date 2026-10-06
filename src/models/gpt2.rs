//! OpenAI GPT-2 architecture (Radford et al., 2019).
//!
//! Features:
//! - Pre-LayerNorm autoregressive Transformer decoder blocks
//! - Learned token and 1D absolute positional embeddings (wte + wpe)
//! - Multi-Head Causal Self-Attention with causal upper-triangular masking
//! - GELU feed-forward network with 4x hidden expansion
//! - Key-Value Cache (`KVCache`) support for $O(1)$ fast autoregressive token generation
//! - Predefined configurations: GPT-2 Small (124M), Medium (355M), Large (774M), XL (1.5B), and Tiny

use crate::autograd::Tensor;
use crate::error::{EngineError, Result};
use crate::nn::activations::GELU;
use crate::nn::attention::MultiHeadAttention;
use crate::nn::embedding::Embedding;
use crate::nn::kv_cache::KVCache;
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::nn::norm::LayerNorm;

/// Configuration hyperparameters for GPT-2 models.
#[derive(Debug, Clone)]
pub struct GPT2Config {
    /// Size of the vocabulary (50,257 for OpenAI GPT-2).
    pub vocab_size: usize,
    /// Maximum context sequence length (1,024 for original GPT-2).
    pub max_position_embeddings: usize,
    /// Model representation embedding dimension ($d_{model}$, e.g. 768 for GPT-2 small).
    pub d_model: usize,
    /// Number of attention heads (e.g. 12 for GPT-2 small).
    pub num_heads: usize,
    /// Number of transformer decoder layers (e.g. 12 for GPT-2 small).
    pub num_layers: usize,
    /// Inner dimension of the feed-forward network ($4 \times d_{model}$).
    pub d_ff: usize,
    /// Epsilon value for LayerNorm.
    pub layer_norm_eps: f32,
}

impl GPT2Config {
    /// Pre-configured compact GPT-2 model for fast local testing and unit tests.
    pub fn tiny() -> Self {
        Self {
            vocab_size: 256,
            max_position_embeddings: 128,
            d_model: 64,
            num_heads: 4,
            num_layers: 2,
            d_ff: 256,
            layer_norm_eps: 1e-5,
        }
    }

    /// OpenAI GPT-2 Small configuration (124M parameters: 12 layers, 768 dim, 12 heads).
    pub fn gpt2_small(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            max_position_embeddings: 1024,
            d_model: 768,
            num_heads: 12,
            num_layers: 12,
            d_ff: 768 * 4,
            layer_norm_eps: 1e-5,
        }
    }

    /// OpenAI GPT-2 Medium configuration (355M parameters: 24 layers, 1024 dim, 16 heads).
    pub fn gpt2_medium(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            max_position_embeddings: 1024,
            d_model: 1024,
            num_heads: 16,
            num_layers: 24,
            d_ff: 1024 * 4,
            layer_norm_eps: 1e-5,
        }
    }

    /// OpenAI GPT-2 Large configuration (774M parameters: 36 layers, 1280 dim, 20 heads).
    pub fn gpt2_large(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            max_position_embeddings: 1024,
            d_model: 1280,
            num_heads: 20,
            num_layers: 36,
            d_ff: 1280 * 4,
            layer_norm_eps: 1e-5,
        }
    }

    /// OpenAI GPT-2 XL configuration (1.5B parameters: 48 layers, 1600 dim, 25 heads).
    pub fn gpt2_xl(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            max_position_embeddings: 1024,
            d_model: 1600,
            num_heads: 25,
            num_layers: 48,
            d_ff: 1600 * 4,
            layer_norm_eps: 1e-5,
        }
    }
}

/// A single GPT-2 Transformer Decoder Block with Pre-LayerNorm causal self-attention and MLP.
///
/// $$x = x + \text{Attn}(\text{LN}_1(x))$$
/// $$x = x + \text{MLP}(\text{LN}_2(x))$$
#[derive(Clone)]
pub struct GPT2Block {
    pub ln_1: LayerNorm,
    pub attn: MultiHeadAttention,
    pub ln_2: LayerNorm,
    pub mlp_fc: Linear,
    pub mlp_gelu: GELU,
    pub mlp_proj: Linear,
}

impl GPT2Block {
    pub fn new(config: &GPT2Config) -> Self {
        Self {
            ln_1: LayerNorm::with_eps(config.d_model, config.layer_norm_eps),
            attn: MultiHeadAttention::new(config.d_model, config.num_heads, true),
            ln_2: LayerNorm::with_eps(config.d_model, config.layer_norm_eps),
            mlp_fc: Linear::new(config.d_model, config.d_ff),
            mlp_gelu: GELU,
            mlp_proj: Linear::new(config.d_ff, config.d_model),
        }
    }

    /// Forward pass with optional Key-Value caching for fast incremental token generation.
    pub fn forward_block_cached(
        &self,
        x: &Tensor,
        cache: Option<&mut (Tensor, Tensor)>,
    ) -> Result<Tensor> {
        // 1. Pre-LayerNorm Attention with residual connection
        let norm_1 = self.ln_1.forward(x)?;
        let attn_out = self.attn.forward_attention_cached(&norm_1, cache)?;
        let x = x.add(&attn_out)?;

        // 2. Pre-LayerNorm Feed-Forward MLP with residual connection
        let norm_2 = self.ln_2.forward(&x)?;
        let h = self.mlp_fc.forward(&norm_2)?;
        let act = self.mlp_gelu.forward(&h)?;
        let ffn_out = self.mlp_proj.forward(&act)?;
        x.add(&ffn_out)
    }

    pub fn forward_block(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_block_cached(x, None)
    }
}

impl Module for GPT2Block {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_block(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.ln_1.parameters());
        params.extend(self.attn.parameters());
        params.extend(self.ln_2.parameters());
        params.extend(self.mlp_fc.parameters());
        params.extend(self.mlp_proj.parameters());
        params
    }
}

/// Full GPT-2 Language Model Architecture for causal autoregressive text generation.
#[derive(Clone)]
pub struct GPT2Model {
    /// Token embeddings table (`wte`).
    pub wte: Embedding,
    /// Position embeddings table (`wpe`).
    pub wpe: Embedding,
    /// Cascade of GPT-2 Transformer decoder blocks.
    pub blocks: Vec<GPT2Block>,
    /// Final LayerNorm before language modeling head (`ln_f`).
    pub ln_f: LayerNorm,
    /// Unembedding Language Model projection head (`lm_head`).
    pub lm_head: Linear,
    /// Model hyperparameters.
    pub config: GPT2Config,
}

impl GPT2Model {
    pub fn new(config: GPT2Config) -> Self {
        let mut blocks = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            blocks.push(GPT2Block::new(&config));
        }

        Self {
            wte: Embedding::new(config.vocab_size, config.d_model),
            wpe: Embedding::new(config.max_position_embeddings, config.d_model),
            blocks,
            ln_f: LayerNorm::with_eps(config.d_model, config.layer_norm_eps),
            lm_head: Linear::new(config.d_model, config.vocab_size),
            config,
        }
    }

    /// Forward pass through GPT-2 returning next-token logits `[B, T, vocab_size]`.
    ///
    /// Accepts 1D or 2D token IDs tensor of shape `[B, T]`.
    pub fn forward_tokens(
        &self,
        token_indices: &Tensor,
        start_pos: usize,
        mut kv_cache: Option<&mut KVCache>,
    ) -> Result<Tensor> {
        let shape = token_indices.shape();
        let (batch_size, seq_len) = if shape.len() == 2 {
            (shape[0], shape[1])
        } else if shape.len() == 1 {
            (1, shape[0])
        } else {
            return Err(EngineError::InvalidArgument(format!(
                "GPT2Model expects 1D [T] or 2D [B, T] token indices, got shape {:?}",
                shape
            )));
        };

        if start_pos + seq_len > self.config.max_position_embeddings {
            return Err(EngineError::InvalidArgument(format!(
                "Sequence end pos {} exceeds max_position_embeddings {}",
                start_pos + seq_len,
                self.config.max_position_embeddings
            )));
        }

        // 1. Token Embeddings: [B, T, d_model]
        let tok_emb = self.wte.forward(token_indices)?;
        let tok_emb = tok_emb.reshape(&[batch_size, seq_len, self.config.d_model])?;

        // 2. Position Embeddings: [1, T, d_model]
        let pos_indices: Vec<usize> = (start_pos..start_pos + seq_len).collect();
        let pos_emb = self.wpe.forward_indices(&pos_indices)?;
        let pos_emb = pos_emb.reshape(&[1, seq_len, self.config.d_model])?;

        // 3. Combined Embeddings
        let mut x = tok_emb.add(&pos_emb)?;

        // 4. Cascade through GPT-2 Blocks with optional KV caching
        for (i, block) in self.blocks.iter().enumerate() {
            let layer_slot = match kv_cache.as_deref_mut() {
                Some(c) => {
                    if let Some(slot) = c.layers.get_mut(i) {
                        if slot.is_none() {
                            *slot = Some((Tensor::zeros(&[0], false), Tensor::zeros(&[0], false)));
                        }
                        slot.as_mut()
                    } else {
                        None
                    }
                }
                None => None,
            };
            x = block.forward_block_cached(&x, layer_slot)?;
        }

        // 5. Final LayerNorm & Unembedding LM Head -> [B, T, vocab_size]
        let x = self.ln_f.forward(&x)?;
        self.lm_head.forward(&x)
    }

    /// Autoregressive greedy text generation loop.
    pub fn generate(
        &self,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        eos_token_id: Option<usize>,
    ) -> Result<Vec<usize>> {
        let mut tokens = prompt_tokens.to_vec();
        let mut cache = KVCache::new(self.config.num_layers);
        for slot in &mut cache.layers {
            *slot = Some((Tensor::zeros(&[0], false), Tensor::zeros(&[0], false)));
        }

        // Prefill pass: process full prompt and fill KV cache
        let prompt_tensor = Tensor::from_slice(
            &prompt_tokens.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            &[1, prompt_tokens.len()],
            false,
        );
        let logits = self.forward_tokens(&prompt_tensor, 0, Some(&mut cache))?;

        // Sample next token from last position
        let last_step_logits = logits.slice(1, prompt_tokens.len() - 1, prompt_tokens.len())?;
        let last_slice = last_step_logits.data().to_contiguous();
        let vocab_slice = last_slice.as_slice();

        let mut next_token = 0;
        let mut max_val = f32::NEG_INFINITY;
        for (idx, &val) in vocab_slice.iter().enumerate() {
            if val > max_val {
                max_val = val;
                next_token = idx;
            }
        }
        tokens.push(next_token);

        if let Some(eos) = eos_token_id {
            if next_token == eos {
                return Ok(tokens);
            }
        }

        // Incremental generation: process 1 token at a time with O(1) step cost
        for _ in 1..max_new_tokens {
            let curr_pos = tokens.len() - 1;
            let single_token_tensor = Tensor::scalar(next_token as f32, false).reshape(&[1, 1])?;
            let logits = self.forward_tokens(&single_token_tensor, curr_pos, Some(&mut cache))?;

            let single_slice = logits.data().to_contiguous();
            let slice = single_slice.as_slice();

            next_token = 0;
            max_val = f32::NEG_INFINITY;
            for (idx, &val) in slice.iter().enumerate() {
                if val > max_val {
                    max_val = val;
                    next_token = idx;
                }
            }
            tokens.push(next_token);

            if let Some(eos) = eos_token_id {
                if next_token == eos {
                    break;
                }
            }
        }

        Ok(tokens)
    }

    /// Autoregressive text generation with streaming callback and advanced sampling
    /// (temperature, top-k, top-p nucleus sampling).
    #[allow(clippy::too_many_arguments)]
    pub fn generate_stream<R: rand::Rng, F: FnMut(usize, &str) -> Result<bool>>(
        &self,
        tokenizer: &crate::tokenizer::HfTokenizer,
        prompt_tokens: &[usize],
        max_new_tokens: usize,
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
        eos_token_id: Option<usize>,
        rng: &mut R,
        mut on_token: F,
    ) -> Result<Vec<usize>> {
        if prompt_tokens.is_empty() {
            return Err(EngineError::InvalidArgument(
                "prompt_tokens cannot be empty".to_string(),
            ));
        }
        let mut tokens = prompt_tokens.to_vec();
        if max_new_tokens == 0 {
            return Ok(tokens);
        }

        let mut cache = KVCache::new(self.config.num_layers);
        for slot in &mut cache.layers {
            *slot = Some((Tensor::zeros(&[0], false), Tensor::zeros(&[0], false)));
        }

        // Prefill pass: process full prompt and fill KV cache
        let prompt_tensor = Tensor::from_slice(
            &prompt_tokens.iter().map(|&x| x as f32).collect::<Vec<_>>(),
            &[1, prompt_tokens.len()],
            false,
        );
        let logits = self.forward_tokens(&prompt_tensor, 0, Some(&mut cache))?;

        let last_step_logits = logits.slice(1, prompt_tokens.len() - 1, prompt_tokens.len())?;
        let last_slice = last_step_logits.data().to_contiguous();
        let mut next_token =
            crate::utils::sample_token(last_slice.as_slice(), temperature, top_k, top_p, rng);
        tokens.push(next_token);

        let piece = tokenizer.decode_token(next_token);
        let should_continue = on_token(next_token, &piece)?;
        if !should_continue {
            return Ok(tokens);
        }
        if let Some(eos) = eos_token_id {
            if next_token == eos {
                return Ok(tokens);
            }
        }

        // Incremental generation: process 1 token at a time with O(1) step cost
        for _ in 1..max_new_tokens {
            let curr_pos = tokens.len() - 1;
            if curr_pos >= self.config.max_position_embeddings {
                break;
            }
            let single_token_tensor = Tensor::scalar(next_token as f32, false).reshape(&[1, 1])?;
            let logits = self.forward_tokens(&single_token_tensor, curr_pos, Some(&mut cache))?;

            let single_slice = logits.data().to_contiguous();
            next_token =
                crate::utils::sample_token(single_slice.as_slice(), temperature, top_k, top_p, rng);
            tokens.push(next_token);

            let piece = tokenizer.decode_token(next_token);
            let should_continue = on_token(next_token, &piece)?;
            if !should_continue {
                break;
            }
            if let Some(eos) = eos_token_id {
                if next_token == eos {
                    break;
                }
            }
        }

        Ok(tokens)
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

        set_tensor(&mut self.wte.weight, "wte.weight")?;
        set_tensor(&mut self.wpe.weight, "wpe.weight")?;

        for (i, block) in self.blocks.iter_mut().enumerate() {
            set_tensor(&mut block.ln_1.weight, &format!("blocks.{}.ln_1.weight", i))?;
            set_tensor(&mut block.ln_1.bias, &format!("blocks.{}.ln_1.bias", i))?;

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

            set_tensor(&mut block.ln_2.weight, &format!("blocks.{}.ln_2.weight", i))?;
            set_tensor(&mut block.ln_2.bias, &format!("blocks.{}.ln_2.bias", i))?;

            set_tensor(
                &mut block.mlp_fc.weight,
                &format!("blocks.{}.mlp_fc.weight", i),
            )?;
            set_opt_tensor(&mut block.mlp_fc.bias, &format!("blocks.{}.mlp_fc.bias", i))?;

            set_tensor(
                &mut block.mlp_proj.weight,
                &format!("blocks.{}.mlp_proj.weight", i),
            )?;
            set_opt_tensor(
                &mut block.mlp_proj.bias,
                &format!("blocks.{}.mlp_proj.bias", i),
            )?;
        }

        set_tensor(&mut self.ln_f.weight, "ln_f.weight")?;
        set_tensor(&mut self.ln_f.bias, "ln_f.bias")?;

        set_tensor(&mut self.lm_head.weight, "lm_head.weight")?;
        if weights.contains_key("lm_head.bias") {
            set_opt_tensor(&mut self.lm_head.bias, "lm_head.bias")?;
        }

        Ok(())
    }

    /// Loads model weights directly from a SafeTensors file.
    pub fn load_safetensors<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<()> {
        let weights = crate::io::load_safetensors(path)?;
        self.load_weights(&weights)
    }
}

impl Module for GPT2Model {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_tokens(input, 0, None)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.wte.parameters());
        params.extend(self.wpe.parameters());
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.ln_f.parameters());
        params.extend(self.lm_head.parameters());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::RawTensor;

    #[test]
    fn test_gpt2_forward_and_backward() {
        let config = GPT2Config::tiny();
        let model = GPT2Model::new(config);

        let input_raw = RawTensor::from_slice(
            &[
                10.0, 20.0, 30.0, 40.0, // Batch 0
                5.0, 15.0, 25.0, 35.0, // Batch 1
            ],
            &[2, 4],
        );
        let input_ids = Tensor::new(input_raw, false);

        let logits = model.forward_tokens(&input_ids, 0, None).unwrap();
        assert_eq!(logits.shape(), &[2, 4, 256]);

        let loss = logits.sum_all();
        loss.backward();

        assert!(model.wte.weight.grad().is_some());
        assert!(model.wpe.weight.grad().is_some());
        assert!(model.blocks[0].attn.q_proj.weight.grad().is_some());
        assert!(model.blocks[0].mlp_fc.weight.grad().is_some());
        assert!(model.lm_head.weight.grad().is_some());
    }

    #[test]
    fn test_gpt2_generate_autoregressive() {
        let config = GPT2Config::tiny();
        let model = GPT2Model::new(config);

        let prompt = vec![1, 2, 3];
        let generated = model.generate(&prompt, 3, None).unwrap();
        assert_eq!(generated.len(), 6);
        assert_eq!(&generated[..3], &prompt[..]);
    }
}
