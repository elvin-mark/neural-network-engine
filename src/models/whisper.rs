//! OpenAI Whisper Encoder-Decoder Sequence-to-Sequence Architecture for Speech Recognition.
//!
//! Features:
//! - 1D Convolutional temporal downsampling of log-mel spectrograms
//! - Transformer Encoder with bidirectional multi-head self-attention
//! - Transformer Decoder with causal self-attention and encoder-decoder cross-attention
//! - Pre-LayerNorm residual connections and GELU feed-forward networks
//! - Autoregressive greedy speech transcription decoding

use crate::autograd::Tensor;
use crate::error::{EngineError, Result};
use crate::nn::attention::MultiHeadAttention;
use crate::nn::conv::Conv2d;
use crate::nn::embedding::Embedding;
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::nn::norm::LayerNorm;
use crate::tensor::RawTensor;
use crate::tokenizer::ByteLevelBPE;

/// Configuration hyperparameters for the Whisper Speech-to-Text Model.
#[derive(Debug, Clone)]
pub struct WhisperConfig {
    /// Number of acoustic Mel frequency bins in input spectrograms (e.g. 64 or 80).
    pub n_mels: usize,
    /// Hidden dimension of transformer token and audio representations.
    pub d_model: usize,
    /// Number of encoder transformer layers.
    pub encoder_layers: usize,
    /// Number of decoder transformer layers.
    pub decoder_layers: usize,
    /// Number of attention heads in encoder self-attention.
    pub encoder_heads: usize,
    /// Number of attention heads in decoder self- and cross-attention.
    pub decoder_heads: usize,
    /// Inner dimension of the Feed-Forward (FFN) blocks.
    pub d_ff: usize,
    /// Target text vocabulary size.
    pub vocab_size: usize,
    /// Maximum number of encoder audio frames after convolutional downsampling.
    pub max_source_positions: usize,
    /// Maximum number of target text tokens.
    pub max_target_positions: usize,
}

impl WhisperConfig {
    /// Pre-configured compact Whisper model for fast training and spoken word recognition.
    pub fn tiny() -> Self {
        Self {
            n_mels: 64,
            d_model: 64,
            encoder_layers: 2,
            decoder_layers: 2,
            encoder_heads: 4,
            decoder_heads: 4,
            d_ff: 160,
            vocab_size: 350,
            max_source_positions: 128,
            max_target_positions: 48,
        }
    }

    /// OpenAI Whisper Tiny official configuration (`openai/whisper-tiny`).
    pub fn whisper_tiny() -> Self {
        Self {
            n_mels: 80,
            d_model: 384,
            encoder_layers: 4,
            decoder_layers: 4,
            encoder_heads: 6,
            decoder_heads: 6,
            d_ff: 1536,
            vocab_size: 51865,
            max_source_positions: 1500,
            max_target_positions: 448,
        }
    }
}

/// A single Transformer Encoder Block for Whisper.
pub struct WhisperEncoderBlock {
    pub self_attn_ln: LayerNorm,
    pub self_attn: MultiHeadAttention,
    pub mlp_ln: LayerNorm,
    pub mlp_fc1: Linear,
    pub mlp_fc2: Linear,
}

impl WhisperEncoderBlock {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        Self {
            self_attn_ln: LayerNorm::new(d_model),
            self_attn: MultiHeadAttention::new(d_model, num_heads, false),
            mlp_ln: LayerNorm::new(d_model),
            mlp_fc1: Linear::new(d_model, d_ff),
            mlp_fc2: Linear::new(d_ff, d_model),
        }
    }

    pub fn forward_block(&self, x: &Tensor) -> Result<Tensor> {
        // 1. Bidirectional Self-Attention with Pre-LayerNorm & Residual
        let norm_x = self.self_attn_ln.forward(x)?;
        let h_attn = self.self_attn.forward_attention(&norm_x)?;
        let x = x.add(&h_attn)?;

        // 2. GELU MLP with Pre-LayerNorm & Residual
        let norm_x2 = self.mlp_ln.forward(&x)?;
        let h_mlp = self.mlp_fc1.forward(&norm_x2)?.gelu()?;
        let h_mlp = self.mlp_fc2.forward(&h_mlp)?;
        x.add(&h_mlp)
    }
}

impl Module for WhisperEncoderBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_block(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.self_attn_ln.parameters());
        params.extend(self.self_attn.parameters());
        params.extend(self.mlp_ln.parameters());
        params.extend(self.mlp_fc1.parameters());
        params.extend(self.mlp_fc2.parameters());
        params
    }
}

/// A single Transformer Decoder Block with Causal Self-Attention and Encoder Cross-Attention.
pub struct WhisperDecoderBlock {
    pub self_attn_ln: LayerNorm,
    pub self_attn: MultiHeadAttention,
    pub cross_attn_ln: LayerNorm,
    pub cross_attn: MultiHeadAttention,
    pub mlp_ln: LayerNorm,
    pub mlp_fc1: Linear,
    pub mlp_fc2: Linear,
}

impl WhisperDecoderBlock {
    pub fn new(d_model: usize, num_heads: usize, d_ff: usize) -> Self {
        Self {
            self_attn_ln: LayerNorm::new(d_model),
            self_attn: MultiHeadAttention::new(d_model, num_heads, true), // Causal
            cross_attn_ln: LayerNorm::new(d_model),
            cross_attn: MultiHeadAttention::new(d_model, num_heads, false), // Cross-attention
            mlp_ln: LayerNorm::new(d_model),
            mlp_fc1: Linear::new(d_model, d_ff),
            mlp_fc2: Linear::new(d_ff, d_model),
        }
    }

    pub fn forward_block(&self, x: &Tensor, memory: &Tensor) -> Result<Tensor> {
        // 1. Masked Causal Self-Attention with Pre-LayerNorm & Residual
        let norm_x = self.self_attn_ln.forward(x)?;
        let h_self = self.self_attn.forward_attention(&norm_x)?;
        let x = x.add(&h_self)?;

        // 2. Multi-Head Cross-Attention with Pre-LayerNorm & Residual
        let norm_x2 = self.cross_attn_ln.forward(&x)?;
        let h_cross = self.cross_attn.forward_cross_attention(&norm_x2, memory)?;
        let x = x.add(&h_cross)?;

        // 3. GELU MLP with Pre-LayerNorm & Residual
        let norm_x3 = self.mlp_ln.forward(&x)?;
        let h_mlp = self.mlp_fc1.forward(&norm_x3)?.gelu()?;
        let h_mlp = self.mlp_fc2.forward(&h_mlp)?;
        x.add(&h_mlp)
    }
}

impl Module for WhisperDecoderBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_block(input, input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.self_attn_ln.parameters());
        params.extend(self.self_attn.parameters());
        params.extend(self.cross_attn_ln.parameters());
        params.extend(self.cross_attn.parameters());
        params.extend(self.mlp_ln.parameters());
        params.extend(self.mlp_fc1.parameters());
        params.extend(self.mlp_fc2.parameters());
        params
    }
}

/// The Whisper Audio Transformer Encoder.
pub struct WhisperEncoder {
    pub conv1: Conv2d,
    pub conv2: Conv2d,
    pub pos_embed: Tensor,
    pub blocks: Vec<WhisperEncoderBlock>,
    pub ln_post: LayerNorm,
    pub config: WhisperConfig,
}

impl WhisperEncoder {
    pub fn new(config: &WhisperConfig) -> Self {
        // Conv1: in_channels=n_mels, out_channels=d_model, kernel=(1, 3), stride=(1, 1), padding=(0, 1)
        let conv1 = Conv2d::with_options(
            config.n_mels,
            config.d_model,
            (1, 3),
            (1, 1),
            (0, 1),
            (1, 1),
            true,
        );

        // Conv2: in_channels=d_model, out_channels=d_model, kernel=(1, 3), stride=(1, 2), padding=(0, 1)
        let conv2 = Conv2d::with_options(
            config.d_model,
            config.d_model,
            (1, 3),
            (1, 2),
            (0, 1),
            (1, 1),
            true,
        );

        let pos_data =
            RawTensor::randn(&[1, config.max_source_positions, config.d_model], 0.0, 0.02);
        let pos_embed = Tensor::new(pos_data, true);

        let mut blocks = Vec::with_capacity(config.encoder_layers);
        for _ in 0..config.encoder_layers {
            blocks.push(WhisperEncoderBlock::new(
                config.d_model,
                config.encoder_heads,
                config.d_ff,
            ));
        }

        Self {
            conv1,
            conv2,
            pos_embed,
            blocks,
            ln_post: LayerNorm::new(config.d_model),
            config: config.clone(),
        }
    }

    /// Encodes a batch of log-mel spectrograms [B, n_mels, T_audio] into encoder memory [B, T_enc, d_model].
    pub fn forward_encoder(&self, mel: &Tensor) -> Result<Tensor> {
        let shape = mel.shape();
        if shape.len() != 3 {
            return Err(EngineError::IncompatibleShapes {
                op: "WhisperEncoder (expected 3D mel input [B, n_mels, T])",
                shapes: vec![shape],
            });
        }

        let (b, n_mels, t) = (shape[0], shape[1], shape[2]);
        if n_mels != self.config.n_mels {
            return Err(EngineError::ShapeMismatch {
                expected: vec![b, self.config.n_mels, t],
                actual: shape,
            });
        }

        // 1. Reshape to 4D [B, n_mels, 1, T] for 2D convolution
        let x_4d = mel.reshape(&[b, n_mels, 1, t])?;

        // 2. Conv downsampling: conv1 + GELU + conv2 + GELU -> [B, d_model, 1, T_enc]
        let c1 = self.conv1.forward(&x_4d)?.gelu()?;
        let c2 = self.conv2.forward(&c1)?.gelu()?;
        let t_enc = c2.shape()[3];

        // 3. Reshape & transpose to [B, T_enc, d_model]
        let mut x = c2
            .reshape(&[b, self.config.d_model, t_enc])?
            .transpose(1, 2)?;

        // 4. Add positional embeddings (truncated to t_enc)
        let pos = self.pos_embed.slice(1, 0, t_enc)?;
        x = x.add(&pos)?;

        // 5. Cascade through Encoder Blocks
        for block in &self.blocks {
            x = block.forward_block(&x)?;
        }

        // 6. Post LayerNorm
        self.ln_post.forward(&x)
    }
}

impl Module for WhisperEncoder {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_encoder(input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.conv2.parameters());
        params.push(self.pos_embed.clone());
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.ln_post.parameters());
        params
    }
}

/// The Whisper Text Transformer Decoder.
pub struct WhisperDecoder {
    pub token_embedding: Embedding,
    pub pos_embed: Tensor,
    pub blocks: Vec<WhisperDecoderBlock>,
    pub ln_post: LayerNorm,
    pub lm_head: Linear,
    pub config: WhisperConfig,
}

impl WhisperDecoder {
    pub fn new(config: &WhisperConfig) -> Self {
        let token_embedding = Embedding::new(config.vocab_size, config.d_model);

        let pos_data =
            RawTensor::randn(&[1, config.max_target_positions, config.d_model], 0.0, 0.02);
        let pos_embed = Tensor::new(pos_data, true);

        let mut blocks = Vec::with_capacity(config.decoder_layers);
        for _ in 0..config.decoder_layers {
            blocks.push(WhisperDecoderBlock::new(
                config.d_model,
                config.decoder_heads,
                config.d_ff,
            ));
        }

        Self {
            token_embedding,
            pos_embed,
            blocks,
            ln_post: LayerNorm::new(config.d_model),
            lm_head: Linear::without_bias(config.d_model, config.vocab_size),
            config: config.clone(),
        }
    }

    /// Decodes target token sequence [B, T_text] given encoder memory [B, T_enc, d_model].
    pub fn forward_decoder(&self, tokens: &Tensor, memory: &Tensor) -> Result<Tensor> {
        let shape = tokens.shape();
        if shape.len() != 2 {
            return Err(EngineError::IncompatibleShapes {
                op: "WhisperDecoder (expected 2D tokens input [B, T])",
                shapes: vec![shape],
            });
        }

        let (_b, t_text) = (shape[0], shape[1]);

        // 1. Token Embeddings -> [B, T_text, d_model]
        let tok_embeds = self.token_embedding.forward(tokens)?;

        // 2. Positional Embeddings
        let pos = self.pos_embed.slice(1, 0, t_text)?;
        let mut x = tok_embeds.add(&pos)?;

        // 3. Cascade through Decoder Blocks with Cross-Attention
        for block in &self.blocks {
            x = block.forward_block(&x, memory)?;
        }

        // 4. Post LayerNorm & Projection Head -> [B, T_text, vocab_size]
        let x = self.ln_post.forward(&x)?;
        self.lm_head.forward(&x)
    }
}

impl Module for WhisperDecoder {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_decoder(input, input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.token_embedding.parameters());
        params.push(self.pos_embed.clone());
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.ln_post.parameters());
        params.extend(self.lm_head.parameters());
        params
    }
}

/// The complete Whisper Encoder-Decoder Model for Automatic Speech Recognition.
pub struct Whisper {
    pub encoder: WhisperEncoder,
    pub decoder: WhisperDecoder,
    pub config: WhisperConfig,
}

impl Whisper {
    pub fn new(config: WhisperConfig) -> Self {
        let encoder = WhisperEncoder::new(&config);
        let decoder = WhisperDecoder::new(&config);
        Self {
            encoder,
            decoder,
            config,
        }
    }

    /// Forward pass: encodes audio spectrogram and decodes text tokens to output vocabulary logits.
    pub fn forward_model(&self, mel: &Tensor, tokens: &Tensor) -> Result<Tensor> {
        let memory = self.encoder.forward_encoder(mel)?;
        self.decoder.forward_decoder(tokens, &memory)
    }

    /// Encodes acoustic spectrogram into encoder memory representations.
    pub fn encode(&self, mel: &Tensor) -> Result<Tensor> {
        self.encoder.forward_encoder(mel)
    }

    /// Decodes text tokens using pre-computed encoder memory representations.
    pub fn decode(&self, tokens: &Tensor, memory: &Tensor) -> Result<Tensor> {
        self.decoder.forward_decoder(tokens, memory)
    }

    /// Autoregressively transcribes an audio spectrogram into text using greedy decoding.
    pub fn generate_transcription(
        &self,
        mel: &Tensor,
        tokenizer: &ByteLevelBPE,
        max_tokens: usize,
    ) -> Result<String> {
        let memory = self.encode(mel)?;

        let bos_id = tokenizer.bos_token_id().unwrap_or(1);
        let eos_id = tokenizer.eos_token_id().unwrap_or(2);

        let mut token_ids = vec![bos_id];

        for _ in 0..max_tokens {
            let cur_len = token_ids.len();
            let tokens_raw = RawTensor::from_vec(
                token_ids.iter().map(|&t| t as f32).collect(),
                vec![1, cur_len],
            );
            let tokens_tensor = Tensor::new(tokens_raw, false);

            let logits = self.decode(&tokens_tensor, &memory)?;
            let slice = logits.data().to_contiguous();
            let num_classes = self.config.vocab_size;
            let last_logits = &slice.as_slice()[(cur_len - 1) * num_classes..cur_len * num_classes];

            // Greedy argmax selection
            let mut best_token = 0;
            let mut best_val = f32::NEG_INFINITY;
            for (idx, &v) in last_logits.iter().enumerate() {
                if v > best_val {
                    best_val = v;
                    best_token = idx;
                }
            }

            if best_token == eos_id {
                break;
            }

            token_ids.push(best_token);
        }

        // Decode generated tokens (skip initial BOS token)
        let generated_slice = if token_ids.len() > 1 {
            &token_ids[1..]
        } else {
            &token_ids[..]
        };

        tokenizer.decode(generated_slice)
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

        // Encoder
        set_tensor(&mut self.encoder.conv1.weight, "encoder.conv1.weight")?;
        set_opt_tensor(&mut self.encoder.conv1.bias, "encoder.conv1.bias")?;
        set_tensor(&mut self.encoder.conv2.weight, "encoder.conv2.weight")?;
        set_opt_tensor(&mut self.encoder.conv2.bias, "encoder.conv2.bias")?;
        set_tensor(
            &mut self.encoder.pos_embed,
            "encoder.embed_positions.weight",
        )?;
        set_tensor(
            &mut self.encoder.ln_post.weight,
            "encoder.layer_norm.weight",
        )?;
        set_tensor(&mut self.encoder.ln_post.bias, "encoder.layer_norm.bias")?;

        for (i, block) in self.encoder.blocks.iter_mut().enumerate() {
            set_tensor(
                &mut block.self_attn.q_proj.weight,
                &format!("encoder.layers.{}.self_attn.q_proj.weight", i),
            )?;
            set_opt_tensor(
                &mut block.self_attn.q_proj.bias,
                &format!("encoder.layers.{}.self_attn.q_proj.bias", i),
            )?;
            set_tensor(
                &mut block.self_attn.k_proj.weight,
                &format!("encoder.layers.{}.self_attn.k_proj.weight", i),
            )?;
            if weights.contains_key(&format!("encoder.layers.{}.self_attn.k_proj.bias", i)) {
                set_opt_tensor(
                    &mut block.self_attn.k_proj.bias,
                    &format!("encoder.layers.{}.self_attn.k_proj.bias", i),
                )?;
            } else {
                block.self_attn.k_proj.bias = None;
            }
            set_tensor(
                &mut block.self_attn.v_proj.weight,
                &format!("encoder.layers.{}.self_attn.v_proj.weight", i),
            )?;
            set_opt_tensor(
                &mut block.self_attn.v_proj.bias,
                &format!("encoder.layers.{}.self_attn.v_proj.bias", i),
            )?;
            set_tensor(
                &mut block.self_attn.out_proj.weight,
                &format!("encoder.layers.{}.self_attn.out_proj.weight", i),
            )?;
            set_opt_tensor(
                &mut block.self_attn.out_proj.bias,
                &format!("encoder.layers.{}.self_attn.out_proj.bias", i),
            )?;

            set_tensor(
                &mut block.self_attn_ln.weight,
                &format!("encoder.layers.{}.self_attn_layer_norm.weight", i),
            )?;
            set_tensor(
                &mut block.self_attn_ln.bias,
                &format!("encoder.layers.{}.self_attn_layer_norm.bias", i),
            )?;

            set_tensor(
                &mut block.mlp_fc1.weight,
                &format!("encoder.layers.{}.fc1.weight", i),
            )?;
            set_opt_tensor(
                &mut block.mlp_fc1.bias,
                &format!("encoder.layers.{}.fc1.bias", i),
            )?;
            set_tensor(
                &mut block.mlp_fc2.weight,
                &format!("encoder.layers.{}.fc2.weight", i),
            )?;
            set_opt_tensor(
                &mut block.mlp_fc2.bias,
                &format!("encoder.layers.{}.fc2.bias", i),
            )?;

            set_tensor(
                &mut block.mlp_ln.weight,
                &format!("encoder.layers.{}.final_layer_norm.weight", i),
            )?;
            set_tensor(
                &mut block.mlp_ln.bias,
                &format!("encoder.layers.{}.final_layer_norm.bias", i),
            )?;
        }

        // Decoder
        set_tensor(
            &mut self.decoder.token_embedding.weight,
            "decoder.embed_tokens.weight",
        )?;
        set_tensor(
            &mut self.decoder.pos_embed,
            "decoder.embed_positions.weight",
        )?;
        set_tensor(
            &mut self.decoder.ln_post.weight,
            "decoder.layer_norm.weight",
        )?;
        set_tensor(&mut self.decoder.ln_post.bias, "decoder.layer_norm.bias")?;
        set_tensor(&mut self.decoder.lm_head.weight, "decoder.lm_head.weight")?;
        if weights.contains_key("decoder.lm_head.bias") {
            set_opt_tensor(&mut self.decoder.lm_head.bias, "decoder.lm_head.bias")?;
        }

        for (i, block) in self.decoder.blocks.iter_mut().enumerate() {
            set_tensor(
                &mut block.self_attn.q_proj.weight,
                &format!("decoder.layers.{}.self_attn.q_proj.weight", i),
            )?;
            set_opt_tensor(
                &mut block.self_attn.q_proj.bias,
                &format!("decoder.layers.{}.self_attn.q_proj.bias", i),
            )?;
            set_tensor(
                &mut block.self_attn.k_proj.weight,
                &format!("decoder.layers.{}.self_attn.k_proj.weight", i),
            )?;
            if weights.contains_key(&format!("decoder.layers.{}.self_attn.k_proj.bias", i)) {
                set_opt_tensor(
                    &mut block.self_attn.k_proj.bias,
                    &format!("decoder.layers.{}.self_attn.k_proj.bias", i),
                )?;
            } else {
                block.self_attn.k_proj.bias = None;
            }
            set_tensor(
                &mut block.self_attn.v_proj.weight,
                &format!("decoder.layers.{}.self_attn.v_proj.weight", i),
            )?;
            set_opt_tensor(
                &mut block.self_attn.v_proj.bias,
                &format!("decoder.layers.{}.self_attn.v_proj.bias", i),
            )?;
            set_tensor(
                &mut block.self_attn.out_proj.weight,
                &format!("decoder.layers.{}.self_attn.out_proj.weight", i),
            )?;
            set_opt_tensor(
                &mut block.self_attn.out_proj.bias,
                &format!("decoder.layers.{}.self_attn.out_proj.bias", i),
            )?;

            set_tensor(
                &mut block.self_attn_ln.weight,
                &format!("decoder.layers.{}.self_attn_layer_norm.weight", i),
            )?;
            set_tensor(
                &mut block.self_attn_ln.bias,
                &format!("decoder.layers.{}.self_attn_layer_norm.bias", i),
            )?;

            set_tensor(
                &mut block.cross_attn.q_proj.weight,
                &format!("decoder.layers.{}.encoder_attn.q_proj.weight", i),
            )?;
            set_opt_tensor(
                &mut block.cross_attn.q_proj.bias,
                &format!("decoder.layers.{}.encoder_attn.q_proj.bias", i),
            )?;
            set_tensor(
                &mut block.cross_attn.k_proj.weight,
                &format!("decoder.layers.{}.encoder_attn.k_proj.weight", i),
            )?;
            if weights.contains_key(&format!("decoder.layers.{}.encoder_attn.k_proj.bias", i)) {
                set_opt_tensor(
                    &mut block.cross_attn.k_proj.bias,
                    &format!("decoder.layers.{}.encoder_attn.k_proj.bias", i),
                )?;
            } else {
                block.cross_attn.k_proj.bias = None;
            }
            set_tensor(
                &mut block.cross_attn.v_proj.weight,
                &format!("decoder.layers.{}.encoder_attn.v_proj.weight", i),
            )?;
            set_opt_tensor(
                &mut block.cross_attn.v_proj.bias,
                &format!("decoder.layers.{}.encoder_attn.v_proj.bias", i),
            )?;
            set_tensor(
                &mut block.cross_attn.out_proj.weight,
                &format!("decoder.layers.{}.encoder_attn.out_proj.weight", i),
            )?;
            set_opt_tensor(
                &mut block.cross_attn.out_proj.bias,
                &format!("decoder.layers.{}.encoder_attn.out_proj.bias", i),
            )?;

            set_tensor(
                &mut block.cross_attn_ln.weight,
                &format!("decoder.layers.{}.encoder_attn_layer_norm.weight", i),
            )?;
            set_tensor(
                &mut block.cross_attn_ln.bias,
                &format!("decoder.layers.{}.encoder_attn_layer_norm.bias", i),
            )?;

            set_tensor(
                &mut block.mlp_fc1.weight,
                &format!("decoder.layers.{}.fc1.weight", i),
            )?;
            set_opt_tensor(
                &mut block.mlp_fc1.bias,
                &format!("decoder.layers.{}.fc1.bias", i),
            )?;
            set_tensor(
                &mut block.mlp_fc2.weight,
                &format!("decoder.layers.{}.fc2.weight", i),
            )?;
            set_opt_tensor(
                &mut block.mlp_fc2.bias,
                &format!("decoder.layers.{}.fc2.bias", i),
            )?;

            set_tensor(
                &mut block.mlp_ln.weight,
                &format!("decoder.layers.{}.final_layer_norm.weight", i),
            )?;
            set_tensor(
                &mut block.mlp_ln.bias,
                &format!("decoder.layers.{}.final_layer_norm.bias", i),
            )?;
        }

        Ok(())
    }

    /// Loads model weights directly from a SafeTensors file.
    pub fn load_safetensors<P: AsRef<std::path::Path>>(&mut self, path: P) -> Result<()> {
        let weights = crate::io::load_safetensors(path)?;
        self.load_weights(&weights)
    }
}

impl Module for Whisper {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_model(input, input)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.encoder.parameters());
        params.extend(self.decoder.parameters());
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_whisper_forward_and_backward() {
        let config = WhisperConfig {
            n_mels: 32,
            d_model: 32,
            encoder_layers: 1,
            decoder_layers: 1,
            encoder_heads: 2,
            decoder_heads: 2,
            d_ff: 64,
            vocab_size: 100,
            max_source_positions: 64,
            max_target_positions: 32,
        };

        let whisper = Whisper::new(config);

        // Input mel: [Batch=2, n_mels=32, T_audio=20]
        let mel = Tensor::randn(&[2, 32, 20], 0.0, 1.0, true);
        // Input text tokens: [Batch=2, T_text=5]
        let tokens_raw = RawTensor::from_slice(
            &[1.0, 10.0, 12.0, 15.0, 2.0, 1.0, 5.0, 8.0, 9.0, 2.0],
            &[2, 5],
        );
        let tokens = Tensor::new(tokens_raw, false);

        let logits = whisper.forward_model(&mel, &tokens).unwrap();
        assert_eq!(logits.shape(), &[2, 5, 100]);

        // Autograd backward verification
        let loss = logits.sum_all();
        loss.backward();

        assert!(mel.grad().is_some());
        assert_eq!(mel.grad().unwrap().shape(), &[2, 32, 20]);
        assert!(whisper.encoder.conv1.weight.grad().is_some());
        assert!(whisper.encoder.conv2.weight.grad().is_some());
        assert!(whisper.decoder.lm_head.weight.grad().is_some());
    }
}
