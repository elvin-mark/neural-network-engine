//! Complete LLaMA 2 / LLaMA 3 architecture featuring Grouped-Query Attention (GQA),
//! Rotary Position Embeddings (RoPE), RMSNorm, and SwiGLU Feed-Forward Networks.

use crate::autograd::Tensor;
use crate::error::{EngineError, Result};
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::nn::norm::RMSNorm;
use crate::tensor::RawTensor;

/// Configuration hyperparameters for LLaMA 2 / 3 models.
#[derive(Clone, Debug)]
pub struct LlamaConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub hidden_dim: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub norm_eps: f32,
    pub rope_theta: f32,
}

impl LlamaConfig {
    /// Creates a small / mini LLaMA 2 configuration suitable for quick training and local testing.
    pub fn mini(vocab_size: usize, max_seq_len: usize) -> Self {
        let d_model = 64usize;
        let num_heads = 4;
        let num_kv_heads = 2; // GQA with 2 KV heads sharing across 4 query heads (G=2)
        let raw_hidden: usize = d_model * 8 / 3;
        let hidden_dim = raw_hidden.div_ceil(32) * 32; // ~160, multiple of 32

        Self {
            vocab_size,
            d_model,
            hidden_dim,
            num_heads,
            num_kv_heads,
            num_layers: 2,
            max_seq_len,
            norm_eps: 1e-6,
            rope_theta: 10000.0,
        }
    }
}

/// Rotary Position Embedding (RoPE) helper table.
#[derive(Clone)]
pub struct RotaryEmbedding {
    pub cos_cached: RawTensor, // [1, 1, max_seq_len, head_dim]
    pub sin_cached: RawTensor, // [1, 1, max_seq_len, head_dim]
    pub head_dim: usize,
    pub max_seq_len: usize,
}

impl RotaryEmbedding {
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f32) -> Self {
        assert_eq!(head_dim % 2, 0, "RoPE head_dim ({}) must be even", head_dim);
        let half_dim = head_dim / 2;

        let mut inv_freq = Vec::with_capacity(half_dim);
        for i in 0..half_dim {
            let exponent = (2 * i) as f32 / head_dim as f32;
            inv_freq.push(1.0 / theta.powf(exponent));
        }

        let mut cos_data = vec![0.0; max_seq_len * head_dim];
        let mut sin_data = vec![0.0; max_seq_len * head_dim];

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = pos as f32 * inv_freq[i];
                let c = freq.cos();
                let s = freq.sin();

                // Interleaved / split layout matching standard LLaMA [x1, x2]
                cos_data[pos * head_dim + i] = c;
                cos_data[pos * head_dim + i + half_dim] = c;

                sin_data[pos * head_dim + i] = s;
                sin_data[pos * head_dim + i + half_dim] = s;
            }
        }

        Self {
            cos_cached: RawTensor::from_vec(cos_data, vec![1, 1, max_seq_len, head_dim]),
            sin_cached: RawTensor::from_vec(sin_data, vec![1, 1, max_seq_len, head_dim]),
            head_dim,
            max_seq_len,
        }
    }

    /// Applies RoPE rotation to 4D tensor `x` of shape `[B, H, T, D]`.
    pub fn apply(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        let shape = x.shape();
        let (_b, _h, t, d) = (shape[0], shape[1], shape[2], shape[3]);
        assert_eq!(d, self.head_dim);
        assert!(start_pos + t <= self.max_seq_len);

        let half_d = d / 2;

        // 1. Slice cos and sin for current positions -> [1, 1, T, D]
        let cos_slice = self.cos_cached.slice(2, start_pos, start_pos + t)?;
        let sin_slice = self.sin_cached.slice(2, start_pos, start_pos + t)?;

        let cos = Tensor::new(cos_slice, false);
        let sin = Tensor::new(sin_slice, false);

        // 2. rotate_half(x): [-x2, x1] where x = [x1, x2]
        let x1 = x.slice(3, 0, half_d)?;
        let x2 = x.slice(3, half_d, d)?;
        let neg_x2 = x2.neg();
        let rotated_half = Tensor::cat(&[&neg_x2, &x1], 3)?;

        // 3. x * cos + rotate_half(x) * sin
        let term1 = x.mul(&cos)?;
        let term2 = rotated_half.mul(&sin)?;
        term1.add(&term2)
    }
}

/// Grouped-Query Attention (GQA) with RoPE and Causal Masking.
pub struct GroupedQueryAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub rope: RotaryEmbedding,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_queries_per_kv: usize, // G = num_heads / num_kv_heads
}

impl GroupedQueryAttention {
    pub fn new(config: &LlamaConfig) -> Self {
        assert_eq!(
            config.d_model % config.num_heads,
            0,
            "d_model must be divisible by num_heads"
        );
        assert_eq!(
            config.num_heads % config.num_kv_heads,
            0,
            "num_heads must be divisible by num_kv_heads for GQA"
        );

        let head_dim = config.d_model / config.num_heads;
        let num_queries_per_kv = config.num_heads / config.num_kv_heads;
        let kv_dim = config.num_kv_heads * head_dim;

        Self {
            q_proj: Linear::without_bias(config.d_model, config.d_model),
            k_proj: Linear::without_bias(config.d_model, kv_dim),
            v_proj: Linear::without_bias(config.d_model, kv_dim),
            o_proj: Linear::without_bias(config.d_model, config.d_model),
            rope: RotaryEmbedding::new(head_dim, config.max_seq_len, config.rope_theta),
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim,
            num_queries_per_kv,
        }
    }

    pub fn forward_gqa(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        let shape = x.shape();
        if shape.len() != 3 {
            return Err(EngineError::IncompatibleShapes {
                op: "GroupedQueryAttention forward (expected 3D input [B, T, C])",
                shapes: vec![shape],
            });
        }

        let (b, t, _c) = (shape[0], shape[1], shape[2]);
        let h_q = self.num_heads;
        let h_kv = self.num_kv_heads;
        let d = self.head_dim;
        let g = self.num_queries_per_kv;

        // 1. Project Q, K, V -> Reshape to 4D
        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q.reshape(&[b, t, h_q, d])?.transpose(1, 2)?; // [B, H_q, T, D]
        let k = k.reshape(&[b, t, h_kv, d])?.transpose(1, 2)?; // [B, H_kv, T, D]
        let v = v.reshape(&[b, t, h_kv, d])?.transpose(1, 2)?; // [B, H_kv, T, D]

        // 2. Apply Rotary Position Embedding (RoPE) to Q and K
        let q = self.rope.apply(&q, start_pos)?;
        let k = self.rope.apply(&k, start_pos)?;

        // 3. Repeat KV heads for GQA if G > 1
        let (k_exp, v_exp) = if g > 1 {
            let zeros = Tensor::zeros(&[b, h_kv, g, t, d], false);
            let k_rep = k.unsqueeze(2)?.add(&zeros)?.reshape(&[b, h_q, t, d])?;
            let v_rep = v.unsqueeze(2)?.add(&zeros)?.reshape(&[b, h_q, t, d])?;
            (k_rep, v_rep)
        } else {
            (k, v)
        };

        // 4. Attention scores: (Q * K^T) / sqrt(D) -> [B, H_q, T, T]
        let k_t = k_exp.transpose(2, 3)?;
        let scores = q.matmul(&k_t)?;
        let scale = 1.0 / (d as f32).sqrt();
        let mut scaled_scores = scores.mul_scalar(scale)?;

        // 5. Causal autoregressive mask
        if t > 1 {
            let mut mask_data = vec![0.0; t * t];
            for r in 0..t {
                for c in 0..t {
                    if c > r {
                        mask_data[r * t + c] = -1e4;
                    }
                }
            }
            let mask = Tensor::new(RawTensor::from_vec(mask_data, vec![1, 1, t, t]), false);
            scaled_scores = scaled_scores.add(&mask)?;
        }

        // 6. Softmax & Context aggregation
        let weights = scaled_scores.softmax(3)?;
        let context = weights.matmul(&v_exp)?; // [B, H_q, T, D]

        // 7. Merge heads and output projection
        let merged = context.transpose(1, 2)?.reshape(&[b, t, h_q * d])?;
        self.o_proj.forward(&merged)
    }
}

impl Module for GroupedQueryAttention {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_gqa(input, 0)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.q_proj.parameters());
        params.extend(self.k_proj.parameters());
        params.extend(self.v_proj.parameters());
        params.extend(self.o_proj.parameters());
        params
    }
}

/// SwiGLU Feed-Forward Network: down_proj(silu(gate_proj(x)) * up_proj(x)).
pub struct SwiGLU {
    pub gate_proj: Linear,
    pub up_proj: Linear,
    pub down_proj: Linear,
}

impl SwiGLU {
    pub fn new(d_model: usize, hidden_dim: usize) -> Self {
        Self {
            gate_proj: Linear::without_bias(d_model, hidden_dim),
            up_proj: Linear::without_bias(d_model, hidden_dim),
            down_proj: Linear::without_bias(hidden_dim, d_model),
        }
    }
}

impl Module for SwiGLU {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
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

/// A single LLaMA 2 Transformer Decoder Block.
pub struct Llama2Block {
    pub attn_norm: RMSNorm,
    pub attn: GroupedQueryAttention,
    pub ffn_norm: RMSNorm,
    pub ffn: SwiGLU,
}

impl Llama2Block {
    pub fn new(config: &LlamaConfig) -> Self {
        Self {
            attn_norm: RMSNorm::with_eps(config.d_model, config.norm_eps),
            attn: GroupedQueryAttention::new(config),
            ffn_norm: RMSNorm::with_eps(config.d_model, config.norm_eps),
            ffn: SwiGLU::new(config.d_model, config.hidden_dim),
        }
    }

    pub fn forward_block(&self, x: &Tensor, start_pos: usize) -> Result<Tensor> {
        // 1. Attention with Pre-RMSNorm and Residual
        let norm_x = self.attn_norm.forward(x)?;
        let h_attn = self.attn.forward_gqa(&norm_x, start_pos)?;
        let x = x.add(&h_attn)?;

        // 2. SwiGLU FFN with Pre-RMSNorm and Residual
        let norm_x2 = self.ffn_norm.forward(&x)?;
        let h_ffn = self.ffn.forward(&norm_x2)?;
        x.add(&h_ffn)
    }
}

impl Module for Llama2Block {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        self.forward_block(input, 0)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.attn_norm.parameters());
        params.extend(self.attn.parameters());
        params.extend(self.ffn_norm.parameters());
        params.extend(self.ffn.parameters());
        params
    }
}

/// Complete LLaMA 2 Decoder-only Language Model with GQA, RoPE, RMSNorm, and SwiGLU.
pub struct Llama2LM {
    pub tok_embeddings: crate::nn::embedding::Embedding,
    pub layers: Vec<Llama2Block>,
    pub norm: RMSNorm,
    pub lm_head: Linear,
    pub config: LlamaConfig,
}

impl Llama2LM {
    pub fn new(config: LlamaConfig) -> Self {
        let mut layers = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            layers.push(Llama2Block::new(&config));
        }

        Self {
            tok_embeddings: crate::nn::embedding::Embedding::new(config.vocab_size, config.d_model),
            layers,
            norm: RMSNorm::with_eps(config.d_model, config.norm_eps),
            lm_head: Linear::without_bias(config.d_model, config.vocab_size),
            config,
        }
    }

    /// Forward pass for batch of token indices [BatchSize, SeqLen].
    pub fn forward_tokens(
        &self,
        token_indices: &[usize],
        batch_size: usize,
        seq_len: usize,
        start_pos: usize,
    ) -> Result<Tensor> {
        assert_eq!(
            token_indices.len(),
            batch_size * seq_len,
            "token_indices length must match batch_size * seq_len"
        );
        assert!(
            start_pos + seq_len <= self.config.max_seq_len,
            "Sequence position exceeds max_seq_len"
        );

        // 1. Token Embeddings -> [B, T, D]
        let tok = self.tok_embeddings.forward_indices(token_indices)?;
        let mut x = tok.reshape(&[batch_size, seq_len, self.config.d_model])?;

        // 2. Cascade through LLaMA 2 Transformer Blocks with RoPE
        for layer in &self.layers {
            x = layer.forward_block(&x, start_pos)?;
        }

        // 3. Final RMSNorm & LM Head Logits -> [B, T, VocabSize]
        let x = self.norm.forward(&x)?;
        self.lm_head.forward(&x)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.tok_embeddings.parameters());
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        params.extend(self.norm.parameters());
        params.extend(self.lm_head.parameters());
        params
    }
}
