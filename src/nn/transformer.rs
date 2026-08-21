//! Transformer Block and Decoder-only Transformer Language Model (nanoGPT architecture).

use crate::autograd::Tensor;
use crate::error::Result;
use crate::nn::activations::GELU;
use crate::nn::attention::MultiHeadAttention;
use crate::nn::embedding::Embedding;
use crate::nn::linear::Linear;
use crate::nn::module::Module;
use crate::nn::norm::LayerNorm;

/// A single Transformer decoder layer consisting of Pre-LayerNorm Causal Self-Attention and MLP.
pub struct TransformerBlock {
    pub ln1: LayerNorm,
    pub attn: MultiHeadAttention,
    pub ln2: LayerNorm,
    pub mlp_fc1: Linear,
    pub mlp_gelu: GELU,
    pub mlp_fc2: Linear,
}

impl TransformerBlock {
    pub fn new(d_model: usize, num_heads: usize, is_causal: bool) -> Self {
        let mlp_hidden = d_model * 4;
        Self {
            ln1: LayerNorm::new(d_model),
            attn: MultiHeadAttention::new(d_model, num_heads, is_causal),
            ln2: LayerNorm::new(d_model),
            mlp_fc1: Linear::new(d_model, mlp_hidden),
            mlp_gelu: GELU,
            mlp_fc2: Linear::new(mlp_hidden, d_model),
        }
    }
}

impl Module for TransformerBlock {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // 1. Attention branch with residual connection: x = x + Attention(LN(x))
        let norm1 = self.ln1.forward(x)?;
        let attn_out = self.attn.forward(&norm1)?;
        let x = x.add(&attn_out)?;

        // 2. Feed-Forward MLP branch with residual connection: x = x + MLP(LN(x))
        let norm2 = self.ln2.forward(&x)?;
        let h1 = self.mlp_fc1.forward(&norm2)?;
        let act = self.mlp_gelu.forward(&h1)?;
        let h2 = self.mlp_fc2.forward(&act)?;
        x.add(&h2)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.ln1.parameters());
        params.extend(self.attn.parameters());
        params.extend(self.ln2.parameters());
        params.extend(self.mlp_fc1.parameters());
        params.extend(self.mlp_fc2.parameters());
        params
    }
}

/// Decoder-only autoregressive Transformer Language Model (nanoGPT style).
pub struct TransformerLM {
    pub tok_emb: Embedding,
    pub pos_emb: Embedding,
    pub blocks: Vec<TransformerBlock>,
    pub ln_f: LayerNorm,
    pub lm_head: Linear,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub d_model: usize,
}

impl TransformerLM {
    pub fn new(
        vocab_size: usize,
        max_seq_len: usize,
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
    ) -> Self {
        let mut blocks = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            blocks.push(TransformerBlock::new(d_model, num_heads, true));
        }

        Self {
            tok_emb: Embedding::new(vocab_size, d_model),
            pos_emb: Embedding::new(max_seq_len, d_model),
            blocks,
            ln_f: LayerNorm::new(d_model),
            lm_head: Linear::new(d_model, vocab_size),
            vocab_size,
            max_seq_len,
            d_model,
        }
    }

    /// Forward pass through the Transformer for a batch of token sequences [BatchSize, SeqLen].
    pub fn forward_tokens(
        &self,
        token_indices: &[usize],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        assert_eq!(
            token_indices.len(),
            batch_size * seq_len,
            "Token indices length must match batch_size * seq_len"
        );
        assert!(
            seq_len <= self.max_seq_len,
            "Sequence length {} exceeds max_seq_len {}",
            seq_len,
            self.max_seq_len
        );

        // 1. Token Embeddings -> [B * T, D] -> [B, T, D]
        let tok = self.tok_emb.forward_indices(token_indices)?;
        let tok = tok.reshape(&[batch_size, seq_len, self.d_model])?;

        // 2. Position Embeddings -> [T, D] -> [1, T, D]
        let pos_indices: Vec<usize> = (0..seq_len).collect();
        let pos = self.pos_emb.forward_indices(&pos_indices)?;
        let pos = pos.reshape(&[1, seq_len, self.d_model])?;

        // 3. Combined Embeddings
        let mut x = tok.add(&pos)?;

        // 4. Cascade through Transformer Blocks
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        // 5. Final LayerNorm & Logits projection -> [B, T, VocabSize]
        let x = self.ln_f.forward(&x)?;
        self.lm_head.forward(&x)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.tok_emb.parameters());
        params.extend(self.pos_emb.parameters());
        for block in &self.blocks {
            params.extend(block.parameters());
        }
        params.extend(self.ln_f.parameters());
        params.extend(self.lm_head.parameters());
        params
    }
}
