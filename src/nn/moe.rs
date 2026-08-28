//! Mixture of Experts (MoE) Layer with Top-K sparse routing.
//!
//! Architecture used in: Mixtral 8x7B, Switch Transformer, DeepSeek-V2/V3, and GShard.
//!
//! ## Design
//! - **Router**: A single `Linear(d_model -> num_experts)` gate network.
//! - **Experts**: `num_experts` independent `SwiGLU` FFN sub-networks.
//! - **Top-K Gating**: Only the top-k highest-scored experts activate per token.
//! - **Load-Balancing Loss**: Auxiliary loss penalizing routing imbalance to prevent expert collapse.

use crate::autograd::Tensor;
use crate::error::Result;
use crate::nn::linear::Linear;
use crate::nn::llama::SwiGLU;
use crate::nn::module::Module;
use crate::tensor::RawTensor;

/// Configuration for the MoE layer.
#[derive(Clone, Debug)]
pub struct MoEConfig {
    /// Model dimensionality (d_model).
    pub d_model: usize,
    /// Hidden dimension inside each expert FFN.
    pub hidden_dim: usize,
    /// Total number of expert networks.
    pub num_experts: usize,
    /// Number of experts activated per token.
    pub top_k: usize,
    /// Coefficient for the auxiliary load-balancing loss (typically 0.01).
    pub aux_loss_coeff: f32,
}

impl MoEConfig {
    /// Standard 8-expert MoE config matching Mixtral architecture ratios.
    pub fn mixtral_style(d_model: usize) -> Self {
        Self {
            d_model,
            hidden_dim: d_model * 4,
            num_experts: 8,
            top_k: 2,
            aux_loss_coeff: 0.01,
        }
    }

    /// Minimal MoE config for fast unit tests.
    pub fn mini(d_model: usize, num_experts: usize) -> Self {
        Self {
            d_model,
            hidden_dim: d_model * 2,
            num_experts,
            top_k: 2.min(num_experts),
            aux_loss_coeff: 0.01,
        }
    }
}

/// Top-K sparse Router network: outputs gate logits and selects the top-k experts per token.
#[derive(Clone)]
pub struct TopKRouter {
    pub gate: Linear,
    pub num_experts: usize,
    pub top_k: usize,
}

impl TopKRouter {
    pub fn new(d_model: usize, num_experts: usize, top_k: usize) -> Self {
        Self {
            gate: Linear::without_bias(d_model, num_experts),
            num_experts,
            top_k,
        }
    }

    /// Forward pass: returns `(gate_weights, expert_indices, router_logits)`
    ///
    /// - `gate_weights`: `[B*T, top_k]` softmax-normalized weights for selected experts
    /// - `expert_indices`: `Vec<usize>` of length `B*T * top_k`, row-local expert indices
    /// - `router_logits`: `[B*T, num_experts]` raw logits (used for aux loss)
    pub fn route(&self, x: &Tensor) -> Result<(Tensor, Vec<usize>, Tensor)> {
        let shape = x.shape();
        let (batch_tokens, d_model) = match shape.len() {
            2 => (shape[0], shape[1]),
            3 => (shape[0] * shape[1], shape[2]),
            _ => (
                shape.iter().rev().skip(1).product::<usize>().max(1),
                shape[shape.len() - 1],
            ),
        };

        let x_2d = if x.shape().len() == 3 {
            x.reshape(&[batch_tokens, d_model])?
        } else {
            x.clone()
        };

        // Gate logits: [B*T, num_experts]
        let router_logits = self.gate.forward(&x_2d)?;

        // Top-K selection: values [B*T, top_k], indices Vec<usize> of len B*T*top_k
        let (top_vals, expert_indices) = router_logits.topk(self.top_k)?;

        // Softmax over selected top-k values -> normalized dispatch weights
        let gate_weights = top_vals.softmax(top_vals.shape().len() - 1)?;

        Ok((gate_weights, expert_indices, router_logits))
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        self.gate.parameters()
    }
}

/// Mixture of Experts Feed-Forward layer.
///
/// Drop-in replacement for a single FFN (SwiGLU / MLP) in a Transformer block.
/// For each token, activates only `top_k` of `num_experts` expert networks.
#[derive(Clone)]
pub struct MoELayer {
    pub router: TopKRouter,
    pub experts: Vec<SwiGLU>,
    pub config: MoEConfig,
}

impl MoELayer {
    pub fn new(config: MoEConfig) -> Self {
        let router = TopKRouter::new(config.d_model, config.num_experts, config.top_k);
        let experts = (0..config.num_experts)
            .map(|_| SwiGLU::new(config.d_model, config.hidden_dim))
            .collect();
        Self {
            router,
            experts,
            config,
        }
    }

    /// Forward pass - returns `(output [B, T, d_model], aux_load_balance_loss scalar)`.
    ///
    /// The auxiliary load-balancing loss should be added to the task loss:
    /// `total_loss = task_loss.add(&aux_loss)?`
    pub fn forward_with_aux(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        let orig_shape = x.shape();
        let d = self.config.d_model;
        let top_k = self.config.top_k;
        let num_experts = self.config.num_experts;
        let bt: usize = orig_shape.iter().product::<usize>() / d;

        let x_2d = if orig_shape.len() == 3 {
            x.reshape(&[bt, d])?
        } else {
            x.clone()
        };

        // 1. Route: [B*T, top_k] gate weights + expert indices + logits
        let (gate_weights, expert_indices, router_logits) = self.router.route(&x_2d)?;

        // 2. Forward through all experts
        let expert_outputs: Vec<Tensor> = self
            .experts
            .iter()
            .map(|expert| expert.forward(&x_2d))
            .collect::<Result<_>>()?;

        // 3. Differentiable combination using gate weight per expert
        let gw_data = gate_weights.data();
        let gw_slice = gw_data.to_contiguous();
        let gw_slice_ref = gw_slice.as_slice();

        let mut output = Tensor::zeros(&[bt, d], x.requires_grad());
        for (e_idx, exp_out) in expert_outputs.iter().enumerate() {
            let col_data: Vec<f32> = (0..bt)
                .map(|t| {
                    let start = t * top_k;
                    let end = start + top_k;
                    (start..end)
                        .position(|i| expert_indices[i] == e_idx)
                        .map(|ki| gw_slice_ref[start + ki])
                        .unwrap_or(0.0f32)
                })
                .collect();

            let col_raw = RawTensor::from_vec(col_data, vec![bt, 1]);
            let col_tensor = Tensor::new(col_raw, gate_weights.requires_grad());
            let scaled = exp_out.mul(&col_tensor)?;
            output = output.add(&scaled)?;
        }

        // Reshape back to original 3D shape if needed
        let out_final = if orig_shape.len() == 3 {
            output.reshape(&orig_shape)?
        } else {
            output
        };

        // 4. Auxiliary load-balancing loss
        let aux_loss =
            self.compute_aux_loss(&router_logits, &expert_indices, bt, num_experts, top_k)?;

        Ok((out_final, aux_loss))
    }

    fn compute_aux_loss(
        &self,
        router_logits: &Tensor,
        expert_indices: &[usize],
        num_tokens: usize,
        num_experts: usize,
        top_k: usize,
    ) -> Result<Tensor> {
        let mut token_counts = vec![0.0f32; num_experts];
        for &exp_id in expert_indices {
            if exp_id < num_experts {
                token_counts[exp_id] += 1.0;
            }
        }
        let total_assignments = (num_tokens * top_k) as f32;
        for c in &mut token_counts {
            *c /= total_assignments.max(1.0);
        }

        let router_probs = router_logits.softmax(1)?;
        let mean_probs = router_probs.mean(0, false)?;

        let f_tensor = Tensor::new(RawTensor::from_vec(token_counts, vec![num_experts]), false);

        let dot = f_tensor.mul(&mean_probs)?;
        let sum_dot = dot.sum_all();
        let coeff = self.config.aux_loss_coeff * num_experts as f32;
        sum_dot.mul_scalar(coeff)
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.router.parameters();
        for expert in &self.experts {
            params.extend(expert.parameters());
        }
        params
    }
}

impl Module for MoELayer {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (out, _aux_loss) = self.forward_with_aux(input)?;
        Ok(out)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.parameters()
    }
}

/// Transformer block where the standard FFN is replaced with a sparse MoE layer.
#[derive(Clone)]
pub struct SparseMoEBlock {
    pub ln1: crate::nn::norm::LayerNorm,
    pub attn: crate::nn::attention::MultiHeadAttention,
    pub ln2: crate::nn::norm::LayerNorm,
    pub moe: MoELayer,
}

impl SparseMoEBlock {
    pub fn new(d_model: usize, num_heads: usize, moe_config: MoEConfig) -> Self {
        Self {
            ln1: crate::nn::norm::LayerNorm::new(d_model),
            attn: crate::nn::attention::MultiHeadAttention::new(d_model, num_heads, true),
            ln2: crate::nn::norm::LayerNorm::new(d_model),
            moe: MoELayer::new(moe_config),
        }
    }

    /// Forward returning `(output, aux_loss)`. Add aux_loss to task loss.
    pub fn forward_with_aux(&self, x: &Tensor) -> Result<(Tensor, Tensor)> {
        // 1. Pre-LN Causal Self-Attention with residual
        let norm1 = self.ln1.forward(x)?;
        let attn_out = self.attn.forward_attention(&norm1)?;
        let x = x.add(&attn_out)?;

        // 2. Pre-LN Sparse MoE FFN with residual
        let norm2 = self.ln2.forward(&x)?;
        let (moe_out, aux_loss) = self.moe.forward_with_aux(&norm2)?;
        let x = x.add(&moe_out)?;

        Ok((x, aux_loss))
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.ln1.parameters());
        params.extend(self.attn.parameters());
        params.extend(self.ln2.parameters());
        params.extend(self.moe.parameters());
        params
    }
}

impl Module for SparseMoEBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (out, _) = self.forward_with_aux(input)?;
        Ok(out)
    }

    fn parameters(&self) -> Vec<Tensor> {
        self.parameters()
    }
}
