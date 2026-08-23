//! Recurrent Neural Network layers: RNN, LSTM, and GRU (cells and multi-layer sequence modules).
//!
//! Provides:
//! - [`RNNCell`]: Single-step Elman RNN cell with Tanh or ReLU non-linearity.
//! - [`RNN`]: Multi-layer, optionally bidirectional sequence Elman RNN module.
//! - [`LSTMCell`]: Single-step Long Short-Term Memory cell with fused gate projections ($i, f, g, o$).
//! - [`LSTM`]: Multi-layer, optionally bidirectional sequence Long Short-Term Memory module.
//! - [`GRUCell`]: Single-step Gated Recurrent Unit cell with reset, update, and candidate gates ($r, z, n$).
//! - [`GRU`]: Multi-layer, optionally bidirectional sequence Gated Recurrent Unit module.

use crate::autograd::Tensor;
use crate::error::{EngineError, Result};
use crate::nn::dropout::Dropout;
use crate::nn::init::{kaiming_uniform, uniform, FanMode, NonLinearity};
use crate::nn::module::Module;

// =========================================================================
// 1. RNN CELL & SEQUENCE MODULE
// =========================================================================

/// Non-linear activation function for the Elman RNN cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RNNActivation {
    Tanh,
    ReLU,
}

/// An Elman RNN cell with Tanh or ReLU non-linearity:
/// $$h' = \text{activation}(x W_{ih}^T + b_{ih} + h W_{hh}^T + b_{hh})$$
#[derive(Clone)]
pub struct RNNCell {
    pub weight_ih: Tensor,
    pub weight_hh: Tensor,
    pub bias_ih: Option<Tensor>,
    pub bias_hh: Option<Tensor>,
    pub input_size: usize,
    pub hidden_size: usize,
    pub activation: RNNActivation,
}

impl RNNCell {
    /// Creates a new `RNNCell`.
    pub fn new(input_size: usize, hidden_size: usize, activation: RNNActivation) -> Self {
        Self::with_bias(input_size, hidden_size, activation, true)
    }

    /// Creates a new `RNNCell` with optional bias.
    pub fn with_bias(
        input_size: usize,
        hidden_size: usize,
        activation: RNNActivation,
        has_bias: bool,
    ) -> Self {
        let non_lin = match activation {
            RNNActivation::Tanh => NonLinearity::Tanh,
            RNNActivation::ReLU => NonLinearity::ReLU,
        };

        let weight_ih = Tensor::new(
            kaiming_uniform(&[hidden_size, input_size], 0.0, FanMode::FanIn, non_lin),
            true,
        );
        let weight_hh = Tensor::new(
            kaiming_uniform(&[hidden_size, hidden_size], 0.0, FanMode::FanIn, non_lin),
            true,
        );

        let (bias_ih, bias_hh) = if has_bias {
            let bound = 1.0 / (hidden_size as f32).sqrt();
            (
                Some(Tensor::uniform(&[hidden_size], -bound, bound, true)),
                Some(Tensor::uniform(&[hidden_size], -bound, bound, true)),
            )
        } else {
            (None, None)
        };

        Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            input_size,
            hidden_size,
            activation,
        }
    }

    /// Computes a single step forward pass:
    /// Returns the new hidden state $h_t \in \mathbb{R}^{B \times H}$.
    pub fn forward_step(&self, input: &Tensor, hidden: Option<&Tensor>) -> Result<Tensor> {
        let mut pre_act = input.matmul_transposed_b(&self.weight_ih)?;
        if let Some(ref b_ih) = self.bias_ih {
            pre_act = pre_act.add(b_ih)?;
        }

        if let Some(h) = hidden {
            let mut h_term = h.matmul_transposed_b(&self.weight_hh)?;
            if let Some(ref b_hh) = self.bias_hh {
                h_term = h_term.add(b_hh)?;
            }
            pre_act = pre_act.add(&h_term)?;
        } else if let Some(ref b_hh) = self.bias_hh {
            pre_act = pre_act.add(b_hh)?;
        }

        match self.activation {
            RNNActivation::Tanh => pre_act.tanh(),
            RNNActivation::ReLU => pre_act.relu(),
        }
    }

    /// Returns learnable parameter tensors.
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight_ih.clone(), self.weight_hh.clone()];
        if let Some(ref b) = self.bias_ih {
            params.push(b.clone());
        }
        if let Some(ref b) = self.bias_hh {
            params.push(b.clone());
        }
        params
    }
}

/// Multi-layer, optionally bidirectional Elman RNN sequence module.
#[derive(Clone)]
pub struct RNN {
    pub forward_cells: Vec<RNNCell>,
    pub backward_cells: Vec<RNNCell>,
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub bidirectional: bool,
    pub dropout: Option<Dropout>,
}

impl RNN {
    /// Creates a new `RNN` layer.
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        activation: RNNActivation,
        bidirectional: bool,
        dropout_p: f32,
    ) -> Self {
        assert!(num_layers > 0, "num_layers must be at least 1");
        let mut forward_cells = Vec::with_capacity(num_layers);
        let mut backward_cells = Vec::with_capacity(if bidirectional { num_layers } else { 0 });

        for layer in 0..num_layers {
            let in_dim = if layer == 0 {
                input_size
            } else if bidirectional {
                hidden_size * 2
            } else {
                hidden_size
            };

            forward_cells.push(RNNCell::new(in_dim, hidden_size, activation));
            if bidirectional {
                backward_cells.push(RNNCell::new(in_dim, hidden_size, activation));
            }
        }

        let dropout = if dropout_p > 0.0 && num_layers > 1 {
            Some(Dropout::new(dropout_p))
        } else {
            None
        };

        Self {
            forward_cells,
            backward_cells,
            input_size,
            hidden_size,
            num_layers,
            bidirectional,
            dropout,
        }
    }

    /// Forward pass through the sequence:
    /// - `input`: `[batch_size, seq_len, input_size]`
    /// - `initial_hidden`: Optional `[num_directions * num_layers, batch_size, hidden_size]`
    ///
    /// Returns:
    /// - `output`: `[batch_size, seq_len, num_directions * hidden_size]`
    /// - `h_n`: `[num_directions * num_layers, batch_size, hidden_size]`
    pub fn forward_seq(
        &self,
        input: &Tensor,
        initial_hidden: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(EngineError::ShapeMismatch {
                expected: vec![0, 0, self.input_size],
                actual: shape,
            });
        }
        let (_batch_size, seq_len, _) = (shape[0], shape[1], shape[2]);
        let mut current_input = input.clone();
        let mut final_hiddens = Vec::new();

        for layer in 0..self.num_layers {
            let mut fwd_outputs = Vec::with_capacity(seq_len);
            let mut fwd_h = if let Some(init_h) = initial_hidden {
                let idx = if self.bidirectional { layer * 2 } else { layer };
                Some(init_h.slice(0, idx, idx + 1)?.squeeze(0)?)
            } else {
                None
            };

            for t in 0..seq_len {
                let x_t = current_input.slice(1, t, t + 1)?.squeeze(1)?;
                let next_h = self.forward_cells[layer].forward_step(&x_t, fwd_h.as_ref())?;
                fwd_h = Some(next_h.clone());
                fwd_outputs.push(next_h.unsqueeze(1)?);
            }

            let fwd_refs: Vec<&Tensor> = fwd_outputs.iter().collect();
            let fwd_seq = Tensor::cat(&fwd_refs, 1)?;
            final_hiddens.push(fwd_h.unwrap().unsqueeze(0)?);

            if self.bidirectional {
                let mut bwd_outputs = Vec::with_capacity(seq_len);
                let mut bwd_h = if let Some(init_h) = initial_hidden {
                    let idx = layer * 2 + 1;
                    Some(init_h.slice(0, idx, idx + 1)?.squeeze(0)?)
                } else {
                    None
                };

                for t in (0..seq_len).rev() {
                    let x_t = current_input.slice(1, t, t + 1)?.squeeze(1)?;
                    let next_h = self.backward_cells[layer].forward_step(&x_t, bwd_h.as_ref())?;
                    bwd_h = Some(next_h.clone());
                    bwd_outputs.push(next_h.unsqueeze(1)?);
                }
                bwd_outputs.reverse();
                let bwd_refs: Vec<&Tensor> = bwd_outputs.iter().collect();
                let bwd_seq = Tensor::cat(&bwd_refs, 1)?;
                final_hiddens.push(bwd_h.unwrap().unsqueeze(0)?);

                let layer_out = Tensor::cat(&[&fwd_seq, &bwd_seq], 2)?;
                current_input = if let Some(ref drop) = self.dropout {
                    if layer < self.num_layers - 1 {
                        drop.forward(&layer_out)?
                    } else {
                        layer_out
                    }
                } else {
                    layer_out
                };
            } else {
                current_input = if let Some(ref drop) = self.dropout {
                    if layer < self.num_layers - 1 {
                        drop.forward(&fwd_seq)?
                    } else {
                        fwd_seq
                    }
                } else {
                    fwd_seq
                };
            }
        }

        let h_refs: Vec<&Tensor> = final_hiddens.iter().collect();
        let h_n = Tensor::cat(&h_refs, 0)?;

        Ok((current_input, h_n))
    }
}

impl Module for RNN {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (output, _) = self.forward_seq(input, None)?;
        Ok(output)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for cell in &self.forward_cells {
            params.extend(cell.parameters());
        }
        for cell in &self.backward_cells {
            params.extend(cell.parameters());
        }
        params
    }

    fn train(&mut self) {
        if let Some(ref mut drop) = self.dropout {
            drop.train();
        }
    }

    fn eval(&mut self) {
        if let Some(ref mut drop) = self.dropout {
            drop.eval();
        }
    }
}

// =========================================================================
// 2. LSTM CELL & SEQUENCE MODULE
// =========================================================================

/// A Long Short-Term Memory (LSTM) cell with fused gate projections:
/// - $i_t = \sigma(W_{ii} x_t + b_{ii} + W_{hi} h_{t-1} + b_{hi})$ (input gate)
/// - $f_t = \sigma(W_{if} x_t + b_{if} + W_{hf} h_{t-1} + b_{hf})$ (forget gate)
/// - $g_t = \tanh(W_{ig} x_t + b_{ig} + W_{hg} h_{t-1} + b_{hg})$ (candidate cell gate)
/// - $o_t = \sigma(W_{io} x_t + b_{io} + W_{ho} h_{t-1} + b_{ho})$ (output gate)
/// - $c_t = f_t \odot c_{t-1} + i_t \odot g_t$ (cell state)
/// - $h_t = o_t \odot \tanh(c_t)$ (hidden state)
#[derive(Clone)]
pub struct LSTMCell {
    pub weight_ih: Tensor,
    pub weight_hh: Tensor,
    pub bias_ih: Option<Tensor>,
    pub bias_hh: Option<Tensor>,
    pub input_size: usize,
    pub hidden_size: usize,
}

impl LSTMCell {
    /// Creates a new `LSTMCell` with bias enabled.
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self::with_bias(input_size, hidden_size, true)
    }

    /// Creates a new `LSTMCell` with optional bias and forget-gate bias initialized to 1.0.
    pub fn with_bias(input_size: usize, hidden_size: usize, has_bias: bool) -> Self {
        let weight_ih = Tensor::new(
            kaiming_uniform(
                &[4 * hidden_size, input_size],
                0.0,
                FanMode::FanIn,
                NonLinearity::Sigmoid,
            ),
            true,
        );
        let weight_hh = Tensor::new(
            kaiming_uniform(
                &[4 * hidden_size, hidden_size],
                0.0,
                FanMode::FanIn,
                NonLinearity::Sigmoid,
            ),
            true,
        );

        let (bias_ih, bias_hh) = if has_bias {
            let bound = 1.0 / (hidden_size as f32).sqrt();
            let mut b_ih_raw = uniform(&[4 * hidden_size], -bound, bound);
            let mut b_hh_raw = uniform(&[4 * hidden_size], -bound, bound);

            // Set forget gate bias to 1.0 (Jozefowicz et al. 2015 best practice for gradient flow)
            let f_start = hidden_size;
            let f_end = 2 * hidden_size;
            for val in &mut b_ih_raw.as_mut_slice()[f_start..f_end] {
                *val = 1.0;
            }
            for val in &mut b_hh_raw.as_mut_slice()[f_start..f_end] {
                *val = 0.0;
            }

            (
                Some(Tensor::new(b_ih_raw, true)),
                Some(Tensor::new(b_hh_raw, true)),
            )
        } else {
            (None, None)
        };

        Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            input_size,
            hidden_size,
        }
    }

    /// Single-step forward computation.
    /// - `input`: `[batch_size, input_size]`
    /// - `state`: Optional `(h_{t-1}, c_{t-1})` where each is `[batch_size, hidden_size]`
    ///
    /// Returns `(h_t, c_t)` tuple.
    pub fn forward_step(
        &self,
        input: &Tensor,
        state: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Tensor)> {
        let h = self.hidden_size;

        let mut gates = input.matmul_transposed_b(&self.weight_ih)?;
        if let Some(ref b_ih) = self.bias_ih {
            gates = gates.add(b_ih)?;
        }

        let (h_prev, c_prev) = match state {
            Some((h_p, c_p)) => (Some(h_p.clone()), Some(c_p.clone())),
            None => (None, None),
        };

        if let Some(ref hp) = h_prev {
            let mut h_term = hp.matmul_transposed_b(&self.weight_hh)?;
            if let Some(ref b_hh) = self.bias_hh {
                h_term = h_term.add(b_hh)?;
            }
            gates = gates.add(&h_term)?;
        } else if let Some(ref b_hh) = self.bias_hh {
            gates = gates.add(b_hh)?;
        }

        // Slice gates: [i, f, g, o]
        let i_gate = gates.slice(1, 0, h)?.sigmoid()?;
        let f_gate = gates.slice(1, h, 2 * h)?.sigmoid()?;
        let g_gate = gates.slice(1, 2 * h, 3 * h)?.tanh()?;
        let o_gate = gates.slice(1, 3 * h, 4 * h)?.sigmoid()?;

        // Cell state: c_t = f_t * c_{t-1} + i_t * g_t
        let c_next = if let Some(ref cp) = c_prev {
            let f_c = f_gate.mul(cp)?;
            let i_g = i_gate.mul(&g_gate)?;
            f_c.add(&i_g)?
        } else {
            i_gate.mul(&g_gate)?
        };

        // Hidden state: h_t = o_t * tanh(c_t)
        let c_tanh = c_next.tanh()?;
        let h_next = o_gate.mul(&c_tanh)?;

        Ok((h_next, c_next))
    }

    /// Returns learnable parameter tensors.
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight_ih.clone(), self.weight_hh.clone()];
        if let Some(ref b) = self.bias_ih {
            params.push(b.clone());
        }
        if let Some(ref b) = self.bias_hh {
            params.push(b.clone());
        }
        params
    }
}

/// Multi-layer, optionally bidirectional Long Short-Term Memory (LSTM) sequence module.
#[derive(Clone)]
pub struct LSTM {
    pub forward_cells: Vec<LSTMCell>,
    pub backward_cells: Vec<LSTMCell>,
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub bidirectional: bool,
    pub dropout: Option<Dropout>,
}

impl LSTM {
    /// Creates a new multi-layer `LSTM` module.
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bidirectional: bool,
        dropout_p: f32,
    ) -> Self {
        assert!(num_layers > 0, "num_layers must be at least 1");
        let mut forward_cells = Vec::with_capacity(num_layers);
        let mut backward_cells = Vec::with_capacity(if bidirectional { num_layers } else { 0 });

        for layer in 0..num_layers {
            let in_dim = if layer == 0 {
                input_size
            } else if bidirectional {
                hidden_size * 2
            } else {
                hidden_size
            };

            forward_cells.push(LSTMCell::new(in_dim, hidden_size));
            if bidirectional {
                backward_cells.push(LSTMCell::new(in_dim, hidden_size));
            }
        }

        let dropout = if dropout_p > 0.0 && num_layers > 1 {
            Some(Dropout::new(dropout_p))
        } else {
            None
        };

        Self {
            forward_cells,
            backward_cells,
            input_size,
            hidden_size,
            num_layers,
            bidirectional,
            dropout,
        }
    }

    /// Forward pass through the sequence:
    /// - `input`: `[batch_size, seq_len, input_size]`
    /// - `initial_state`: Optional `(h_0, c_0)` where each is `[num_directions * num_layers, batch_size, hidden_size]`
    ///
    /// Returns:
    /// - `(output, (h_n, c_n))`
    /// - `output`: `[batch_size, seq_len, num_directions * hidden_size]`
    /// - `h_n`: `[num_directions * num_layers, batch_size, hidden_size]`
    /// - `c_n`: `[num_directions * num_layers, batch_size, hidden_size]`
    pub fn forward_seq(
        &self,
        input: &Tensor,
        initial_state: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, (Tensor, Tensor))> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(EngineError::ShapeMismatch {
                expected: vec![0, 0, self.input_size],
                actual: shape,
            });
        }
        let (_batch_size, seq_len, _) = (shape[0], shape[1], shape[2]);
        let mut current_input = input.clone();
        let mut final_hiddens = Vec::new();
        let mut final_cells = Vec::new();

        for layer in 0..self.num_layers {
            let mut fwd_outputs = Vec::with_capacity(seq_len);
            let (mut fwd_h, mut fwd_c) = if let Some((init_h, init_c)) = initial_state {
                let idx = if self.bidirectional { layer * 2 } else { layer };
                (
                    Some(init_h.slice(0, idx, idx + 1)?.squeeze(0)?),
                    Some(init_c.slice(0, idx, idx + 1)?.squeeze(0)?),
                )
            } else {
                (None, None)
            };

            for t in 0..seq_len {
                let x_t = current_input.slice(1, t, t + 1)?.squeeze(1)?;
                let state_ref = match (&fwd_h, &fwd_c) {
                    (Some(ref h), Some(ref c)) => Some((h, c)),
                    _ => None,
                };
                let (next_h, next_c) = self.forward_cells[layer].forward_step(&x_t, state_ref)?;
                fwd_h = Some(next_h.clone());
                fwd_c = Some(next_c);
                fwd_outputs.push(next_h.unsqueeze(1)?);
            }

            let fwd_refs: Vec<&Tensor> = fwd_outputs.iter().collect();
            let fwd_seq = Tensor::cat(&fwd_refs, 1)?;
            final_hiddens.push(fwd_h.unwrap().unsqueeze(0)?);
            final_cells.push(fwd_c.unwrap().unsqueeze(0)?);

            if self.bidirectional {
                let mut bwd_outputs = Vec::with_capacity(seq_len);
                let (mut bwd_h, mut bwd_c) = if let Some((init_h, init_c)) = initial_state {
                    let idx = layer * 2 + 1;
                    (
                        Some(init_h.slice(0, idx, idx + 1)?.squeeze(0)?),
                        Some(init_c.slice(0, idx, idx + 1)?.squeeze(0)?),
                    )
                } else {
                    (None, None)
                };

                for t in (0..seq_len).rev() {
                    let x_t = current_input.slice(1, t, t + 1)?.squeeze(1)?;
                    let state_ref = match (&bwd_h, &bwd_c) {
                        (Some(ref h), Some(ref c)) => Some((h, c)),
                        _ => None,
                    };
                    let (next_h, next_c) =
                        self.backward_cells[layer].forward_step(&x_t, state_ref)?;
                    bwd_h = Some(next_h.clone());
                    bwd_c = Some(next_c);
                    bwd_outputs.push(next_h.unsqueeze(1)?);
                }
                bwd_outputs.reverse();
                let bwd_refs: Vec<&Tensor> = bwd_outputs.iter().collect();
                let bwd_seq = Tensor::cat(&bwd_refs, 1)?;
                final_hiddens.push(bwd_h.unwrap().unsqueeze(0)?);
                final_cells.push(bwd_c.unwrap().unsqueeze(0)?);

                let layer_out = Tensor::cat(&[&fwd_seq, &bwd_seq], 2)?;
                current_input = if let Some(ref drop) = self.dropout {
                    if layer < self.num_layers - 1 {
                        drop.forward(&layer_out)?
                    } else {
                        layer_out
                    }
                } else {
                    layer_out
                };
            } else {
                current_input = if let Some(ref drop) = self.dropout {
                    if layer < self.num_layers - 1 {
                        drop.forward(&fwd_seq)?
                    } else {
                        fwd_seq
                    }
                } else {
                    fwd_seq
                };
            }
        }

        let h_refs: Vec<&Tensor> = final_hiddens.iter().collect();
        let c_refs: Vec<&Tensor> = final_cells.iter().collect();
        let h_n = Tensor::cat(&h_refs, 0)?;
        let c_n = Tensor::cat(&c_refs, 0)?;

        Ok((current_input, (h_n, c_n)))
    }
}

impl Module for LSTM {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (output, _) = self.forward_seq(input, None)?;
        Ok(output)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for cell in &self.forward_cells {
            params.extend(cell.parameters());
        }
        for cell in &self.backward_cells {
            params.extend(cell.parameters());
        }
        params
    }

    fn train(&mut self) {
        if let Some(ref mut drop) = self.dropout {
            drop.train();
        }
    }

    fn eval(&mut self) {
        if let Some(ref mut drop) = self.dropout {
            drop.eval();
        }
    }
}

// =========================================================================
// 3. GRU CELL & SEQUENCE MODULE
// =========================================================================

/// A Gated Recurrent Unit (GRU) cell:
/// - $r_t = \sigma(W_{ir} x_t + b_{ir} + W_{hr} h_{t-1} + b_{hr})$ (reset gate)
/// - $z_t = \sigma(W_{iz} x_t + b_{iz} + W_{hz} h_{t-1} + b_{hz})$ (update gate)
/// - $n_t = \tanh(W_{in} x_t + b_{in} + r_t \odot (W_{hn} h_{t-1} + b_{hn}))$ (candidate state)
/// - $h_t = (1 - z_t) \odot n_t + z_t \odot h_{t-1}$ (hidden state)
#[derive(Clone)]
pub struct GRUCell {
    pub weight_ih: Tensor,
    pub weight_hh: Tensor,
    pub bias_ih: Option<Tensor>,
    pub bias_hh: Option<Tensor>,
    pub input_size: usize,
    pub hidden_size: usize,
}

impl GRUCell {
    /// Creates a new `GRUCell` with bias enabled.
    pub fn new(input_size: usize, hidden_size: usize) -> Self {
        Self::with_bias(input_size, hidden_size, true)
    }

    /// Creates a new `GRUCell` with optional bias.
    pub fn with_bias(input_size: usize, hidden_size: usize, has_bias: bool) -> Self {
        let weight_ih = Tensor::new(
            kaiming_uniform(
                &[3 * hidden_size, input_size],
                0.0,
                FanMode::FanIn,
                NonLinearity::Sigmoid,
            ),
            true,
        );
        let weight_hh = Tensor::new(
            kaiming_uniform(
                &[3 * hidden_size, hidden_size],
                0.0,
                FanMode::FanIn,
                NonLinearity::Sigmoid,
            ),
            true,
        );

        let (bias_ih, bias_hh) = if has_bias {
            let bound = 1.0 / (hidden_size as f32).sqrt();
            (
                Some(Tensor::uniform(&[3 * hidden_size], -bound, bound, true)),
                Some(Tensor::uniform(&[3 * hidden_size], -bound, bound, true)),
            )
        } else {
            (None, None)
        };

        Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            input_size,
            hidden_size,
        }
    }

    /// Single-step forward computation.
    /// - `input`: `[batch_size, input_size]`
    /// - `hidden`: Optional `h_{t-1}` of shape `[batch_size, hidden_size]`
    ///
    /// Returns `h_t \in \mathbb{R}^{B \times H}`.
    pub fn forward_step(&self, input: &Tensor, hidden: Option<&Tensor>) -> Result<Tensor> {
        let batch_size = input.shape()[0];
        let h = self.hidden_size;

        let mut x_gates = input.matmul_transposed_b(&self.weight_ih)?;
        if let Some(ref b_ih) = self.bias_ih {
            x_gates = x_gates.add(b_ih)?;
        }

        let h_gates = if let Some(h_prev) = hidden {
            let mut h_proj = h_prev.matmul_transposed_b(&self.weight_hh)?;
            if let Some(ref b_hh) = self.bias_hh {
                h_proj = h_proj.add(b_hh)?;
            }
            h_proj
        } else if let Some(ref b_hh) = self.bias_hh {
            Tensor::zeros(&[batch_size, 3 * h], false).add(b_hh)?
        } else {
            Tensor::zeros(&[batch_size, 3 * h], false)
        };

        let x_r = x_gates.slice(1, 0, h)?;
        let x_z = x_gates.slice(1, h, 2 * h)?;
        let x_n = x_gates.slice(1, 2 * h, 3 * h)?;

        let h_r = h_gates.slice(1, 0, h)?;
        let h_z = h_gates.slice(1, h, 2 * h)?;
        let h_n = h_gates.slice(1, 2 * h, 3 * h)?;

        // Reset gate: r = sigmoid(x_r + h_r)
        let r = x_r.add(&h_r)?.sigmoid()?;

        // Update gate: z = sigmoid(x_z + h_z)
        let z = x_z.add(&h_z)?.sigmoid()?;

        // Candidate state: n = tanh(x_n + r * h_n)
        let r_hn = r.mul(&h_n)?;
        let n = x_n.add(&r_hn)?.tanh()?;

        // Hidden state: h_t = (1 - z) * n + z * h_{t-1}
        let ones = Tensor::ones(z.shape().as_slice(), false);
        let one_minus_z = ones.sub(&z)?;
        let term1 = one_minus_z.mul(&n)?;

        let h_next = if let Some(h_prev) = hidden {
            let term2 = z.mul(h_prev)?;
            term1.add(&term2)?
        } else {
            term1
        };

        Ok(h_next)
    }

    /// Returns learnable parameter tensors.
    pub fn parameters(&self) -> Vec<Tensor> {
        let mut params = vec![self.weight_ih.clone(), self.weight_hh.clone()];
        if let Some(ref b) = self.bias_ih {
            params.push(b.clone());
        }
        if let Some(ref b) = self.bias_hh {
            params.push(b.clone());
        }
        params
    }
}

/// Multi-layer, optionally bidirectional Gated Recurrent Unit (GRU) sequence module.
#[derive(Clone)]
pub struct GRU {
    pub forward_cells: Vec<GRUCell>,
    pub backward_cells: Vec<GRUCell>,
    pub input_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub bidirectional: bool,
    pub dropout: Option<Dropout>,
}

impl GRU {
    /// Creates a new multi-layer `GRU` module.
    pub fn new(
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
        bidirectional: bool,
        dropout_p: f32,
    ) -> Self {
        assert!(num_layers > 0, "num_layers must be at least 1");
        let mut forward_cells = Vec::with_capacity(num_layers);
        let mut backward_cells = Vec::with_capacity(if bidirectional { num_layers } else { 0 });

        for layer in 0..num_layers {
            let in_dim = if layer == 0 {
                input_size
            } else if bidirectional {
                hidden_size * 2
            } else {
                hidden_size
            };

            forward_cells.push(GRUCell::new(in_dim, hidden_size));
            if bidirectional {
                backward_cells.push(GRUCell::new(in_dim, hidden_size));
            }
        }

        let dropout = if dropout_p > 0.0 && num_layers > 1 {
            Some(Dropout::new(dropout_p))
        } else {
            None
        };

        Self {
            forward_cells,
            backward_cells,
            input_size,
            hidden_size,
            num_layers,
            bidirectional,
            dropout,
        }
    }

    /// Forward pass through the sequence:
    /// - `input`: `[batch_size, seq_len, input_size]`
    /// - `initial_hidden`: Optional `[num_directions * num_layers, batch_size, hidden_size]`
    ///
    /// Returns `(output, h_n)`:
    /// - `output`: `[batch_size, seq_len, num_directions * hidden_size]`
    /// - `h_n`: `[num_directions * num_layers, batch_size, hidden_size]`
    pub fn forward_seq(
        &self,
        input: &Tensor,
        initial_hidden: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(EngineError::ShapeMismatch {
                expected: vec![0, 0, self.input_size],
                actual: shape,
            });
        }
        let (_batch_size, seq_len, _) = (shape[0], shape[1], shape[2]);
        let mut current_input = input.clone();
        let mut final_hiddens = Vec::new();

        for layer in 0..self.num_layers {
            let mut fwd_outputs = Vec::with_capacity(seq_len);
            let mut fwd_h = if let Some(init_h) = initial_hidden {
                let idx = if self.bidirectional { layer * 2 } else { layer };
                Some(init_h.slice(0, idx, idx + 1)?.squeeze(0)?)
            } else {
                None
            };

            for t in 0..seq_len {
                let x_t = current_input.slice(1, t, t + 1)?.squeeze(1)?;
                let next_h = self.forward_cells[layer].forward_step(&x_t, fwd_h.as_ref())?;
                fwd_h = Some(next_h.clone());
                fwd_outputs.push(next_h.unsqueeze(1)?);
            }

            let fwd_refs: Vec<&Tensor> = fwd_outputs.iter().collect();
            let fwd_seq = Tensor::cat(&fwd_refs, 1)?;
            final_hiddens.push(fwd_h.unwrap().unsqueeze(0)?);

            if self.bidirectional {
                let mut bwd_outputs = Vec::with_capacity(seq_len);
                let mut bwd_h = if let Some(init_h) = initial_hidden {
                    let idx = layer * 2 + 1;
                    Some(init_h.slice(0, idx, idx + 1)?.squeeze(0)?)
                } else {
                    None
                };

                for t in (0..seq_len).rev() {
                    let x_t = current_input.slice(1, t, t + 1)?.squeeze(1)?;
                    let next_h = self.backward_cells[layer].forward_step(&x_t, bwd_h.as_ref())?;
                    bwd_h = Some(next_h.clone());
                    bwd_outputs.push(next_h.unsqueeze(1)?);
                }
                bwd_outputs.reverse();
                let bwd_refs: Vec<&Tensor> = bwd_outputs.iter().collect();
                let bwd_seq = Tensor::cat(&bwd_refs, 1)?;
                final_hiddens.push(bwd_h.unwrap().unsqueeze(0)?);

                let layer_out = Tensor::cat(&[&fwd_seq, &bwd_seq], 2)?;
                current_input = if let Some(ref drop) = self.dropout {
                    if layer < self.num_layers - 1 {
                        drop.forward(&layer_out)?
                    } else {
                        layer_out
                    }
                } else {
                    layer_out
                };
            } else {
                current_input = if let Some(ref drop) = self.dropout {
                    if layer < self.num_layers - 1 {
                        drop.forward(&fwd_seq)?
                    } else {
                        fwd_seq
                    }
                } else {
                    fwd_seq
                };
            }
        }

        let h_refs: Vec<&Tensor> = final_hiddens.iter().collect();
        let h_n = Tensor::cat(&h_refs, 0)?;

        Ok((current_input, h_n))
    }
}

impl Module for GRU {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (output, _) = self.forward_seq(input, None)?;
        Ok(output)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        for cell in &self.forward_cells {
            params.extend(cell.parameters());
        }
        for cell in &self.backward_cells {
            params.extend(cell.parameters());
        }
        params
    }

    fn train(&mut self) {
        if let Some(ref mut drop) = self.dropout {
            drop.train();
        }
    }

    fn eval(&mut self) {
        if let Some(ref mut drop) = self.dropout {
            drop.eval();
        }
    }
}
