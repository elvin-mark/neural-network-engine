//! Weight and parameter initialization strategies for neural networks.
//!
//! Includes mathematically principled initializations:
//! - **Xavier / Glorot** (Uniform and Normal) for linear, sigmoid, tanh activations.
//! - **Kaiming / He** (Uniform and Normal) for ReLU, GELU, SiLU activations.
//! - **Orthogonal** initialization via QR/Gram-Schmidt for deep layers and attention projections.
//! - Constant, Uniform, Normal, and Gain calculations.

use crate::error::{EngineError, Result};
use crate::tensor::RawTensor;
use rand_distr::{Distribution, Normal as RandNormal, Uniform as RandUniform};

/// Non-linear activation function type for calculating optimal initialization gains.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NonLinearity {
    Linear,
    Identity,
    Sigmoid,
    Tanh,
    ReLU,
    LeakyReLU(f32),
    SiLU,
    GELU,
}

/// Mode for calculating fan in / fan out in Kaiming initialization.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FanMode {
    FanIn,
    FanOut,
}

/// Calculates the recommended gain value for a given non-linear activation.
pub fn calculate_gain(non_linearity: NonLinearity) -> f32 {
    match non_linearity {
        NonLinearity::Linear | NonLinearity::Identity | NonLinearity::Sigmoid => 1.0,
        NonLinearity::Tanh => 5.0 / 3.0,
        NonLinearity::ReLU | NonLinearity::SiLU | NonLinearity::GELU => 2.0f32.sqrt(),
        NonLinearity::LeakyReLU(negative_slope) => {
            (2.0 / (1.0 + negative_slope * negative_slope)).sqrt()
        }
    }
}

/// Calculates the number of input and output units for a tensor shape.
pub fn calculate_fan_in_and_fan_out(shape: &[usize]) -> (usize, usize) {
    if shape.is_empty() {
        return (1, 1);
    }
    if shape.len() == 1 {
        return (shape[0], shape[0]);
    }
    if shape.len() == 2 {
        // [out_features, in_features] or [in_features, out_features]
        return (shape[1], shape[0]);
    }
    // For Conv2D: [out_channels, in_channels, kH, kW]
    let receptive_field: usize = shape[2..].iter().product();
    let fan_in = shape[1] * receptive_field;
    let fan_out = shape[0] * receptive_field;
    (fan_in, fan_out)
}

/// Fills an existing tensor with values from a uniform distribution `[low, high)`.
pub fn uniform_(tensor: &mut RawTensor, low: f32, high: f32) {
    let mut rng = rand::thread_rng();
    let dist = RandUniform::new(low, high);
    let slice = tensor.as_mut_slice();
    for val in slice.iter_mut() {
        *val = dist.sample(&mut rng);
    }
}

/// Creates a new tensor filled with values from a uniform distribution `[low, high)`.
pub fn uniform(shape: &[usize], low: f32, high: f32) -> RawTensor {
    let mut t = RawTensor::zeros(shape);
    uniform_(&mut t, low, high);
    t
}

/// Fills an existing tensor with values from a normal distribution `N(mean, std^2)`.
pub fn normal_(tensor: &mut RawTensor, mean: f32, std: f32) {
    let mut rng = rand::thread_rng();
    let dist = RandNormal::new(mean, std).expect("Standard deviation must be non-negative");
    let slice = tensor.as_mut_slice();
    for val in slice.iter_mut() {
        *val = dist.sample(&mut rng);
    }
}

/// Creates a new tensor filled with values from a normal distribution `N(mean, std^2)`.
pub fn normal(shape: &[usize], mean: f32, std: f32) -> RawTensor {
    let mut t = RawTensor::zeros(shape);
    normal_(&mut t, mean, std);
    t
}

/// Fills an existing tensor with a constant value.
pub fn constant_(tensor: &mut RawTensor, val: f32) {
    let slice = tensor.as_mut_slice();
    for elem in slice.iter_mut() {
        *elem = val;
    }
}

/// Creates a new tensor filled with a constant value.
pub fn constant(shape: &[usize], val: f32) -> RawTensor {
    let mut t = RawTensor::zeros(shape);
    constant_(&mut t, val);
    t
}

/// Fills an existing tensor with zeros.
pub fn zeros_(tensor: &mut RawTensor) {
    constant_(tensor, 0.0);
}

/// Fills an existing tensor with ones.
pub fn ones_(tensor: &mut RawTensor) {
    constant_(tensor, 1.0);
}

/// Xavier (Glorot) Uniform initialization in-place.
/// Values sampled from $U(-a, a)$ where $a = \text{gain} \times \sqrt{\frac{6}{\text{fan\_in} + \text{fan\_out}}}$.
pub fn xavier_uniform_(tensor: &mut RawTensor, gain: f32) {
    let (fan_in, fan_out) = calculate_fan_in_and_fan_out(tensor.shape());
    let std = gain * (2.0 / (fan_in + fan_out) as f32).sqrt();
    let a = 3.0f32.sqrt() * std;
    uniform_(tensor, -a, a);
}

/// Creates a new tensor with Xavier (Glorot) Uniform initialization.
pub fn xavier_uniform(shape: &[usize], gain: f32) -> RawTensor {
    let mut t = RawTensor::zeros(shape);
    xavier_uniform_(&mut t, gain);
    t
}

/// Xavier (Glorot) Normal initialization in-place.
/// Values sampled from $N(0, \text{std}^2)$ where $\text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}$.
pub fn xavier_normal_(tensor: &mut RawTensor, gain: f32) {
    let (fan_in, fan_out) = calculate_fan_in_and_fan_out(tensor.shape());
    let std = gain * (2.0 / (fan_in + fan_out) as f32).sqrt();
    normal_(tensor, 0.0, std);
}

/// Creates a new tensor with Xavier (Glorot) Normal initialization.
pub fn xavier_normal(shape: &[usize], gain: f32) -> RawTensor {
    let mut t = RawTensor::zeros(shape);
    xavier_normal_(&mut t, gain);
    t
}

/// Kaiming (He) Uniform initialization in-place.
/// Values sampled from $U(-bound, bound)$ where $bound = \text{gain} \times \sqrt{\frac{3}{\text{fan}}}$.
pub fn kaiming_uniform_(
    tensor: &mut RawTensor,
    a: f32,
    mode: FanMode,
    non_linearity: NonLinearity,
) {
    let (fan_in, fan_out) = calculate_fan_in_and_fan_out(tensor.shape());
    let fan = match mode {
        FanMode::FanIn => fan_in,
        FanMode::FanOut => fan_out,
    };
    let gain = match non_linearity {
        NonLinearity::LeakyReLU(_) => calculate_gain(NonLinearity::LeakyReLU(a)),
        other => calculate_gain(other),
    };
    let std = gain / (fan as f32).sqrt();
    let bound = 3.0f32.sqrt() * std;
    uniform_(tensor, -bound, bound);
}

/// Creates a new tensor with Kaiming (He) Uniform initialization.
pub fn kaiming_uniform(
    shape: &[usize],
    a: f32,
    mode: FanMode,
    non_linearity: NonLinearity,
) -> RawTensor {
    let mut t = RawTensor::zeros(shape);
    kaiming_uniform_(&mut t, a, mode, non_linearity);
    t
}

/// Kaiming (He) Normal initialization in-place.
/// Values sampled from $N(0, \text{std}^2)$ where $\text{std} = \frac{\text{gain}}{\sqrt{\text{fan}}}$.
pub fn kaiming_normal_(tensor: &mut RawTensor, a: f32, mode: FanMode, non_linearity: NonLinearity) {
    let (fan_in, fan_out) = calculate_fan_in_and_fan_out(tensor.shape());
    let fan = match mode {
        FanMode::FanIn => fan_in,
        FanMode::FanOut => fan_out,
    };
    let gain = match non_linearity {
        NonLinearity::LeakyReLU(_) => calculate_gain(NonLinearity::LeakyReLU(a)),
        other => calculate_gain(other),
    };
    let std = gain / (fan as f32).sqrt();
    normal_(tensor, 0.0, std);
}

/// Creates a new tensor with Kaiming (He) Normal initialization.
pub fn kaiming_normal(
    shape: &[usize],
    a: f32,
    mode: FanMode,
    non_linearity: NonLinearity,
) -> RawTensor {
    let mut t = RawTensor::zeros(shape);
    kaiming_normal_(&mut t, a, mode, non_linearity);
    t
}

/// Orthogonal initialization in-place via Gram-Schmidt QR decomposition.
pub fn orthogonal_(tensor: &mut RawTensor, gain: f32) -> Result<()> {
    let shape = tensor.shape();
    if shape.len() < 2 {
        return Err(EngineError::InvalidArgument(
            "Orthogonal initialization requires a tensor with at least 2 dimensions".to_string(),
        ));
    }
    let rows = shape[0];
    let cols: usize = shape[1..].iter().product();

    // Generate random Gaussian matrix of size [rows, cols]
    let mut matrix = vec![0.0f32; rows * cols];
    let mut rng = rand::thread_rng();
    let dist = RandNormal::new(0.0, 1.0).unwrap();
    for val in matrix.iter_mut() {
        *val = dist.sample(&mut rng);
    }

    // Perform Gram-Schmidt orthogonalization
    let (m, n) = (rows, cols);
    if m >= n {
        // Orthogonalize columns
        for j in 0..n {
            for k in 0..j {
                let mut dot = 0.0f32;
                for i in 0..m {
                    dot += matrix[i * n + j] * matrix[i * n + k];
                }
                for i in 0..m {
                    matrix[i * n + j] -= dot * matrix[i * n + k];
                }
            }
            let mut norm = 0.0f32;
            for i in 0..m {
                norm += matrix[i * n + j] * matrix[i * n + j];
            }
            let norm = norm.sqrt().max(1e-8);
            for i in 0..m {
                matrix[i * n + j] = (matrix[i * n + j] / norm) * gain;
            }
        }
    } else {
        // Orthogonalize rows
        for i in 0..m {
            for k in 0..i {
                let mut dot = 0.0f32;
                for j in 0..n {
                    dot += matrix[i * n + j] * matrix[k * n + j];
                }
                for j in 0..n {
                    matrix[i * n + j] -= dot * matrix[k * n + j];
                }
            }
            let mut norm = 0.0f32;
            for j in 0..m {
                norm += matrix[i * n + j] * matrix[i * n + j];
            }
            let norm = norm.sqrt().max(1e-8);
            for j in 0..m {
                matrix[i * n + j] = (matrix[i * n + j] / norm) * gain;
            }
        }
    }

    let slice = tensor.as_mut_slice();
    slice.copy_from_slice(&matrix[..slice.len()]);
    Ok(())
}

/// Creates a new tensor with Orthogonal initialization.
pub fn orthogonal(shape: &[usize], gain: f32) -> Result<RawTensor> {
    let mut t = RawTensor::zeros(shape);
    orthogonal_(&mut t, gain)?;
    Ok(t)
}
