//! Gradient clipping utilities to prevent exploding gradients and stabilize neural network training.
//!
//! Provides:
//! - `clip_grad_norm`: Global $L_2$ norm clipping across all parameter tensors ($\|\mathbf{g}\| = \sqrt{\sum \|\mathbf{g}_i\|_2^2}$).
//! - `clip_grad_value`: Element-wise clamping of gradients to $[-c, +c]$.

use crate::autograd::Tensor;

/// Clips gradient norm of an iterable of parameters.
///
/// The norm is computed over all gradients together, as if they were
/// concatenated into a single vector. Gradients are modified in-place.
///
/// # Arguments
/// * `parameters` - Slice of parameter `Tensor`s whose gradients should be clipped.
/// * `max_norm` - Maximum allowable norm of the gradients.
///
/// # Returns
/// The total unclipped $L_2$ norm of the parameters (viewed as a single vector).
pub fn clip_grad_norm(parameters: &[Tensor], max_norm: f32) -> f32 {
    assert!(max_norm > 0.0, "max_norm must be strictly positive");

    let mut total_norm_sq = 0.0f32;

    for param in parameters {
        if let Some(grad) = param.grad() {
            let slice = grad.to_contiguous();
            for &val in slice.as_slice() {
                total_norm_sq += val * val;
            }
        }
    }

    let total_norm = total_norm_sq.sqrt();
    let clip_coef = max_norm / (total_norm + 1e-6);

    if clip_coef < 1.0 {
        for param in parameters {
            if let Some(grad) = param.grad() {
                let mut grad_mut = grad;
                let slice = grad_mut.as_mut_slice();
                for val in slice.iter_mut() {
                    *val *= clip_coef;
                }
                param.set_grad(Some(grad_mut));
            }
        }
    }

    total_norm
}

/// Clips the values of gradients of an iterable of parameters element-wise.
///
/// Gradients are modified in-place to lie within $[-\text{clip\_value}, \text{clip\_value}]$.
///
/// # Arguments
/// * `parameters` - Slice of parameter `Tensor`s whose gradients should be clipped.
/// * `clip_value` - Maximum absolute value allowed for any individual gradient element.
pub fn clip_grad_value(parameters: &[Tensor], clip_value: f32) {
    assert!(clip_value > 0.0, "clip_value must be strictly positive");

    for param in parameters {
        if let Some(grad) = param.grad() {
            let mut grad_mut = grad;
            let slice = grad_mut.as_mut_slice();
            for val in slice.iter_mut() {
                *val = val.clamp(-clip_value, clip_value);
            }
            param.set_grad(Some(grad_mut));
        }
    }
}
