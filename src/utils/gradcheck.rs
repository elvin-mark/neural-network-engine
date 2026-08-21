//! Finite-difference numerical gradient verification (gradcheck).

use crate::autograd::Tensor;
use crate::error::{EngineError, Result};
use crate::tensor::RawTensor;

/// Checks analytical autograd gradients against numerical finite-difference gradients.
///
/// Returns `Ok(max_rel_error)` if relative error <= tolerance, or `Err(EngineError::GradientError)` if mismatched.
pub fn gradcheck<F>(f: F, input: &Tensor, eps: f32, tol: f32) -> Result<f32>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    // 1. Compute analytical gradient via autograd
    input.zero_grad();
    input.set_requires_grad(true);
    let output = f(input)?;
    output.backward();

    let analytical_grad = input.grad().ok_or_else(|| {
        EngineError::GradientError("Input tensor did not receive any gradient".to_string())
    })?;

    let numel = input.numel();
    let mut numerical_grad_data = vec![0.0; numel];
    let input_shape = input.shape();

    // 2. Compute numerical gradient via central finite difference: (f(x + eps) - f(x - eps)) / (2 * eps)
    for i in 0..numel {
        // x + eps
        let mut x_plus_data = input.data();
        let orig_val = x_plus_data.get_by_flat_index(i);
        {
            let slice = x_plus_data.as_mut_slice();
            slice[i] = orig_val + eps;
        }
        let x_plus = Tensor::new(x_plus_data, false);
        let out_plus = f(&x_plus)?.item();

        // x - eps
        let mut x_minus_data = input.data();
        {
            let slice = x_minus_data.as_mut_slice();
            slice[i] = orig_val - eps;
        }
        let x_minus = Tensor::new(x_minus_data, false);
        let out_minus = f(&x_minus)?.item();

        let num_grad = (out_plus - out_minus) / (2.0 * eps);
        numerical_grad_data[i] = num_grad;
    }

    let numerical_grad = RawTensor::from_vec(numerical_grad_data, input_shape);

    // 3. Compute relative error: ||g_ana - g_num|| / max(||g_ana|| + ||g_num||, 1e-7)
    let diff = analytical_grad.sub(&numerical_grad)?;
    let diff_norm = diff.as_slice().iter().map(|&x| x * x).sum::<f32>().sqrt();

    let ana_norm = analytical_grad
        .as_slice()
        .iter()
        .map(|&x| x * x)
        .sum::<f32>()
        .sqrt();
    let num_norm = numerical_grad
        .as_slice()
        .iter()
        .map(|&x| x * x)
        .sum::<f32>()
        .sqrt();

    let denom = (ana_norm + num_norm).max(1e-7);
    let rel_error = diff_norm / denom;

    if rel_error > tol {
        return Err(EngineError::GradientError(format!(
            "Gradient check failed! Relative error: {:.6e} > tolerance: {:.6e}.\nAnalytical:\n{}\nNumerical:\n{}",
            rel_error, tol, analytical_grad, numerical_grad
        )));
    }

    Ok(rel_error)
}
