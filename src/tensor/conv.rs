//! 2D Spatial Convolution and Pooling primitives (im2col, col2im, max_pool2d, avg_pool2d).

use crate::error::{EngineError, Result};
use crate::tensor::matmul::gemm_2d_contiguous;
use crate::tensor::RawTensor;
use rayon::prelude::*;

/// 2D Convolution configuration parameters.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv2dParams {
    pub stride: (usize, usize),
    pub padding: (usize, usize),
    pub dilation: (usize, usize),
}

impl Default for Conv2dParams {
    fn default() -> Self {
        Self {
            stride: (1, 1),
            padding: (0, 0),
            dilation: (1, 1),
        }
    }
}

/// Computes the output spatial dimensions for 2D convolution.
pub fn conv2d_output_dims(
    h_in: usize,
    w_in: usize,
    k_h: usize,
    k_w: usize,
    params: Conv2dParams,
) -> Result<(usize, usize)> {
    let (s_h, s_w) = params.stride;
    let (p_h, p_w) = params.padding;
    let (d_h, d_w) = params.dilation;

    if s_h == 0 || s_w == 0 {
        return Err(EngineError::InvalidConvParams {
            details: "Stride must be greater than zero".to_string(),
        });
    }

    let eff_k_h = d_h * (k_h - 1) + 1;
    let eff_k_w = d_w * (k_w - 1) + 1;

    if h_in + 2 * p_h < eff_k_h || w_in + 2 * p_w < eff_k_w {
        return Err(EngineError::InvalidConvParams {
            details: format!(
                "Kernel size ({k_h}x{k_w}) with dilation ({d_h}x{d_w}) is too large for input ({h_in}x{w_in}) with padding ({p_h}x{p_w})"
            ),
        });
    }

    let h_out = (h_in + 2 * p_h - eff_k_h) / s_h + 1;
    let w_out = (w_in + 2 * p_w - eff_k_w) / s_w + 1;

    Ok((h_out, w_out))
}

/// Converts a single image (C_in, H, W) into column matrix (C_in * K_h * K_w, H_out * W_out).
#[allow(clippy::too_many_arguments)]
pub fn im2col(
    image: &[f32],
    c_in: usize,
    h_in: usize,
    w_in: usize,
    k_h: usize,
    k_w: usize,
    h_out: usize,
    w_out: usize,
    params: Conv2dParams,
    cols: &mut [f32],
) {
    let (s_h, s_w) = params.stride;
    let (p_h, p_w) = params.padding;
    let (d_h, d_w) = params.dilation;
    let col_width = h_out * w_out;

    for c in 0..c_in {
        let img_c_offset = c * h_in * w_in;
        for kh in 0..k_h {
            for kw in 0..k_w {
                let row_idx = (c * k_h + kh) * k_w + kw;
                let col_row_offset = row_idx * col_width;

                for out_h in 0..h_out {
                    let in_h = (out_h * s_h) as isize - p_h as isize + (kh * d_h) as isize;
                    let out_row_offset = out_h * w_out;

                    for out_w in 0..w_out {
                        let in_w = (out_w * s_w) as isize - p_w as isize + (kw * d_w) as isize;
                        let out_idx = col_row_offset + out_row_offset + out_w;

                        if in_h >= 0 && in_h < h_in as isize && in_w >= 0 && in_w < w_in as isize {
                            let img_idx = img_c_offset + (in_h as usize) * w_in + (in_w as usize);
                            cols[out_idx] = image[img_idx];
                        } else {
                            cols[out_idx] = 0.0;
                        }
                    }
                }
            }
        }
    }
}

/// Accumulates column matrix (C_in * K_h * K_w, H_out * W_out) back into image spatial gradient (C_in, H, W).
#[allow(clippy::too_many_arguments)]
pub fn col2im(
    cols: &[f32],
    c_in: usize,
    h_in: usize,
    w_in: usize,
    k_h: usize,
    k_w: usize,
    h_out: usize,
    w_out: usize,
    params: Conv2dParams,
    image_grad: &mut [f32],
) {
    let (s_h, s_w) = params.stride;
    let (p_h, p_w) = params.padding;
    let (d_h, d_w) = params.dilation;
    let col_width = h_out * w_out;

    for c in 0..c_in {
        let img_c_offset = c * h_in * w_in;
        for kh in 0..k_h {
            for kw in 0..k_w {
                let row_idx = (c * k_h + kh) * k_w + kw;
                let col_row_offset = row_idx * col_width;

                for out_h in 0..h_out {
                    let in_h = (out_h * s_h) as isize - p_h as isize + (kh * d_h) as isize;
                    let out_row_offset = out_h * w_out;

                    for out_w in 0..w_out {
                        let in_w = (out_w * s_w) as isize - p_w as isize + (kw * d_w) as isize;
                        if in_h >= 0 && in_h < h_in as isize && in_w >= 0 && in_w < w_in as isize {
                            let col_idx = col_row_offset + out_row_offset + out_w;
                            let img_idx = img_c_offset + (in_h as usize) * w_in + (in_w as usize);
                            image_grad[img_idx] += cols[col_idx];
                        }
                    }
                }
            }
        }
    }
}

/// Forward Conv2D pass computing output = weight * im2col(input) + bias.
pub fn conv2d_forward(
    input: &RawTensor,
    weight: &RawTensor,
    bias: Option<&RawTensor>,
    params: Conv2dParams,
) -> Result<RawTensor> {
    let in_shape = input.shape();
    let w_shape = weight.shape();

    if in_shape.len() != 4 || w_shape.len() != 4 {
        return Err(EngineError::IncompatibleShapes {
            op: "conv2d_forward (expected 4D inputs)",
            shapes: vec![in_shape.to_vec(), w_shape.to_vec()],
        });
    }

    let (batch, c_in, h_in, w_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
    let (c_out, w_cin, k_h, k_w) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);

    if c_in != w_cin {
        return Err(EngineError::IncompatibleShapes {
            op: "conv2d_forward (channel mismatch)",
            shapes: vec![in_shape.to_vec(), w_shape.to_vec()],
        });
    }

    let (h_out, w_out) = conv2d_output_dims(h_in, w_in, k_h, k_w, params)?;

    let input_contig = input.to_contiguous();
    let weight_contig = weight.to_contiguous();
    let in_slice = input_contig.as_slice();
    let w_slice = weight_contig.as_slice();

    let col_k = c_in * k_h * k_w;
    let col_n = h_out * w_out;
    let out_elements_per_batch = c_out * col_n;

    let mut out_data = vec![0.0; batch * out_elements_per_batch];

    // Parallelize over the batch items
    out_data
        .par_chunks_mut(out_elements_per_batch)
        .enumerate()
        .for_each(|(b, out_batch_slice)| {
            let img_offset = b * c_in * h_in * w_in;
            let img = &in_slice[img_offset..img_offset + c_in * h_in * w_in];

            let mut cols = vec![0.0; col_k * col_n];
            im2col(
                img, c_in, h_in, w_in, k_h, k_w, h_out, w_out, params, &mut cols,
            );

            // Matrix multiply: W (c_out, col_k) * cols (col_k, col_n) -> out (c_out, col_n)
            gemm_2d_contiguous(c_out, col_k, col_n, w_slice, &cols, out_batch_slice);
        });

    let mut out_tensor = RawTensor::from_vec(out_data, vec![batch, c_out, h_out, w_out]);

    if let Some(b) = bias {
        let b_reshaped = b.reshape(&[1, c_out, 1, 1])?;
        out_tensor = out_tensor.add(&b_reshaped)?;
    }

    Ok(out_tensor)
}

/// Backward Conv2D pass computing gradients w.r.t input, weight, and bias.
pub fn conv2d_backward(
    grad_output: &RawTensor,
    input: &RawTensor,
    weight: &RawTensor,
    params: Conv2dParams,
) -> Result<(RawTensor, RawTensor, Option<RawTensor>)> {
    let in_shape = input.shape();
    let w_shape = weight.shape();
    let grad_out_shape = grad_output.shape();

    let (batch, c_in, h_in, w_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
    let (c_out, _, k_h, k_w) = (w_shape[0], w_shape[1], w_shape[2], w_shape[3]);
    let (h_out, w_out) = (grad_out_shape[2], grad_out_shape[3]);

    let input_contig = input.to_contiguous();
    let weight_contig = weight.to_contiguous();
    let grad_out_contig = grad_output.to_contiguous();

    let in_slice = input_contig.as_slice();
    let w_slice = weight_contig.as_slice();
    let grad_out_slice = grad_out_contig.as_slice();

    let col_k = c_in * k_h * k_w;
    let col_n = h_out * w_out;

    let mut grad_input_data = vec![0.0; batch * c_in * h_in * w_in];
    let mut grad_weight_data = vec![0.0; c_out * col_k];
    let mut grad_bias_data = vec![0.0; c_out];

    // Transpose weight matrix (c_out, col_k) -> w_t (col_k, c_out)
    let mut w_t = vec![0.0; col_k * c_out];
    for r in 0..c_out {
        for c in 0..col_k {
            w_t[c * c_out + r] = w_slice[r * col_k + c];
        }
    }

    // Accumulate gradients across batch
    for b in 0..batch {
        let img_offset = b * c_in * h_in * w_in;
        let img = &in_slice[img_offset..img_offset + c_in * h_in * w_in];
        let grad_out_b = &grad_out_slice[b * c_out * col_n..(b + 1) * c_out * col_n];

        // 1. im2col of input
        let mut cols = vec![0.0; col_k * col_n];
        im2col(
            img, c_in, h_in, w_in, k_h, k_w, h_out, w_out, params, &mut cols,
        );

        // 2. grad_weight += grad_out_b (c_out, col_n) * cols_T (col_n, col_k)
        // Transpose cols to cols_t (col_n, col_k)
        let mut cols_t = vec![0.0; col_n * col_k];
        for r in 0..col_k {
            for c in 0..col_n {
                cols_t[c * col_k + r] = cols[r * col_n + c];
            }
        }
        let mut d_w_batch = vec![0.0; c_out * col_k];
        gemm_2d_contiguous(c_out, col_n, col_k, grad_out_b, &cols_t, &mut d_w_batch);
        for i in 0..grad_weight_data.len() {
            grad_weight_data[i] += d_w_batch[i];
        }

        // 3. grad_cols = w_t (col_k, c_out) * grad_out_b (c_out, col_n) -> (col_k, col_n)
        let mut grad_cols = vec![0.0; col_k * col_n];
        gemm_2d_contiguous(col_k, c_out, col_n, &w_t, grad_out_b, &mut grad_cols);

        // 4. col2im into grad_input
        let grad_in_b = &mut grad_input_data[img_offset..img_offset + c_in * h_in * w_in];
        col2im(
            &grad_cols, c_in, h_in, w_in, k_h, k_w, h_out, w_out, params, grad_in_b,
        );

        // 5. grad_bias accumulation
        for c in 0..c_out {
            let row = &grad_out_b[c * col_n..(c + 1) * col_n];
            let sum: f32 = row.iter().sum();
            grad_bias_data[c] += sum;
        }
    }

    let grad_input = RawTensor::from_vec(grad_input_data, in_shape.to_vec());
    let grad_weight = RawTensor::from_vec(grad_weight_data, w_shape.to_vec());
    let grad_bias = Some(RawTensor::from_vec(grad_bias_data, vec![c_out]));

    Ok((grad_input, grad_weight, grad_bias))
}

/// MaxPool2D forward and backward operations.
pub fn max_pool2d_forward(
    input: &RawTensor,
    kernel_size: (usize, usize),
    stride: (usize, usize),
) -> Result<(RawTensor, Vec<usize>)> {
    let shape = input.shape();
    if shape.len() != 4 {
        return Err(EngineError::IncompatibleShapes {
            op: "max_pool2d (expected 4D input)",
            shapes: vec![shape.to_vec()],
        });
    }

    let (batch, channels, h_in, w_in) = (shape[0], shape[1], shape[2], shape[3]);
    let (kh, kw) = kernel_size;
    let (sh, sw) = stride;

    let h_out = (h_in.saturating_sub(kh)) / sh + 1;
    let w_out = (w_in.saturating_sub(kw)) / sw + 1;

    let in_contig = input.to_contiguous();
    let in_slice = in_contig.as_slice();

    let out_numel = batch * channels * h_out * w_out;
    let mut out_data = vec![0.0; out_numel];
    let mut argmax_data = vec![0; out_numel];

    for b in 0..batch {
        for c in 0..channels {
            let in_bc_offset = (b * channels + c) * h_in * w_in;
            let out_bc_offset = (b * channels + c) * h_out * w_out;

            for oh in 0..h_out {
                let ih_start = oh * sh;
                for ow in 0..w_out {
                    let iw_start = ow * sw;

                    let mut max_val = f32::NEG_INFINITY;
                    let mut max_idx = 0;

                    for kh_i in 0..kh {
                        let ih = ih_start + kh_i;
                        for kw_i in 0..kw {
                            let iw = iw_start + kw_i;
                            let in_idx = in_bc_offset + ih * w_in + iw;
                            let val = in_slice[in_idx];
                            if val > max_val {
                                max_val = val;
                                max_idx = in_idx;
                            }
                        }
                    }

                    let out_idx = out_bc_offset + oh * w_out + ow;
                    out_data[out_idx] = max_val;
                    argmax_data[out_idx] = max_idx;
                }
            }
        }
    }

    Ok((
        RawTensor::from_vec(out_data, vec![batch, channels, h_out, w_out]),
        argmax_data,
    ))
}

/// MaxPool2D backward operation using stored argmax indices.
pub fn max_pool2d_backward(
    grad_output: &RawTensor,
    input_shape: &[usize],
    argmax_indices: &[usize],
) -> Result<RawTensor> {
    let grad_contig = grad_output.to_contiguous();
    let grad_slice = grad_contig.as_slice();

    let in_numel = input_shape.iter().product();
    let mut grad_input_data = vec![0.0; in_numel];

    for (out_idx, &in_idx) in argmax_indices.iter().enumerate() {
        grad_input_data[in_idx] += grad_slice[out_idx];
    }

    Ok(RawTensor::from_vec(grad_input_data, input_shape.to_vec()))
}
