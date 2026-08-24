//! Vision Data Augmentation and Preprocessing Pipeline.
//!
//! Provides:
//! - [`Transform`]: Core trait for spatial, photometric, and normalization image operations.
//! - [`RandomHorizontalFlip`]: Probabilistic horizontal image mirroring.
//! - [`RandomVerticalFlip`]: Probabilistic vertical image mirroring.
//! - [`RandomCrop`]: Spatial zero-padding followed by random rectangular window cropping.
//! - [`Normalize`]: Channel-wise mean subtraction and standard deviation scaling (with ImageNet/CIFAR presets).
//! - [`ColorJitter`]: Random brightness and contrast adjustment.
//! - [`RandomRotation90`]: Random orthogonal 90-degree rotations.
//! - [`Compose`]: Sequential pipeline container for chaining multiple transforms.

use crate::autograd::Tensor;
use crate::error::{EngineError, Result};
use crate::tensor::RawTensor;
use rand::Rng;

/// Common trait for all computer vision data augmentation and preprocessing transforms.
pub trait Transform: Send + Sync {
    /// Applies the transformation to an unbatched [C, H, W] or batched [B, C, H, W] raw tensor.
    fn apply_raw(&self, tensor: &RawTensor) -> Result<RawTensor>;

    /// Applies the transformation to an autograd Tensor.
    fn apply(&self, tensor: &Tensor) -> Result<Tensor> {
        let raw_out = self.apply_raw(&tensor.data())?;
        Ok(Tensor::new(raw_out, tensor.requires_grad()))
    }
}

/// Randomly flips the image horizontally with probability `p` (default 0.5).
#[derive(Debug, Clone)]
pub struct RandomHorizontalFlip {
    pub p: f32,
}

impl RandomHorizontalFlip {
    pub fn new(p: f32) -> Self {
        Self { p }
    }

    pub fn default_prob() -> Self {
        Self { p: 0.5 }
    }
}

impl Transform for RandomHorizontalFlip {
    fn apply_raw(&self, tensor: &RawTensor) -> Result<RawTensor> {
        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() >= self.p {
            return Ok(tensor.clone());
        }

        let shape = tensor.shape();
        let ndim = shape.len();
        if ndim < 2 {
            return Err(EngineError::InvalidArgument(format!(
                "RandomHorizontalFlip expects at least 2D image tensor, got {:?}",
                shape
            )));
        }

        let w = shape[ndim - 1];
        let h = shape[ndim - 2];
        let prefix_len: usize = shape[..ndim - 2].iter().product();
        let contig = tensor.to_contiguous();
        let src = contig.as_slice();

        let mut out = vec![0.0f32; src.len()];
        let plane_size = h * w;

        for p in 0..prefix_len {
            let p_off = p * plane_size;
            for row in 0..h {
                let r_off = p_off + row * w;
                for col in 0..w {
                    let flipped_col = w - 1 - col;
                    out[r_off + col] = src[r_off + flipped_col];
                }
            }
        }

        Ok(RawTensor::from_vec(out, shape.to_vec()))
    }
}

/// Randomly flips the image vertically with probability `p` (default 0.5).
#[derive(Debug, Clone)]
pub struct RandomVerticalFlip {
    pub p: f32,
}

impl RandomVerticalFlip {
    pub fn new(p: f32) -> Self {
        Self { p }
    }

    pub fn default_prob() -> Self {
        Self { p: 0.5 }
    }
}

impl Transform for RandomVerticalFlip {
    fn apply_raw(&self, tensor: &RawTensor) -> Result<RawTensor> {
        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() >= self.p {
            return Ok(tensor.clone());
        }

        let shape = tensor.shape();
        let ndim = shape.len();
        if ndim < 2 {
            return Err(EngineError::InvalidArgument(format!(
                "RandomVerticalFlip expects at least 2D image tensor, got {:?}",
                shape
            )));
        }

        let w = shape[ndim - 1];
        let h = shape[ndim - 2];
        let prefix_len: usize = shape[..ndim - 2].iter().product();
        let contig = tensor.to_contiguous();
        let src = contig.as_slice();

        let mut out = vec![0.0f32; src.len()];
        let plane_size = h * w;

        for p in 0..prefix_len {
            let p_off = p * plane_size;
            for row in 0..h {
                let flipped_row = h - 1 - row;
                for col in 0..w {
                    out[p_off + row * w + col] = src[p_off + flipped_row * w + col];
                }
            }
        }

        Ok(RawTensor::from_vec(out, shape.to_vec()))
    }
}

/// Zero-pads image borders by `padding` pixels and extracts a random `[target_h, target_w]` crop.
#[derive(Debug, Clone)]
pub struct RandomCrop {
    pub target_h: usize,
    pub target_w: usize,
    pub padding: usize,
}

impl RandomCrop {
    pub fn new(target_h: usize, target_w: usize, padding: usize) -> Self {
        Self {
            target_h,
            target_w,
            padding,
        }
    }
}

impl Transform for RandomCrop {
    fn apply_raw(&self, tensor: &RawTensor) -> Result<RawTensor> {
        let shape = tensor.shape();
        let ndim = shape.len();
        if ndim < 2 {
            return Err(EngineError::InvalidArgument(format!(
                "RandomCrop expects at least 2D image tensor, got {:?}",
                shape
            )));
        }

        let orig_w = shape[ndim - 1];
        let orig_h = shape[ndim - 2];
        let pad = self.padding;

        let padded_h = orig_h + 2 * pad;
        let padded_w = orig_w + 2 * pad;

        if padded_h < self.target_h || padded_w < self.target_w {
            return Err(EngineError::InvalidArgument(format!(
                "Padded size [{}, {}] is smaller than target crop size [{}, {}]",
                padded_h, padded_w, self.target_h, self.target_w
            )));
        }

        let prefix_len: usize = shape[..ndim - 2].iter().product();
        let contig = tensor.to_contiguous();
        let src = contig.as_slice();

        let mut rng = rand::thread_rng();
        let offset_y = rng.gen_range(0..=(padded_h - self.target_h));
        let offset_x = rng.gen_range(0..=(padded_w - self.target_w));

        let mut out_shape = shape.to_vec();
        out_shape[ndim - 2] = self.target_h;
        out_shape[ndim - 1] = self.target_w;
        let out_len = prefix_len * self.target_h * self.target_w;
        let mut out = vec![0.0f32; out_len];

        let in_plane = orig_h * orig_w;
        let out_plane = self.target_h * self.target_w;

        for p in 0..prefix_len {
            let src_p_off = p * in_plane;
            let dst_p_off = p * out_plane;

            for r in 0..self.target_h {
                let padded_r = offset_y + r;
                for c in 0..self.target_w {
                    let padded_c = offset_x + c;

                    // Check if inside original unpadded bounds
                    if padded_r >= pad
                        && padded_r < pad + orig_h
                        && padded_c >= pad
                        && padded_c < pad + orig_w
                    {
                        let orig_r = padded_r - pad;
                        let orig_c = padded_c - pad;
                        out[dst_p_off + r * self.target_w + c] =
                            src[src_p_off + orig_r * orig_w + orig_c];
                    } else {
                        // Padding pixel (zero)
                        out[dst_p_off + r * self.target_w + c] = 0.0;
                    }
                }
            }
        }

        Ok(RawTensor::from_vec(out, out_shape))
    }
}

/// Normalizes a tensor with per-channel mean and standard deviation:
/// $$x_c = \frac{x_c - \mu_c}{\sigma_c}$$
#[derive(Debug, Clone)]
pub struct Normalize {
    pub mean: Vec<f32>,
    pub std: Vec<f32>,
}

impl Normalize {
    pub fn new(mean: Vec<f32>, std: Vec<f32>) -> Self {
        assert_eq!(
            mean.len(),
            std.len(),
            "mean and std lengths must match channel count"
        );
        Self { mean, std }
    }

    /// Standard ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225].
    pub fn imagenet() -> Self {
        Self::new(vec![0.485, 0.456, 0.406], vec![0.229, 0.224, 0.225])
    }

    /// Standard CIFAR-10 normalization: mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010].
    pub fn cifar10() -> Self {
        Self::new(vec![0.4914, 0.4822, 0.4465], vec![0.2023, 0.1994, 0.2010])
    }

    /// Standard CIFAR-100 normalization: mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761].
    pub fn cifar100() -> Self {
        Self::new(vec![0.5071, 0.4867, 0.4408], vec![0.2675, 0.2565, 0.2761])
    }
}

impl Transform for Normalize {
    fn apply_raw(&self, tensor: &RawTensor) -> Result<RawTensor> {
        let shape = tensor.shape();
        let ndim = shape.len();
        if ndim < 3 {
            return Err(EngineError::InvalidArgument(format!(
                "Normalize expects at least 3D tensor [Channels, H, W] or [Batch, Channels, H, W], got {:?}",
                shape
            )));
        }

        let num_channels = shape[ndim - 3];
        if num_channels != self.mean.len() {
            return Err(EngineError::ShapeMismatch {
                expected: vec![self.mean.len()],
                actual: vec![num_channels],
            });
        }

        let h = shape[ndim - 2];
        let w = shape[ndim - 1];
        let spatial_size = h * w;
        let batch_size: usize = shape[..ndim - 3].iter().product();

        let contig = tensor.to_contiguous();
        let src = contig.as_slice();
        let mut out = vec![0.0f32; src.len()];

        for b in 0..batch_size {
            let b_off = b * num_channels * spatial_size;
            for c in 0..num_channels {
                let c_off = b_off + c * spatial_size;
                let m = self.mean[c];
                let inv_s = 1.0 / self.std[c];
                for i in 0..spatial_size {
                    out[c_off + i] = (src[c_off + i] - m) * inv_s;
                }
            }
        }

        Ok(RawTensor::from_vec(out, shape.to_vec()))
    }
}

/// Randomly shifts pixel values by brightness and contrast factors.
#[derive(Debug, Clone)]
pub struct ColorJitter {
    pub brightness: f32,
    pub contrast: f32,
}

impl ColorJitter {
    pub fn new(brightness: f32, contrast: f32) -> Self {
        Self {
            brightness,
            contrast,
        }
    }
}

impl Transform for ColorJitter {
    fn apply_raw(&self, tensor: &RawTensor) -> Result<RawTensor> {
        let mut rng = rand::thread_rng();
        let b_factor = 1.0 + rng.gen_range(-self.brightness..=self.brightness);
        let c_factor = 1.0 + rng.gen_range(-self.contrast..=self.contrast);

        let contig = tensor.to_contiguous();
        let src = contig.as_slice();
        let mut out = Vec::with_capacity(src.len());

        let mean = src.iter().sum::<f32>() / (src.len() as f32);

        for &val in src {
            // Apply contrast around mean, then brightness
            let adjusted = (val - mean) * c_factor + mean;
            out.push(adjusted * b_factor);
        }

        Ok(RawTensor::from_vec(out, tensor.shape().to_vec()))
    }
}

/// Randomly rotates image spatial dimensions by 0, 90, 180, or 270 degrees.
#[derive(Debug, Clone)]
pub struct RandomRotation90 {
    pub p: f32,
}

impl RandomRotation90 {
    pub fn new(p: f32) -> Self {
        Self { p }
    }

    pub fn default_prob() -> Self {
        Self { p: 0.5 }
    }
}

impl Transform for RandomRotation90 {
    fn apply_raw(&self, tensor: &RawTensor) -> Result<RawTensor> {
        let mut rng = rand::thread_rng();
        if rng.gen::<f32>() >= self.p {
            return Ok(tensor.clone());
        }

        let k = rng.gen_range(1..=3); // 1 = 90 deg, 2 = 180 deg, 3 = 270 deg
        let shape = tensor.shape();
        let ndim = shape.len();
        if ndim < 2 {
            return Err(EngineError::InvalidArgument(format!(
                "RandomRotation90 expects at least 2D image tensor, got {:?}",
                shape
            )));
        }

        let w = shape[ndim - 1];
        let h = shape[ndim - 2];
        let prefix_len: usize = shape[..ndim - 2].iter().product();
        let contig = tensor.to_contiguous();
        let src = contig.as_slice();

        let (out_h, out_w) = if k % 2 == 1 { (w, h) } else { (h, w) };
        let mut out_shape = shape.to_vec();
        out_shape[ndim - 2] = out_h;
        out_shape[ndim - 1] = out_w;
        let mut out = vec![0.0f32; src.len()];

        let in_plane = h * w;
        let out_plane = out_h * out_w;

        for p in 0..prefix_len {
            let src_off = p * in_plane;
            let dst_off = p * out_plane;

            for r in 0..h {
                for c in 0..w {
                    let val = src[src_off + r * w + c];
                    let (dst_r, dst_c) = match k {
                        1 => (c, h - 1 - r),         // 90 deg clockwise
                        2 => (h - 1 - r, w - 1 - c), // 180 deg
                        3 => (w - 1 - c, r),         // 270 deg
                        _ => (r, c),
                    };
                    out[dst_off + dst_r * out_w + dst_c] = val;
                }
            }
        }

        Ok(RawTensor::from_vec(out, out_shape))
    }
}

/// Composable container that applies a sequence of [`Transform`] operations in order.
pub struct Compose {
    pub transforms: Vec<Box<dyn Transform>>,
}

impl Compose {
    pub fn new(transforms: Vec<Box<dyn Transform>>) -> Self {
        Self { transforms }
    }
}

impl Transform for Compose {
    fn apply_raw(&self, tensor: &RawTensor) -> Result<RawTensor> {
        let mut current = tensor.clone();
        for t in &self.transforms {
            current = t.apply_raw(&current)?;
        }
        Ok(current)
    }

    fn apply(&self, tensor: &Tensor) -> Result<Tensor> {
        let mut current = tensor.clone();
        for t in &self.transforms {
            current = t.apply(&current)?;
        }
        Ok(current)
    }
}
