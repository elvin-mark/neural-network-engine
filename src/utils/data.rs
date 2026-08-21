//! Dataset loaders, batching, shuffling, and synthetic data generation.

use crate::autograd::Tensor;
use crate::error::{EngineError, Result};
use crate::tensor::RawTensor;
use rand::seq::SliceRandom;
use rand::Rng;

/// In-memory dataset holding feature and target tensors.
pub struct TensorDataset {
    pub features: RawTensor,
    pub targets: RawTensor,
    pub length: usize,
}

impl TensorDataset {
    pub fn new(features: RawTensor, targets: RawTensor) -> Result<Self> {
        let f_len = features.shape().first().copied().unwrap_or(0);
        let t_len = targets.shape().first().copied().unwrap_or(0);

        if f_len != t_len {
            return Err(EngineError::ShapeMismatch {
                expected: vec![f_len],
                actual: vec![t_len],
            });
        }

        Ok(Self {
            features,
            targets,
            length: f_len,
        })
    }
}

/// Mini-batch data loader with optional shuffling.
pub struct DataLoader<'a> {
    pub dataset: &'a TensorDataset,
    pub batch_size: usize,
    pub shuffle: bool,
    pub indices: Vec<usize>,
    pub current_idx: usize,
}

impl<'a> DataLoader<'a> {
    pub fn new(dataset: &'a TensorDataset, batch_size: usize, shuffle: bool) -> Self {
        let mut indices: Vec<usize> = (0..dataset.length).collect();
        if shuffle {
            indices.shuffle(&mut rand::thread_rng());
        }

        Self {
            dataset,
            batch_size,
            shuffle,
            indices,
            current_idx: 0,
        }
    }

    /// Resets the loader and reshuffles if enabled.
    pub fn reset(&mut self) {
        self.current_idx = 0;
        if self.shuffle {
            self.indices.shuffle(&mut rand::thread_rng());
        }
    }
}

impl<'a> Iterator for DataLoader<'a> {
    type Item = (Tensor, Tensor);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_idx >= self.dataset.length {
            return None;
        }

        let end_idx = (self.current_idx + self.batch_size).min(self.dataset.length);
        let batch_indices = &self.indices[self.current_idx..end_idx];
        let actual_batch_size = batch_indices.len();

        let f_shape = self.dataset.features.shape();
        let t_shape = self.dataset.targets.shape();

        let mut b_f_shape = f_shape.to_vec();
        b_f_shape[0] = actual_batch_size;
        let mut b_t_shape = t_shape.to_vec();
        b_t_shape[0] = actual_batch_size;

        let f_sample_size: usize = f_shape[1..].iter().product();
        let t_sample_size: usize = t_shape[1..].iter().product();

        let f_slice = self.dataset.features.as_slice();
        let t_slice = self.dataset.targets.as_slice();

        let mut b_f_data = vec![0.0; actual_batch_size * f_sample_size];
        let mut b_t_data = vec![0.0; actual_batch_size * t_sample_size];

        for (i, &idx) in batch_indices.iter().enumerate() {
            let f_src = idx * f_sample_size;
            let f_dst = i * f_sample_size;
            b_f_data[f_dst..f_dst + f_sample_size]
                .copy_from_slice(&f_slice[f_src..f_src + f_sample_size]);

            let t_src = idx * t_sample_size;
            let t_dst = i * t_sample_size;
            b_t_data[t_dst..t_dst + t_sample_size]
                .copy_from_slice(&t_slice[t_src..t_src + t_sample_size]);
        }

        self.current_idx = end_idx;

        Some((
            Tensor::new(RawTensor::from_vec(b_f_data, b_f_shape), false),
            Tensor::new(RawTensor::from_vec(b_t_data, b_t_shape), false),
        ))
    }
}

/// Generates synthetic non-linear 2D spiral classification dataset.
pub fn generate_spiral_dataset(
    points_per_arm: usize,
    num_arms: usize,
    noise: f32,
) -> (RawTensor, Vec<usize>) {
    let total_points = points_per_arm * num_arms;
    let mut x_data = vec![0.0; total_points * 2];
    let mut labels = vec![0; total_points];
    let mut rng = rand::thread_rng();

    for j in 0..num_arms {
        for i in 0..points_per_arm {
            let r = (i as f32) / (points_per_arm as f32); // radius
            let theta = (j as f32) * 4.0 + (r * 4.0) + (rng.gen::<f32>() - 0.5) * noise * 2.0;

            let idx = j * points_per_arm + i;
            x_data[idx * 2] = r * (theta).sin();
            x_data[idx * 2 + 1] = r * (theta).cos();
            labels[idx] = j;
        }
    }

    (RawTensor::from_vec(x_data, vec![total_points, 2]), labels)
}

/// Generates synthetic 2D XOR classification dataset.
pub fn generate_xor_dataset(num_points: usize, noise: f32) -> (RawTensor, Vec<usize>) {
    let mut x_data = vec![0.0; num_points * 2];
    let mut labels = vec![0; num_points];
    let mut rng = rand::thread_rng();

    for i in 0..num_points {
        let x1 = if rng.gen::<bool>() { 1.0 } else { -1.0 } + (rng.gen::<f32>() - 0.5) * noise;
        let x2 = if rng.gen::<bool>() { 1.0 } else { -1.0 } + (rng.gen::<f32>() - 0.5) * noise;

        x_data[i * 2] = x1;
        x_data[i * 2 + 1] = x2;

        let label = if (x1 > 0.0) ^ (x2 > 0.0) { 1 } else { 0 };
        labels[i] = label;
    }

    (RawTensor::from_vec(x_data, vec![num_points, 2]), labels)
}
