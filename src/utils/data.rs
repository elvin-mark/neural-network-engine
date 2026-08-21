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
        let f_contig = features.to_contiguous();
        let t_contig = targets.to_contiguous();
        let f_len = f_contig.shape().first().copied().unwrap_or(0);
        let t_len = t_contig.shape().first().copied().unwrap_or(0);

        if f_len != t_len {
            return Err(EngineError::ShapeMismatch {
                expected: vec![f_len],
                actual: vec![t_len],
            });
        }

        Ok(Self {
            features: f_contig,
            targets: t_contig,
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
        assert!(
            batch_size > 0,
            "DataLoader batch_size must be greater than 0"
        );
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

use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Loads Fisher's Iris dataset from a CSV/data file.
/// Format per line: `sepal_length,sepal_width,petal_length,petal_width,class_name`
pub fn load_iris_from_csv<P: AsRef<Path>>(path: P) -> Result<(RawTensor, Vec<usize>)> {
    let file = File::open(&path).map_err(|e| {
        EngineError::SerializationError(format!(
            "Failed to open Iris dataset file '{}': {}",
            path.as_ref().display(),
            e
        ))
    })?;
    let reader = BufReader::new(file);

    let mut features = Vec::new();
    let mut labels = Vec::new();

    for (line_num, line_res) in reader.lines().enumerate() {
        let line = line_res.map_err(|e| {
            EngineError::SerializationError(format!("Error reading line {}: {}", line_num + 1, e))
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let parts: Vec<&str> = trimmed.split(',').map(|s| s.trim()).collect();
        if parts.len() < 5 {
            return Err(EngineError::SerializationError(format!(
                "Invalid Iris line {} in '{}': expected 5 fields, found {}",
                line_num + 1,
                path.as_ref().display(),
                parts.len()
            )));
        }

        for p in &parts[..4] {
            let val: f32 = p.parse().map_err(|e| {
                EngineError::SerializationError(format!(
                    "Failed to parse float '{}' at line {}: {}",
                    p,
                    line_num + 1,
                    e
                ))
            })?;
            features.push(val);
        }

        let class_str = parts[4].to_lowercase();
        let label = if class_str.contains("setosa") || class_str == "0" {
            0
        } else if class_str.contains("versicolor") || class_str == "1" {
            1
        } else if class_str.contains("virginica") || class_str == "2" {
            2
        } else {
            return Err(EngineError::SerializationError(format!(
                "Unknown Iris class '{}' at line {}",
                parts[4],
                line_num + 1
            )));
        };
        labels.push(label);
    }

    let n = labels.len();
    if n == 0 {
        return Err(EngineError::SerializationError(format!(
            "Iris dataset file '{}' is empty",
            path.as_ref().display()
        )));
    }

    Ok((RawTensor::from_vec(features, vec![n, 4]), labels))
}

/// Loads Optical Handwritten Digits dataset from a CSV/data file (e.g. `optdigits.tra` or `optdigits.tes`).
/// Format per line: 64 comma-separated pixel values (0..16) followed by 1 class label (0..9).
pub fn load_digits_from_csv<P: AsRef<Path>>(
    path: P,
    max_samples: Option<usize>,
) -> Result<(RawTensor, Vec<usize>)> {
    let file = File::open(&path).map_err(|e| {
        EngineError::SerializationError(format!(
            "Failed to open Digits dataset file '{}': {}",
            path.as_ref().display(),
            e
        ))
    })?;
    let reader = BufReader::new(file);

    let mut images = Vec::new();
    let mut labels = Vec::new();

    for (line_num, line_res) in reader.lines().enumerate() {
        if let Some(max_s) = max_samples {
            if labels.len() >= max_s {
                break;
            }
        }

        let line = line_res.map_err(|e| {
            EngineError::SerializationError(format!("Error reading line {}: {}", line_num + 1, e))
        })?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let parts: Vec<&str> = trimmed.split(',').map(|s| s.trim()).collect();
        if parts.len() < 65 {
            return Err(EngineError::SerializationError(format!(
                "Invalid digits line {} in '{}': expected 65 fields, found {}",
                line_num + 1,
                path.as_ref().display(),
                parts.len()
            )));
        }

        // 64 pixel floats normalized to [0.0, 1.0] (0..16 -> 0.0..1.0)
        for p in &parts[..64] {
            let pixel_val: f32 = p.parse().map_err(|e| {
                EngineError::SerializationError(format!(
                    "Failed to parse pixel '{}' at line {}: {}",
                    p,
                    line_num + 1,
                    e
                ))
            })?;
            images.push((pixel_val / 16.0).clamp(0.0, 1.0));
        }

        let label: usize = parts[64].parse().map_err(|e| {
            EngineError::SerializationError(format!(
                "Failed to parse digit label '{}' at line {}: {}",
                parts[64],
                line_num + 1,
                e
            ))
        })?;
        if label > 9 {
            return Err(EngineError::SerializationError(format!(
                "Invalid digit label {} at line {} (expected 0..9)",
                label,
                line_num + 1
            )));
        }
        labels.push(label);
    }

    let n = labels.len();
    if n == 0 {
        return Err(EngineError::SerializationError(format!(
            "Digits dataset file '{}' is empty",
            path.as_ref().display()
        )));
    }

    Ok((RawTensor::from_vec(images, vec![n, 1, 8, 8]), labels))
}

/// Fisher's Iris Dataset (150 samples, 4 features, 3 classes).
/// Checks `data/iris.data` or `data/iris.csv` first if downloaded, otherwise uses the canonical embedded dataset.
pub fn load_iris_dataset() -> (RawTensor, Vec<usize>) {
    for candidate in &[
        "data/iris.data",
        "data/iris.csv",
        "../data/iris.data",
        "../data/iris.csv",
    ] {
        if Path::new(candidate).exists() {
            if let Ok(res) = load_iris_from_csv(candidate) {
                return res;
            }
        }
    }

    #[rustfmt::skip]
    const IRIS_DATA: [f32; 600] = [
        // Setosa (class 0)
        5.1, 3.5, 1.4, 0.2,  4.9, 3.0, 1.4, 0.2,  4.7, 3.2, 1.3, 0.2,  4.6, 3.1, 1.5, 0.2,
        5.0, 3.6, 1.4, 0.2,  5.4, 3.9, 1.7, 0.4,  4.6, 3.4, 1.4, 0.3,  5.0, 3.4, 1.5, 0.2,
        4.4, 2.9, 1.4, 0.2,  4.9, 3.1, 1.5, 0.1,  5.4, 3.7, 1.5, 0.2,  4.8, 3.4, 1.6, 0.2,
        4.8, 3.0, 1.4, 0.1,  4.3, 3.0, 1.1, 0.1,  5.8, 4.0, 1.2, 0.2,  5.7, 4.4, 1.5, 0.4,
        5.4, 3.9, 1.3, 0.4,  5.1, 3.5, 1.4, 0.3,  5.7, 3.8, 1.7, 0.3,  5.1, 3.8, 1.5, 0.3,
        5.4, 3.4, 1.7, 0.2,  5.1, 3.7, 1.5, 0.4,  4.6, 3.6, 1.0, 0.2,  5.1, 3.3, 1.7, 0.5,
        4.8, 3.4, 1.9, 0.2,  5.0, 3.0, 1.6, 0.2,  5.0, 3.4, 1.6, 0.4,  5.2, 3.5, 1.5, 0.2,
        5.2, 3.4, 1.4, 0.2,  4.7, 3.2, 1.6, 0.2,  4.8, 3.1, 1.6, 0.2,  5.4, 3.4, 1.5, 0.4,
        5.2, 4.1, 1.5, 0.1,  5.5, 4.2, 1.4, 0.2,  4.9, 3.1, 1.5, 0.2,  5.0, 3.2, 1.2, 0.2,
        5.5, 3.5, 1.3, 0.2,  4.9, 3.6, 1.4, 0.1,  4.4, 3.0, 1.3, 0.2,  5.1, 3.4, 1.5, 0.2,
        5.0, 3.5, 1.3, 0.3,  4.5, 2.3, 1.3, 0.3,  4.4, 3.2, 1.3, 0.2,  5.0, 3.5, 1.6, 0.6,
        5.1, 3.8, 1.9, 0.4,  4.8, 3.0, 1.4, 0.3,  5.1, 3.8, 1.6, 0.2,  4.6, 3.2, 1.4, 0.2,
        5.3, 3.7, 1.5, 0.2,  5.0, 3.3, 1.4, 0.2,
        // Versicolor (class 1)
        7.0, 3.2, 4.7, 1.4,  6.4, 3.2, 4.5, 1.5,  6.9, 3.1, 4.9, 1.5,  5.5, 2.3, 4.0, 1.3,
        6.5, 2.8, 4.6, 1.5,  5.7, 2.8, 4.5, 1.3,  6.3, 3.3, 4.7, 1.6,  4.9, 2.4, 3.3, 1.0,
        6.6, 2.9, 4.6, 1.3,  5.2, 2.7, 3.9, 1.4,  5.0, 2.0, 3.5, 1.0,  5.9, 3.0, 4.2, 1.5,
        6.0, 2.2, 4.0, 1.0,  6.1, 2.9, 4.7, 1.4,  5.6, 2.9, 3.6, 1.3,  6.7, 3.1, 4.4, 1.4,
        5.6, 3.0, 4.5, 1.5,  5.8, 2.7, 4.1, 1.0,  6.2, 2.2, 4.5, 1.5,  5.6, 2.5, 3.9, 1.1,
        5.9, 3.2, 4.8, 1.8,  6.1, 2.8, 4.0, 1.3,  6.3, 2.5, 4.9, 1.5,  6.1, 2.8, 4.7, 1.2,
        6.4, 2.9, 4.3, 1.3,  6.6, 3.0, 4.4, 1.4,  6.8, 2.8, 4.8, 1.4,  6.7, 3.0, 5.0, 1.7,
        6.0, 2.9, 4.5, 1.5,  5.7, 2.6, 3.5, 1.0,  5.5, 2.4, 3.8, 1.1,  5.5, 2.4, 3.7, 1.0,
        5.8, 2.7, 3.9, 1.2,  6.0, 2.7, 5.1, 1.6,  5.4, 3.0, 4.5, 1.5,  6.0, 3.4, 4.5, 1.6,
        6.7, 3.1, 4.7, 1.5,  6.3, 2.3, 4.4, 1.3,  5.6, 3.0, 4.1, 1.3,  5.5, 2.5, 4.0, 1.3,
        5.5, 2.6, 4.4, 1.2,  6.1, 3.0, 4.6, 1.4,  5.8, 2.6, 4.0, 1.2,  5.0, 2.3, 3.3, 1.0,
        5.6, 2.7, 4.2, 1.3,  5.7, 3.0, 4.2, 1.2,  5.7, 2.9, 4.2, 1.3,  6.2, 2.9, 4.3, 1.3,
        5.1, 2.5, 3.0, 1.1,  5.7, 2.8, 4.1, 1.3,
        // Virginica (class 2)
        6.3, 3.3, 6.0, 2.5,  5.8, 2.7, 5.1, 1.9,  7.1, 3.0, 5.9, 2.1,  6.3, 2.9, 5.6, 1.8,
        6.5, 3.0, 5.8, 2.2,  7.6, 3.0, 6.6, 2.1,  4.9, 2.5, 4.5, 1.7,  7.3, 2.9, 6.3, 1.8,
        6.7, 2.5, 5.8, 1.8,  7.2, 3.6, 6.1, 2.5,  6.5, 3.2, 5.1, 2.0,  6.4, 2.7, 5.3, 1.9,
        6.8, 3.0, 5.5, 2.1,  5.7, 2.5, 5.0, 2.0,  5.8, 2.8, 5.1, 2.4,  6.4, 3.2, 5.3, 2.3,
        6.5, 3.0, 5.5, 1.8,  7.7, 3.8, 6.7, 2.2,  7.7, 2.6, 6.9, 2.3,  6.0, 2.2, 5.0, 1.5,
        6.9, 3.2, 5.7, 2.3,  5.6, 2.8, 4.9, 2.0,  7.7, 2.8, 6.7, 2.0,  6.3, 2.7, 4.9, 1.8,
        6.7, 3.3, 5.7, 2.1,  7.2, 3.2, 6.0, 1.8,  6.2, 2.8, 4.8, 1.8,  6.1, 3.0, 4.9, 1.8,
        6.4, 2.8, 5.6, 2.1,  7.2, 3.0, 5.8, 1.6,  7.4, 2.8, 6.1, 1.9,  7.9, 3.8, 6.4, 2.0,
        6.4, 2.8, 5.6, 2.2,  6.3, 2.8, 5.1, 1.5,  6.1, 2.6, 5.6, 1.4,  7.7, 3.0, 6.1, 2.3,
        6.3, 3.4, 5.6, 2.4,  6.4, 3.1, 5.5, 1.8,  6.0, 3.0, 4.8, 1.8,  6.9, 3.1, 5.4, 2.1,
        6.7, 3.1, 5.6, 2.4,  6.9, 3.1, 5.1, 2.3,  5.8, 2.7, 5.1, 1.9,  6.8, 3.2, 5.9, 2.3,
        6.7, 3.3, 5.7, 2.5,  6.7, 3.0, 5.2, 2.3,  6.3, 2.5, 5.0, 1.9,  6.5, 3.0, 5.2, 2.0,
        6.2, 3.4, 5.4, 2.3,  5.9, 3.0, 5.1, 1.8,
    ];

    let mut labels = Vec::with_capacity(150);
    labels.extend(vec![0; 50]);
    labels.extend(vec![1; 50]);
    labels.extend(vec![2; 50]);

    (
        RawTensor::from_vec(IRIS_DATA.to_vec(), vec![150, 4]),
        labels,
    )
}

/// Loads 8x8 Optical Handwritten Digits dataset (0..9).
/// Checks `data/optdigits.tra` or `data/digits.csv` first if downloaded, otherwise generates synthetic samples.
pub fn load_digits_dataset(max_samples: Option<usize>) -> (RawTensor, Vec<usize>) {
    for candidate in &[
        "data/optdigits.tra",
        "data/digits.csv",
        "../data/optdigits.tra",
        "../data/digits.csv",
    ] {
        if Path::new(candidate).exists() {
            if let Ok(res) = load_digits_from_csv(candidate, max_samples) {
                return res;
            }
        }
    }

    generate_digits_dataset(max_samples.unwrap_or(600), 0.08)
}

/// Generates an 8x8 handwritten optical digit recognition dataset (0..9).
/// Produces tensors formatted as `[num_samples, 1, 8, 8]` with integer labels `0..10`.
pub fn generate_digits_dataset(num_samples: usize, noise: f32) -> (RawTensor, Vec<usize>) {
    #[rustfmt::skip]
    const DIGIT_BITMAPS: [[u8; 8]; 10] = [
        // 0
        [0b00111100,
         0b01100110,
         0b01100110,
         0b01100110,
         0b01100110,
         0b01100110,
         0b01100110,
         0b00111100],
        // 1
        [0b00011000,
         0b00111000,
         0b00011000,
         0b00011000,
         0b00011000,
         0b00011000,
         0b00011000,
         0b00111100],
        // 2
        [0b00111100,
         0b01100110,
         0b00000110,
         0b00001100,
         0b00011000,
         0b00110000,
         0b01100000,
         0b01111110],
        // 3
        [0b00111100,
         0b01100110,
         0b00000110,
         0b00011100,
         0b00000110,
         0b00000110,
         0b01100110,
         0b00111100],
        // 4
        [0b00001100,
         0b00011100,
         0b00101100,
         0b01001100,
         0b01111110,
         0b00001100,
         0b00001100,
         0b00001100],
        // 5
        [0b01111110,
         0b01100000,
         0b01111100,
         0b00000110,
         0b00000110,
         0b00000110,
         0b01100110,
         0b00111100],
        // 6
        [0b00111100,
         0b01100000,
         0b01100000,
         0b01111100,
         0b01100110,
         0b01100110,
         0b01100110,
         0b00111100],
        // 7
        [0b01111110,
         0b00000110,
         0b00001100,
         0b00011000,
         0b00110000,
         0b00110000,
         0b00110000,
         0b00110000],
        // 8
        [0b00111100,
         0b01100110,
         0b01100110,
         0b00111100,
         0b01100110,
         0b01100110,
         0b01100110,
         0b00111100],
        // 9
        [0b00111100,
         0b01100110,
         0b01100110,
         0b00111110,
         0b00000110,
         0b00000110,
         0b01100110,
         0b00111100],
    ];

    let mut images = vec![0.0f32; num_samples * 8 * 8];
    let mut labels = vec![0usize; num_samples];
    let mut rng = rand::thread_rng();

    for (i, label) in labels.iter_mut().enumerate().take(num_samples) {
        let digit = i % 10;
        *label = digit;
        let base_bitmap = &DIGIT_BITMAPS[digit];
        let img_offset = i * 64;

        // Random affine-like shift / stroke perturbation
        let shift_r: i32 = if rng.gen_bool(0.3) {
            rng.gen_range(-1..=1)
        } else {
            0
        };
        let shift_c: i32 = if rng.gen_bool(0.3) {
            rng.gen_range(-1..=1)
        } else {
            0
        };
        let intensity_scale: f32 = rng.gen_range(0.8..=1.2);

        for r in 0..8 {
            for c in 0..8 {
                let src_r = (r as i32) - shift_r;
                let src_c = (c as i32) - shift_c;

                let is_on = if (0..8).contains(&src_r) && (0..8).contains(&src_c) {
                    (base_bitmap[src_r as usize] & (1 << (7 - src_c))) != 0
                } else {
                    false
                };

                let base_val = if is_on { 1.0 * intensity_scale } else { 0.0 };

                let sample_noise: f32 = (rng.gen::<f32>() - 0.5) * 2.0 * noise;
                let pixel = (base_val + sample_noise).clamp(0.0, 1.0);
                images[img_offset + r * 8 + c] = pixel;
            }
        }
    }

    (
        RawTensor::from_vec(images, vec![num_samples, 1, 8, 8]),
        labels,
    )
}

/// Splits features and labels into training and test splits according to `test_ratio`.
pub fn train_test_split(
    features: &RawTensor,
    labels: &[usize],
    test_ratio: f32,
    shuffle: bool,
) -> (RawTensor, Vec<usize>, RawTensor, Vec<usize>) {
    let contig_features = features.to_contiguous();
    let n = labels.len();
    assert_eq!(contig_features.shape()[0], n);

    let mut indices: Vec<usize> = (0..n).collect();
    if shuffle {
        indices.shuffle(&mut rand::thread_rng());
    }

    let test_size = ((n as f32) * test_ratio).round() as usize;
    let train_size = n - test_size;

    let sample_elements: usize = contig_features.shape()[1..].iter().product();
    let f_slice = contig_features.as_slice();

    let mut train_f = vec![0.0f32; train_size * sample_elements];
    let mut train_l = Vec::with_capacity(train_size);
    for (i, &idx) in indices[..train_size].iter().enumerate() {
        train_l.push(labels[idx]);
        let src = idx * sample_elements;
        let dst = i * sample_elements;
        train_f[dst..dst + sample_elements].copy_from_slice(&f_slice[src..src + sample_elements]);
    }

    let mut test_f = vec![0.0f32; test_size * sample_elements];
    let mut test_l = Vec::with_capacity(test_size);
    for (i, &idx) in indices[train_size..].iter().enumerate() {
        test_l.push(labels[idx]);
        let src = idx * sample_elements;
        let dst = i * sample_elements;
        test_f[dst..dst + sample_elements].copy_from_slice(&f_slice[src..src + sample_elements]);
    }

    let mut train_shape = contig_features.shape().to_vec();
    train_shape[0] = train_size;
    let mut test_shape = contig_features.shape().to_vec();
    test_shape[0] = test_size;

    (
        RawTensor::from_vec(train_f, train_shape),
        train_l,
        RawTensor::from_vec(test_f, test_shape),
        test_l,
    )
}

/// Standardizes features along each feature column (zero mean, unit variance).
pub fn standardize(features: &RawTensor) -> (RawTensor, Vec<f32>, Vec<f32>) {
    let contig = features.to_contiguous();
    let shape = contig.shape();
    assert_eq!(shape.len(), 2, "Standardize expects 2D [N, D] tensor");
    let n = shape[0] as f32;
    let d = shape[1];
    let slice = contig.as_slice();

    let mut mean = vec![0.0f32; d];
    let mut std = vec![0.0f32; d];

    for row in 0..shape[0] {
        for col in 0..d {
            mean[col] += slice[row * d + col];
        }
    }
    for m in mean.iter_mut().take(d) {
        *m /= n;
    }

    for row in 0..shape[0] {
        for col in 0..d {
            let diff = slice[row * d + col] - mean[col];
            std[col] += diff * diff;
        }
    }
    for s in std.iter_mut().take(d) {
        *s = (*s / n).sqrt().max(1e-7);
    }

    let mut out_data = vec![0.0f32; shape[0] * d];
    for row in 0..shape[0] {
        for col in 0..d {
            out_data[row * d + col] = (slice[row * d + col] - mean[col]) / std[col];
        }
    }

    (RawTensor::from_vec(out_data, shape.to_vec()), mean, std)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iris_dataset_loader() {
        let (x, y) = load_iris_dataset();
        assert_eq!(x.shape(), &[150, 4]);
        assert_eq!(y.len(), 150);
        assert_eq!(y.iter().filter(|&&l| l == 0).count(), 50);
        assert_eq!(y.iter().filter(|&&l| l == 1).count(), 50);
        assert_eq!(y.iter().filter(|&&l| l == 2).count(), 50);
    }

    #[test]
    fn test_digits_dataset_generator() {
        let (x, y) = generate_digits_dataset(100, 0.05);
        assert_eq!(x.shape(), &[100, 1, 8, 8]);
        assert_eq!(y.len(), 100);
        for &l in &y {
            assert!(l < 10);
        }
    }

    #[test]
    fn test_train_test_split() {
        let (x, y) = load_iris_dataset();
        let (train_x, train_y, test_x, test_y) = train_test_split(&x, &y, 0.2, true);
        assert_eq!(train_x.shape(), &[120, 4]);
        assert_eq!(train_y.len(), 120);
        assert_eq!(test_x.shape(), &[30, 4]);
        assert_eq!(test_y.len(), 30);
    }

    #[test]
    fn test_standardize() {
        let (x, _) = load_iris_dataset();
        let (norm_x, mean, std) = standardize(&x);
        assert_eq!(norm_x.shape(), &[150, 4]);
        assert_eq!(mean.len(), 4);
        assert_eq!(std.len(), 4);
    }
}
