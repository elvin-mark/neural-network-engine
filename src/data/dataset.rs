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
use std::io::{BufRead, BufReader, Read};
use std::path::Path;

/// Canonical class names for CIFAR-10 (10 categories).
pub const CIFAR10_CLASSES: [&str; 10] = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
];

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

/// Loads MNIST dataset from binary IDX format files (e.g. `train-images-idx3-ubyte` and `train-labels-idx1-ubyte`).
pub fn load_mnist_from_idx<P1: AsRef<Path>, P2: AsRef<Path>>(
    images_path: P1,
    labels_path: P2,
    max_samples: Option<usize>,
) -> Result<(RawTensor, Vec<usize>)> {
    let mut img_file = File::open(&images_path).map_err(|e| {
        EngineError::SerializationError(format!(
            "Failed to open MNIST images file '{}': {}",
            images_path.as_ref().display(),
            e
        ))
    })?;
    let mut lbl_file = File::open(&labels_path).map_err(|e| {
        EngineError::SerializationError(format!(
            "Failed to open MNIST labels file '{}': {}",
            labels_path.as_ref().display(),
            e
        ))
    })?;

    let mut img_header = [0u8; 16];
    img_file.read_exact(&mut img_header).map_err(|e| {
        EngineError::SerializationError(format!("Failed to read MNIST images header: {}", e))
    })?;
    let img_magic = u32::from_be_bytes(img_header[0..4].try_into().unwrap());
    if img_magic != 2051 {
        return Err(EngineError::SerializationError(format!(
            "Invalid MNIST images magic number: expected 2051 (0x803), found {}",
            img_magic
        )));
    }
    let total_images = u32::from_be_bytes(img_header[4..8].try_into().unwrap()) as usize;
    let rows = u32::from_be_bytes(img_header[8..12].try_into().unwrap()) as usize;
    let cols = u32::from_be_bytes(img_header[12..16].try_into().unwrap()) as usize;
    if rows != 28 || cols != 28 {
        return Err(EngineError::SerializationError(format!(
            "Invalid MNIST image dimensions: expected 28x28, found {}x{}",
            rows, cols
        )));
    }

    let mut lbl_header = [0u8; 8];
    lbl_file.read_exact(&mut lbl_header).map_err(|e| {
        EngineError::SerializationError(format!("Failed to read MNIST labels header: {}", e))
    })?;
    let lbl_magic = u32::from_be_bytes(lbl_header[0..4].try_into().unwrap());
    if lbl_magic != 2049 {
        return Err(EngineError::SerializationError(format!(
            "Invalid MNIST labels magic number: expected 2049 (0x801), found {}",
            lbl_magic
        )));
    }
    let total_labels = u32::from_be_bytes(lbl_header[4..8].try_into().unwrap()) as usize;
    if total_images != total_labels {
        return Err(EngineError::SerializationError(format!(
            "MNIST count mismatch: {} images vs {} labels",
            total_images, total_labels
        )));
    }

    let n = total_images.min(max_samples.unwrap_or(total_images));
    let mut raw_pixels = vec![0u8; n * 28 * 28];
    img_file.read_exact(&mut raw_pixels).map_err(|e| {
        EngineError::SerializationError(format!("Failed to read MNIST pixels: {}", e))
    })?;

    let mut raw_labels = vec![0u8; n];
    lbl_file.read_exact(&mut raw_labels).map_err(|e| {
        EngineError::SerializationError(format!("Failed to read MNIST labels: {}", e))
    })?;

    let mut images = Vec::with_capacity(n * 28 * 28);
    for &p in &raw_pixels {
        images.push(p as f32 / 255.0);
    }
    let labels: Vec<usize> = raw_labels.into_iter().map(|l| l as usize).collect();

    Ok((RawTensor::from_vec(images, vec![n, 1, 28, 28]), labels))
}

/// Generates a synthetic 28x28 handwritten digit dataset (0..9).
/// Produces tensors formatted as `[num_samples, 1, 28, 28]` with integer labels `0..10`.
pub fn generate_mnist_dataset(num_samples: usize, noise: f32) -> (RawTensor, Vec<usize>) {
    let (digits_8x8, labels) = generate_digits_dataset(num_samples, noise * 0.5);
    let slice_8x8 = digits_8x8.as_slice();

    let mut images = vec![0.0f32; num_samples * 28 * 28];
    let mut rng = rand::thread_rng();

    for i in 0..num_samples {
        let src_offset = i * 64;
        let dst_offset = i * 28 * 28;

        for r in 0..28 {
            for c in 0..28 {
                let dst_idx = dst_offset + r * 28 + c;

                let r_f = (r as f32 - 4.0) / 2.5;
                let c_f = (c as f32 - 4.0) / 2.5;

                let pixel_val = if (0.0..7.0).contains(&r_f) && (0.0..7.0).contains(&c_f) {
                    let r0 = r_f.floor() as usize;
                    let c0 = c_f.floor() as usize;
                    let r1 = (r0 + 1).min(7);
                    let c1 = (c0 + 1).min(7);
                    let dr = r_f - r0 as f32;
                    let dc = c_f - c0 as f32;

                    let p00 = slice_8x8[src_offset + r0 * 8 + c0];
                    let p01 = slice_8x8[src_offset + r0 * 8 + c1];
                    let p10 = slice_8x8[src_offset + r1 * 8 + c0];
                    let p11 = slice_8x8[src_offset + r1 * 8 + c1];

                    let top = p00 * (1.0 - dc) + p01 * dc;
                    let bot = p10 * (1.0 - dc) + p11 * dc;
                    top * (1.0 - dr) + bot * dr
                } else {
                    0.0
                };

                let sample_noise = (rng.gen::<f32>() - 0.5) * noise;
                images[dst_idx] = (pixel_val + sample_noise).clamp(0.0, 1.0);
            }
        }
    }

    (
        RawTensor::from_vec(images, vec![num_samples, 1, 28, 28]),
        labels,
    )
}

/// Loads MNIST dataset. Checks for downloaded IDX files in `data/`, otherwise falls back to synthetic dataset.
pub fn load_mnist_dataset(max_samples: Option<usize>) -> (RawTensor, Vec<usize>) {
    let candidates = [
        (
            "data/train-images-idx3-ubyte",
            "data/train-labels-idx1-ubyte",
        ),
        (
            "data/mnist/train-images-idx3-ubyte",
            "data/mnist/train-labels-idx1-ubyte",
        ),
        (
            "../data/train-images-idx3-ubyte",
            "../data/train-labels-idx1-ubyte",
        ),
    ];

    for (img_path, lbl_path) in &candidates {
        if Path::new(img_path).exists() && Path::new(lbl_path).exists() {
            if let Ok(res) = load_mnist_from_idx(img_path, lbl_path, max_samples) {
                return res;
            }
        }
    }

    generate_mnist_dataset(max_samples.unwrap_or(1000), 0.05)
}

/// Loads CIFAR-10 dataset from binary batch file (e.g. `data_batch_1.bin` or `test_batch.bin`).
/// Format per sample: 1 byte label (0..9) + 3072 bytes (1024 R, 1024 G, 1024 B).
pub fn load_cifar10_from_binary<P: AsRef<Path>>(
    path: P,
    max_samples: Option<usize>,
) -> Result<(RawTensor, Vec<usize>)> {
    let mut file = File::open(&path).map_err(|e| {
        EngineError::SerializationError(format!(
            "Failed to open CIFAR-10 binary file '{}': {}",
            path.as_ref().display(),
            e
        ))
    })?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).map_err(|e| {
        EngineError::SerializationError(format!("Failed to read CIFAR-10 file: {}", e))
    })?;

    const RECORD_BYTES: usize = 1 + 3072;
    let total_records = buffer.len() / RECORD_BYTES;
    if total_records == 0 {
        return Err(EngineError::SerializationError(format!(
            "CIFAR-10 binary file '{}' is empty or invalid size ({})",
            path.as_ref().display(),
            buffer.len()
        )));
    }

    let n = total_records.min(max_samples.unwrap_or(total_records));
    let mut images = Vec::with_capacity(n * 3 * 32 * 32);
    let mut labels = Vec::with_capacity(n);

    for i in 0..n {
        let offset = i * RECORD_BYTES;
        let label = buffer[offset] as usize;
        if label > 9 {
            return Err(EngineError::SerializationError(format!(
                "Invalid CIFAR-10 label {} at record {}",
                label, i
            )));
        }
        labels.push(label);

        let pixel_bytes = &buffer[offset + 1..offset + RECORD_BYTES];
        for &p in pixel_bytes {
            images.push(p as f32 / 255.0);
        }
    }

    Ok((RawTensor::from_vec(images, vec![n, 3, 32, 32]), labels))
}

/// Generates synthetic 3x32x32 color image dataset across 10 classes with distinct color geometries.
pub fn generate_cifar10_dataset(num_samples: usize, noise: f32) -> (RawTensor, Vec<usize>) {
    let mut images = vec![0.0f32; num_samples * 3 * 32 * 32];
    let mut labels = Vec::with_capacity(num_samples);
    let mut rng = rand::thread_rng();

    const PALETTES: [[f32; 3]; 10] = [
        [0.2, 0.6, 0.9], // 0: Airplane
        [0.9, 0.2, 0.2], // 1: Automobile
        [0.3, 0.8, 0.4], // 2: Bird
        [0.9, 0.6, 0.2], // 3: Cat
        [0.6, 0.4, 0.2], // 4: Deer
        [0.8, 0.7, 0.5], // 5: Dog
        [0.1, 0.9, 0.2], // 6: Frog
        [0.4, 0.3, 0.2], // 7: Horse
        [0.1, 0.3, 0.8], // 8: Ship
        [0.7, 0.7, 0.8], // 9: Truck
    ];

    for i in 0..num_samples {
        let class = i % 10;
        labels.push(class);
        let color = PALETTES[class];
        let img_offset = i * 3 * 32 * 32;

        let center_r = 16.0f32 + (rng.gen::<f32>() - 0.5) * 4.0;
        let center_c = 16.0f32 + (rng.gen::<f32>() - 0.5) * 4.0;
        let radius = 9.0f32 + (rng.gen::<f32>() - 0.5) * 2.0;

        for r in 0..32 {
            for c in 0..32 {
                let dr = (r as f32) - center_r;
                let dc = (c as f32) - center_c;
                let dist_sq = dr * dr + dc * dc;
                let is_foreground = dist_sq <= radius * radius;

                for (ch, &channel_col) in color.iter().enumerate() {
                    let base_val = if is_foreground {
                        channel_col
                    } else {
                        channel_col * 0.25 + 0.1
                    };
                    let n_val = (rng.gen::<f32>() - 0.5) * 2.0 * noise;
                    let pixel_idx = img_offset + ch * 1024 + r * 32 + c;
                    images[pixel_idx] = (base_val + n_val).clamp(0.0, 1.0);
                }
            }
        }
    }

    (
        RawTensor::from_vec(images, vec![num_samples, 3, 32, 32]),
        labels,
    )
}

/// Loads CIFAR-10 dataset. Checks for downloaded binary batch in `data/`, otherwise falls back to synthetic dataset.
pub fn load_cifar10_dataset(max_samples: Option<usize>) -> (RawTensor, Vec<usize>) {
    let candidates = [
        "data/cifar-10-batches-bin/data_batch_1.bin",
        "data/data_batch_1.bin",
        "data/cifar10/data_batch_1.bin",
        "../data/cifar-10-batches-bin/data_batch_1.bin",
    ];

    for candidate in &candidates {
        if Path::new(candidate).exists() {
            if let Ok(res) = load_cifar10_from_binary(candidate, max_samples) {
                return res;
            }
        }
    }

    generate_cifar10_dataset(max_samples.unwrap_or(800), 0.05)
}

/// Loads CIFAR-100 dataset from binary file (e.g. `train.bin` or `test.bin`).
/// Format per sample: 1 byte coarse label (0..19) + 1 byte fine label (0..99) + 3072 bytes (1024 R, 1024 G, 1024 B).
pub fn load_cifar100_from_binary<P: AsRef<Path>>(
    path: P,
    max_samples: Option<usize>,
) -> Result<(RawTensor, Vec<usize>)> {
    let mut file = File::open(&path).map_err(|e| {
        EngineError::SerializationError(format!(
            "Failed to open CIFAR-100 binary file '{}': {}",
            path.as_ref().display(),
            e
        ))
    })?;

    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).map_err(|e| {
        EngineError::SerializationError(format!("Failed to read CIFAR-100 file: {}", e))
    })?;

    const RECORD_BYTES: usize = 2 + 3072;
    let total_records = buffer.len() / RECORD_BYTES;
    if total_records == 0 {
        return Err(EngineError::SerializationError(format!(
            "CIFAR-100 binary file '{}' is empty or invalid size ({})",
            path.as_ref().display(),
            buffer.len()
        )));
    }

    let n = total_records.min(max_samples.unwrap_or(total_records));
    let mut images = Vec::with_capacity(n * 3 * 32 * 32);
    let mut labels = Vec::with_capacity(n);

    for i in 0..n {
        let offset = i * RECORD_BYTES;
        let fine_label = buffer[offset + 1] as usize;
        if fine_label > 99 {
            return Err(EngineError::SerializationError(format!(
                "Invalid CIFAR-100 fine label {} at record {}",
                fine_label, i
            )));
        }
        labels.push(fine_label);

        let pixel_bytes = &buffer[offset + 2..offset + RECORD_BYTES];
        for &p in pixel_bytes {
            images.push(p as f32 / 255.0);
        }
    }

    Ok((RawTensor::from_vec(images, vec![n, 3, 32, 32]), labels))
}

/// Generates synthetic 3x32x32 color image dataset across 100 fine categories.
pub fn generate_cifar100_dataset(num_samples: usize, noise: f32) -> (RawTensor, Vec<usize>) {
    let mut images = vec![0.0f32; num_samples * 3 * 32 * 32];
    let mut labels = Vec::with_capacity(num_samples);
    let mut rng = rand::thread_rng();

    for i in 0..num_samples {
        let class = i % 100;
        labels.push(class);
        let img_offset = i * 3 * 32 * 32;

        let angle = (class as f32 / 100.0) * std::f32::consts::TAU;
        let r_val = (angle.sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        let g_val = ((angle + 2.094).sin() * 0.5 + 0.5).clamp(0.0, 1.0);
        let b_val = ((angle + 4.188).sin() * 0.5 + 0.5).clamp(0.0, 1.0);

        for r in 0..32 {
            for c in 0..32 {
                let pattern = (((r * (class + 1)) % 16) as f32 / 16.0) * 0.5;
                for ch in 0..3 {
                    let base = match ch {
                        0 => r_val + pattern,
                        1 => g_val + pattern * 0.5,
                        _ => b_val,
                    };
                    let n_val = (rng.gen::<f32>() - 0.5) * noise;
                    let pixel_idx = img_offset + ch * 1024 + r * 32 + c;
                    images[pixel_idx] = (base * 0.7 + 0.1 + n_val).clamp(0.0, 1.0);
                }
            }
        }
    }

    (
        RawTensor::from_vec(images, vec![num_samples, 3, 32, 32]),
        labels,
    )
}

/// Loads CIFAR-100 dataset. Checks for downloaded binary in `data/`, otherwise falls back to synthetic dataset.
pub fn load_cifar100_dataset(max_samples: Option<usize>) -> (RawTensor, Vec<usize>) {
    let candidates = [
        "data/cifar-100-binary/train.bin",
        "data/train.bin",
        "data/cifar100/train.bin",
        "../data/cifar-100-binary/train.bin",
    ];

    for candidate in &candidates {
        if Path::new(candidate).exists() {
            if let Ok(res) = load_cifar100_from_binary(candidate, max_samples) {
                return res;
            }
        }
    }

    generate_cifar100_dataset(max_samples.unwrap_or(1000), 0.05)
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
    assert_eq!(
        contig_features.shape()[0],
        n,
        "Features row count must match labels count"
    );
    assert!(
        test_ratio > 0.0 && test_ratio < 1.0,
        "test_ratio must be strictly between 0.0 and 1.0, got {}",
        test_ratio
    );

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
    let shape = contig.shape().to_vec();
    assert_eq!(shape.len(), 2, "Standardize expects 2D [N, D] tensor");
    if shape[0] == 0 {
        return (contig, vec![0.0; shape[1]], vec![1.0; shape[1]]);
    }
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

/// Loads the TinyStories dataset from disk (e.g. `data/tinystories.txt`),
/// or falls back to generating a synthetic story collection if the file is absent.
pub fn load_tinystories_dataset(max_chars: Option<usize>) -> String {
    let candidate_paths = [
        "data/tinystories.txt",
        "../data/tinystories.txt",
        "tinystories.txt",
    ];
    for path in &candidate_paths {
        if let Ok(content) = std::fs::read_to_string(path) {
            let trimmed = content.trim();
            if !trimmed.is_empty() {
                if let Some(limit) = max_chars {
                    return trimmed.chars().take(limit).collect();
                }
                return trimmed.to_string();
            }
        }
    }

    generate_tinystories_dataset(max_chars)
}

/// Generates a rich synthetic TinyStories corpus containing child-friendly narrative stories.
pub fn generate_tinystories_dataset(max_chars: Option<usize>) -> String {
    let stories = [
        "Once upon a time, there was a little girl named Lily. Lily loved to explore the bright garden behind her house. One sunny morning, Lily saw a tiny blue bird sitting on a wooden fence. The bird was singing a cheerful song. Lily smiled and waved at the bird. \"Hello, little bird!\" she said softly. The bird chirped happily and flew down to the green grass. Lily shared some breadcrumbs with her new feathery friend. From that day on, the bird visited Lily every single morning.",
        "Tim was a playful boy who had a big fluffy dog named Barnaby. Barnaby loved chasing red balls in the park. One afternoon, Tim threw the ball very high into the clear blue sky. The ball bounced under a large oak tree. Barnaby ran quickly, wagging his tail with pure joy. He gently picked up the ball and brought it back to Tim. Tim patted Barnaby on the head and gave him a crunchy treat. They walked home together, happy and tired.",
        "Mia and her brother Leo decided to build a grand sandcastle on the sandy beach. They gathered wet sand in their colorful plastic buckets and shaped tall towers. Leo found smooth white seashells to decorate the castle walls. Mia placed a small green leaf at the very top as a royal flag. \"Look at our magnificent castle!\" shouted Leo. Suddenly, a gentle wave washed over the shore and touched their feet. They laughed and started building another even bigger castle together.",
        "In a cozy green forest, a curious little squirrel named Sammy was searching for sweet acorns. Winter was coming soon, and Sammy wanted to prepare a warm nest. While hopping from branch to branch, Sammy found a shiny silver key near a hollow tree. He wondered what the key could open. He took the key to Oliver the wise old owl. Oliver smiled wisely and said, \"This key opens the secret door to the library of the forest.\" Sammy was thrilled to discover endless stories.",
        "Emma loved painting colorful pictures of stars, rainbows, and friendly dragons. Her favorite color was bright yellow because it reminded her of warm sunshine. One rainy day, Emma could not play outside. She sat by the window with her wooden easel and painted a giant golden sun. Her mother walked into the room and gasped with delight. \"Your painting brought the sunshine right inside our home!\" her mother said with a warm hug.",
        "A little red robot named Beep lived in a quiet workshop on the hill. Beep liked helping everyone fix their broken toys. One day, a crying boy came to the workshop holding a broken toy train. Beep carefully tightened the small gears and polished the wheels. The train started rolling smoothly again, making a cheerful whistle sound. The boy clapped his hands with excitement. Beep's little heart light glowed bright green with happiness.",
        "Daisy the duckling was learning how to swim across the calm pond. At first, the water felt cold and scary. Her mother quacked encouragingly, \"You can do it, Daisy! Just paddle your little feet.\" Daisy took a deep breath, flapped her wings, and jumped into the cool water. To her amazement, she was floating! Daisy paddled across the pond to her brothers and sisters, proud of her brave adventure.",
        "Max had a magical garden where vegetables grew unusually big. There were giant orange carrots, round red tomatoes, and purple eggplants that shimmered in the moonlight. Max took care of each plant with fresh water and kind words. When harvest day arrived, Max shared his enormous vegetables with all his neighbors in the village. Everyone gathered for a joyful feast under the sparkling evening stars.",
    ];

    let mut full_text = String::new();
    let mut repeat = 0;
    while full_text.len() < max_chars.unwrap_or(30_000) {
        for story in &stories {
            full_text.push_str(story);
            full_text.push_str("\n\n");
        }
        repeat += 1;
        if repeat > 20 {
            break;
        }
    }

    if let Some(limit) = max_chars {
        full_text.chars().take(limit).collect()
    } else {
        full_text
    }
}

/// A Question Answering sample containing the question, context paragraph, and answer text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QASample {
    pub question: String,
    pub context: String,
    pub answer: String,
}

/// Generates a structured multi-domain Extractive Question Answering dataset.
pub fn generate_qa_dataset() -> Vec<QASample> {
    vec![
        QASample {
            question: "What is Rust?".to_string(),
            context: "Rust is a systems programming language that focuses on safety and performance.".to_string(),
            answer: "a systems programming language".to_string(),
        },
        QASample {
            question: "What does Autograd provide?".to_string(),
            context: "Autograd provides reverse-mode automatic differentiation for neural networks.".to_string(),
            answer: "reverse-mode automatic differentiation".to_string(),
        },
        QASample {
            question: "Who developed the transformer architecture?".to_string(),
            context: "The transformer architecture was introduced by Vaswani and colleagues in 2017.".to_string(),
            answer: "Vaswani and colleagues".to_string(),
        },
        QASample {
            question: "What is the primary feature of BERT?".to_string(),
            context: "BERT uses bidirectional transformer encoders to learn deep contextual representations.".to_string(),
            answer: "bidirectional transformer encoders".to_string(),
        },
        QASample {
            question: "What is GELU?".to_string(),
            context: "GELU is a smooth nonlinear activation function widely used in modern transformers.".to_string(),
            answer: "a smooth nonlinear activation function".to_string(),
        },
        QASample {
            question: "What optimizer uses adaptive learning rates?".to_string(),
            context: "Adam computes adaptive learning rates for each parameter using first and second moments.".to_string(),
            answer: "Adam".to_string(),
        },
        QASample {
            question: "What does SafeTensors ensure?".to_string(),
            context: "SafeTensors is a secure and fast serialization format for deep learning tensor weights.".to_string(),
            answer: "a secure and fast serialization format".to_string(),
        },
        QASample {
            question: "What is Whisper designed for?".to_string(),
            context: "Whisper is an encoder-decoder sequence-to-sequence model designed for speech recognition.".to_string(),
            answer: "an encoder-decoder sequence-to-sequence model".to_string(),
        },
        QASample {
            question: "Where do neural network weights reside?".to_string(),
            context: "Neural network weights are stored inside dense linear and convolutional layers.".to_string(),
            answer: "inside dense linear and convolutional layers".to_string(),
        },
        QASample {
            question: "How does backpropagation work?".to_string(),
            context: "Backpropagation applies the mathematical chain rule backwards through the computational DAG.".to_string(),
            answer: "the mathematical chain rule".to_string(),
        },
        QASample {
            question: "What is LayerNorm?".to_string(),
            context: "LayerNorm normalizes features across the hidden dimension with learnable scale and shift parameters.".to_string(),
            answer: "normalizes features across the hidden dimension".to_string(),
        },
        QASample {
            question: "What does cross-attention connect?".to_string(),
            context: "Cross-attention connects queries from the decoder to key and value states from the encoder.".to_string(),
            answer: "queries from the decoder to key and value states from the encoder".to_string(),
        },
    ]
}

/// Generates paired sentences with semantic similarity labels (1.0 for related/paraphrase, 0.0 for unrelated).
pub fn generate_semantic_similarity_dataset() -> Vec<(String, String, f32)> {
    vec![
        (
            "Rust is a fast systems programming language.".to_string(),
            "Rust provides high performance and memory safety.".to_string(),
            1.0,
        ),
        (
            "BERT uses bidirectional self-attention.".to_string(),
            "BERT learns contextual representations from both directions.".to_string(),
            1.0,
        ),
        (
            "Neural networks learn through gradient descent.".to_string(),
            "Optimizers update model parameters using backpropagated gradients.".to_string(),
            1.0,
        ),
        (
            "Whisper performs speech-to-text recognition.".to_string(),
            "Whisper transcribes acoustic audio into text sentences.".to_string(),
            1.0,
        ),
        (
            "Apples and oranges are sweet fruits.".to_string(),
            "Quantum computers process qubits using superposition.".to_string(),
            0.0,
        ),
        (
            "The dog ran across the sunny park.".to_string(),
            "Matrix multiplication is a fundamental linear algebra operation.".to_string(),
            0.0,
        ),
        (
            "Deep learning models require training datasets.".to_string(),
            "Astronomers study distant galaxies and black holes.".to_string(),
            0.0,
        ),
        (
            "Convolutional layers extract local spatial features.".to_string(),
            "Conv2d filters scan image pixels to detect visual patterns.".to_string(),
            1.0,
        ),
    ]
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

    #[test]
    fn test_mnist_generator() {
        let (x, y) = generate_mnist_dataset(50, 0.05);
        assert_eq!(x.shape(), &[50, 1, 28, 28]);
        assert_eq!(y.len(), 50);
        for &l in &y {
            assert!(l < 10);
        }
    }

    #[test]
    fn test_cifar10_generator() {
        let (x, y) = generate_cifar10_dataset(50, 0.05);
        assert_eq!(x.shape(), &[50, 3, 32, 32]);
        assert_eq!(y.len(), 50);
        for &l in &y {
            assert!(l < 10);
        }
        assert_eq!(CIFAR10_CLASSES.len(), 10);
    }

    #[test]
    fn test_cifar100_generator() {
        let (x, y) = generate_cifar100_dataset(50, 0.05);
        assert_eq!(x.shape(), &[50, 3, 32, 32]);
        assert_eq!(y.len(), 50);
        for &l in &y {
            assert!(l < 100);
        }
    }

    #[test]
    fn test_tinystories_dataset_loader() {
        let text = load_tinystories_dataset(Some(500));
        assert!(text.len() <= 500);
        assert!(text.contains("Lily") || text.contains("Once upon a time"));
    }

    #[test]
    fn test_qa_dataset_generator() {
        let qa = generate_qa_dataset();
        assert!(!qa.is_empty());
        for sample in &qa {
            assert!(sample.context.contains(&sample.answer));
        }

        let sim = generate_semantic_similarity_dataset();
        assert!(!sim.is_empty());
    }
}
