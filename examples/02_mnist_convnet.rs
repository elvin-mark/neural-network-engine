//! Example 2: Convolutional Neural Network (CNN / ConvNet) Image Classifier with SafeTensors Serialization.

use neural_network_engine::prelude::*;
use rand::Rng;
use std::collections::HashMap;

/// ConvNet model for 16x16 image classification.
struct ConvNet {
    conv1: Conv2d,
    pool1: MaxPool2d,
    conv2: Conv2d,
    pool2: MaxPool2d,
    fc1: Linear,
    fc2: Linear,
}

impl ConvNet {
    pub fn new() -> Self {
        Self {
            conv1: Conv2d::with_options(1, 8, (3, 3), (1, 1), (1, 1), (1, 1), true),
            pool1: MaxPool2d::square(2),
            conv2: Conv2d::with_options(8, 16, (3, 3), (1, 1), (1, 1), (1, 1), true),
            pool2: MaxPool2d::square(2),
            fc1: Linear::new(16 * 4 * 4, 32),
            fc2: Linear::new(32, 10),
        }
    }
}

impl Module for ConvNet {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // input: [B, 1, 16, 16]
        let c1 = self.conv1.forward(input)?.relu()?;
        let p1 = self.pool1.forward(&c1)?; // [B, 8, 8, 8]

        let c2 = self.conv2.forward(&p1)?.relu()?;
        let p2 = self.pool2.forward(&c2)?; // [B, 16, 4, 4]

        let b = input.shape()[0];
        let flat = p2.reshape(&[b, 16 * 4 * 4])?;

        let h = self.fc1.forward(&flat)?.relu()?;
        self.fc2.forward(&h)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params
    }
}

/// Generates synthetic 16x16 multi-class digit/pattern image dataset.
fn generate_image_dataset(num_samples: usize) -> (RawTensor, Vec<usize>) {
    let mut images = vec![0.0; num_samples * 16 * 16];
    let mut labels = vec![0; num_samples];
    let mut rng = rand::thread_rng();

    for (i, label_val) in labels.iter_mut().enumerate().take(num_samples) {
        let class = i % 10;
        *label_val = class;
        let img_offset = i * 16 * 16;

        // Draw characteristic spatial patterns based on class
        for r in 0..16 {
            for c in 0..16 {
                let pixel_idx = img_offset + r * 16 + c;
                let noise: f32 = rng.gen_range(0.0..0.1);
                let is_pattern = match class {
                    0 => (r == 4 || r == 11) && (4..=11).contains(&c),
                    1 => c == 8,
                    2 => r == 8,
                    3 => r == c,
                    4 => r + c == 15,
                    5 => (r as isize - 8).pow(2) + (c as isize - 8).pow(2) <= 16,
                    6 => r % 4 == 0,
                    7 => c % 4 == 0,
                    8 => (r == 4 || r == 8 || r == 12) && (4..=11).contains(&c),
                    _ => (6..=10).contains(&r) && (6..=10).contains(&c),
                };
                images[pixel_idx] = if is_pattern { 1.0 } else { 0.0 } + noise;
            }
        }
    }

    (
        RawTensor::from_vec(images, vec![num_samples, 1, 16, 16]),
        labels,
    )
}

fn main() -> Result<()> {
    println!("============================================================");
    println!("   02_mnist_convnet: CNN Image Classification & SafeTensors ");
    println!("============================================================\n");

    let total_samples = 200;
    let (img_raw, labels) = generate_image_dataset(total_samples);
    println!(
        "Generated synthetic image dataset: shape {:?}, {} classes",
        img_raw.shape(),
        10
    );

    let model = ConvNet::new();
    let mut optimizer = Adam::new(model.parameters(), 0.005);

    let batch_size = 32;
    let num_batches = total_samples.div_ceil(batch_size);

    println!(
        "\nTraining ConvNet (15 epochs, batch size {})...",
        batch_size
    );

    for epoch in 1..=15 {
        let mut total_loss = 0.0;
        let mut correct = 0;

        for b in 0..num_batches {
            let start = b * batch_size;
            let end = (start + batch_size).min(total_samples);
            let b_len = end - start;

            let b_img = img_raw.slice(0, start, end)?;
            let b_labels = &labels[start..end];

            let x = Tensor::new(b_img, false);
            let logits = model.forward(&x)?;
            let loss = CrossEntropyLoss::forward_with_indices(&logits, b_labels)?;

            optimizer.zero_grad();
            loss.backward();
            optimizer.step()?;

            total_loss += loss.item() * (b_len as f32);

            let preds = logits.data().argmax(1)?;
            for (&p, &l) in preds.iter().zip(b_labels.iter()) {
                if p == l {
                    correct += 1;
                }
            }
        }

        let avg_loss = total_loss / (total_samples as f32);
        let acc = (correct as f32) / (total_samples as f32) * 100.0;

        if epoch % 3 == 0 || epoch == 1 {
            println!(
                "Epoch {:2}/15 | Avg Loss: {:7.4} | Accuracy: {:6.2}% ({}/{})",
                epoch, avg_loss, acc, correct, total_samples
            );
        }
    }

    // Demonstrate SafeTensors serialization
    println!("\nSaving trained model weights to SafeTensors format...");
    let mut tensor_map = HashMap::new();
    tensor_map.insert("conv1.weight".to_string(), model.conv1.weight.data());
    if let Some(ref b) = model.conv1.bias {
        tensor_map.insert("conv1.bias".to_string(), b.data());
    }
    tensor_map.insert("conv2.weight".to_string(), model.conv2.weight.data());
    if let Some(ref b) = model.conv2.bias {
        tensor_map.insert("conv2.bias".to_string(), b.data());
    }
    tensor_map.insert("fc1.weight".to_string(), model.fc1.weight.data());
    if let Some(ref b) = model.fc1.bias {
        tensor_map.insert("fc1.bias".to_string(), b.data());
    }
    tensor_map.insert("fc2.weight".to_string(), model.fc2.weight.data());
    if let Some(ref b) = model.fc2.bias {
        tensor_map.insert("fc2.bias".to_string(), b.data());
    }

    let save_path = "model_weights.safetensors";
    save_safetensors(&tensor_map, save_path)?;
    println!("Successfully saved weights to '{}'", save_path);

    let loaded = load_safetensors(save_path)?;
    println!(
        "Successfully reloaded {} tensors from SafeTensors format!",
        loaded.len()
    );

    // Clean up temporary weights file
    let _ = std::fs::remove_file(save_path);

    println!("\nConvNet training & SafeTensors verification succeeded!");
    Ok(())
}
