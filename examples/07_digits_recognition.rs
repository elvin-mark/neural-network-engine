//! Example 7: 8x8 Optical Handwritten Digits Recognition using CNN & MLP.
//!
//! Demonstrates:
//! - Generating and loading 8x8 grayscale digit images (0..9) with transformations & noise.
//! - Building a Convolutional Neural Network (Conv2d, MaxPool2d, Linear, ReLU).
//! - Training with Adam optimizer and mini-batch DataLoader.
//! - Measuring test accuracy across all 10 digit classes.
//! - Visualizing ASCII digit predictions and exporting model weights via SafeTensors.

use neural_network_engine::prelude::*;
use std::collections::HashMap;

/// ConvNet classifier tailored for 8x8 single-channel image recognition.
struct DigitConvNet {
    conv1: Conv2d,
    pool1: MaxPool2d,
    conv2: Conv2d,
    pool2: MaxPool2d,
    fc1: Linear,
    fc2: Linear,
}

impl DigitConvNet {
    pub fn new() -> Self {
        Self {
            // Conv1: 1 in_channel -> 8 out_channels, 3x3 kernel, pad=1 -> [B, 8, 8, 8]
            conv1: Conv2d::with_options(1, 8, (3, 3), (1, 1), (1, 1), (1, 1), true),
            pool1: MaxPool2d::square(2), // -> [B, 8, 4, 4]
            // Conv2: 8 in_channels -> 16 out_channels, 3x3 kernel, pad=1 -> [B, 16, 4, 4]
            conv2: Conv2d::with_options(8, 16, (3, 3), (1, 1), (1, 1), (1, 1), true),
            pool2: MaxPool2d::square(2), // -> [B, 16, 2, 2] = 64 features
            fc1: Linear::new(16 * 2 * 2, 32),
            fc2: Linear::new(32, 10),
        }
    }
}

impl Module for DigitConvNet {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let c1 = self.conv1.forward(input)?.relu()?;
        let p1 = self.pool1.forward(&c1)?;

        let c2 = self.conv2.forward(&p1)?.relu()?;
        let p2 = self.pool2.forward(&c2)?;

        let b = input.shape()[0];
        let flat = p2.reshape(&[b, 16 * 2 * 2])?;

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

fn compute_accuracy(model: &DigitConvNet, x: &RawTensor, y: &[usize]) -> Result<f32> {
    let _guard = NoGradGuard::new();
    let x_t = Tensor::new(x.clone(), false);
    let logits = model.forward(&x_t)?;
    let slice = logits.data().as_slice().to_vec();
    let num_classes = 10;
    let n = y.len();

    let mut correct = 0;
    for i in 0..n {
        let row = &slice[i * num_classes..(i + 1) * num_classes];
        let mut max_idx = 0;
        let mut max_val = f32::NEG_INFINITY;
        for (c, &val) in row.iter().enumerate() {
            if val > max_val {
                max_val = val;
                max_idx = c;
            }
        }
        if max_idx == y[i] {
            correct += 1;
        }
    }

    Ok((correct as f32) / (n as f32) * 100.0)
}

/// Prints an 8x8 digit image in ASCII format.
fn render_ascii_digit(slice: &[f32]) {
    for r in 0..8 {
        print!("    ");
        for c in 0..8 {
            let val = slice[r * 8 + c];
            let ch = if val > 0.6 {
                "██"
            } else if val > 0.3 {
                "▒▒"
            } else if val > 0.1 {
                "░░"
            } else {
                "  "
            };
            print!("{}", ch);
        }
        println!();
    }
}

fn main() -> Result<()> {
    println!("============================================================");
    println!("   07_digits_recognition: CNN on 8x8 Handwritten Digits     ");
    println!("============================================================\n");

    let num_samples = 600;
    let (dataset_x, dataset_y) = generate_digits_dataset(num_samples, 0.08);
    println!(
        "Generated {} samples of 8x8 digits across 10 classes (0-9)",
        num_samples
    );

    // 80% train = 480, 20% test = 120
    let (train_x, train_y, test_x, test_y) = train_test_split(&dataset_x, &dataset_y, 0.20, true);
    println!(
        "Train set size: {} images | Test set size: {} images\n",
        train_y.len(),
        test_y.len()
    );

    let model = DigitConvNet::new();
    let mut optimizer = Adam::adamw(model.parameters(), 0.005, 1e-4);

    let train_dataset = TensorDataset::new(
        train_x.clone(),
        RawTensor::from_vec(
            train_y.iter().map(|&l| l as f32).collect(),
            vec![train_y.len(), 1],
        ),
    )?;

    println!("Training ConvNet for 30 epochs...");
    let batch_size = 32;
    let epochs = 30;

    for epoch in 1..=epochs {
        let mut loader = DataLoader::new(&train_dataset, batch_size, true);
        let mut total_loss = 0.0;
        let mut batches = 0;

        while let Some((batch_x, batch_y)) = loader.next() {
            let b_labels: Vec<usize> = batch_y
                .data()
                .as_slice()
                .iter()
                .map(|&v| v as usize)
                .collect();

            optimizer.zero_grad();
            let logits = model.forward(&batch_x)?;
            let loss = CrossEntropyLoss::forward_with_indices(&logits, &b_labels)?;
            loss.backward();
            optimizer.step()?;

            total_loss += loss.item();
            batches += 1;
        }

        if epoch % 5 == 0 || epoch == epochs {
            let avg_loss = total_loss / (batches as f32);
            let train_acc = compute_accuracy(&model, &train_x, &train_y)?;
            let test_acc = compute_accuracy(&model, &test_x, &test_y)?;
            println!(
                "Epoch {:2}/{} | Avg Loss: {:.4} | Train Acc: {:5.1}% | Test Acc: {:5.1}%",
                epoch, epochs, avg_loss, train_acc, test_acc
            );
        }
    }

    // Evaluate final test accuracy
    let final_acc = compute_accuracy(&model, &test_x, &test_y)?;
    println!("\nFinal 8x8 Digits Test Accuracy: {:.2}%\n", final_acc);

    // Visualize predictions on 3 random test samples
    println!("------------------------------------------------------------");
    println!("              Sample Test Digits & Predictions             ");
    println!("------------------------------------------------------------");

    let _guard = NoGradGuard::new();
    let test_logits = model.forward(&Tensor::new(test_x.clone(), false))?;
    let test_probs = test_logits.softmax(1)?;
    let prob_slice = test_probs.data().as_slice().to_vec();
    let test_img_slice = test_x.as_slice();

    for i in 0..4.min(test_y.len()) {
        let actual = test_y[i];
        let row = &prob_slice[i * 10..(i + 1) * 10];
        let mut pred = 0;
        let mut max_prob = 0.0f32;
        for (c, &p) in row.iter().enumerate() {
            if p > max_prob {
                max_prob = p;
                pred = c;
            }
        }

        println!(
            "\nTest Image #{} -> Actual: {} | Predicted: {} (Confidence: {:.1}%)",
            i + 1,
            actual,
            pred,
            max_prob * 100.0
        );
        render_ascii_digit(&test_img_slice[i * 64..(i + 1) * 64]);
    }

    // Save model to SafeTensors format
    let mut state_dict: HashMap<String, RawTensor> = HashMap::new();
    let params = model.parameters();
    for (i, p) in params.iter().enumerate() {
        state_dict.insert(format!("layer_{}", i), p.data().clone());
    }

    let save_path = "target/digits_model.safetensors";
    save_safetensors(&state_dict, save_path)?;
    println!("\nSaved trained weights to {}", save_path);

    // Verify loading roundtrip
    let loaded = load_safetensors(save_path)?;
    println!(
        "Successfully reloaded {} weight tensors from SafeTensors!",
        loaded.len()
    );

    println!("\nDigits recognition training completed successfully!");
    Ok(())
}
