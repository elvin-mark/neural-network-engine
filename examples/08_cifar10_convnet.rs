//! Example 8: Convolutional Neural Network (CNN / ConvNet) on 32x32 RGB CIFAR-10 Image Dataset.

use neural_network_engine::prelude::*;
use std::collections::HashMap;

/// ConvNet for 3-channel 32x32 RGB CIFAR-10 image classification.
struct Cifar10ConvNet {
    conv1: Conv2d,
    pool1: MaxPool2d,
    conv2: Conv2d,
    pool2: MaxPool2d,
    conv3: Conv2d,
    pool3: MaxPool2d,
    fc1: Linear,
    fc2: Linear,
}

impl Cifar10ConvNet {
    pub fn new() -> Self {
        Self {
            // [B, 3, 32, 32] -> conv1 -> [B, 16, 32, 32] -> pool1 -> [B, 16, 16, 16]
            conv1: Conv2d::with_options(3, 16, (3, 3), (1, 1), (1, 1), (1, 1), true),
            pool1: MaxPool2d::square(2),
            // [B, 16, 16, 16] -> conv2 -> [B, 32, 16, 16] -> pool2 -> [B, 32, 8, 8]
            conv2: Conv2d::with_options(16, 32, (3, 3), (1, 1), (1, 1), (1, 1), true),
            pool2: MaxPool2d::square(2),
            // [B, 32, 8, 8] -> conv3 -> [B, 64, 8, 8] -> pool3 -> [B, 64, 4, 4]
            conv3: Conv2d::with_options(32, 64, (3, 3), (1, 1), (1, 1), (1, 1), true),
            pool3: MaxPool2d::square(2),
            fc1: Linear::new(64 * 4 * 4, 128),
            fc2: Linear::new(128, 10),
        }
    }
}

impl Module for Cifar10ConvNet {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let c1 = self.conv1.forward(input)?.relu()?;
        let p1 = self.pool1.forward(&c1)?;

        let c2 = self.conv2.forward(&p1)?.relu()?;
        let p2 = self.pool2.forward(&c2)?;

        let c3 = self.conv3.forward(&p2)?.relu()?;
        let p3 = self.pool3.forward(&c3)?;

        let b = input.shape()[0];
        let flat = p3.reshape(&[b, 64 * 4 * 4])?;

        let h = self.fc1.forward(&flat)?.relu()?;
        self.fc2.forward(&h)
    }

    fn parameters(&self) -> Vec<Tensor> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.conv3.parameters());
        params.extend(self.fc1.parameters());
        params.extend(self.fc2.parameters());
        params
    }
}

fn main() -> Result<()> {
    println!("============================================================");
    println!("   08_cifar10_convnet: 3-Channel RGB ConvNet on CIFAR-10    ");
    println!("============================================================\n");

    let max_samples = 600;
    let (dataset_x, dataset_y) = load_cifar10_dataset(Some(max_samples));
    let total_samples = dataset_y.len();
    println!(
        "Loaded {} samples of 3x32x32 RGB CIFAR-10 images across 10 classes",
        total_samples
    );

    // 80% train, 20% test
    let (train_x, train_y, test_x, test_y) = train_test_split(&dataset_x, &dataset_y, 0.20, true);
    let train_len = train_y.len();
    let test_len = test_y.len();
    println!(
        "Train set: {} images | Test set: {} images\n",
        train_len, test_len
    );

    let model = Cifar10ConvNet::new();
    let mut optimizer = Adam::new(model.parameters(), 0.003);

    let batch_size = 32;
    let num_batches = train_len.div_ceil(batch_size);

    println!(
        "Training CIFAR-10 ConvNet for 15 epochs (batch size {})...",
        batch_size
    );

    for epoch in 1..=15 {
        let mut total_loss = 0.0;
        let mut train_correct = 0;

        for b in 0..num_batches {
            let start = b * batch_size;
            let end = (start + batch_size).min(train_len);
            let b_len = end - start;

            let b_img = train_x.slice(0, start, end)?;
            let b_labels = &train_y[start..end];

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
                    train_correct += 1;
                }
            }
        }

        // Test evaluation
        let test_x_tensor = Tensor::new(test_x.clone(), false);
        let test_logits = model.forward(&test_x_tensor)?;
        let test_preds = test_logits.data().argmax(1)?;
        let mut test_correct = 0;
        for (&p, &l) in test_preds.iter().zip(test_y.iter()) {
            if p == l {
                test_correct += 1;
            }
        }

        let avg_loss = total_loss / (train_len as f32);
        let train_acc = (train_correct as f32) / (train_len as f32) * 100.0;
        let test_acc = (test_correct as f32) / (test_len as f32) * 100.0;

        if epoch % 3 == 0 || epoch == 1 || epoch == 15 {
            println!(
                "Epoch {:2}/15 | Avg Loss: {:6.4} | Train Acc: {:5.1}% | Test Acc: {:5.1}%",
                epoch, avg_loss, train_acc, test_acc
            );
        }
    }

    // Evaluate final test predictions
    let test_x_tensor = Tensor::new(test_x.clone(), false);
    let test_logits = model.forward(&test_x_tensor)?;
    let test_probs = test_logits.softmax(1)?;
    let test_preds = test_logits.data().argmax(1)?;

    println!("\n------------------------------------------------------------");
    println!("             Sample Test CIFAR-10 Predictions              ");
    println!("------------------------------------------------------------\n");

    let num_show = 6.min(test_len);
    for i in 0..num_show {
        let actual_idx = test_y[i];
        let pred_idx = test_preds[i];
        let conf = test_probs.data().get(&[i, pred_idx]) * 100.0;

        let status = if actual_idx == pred_idx { "✓" } else { "✗" };
        println!(
            "Sample {:2}: Actual = {:12} | Predicted = {:12} ({:5.1}% conf) [{}]",
            i + 1,
            CIFAR10_CLASSES[actual_idx],
            CIFAR10_CLASSES[pred_idx],
            conf,
            status
        );
    }

    // Save model weights to SafeTensors
    let _ = std::fs::create_dir_all("target");
    let save_path = "target/cifar10_model.safetensors";
    let mut tensor_map = HashMap::new();
    tensor_map.insert("conv1.weight".to_string(), model.conv1.weight.data());
    if let Some(ref b) = model.conv1.bias {
        tensor_map.insert("conv1.bias".to_string(), b.data());
    }
    tensor_map.insert("conv2.weight".to_string(), model.conv2.weight.data());
    if let Some(ref b) = model.conv2.bias {
        tensor_map.insert("conv2.bias".to_string(), b.data());
    }
    tensor_map.insert("conv3.weight".to_string(), model.conv3.weight.data());
    if let Some(ref b) = model.conv3.bias {
        tensor_map.insert("conv3.bias".to_string(), b.data());
    }
    tensor_map.insert("fc1.weight".to_string(), model.fc1.weight.data());
    if let Some(ref b) = model.fc1.bias {
        tensor_map.insert("fc1.bias".to_string(), b.data());
    }
    tensor_map.insert("fc2.weight".to_string(), model.fc2.weight.data());
    if let Some(ref b) = model.fc2.bias {
        tensor_map.insert("fc2.bias".to_string(), b.data());
    }

    save_safetensors(&tensor_map, save_path)?;
    println!("\nSaved trained CIFAR-10 model weights to {}", save_path);

    let loaded = load_safetensors(save_path)?;
    println!(
        "Successfully reloaded {} weight tensors from SafeTensors!",
        loaded.len()
    );

    println!("\nCIFAR-10 ConvNet training completed successfully!");
    Ok(())
}
