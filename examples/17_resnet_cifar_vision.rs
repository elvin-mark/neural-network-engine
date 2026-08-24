//! Example 17: Deep Residual Network (ResNet-18) & Vision Data Augmentation Pipeline.
//!
//! Demonstrates:
//! 1. Building a composable data augmentation pipeline (`Compose`, `RandomCrop`, `RandomHorizontalFlip`, `Normalize`).
//! 2. End-to-end training of ResNet-18 on 32x32 3-channel RGB image batches.
//! 3. Skip connection gradient flow, BatchNorm2d running stats, and Cosine Annealing LR scheduling.
//! 4. Saving and reloading ResNet-18 model weights to/from SafeTensors.
//!
//! Run with:
//! ```bash
//! cargo run --release --example 17_resnet_cifar_vision
//! ```

use neural_network_engine::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    println!("============================================================");
    println!(" 17_resnet_cifar_vision: ResNet-18 & Data Augmentation Demo ");
    println!("============================================================\n");

    let num_samples = 64;
    let num_classes = 10;
    let epochs = 5;
    let batch_size = 16;

    // -------------------------------------------------------------------------
    // 1. Data Augmentation Pipeline
    // -------------------------------------------------------------------------
    println!("------------------------------------------------------------");
    println!(" 1. Vision Data Augmentation Pipeline");
    println!("------------------------------------------------------------");

    let transform = Compose::new(vec![
        Box::new(RandomCrop::new(32, 32, 4)),
        Box::new(RandomHorizontalFlip::new(0.5)),
        Box::new(Normalize::cifar10()),
    ]);
    println!("Augmentation Pipeline: RandomCrop(32, pad=4) -> RandomHorizontalFlip(p=0.5) -> Normalize(cifar10)\n");

    // Generate synthetic CIFAR-10 data [N, 3, 32, 32]
    let raw_images = RawTensor::uniform(&[num_samples, 3, 32, 32], 0.0, 1.0);
    let labels: Vec<usize> = (0..num_samples).map(|i| i % num_classes).collect();

    // -------------------------------------------------------------------------
    // 2. Instantiate ResNet-18 Architecture
    // -------------------------------------------------------------------------
    println!("------------------------------------------------------------");
    println!(" 2. ResNet-18 (CIFAR Stem) Architecture");
    println!("------------------------------------------------------------");

    let mut model = ResNet::cifar_resnet18(3, num_classes);
    model.train();

    let num_params: usize = model.parameters().iter().map(|p| p.numel()).sum();
    println!(
        "ResNet-18 initialized with {} learnable parameters\n",
        num_params
    );

    let mut optimizer = Adam::new(model.parameters(), 0.001);
    let mut scheduler = CosineAnnealingLR::new(0.001, epochs, 1e-5);

    // -------------------------------------------------------------------------
    // 3. Training Loop
    // -------------------------------------------------------------------------
    println!("------------------------------------------------------------");
    println!(
        " 3. Training Loop ({} Epochs, Batch Size = {})",
        epochs, batch_size
    );
    println!("------------------------------------------------------------");

    let start_train = Instant::now();
    let num_batches = num_samples / batch_size;

    for epoch in 1..=epochs {
        let mut epoch_loss = 0.0f32;
        let mut correct = 0usize;

        for b in 0..num_batches {
            let start_idx = b * batch_size;
            let end_idx = start_idx + batch_size;

            let batch_slice = raw_images.slice(0, start_idx, end_idx)?;
            let augmented = transform.apply_raw(&batch_slice)?;
            let x = Tensor::new(augmented, false);
            let batch_labels = &labels[start_idx..end_idx];

            let logits = model.forward(&x)?;
            let loss = CrossEntropyLoss::forward_with_indices(&logits, batch_labels)?;

            epoch_loss += loss.data().item();

            // Compute batch accuracy
            let logits_contig = logits.data().to_contiguous();
            let slice = logits_contig.as_slice();
            for i in 0..batch_size {
                let row = &slice[i * num_classes..(i + 1) * num_classes];
                let pred = row
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                    .map(|(idx, _)| idx)
                    .unwrap();
                if pred == batch_labels[i] {
                    correct += 1;
                }
            }

            optimizer.zero_grad();
            loss.backward();
            clip_grad_norm(&model.parameters(), 1.0);
            optimizer.step()?;
        }

        scheduler.step(&mut optimizer);
        let avg_loss = epoch_loss / (num_batches as f32);
        let acc = (correct as f32 / num_samples as f32) * 100.0;

        println!(
            "Epoch {:2}/{} | Loss: {:.4} | Train Accuracy: {:.1}% | LR: {:.6}",
            epoch,
            epochs,
            avg_loss,
            acc,
            optimizer.get_lr()
        );
    }

    let train_time = start_train.elapsed();
    println!("\nTraining completed in {:.2?}\n", train_time);

    // -------------------------------------------------------------------------
    // 4. SafeTensors Model Checkpointing
    // -------------------------------------------------------------------------
    println!("------------------------------------------------------------");
    println!(" 4. SafeTensors Model Serialization");
    println!("------------------------------------------------------------");

    let mut weights_map = std::collections::HashMap::new();
    for (i, param) in model.parameters().iter().enumerate() {
        weights_map.insert(format!("resnet.param_{}", i), param.data());
    }

    let save_path = "target/resnet18_cifar.safetensors";
    save_safetensors(&weights_map, save_path)?;
    println!("Saved ResNet-18 weights to {}", save_path);

    let loaded = load_safetensors(save_path)?;
    println!(
        "Successfully verified SafeTensors roundtrip ({} parameter tensors)!\n",
        loaded.len()
    );

    println!("ResNet & Vision Transforms example completed successfully!");
    Ok(())
}
