//! Example 10: Vision Transformer (ViT) on 32x32 RGB CIFAR-10 Image Dataset.

use neural_network_engine::prelude::*;
use std::collections::HashMap;

fn main() -> Result<()> {
    println!("============================================================");
    println!("   10_cifar10_vit: Vision Transformer (ViT) on CIFAR-10    ");
    println!("============================================================\n");

    let max_samples = 600;
    let (dataset_x, dataset_y) = load_cifar10_dataset(Some(max_samples));
    let total_samples = dataset_y.len();
    println!(
        "Loaded {} samples of 3x32x32 RGB CIFAR-10 images across 10 categories",
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

    // ViT Hyperparameters:
    // Image: 32x32, Patches: 4x4 -> 64 tokens, d_model = 64, 3 layers, 4 attention heads
    let config = ViTConfig::cifar10();
    println!("ViT Architecture Configuration:");
    println!(
        "  • Input: {}x{} RGB (3 channels)",
        config.image_size, config.image_size
    );
    println!(
        "  • Patch Size: {}x{} (Total {} patches)",
        config.patch_size,
        config.patch_size,
        config.num_patches()
    );
    println!("  • Latent Dimension (d_model): {}", config.d_model);
    println!("  • Transformer Encoder Layers: {}", config.num_layers);
    println!("  • Self-Attention Heads: {}", config.num_heads);
    println!("  • MLP Hidden Dimension: {}\n", config.mlp_dim);

    let model = VisionTransformer::new(config);
    let mut optimizer = Adam::new(model.parameters(), 0.002);

    let batch_size = 32;
    let num_batches = train_len.div_ceil(batch_size);

    println!(
        "Training Vision Transformer for 15 epochs (batch size {})...",
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

    // Final evaluation & sample predictions
    let test_x_tensor = Tensor::new(test_x.clone(), false);
    let test_logits = model.forward(&test_x_tensor)?;
    let test_probs = test_logits.softmax(1)?;
    let test_preds = test_logits.data().argmax(1)?;

    println!("\n------------------------------------------------------------");
    println!("          Sample Test Predictions (Vision Transformer)      ");
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

    // Save model weights to SafeTensors format
    let _ = std::fs::create_dir_all("target");
    let save_path = "target/cifar10_vit_model.safetensors";
    let mut tensor_map = HashMap::new();

    tensor_map.insert(
        "patch_embed.weight".to_string(),
        model.patch_embed.weight.data(),
    );
    if let Some(ref b) = model.patch_embed.bias {
        tensor_map.insert("patch_embed.bias".to_string(), b.data());
    }
    tensor_map.insert("pos_embed".to_string(), model.pos_embed.data());

    for (idx, block) in model.blocks.iter().enumerate() {
        tensor_map.insert(
            format!("blocks.{}.ln1.weight", idx),
            block.ln1.weight.data(),
        );
        tensor_map.insert(format!("blocks.{}.ln1.bias", idx), block.ln1.bias.data());
        tensor_map.insert(
            format!("blocks.{}.attn.q_proj.weight", idx),
            block.attn.q_proj.weight.data(),
        );
        tensor_map.insert(
            format!("blocks.{}.attn.k_proj.weight", idx),
            block.attn.k_proj.weight.data(),
        );
        tensor_map.insert(
            format!("blocks.{}.attn.v_proj.weight", idx),
            block.attn.v_proj.weight.data(),
        );
        tensor_map.insert(
            format!("blocks.{}.attn.out_proj.weight", idx),
            block.attn.out_proj.weight.data(),
        );
        tensor_map.insert(
            format!("blocks.{}.ln2.weight", idx),
            block.ln2.weight.data(),
        );
        tensor_map.insert(format!("blocks.{}.ln2.bias", idx), block.ln2.bias.data());
        tensor_map.insert(
            format!("blocks.{}.mlp_fc1.weight", idx),
            block.mlp_fc1.weight.data(),
        );
        tensor_map.insert(
            format!("blocks.{}.mlp_fc2.weight", idx),
            block.mlp_fc2.weight.data(),
        );
    }

    tensor_map.insert("norm.weight".to_string(), model.norm.weight.data());
    tensor_map.insert("norm.bias".to_string(), model.norm.bias.data());
    tensor_map.insert("head.weight".to_string(), model.head.weight.data());
    if let Some(ref b) = model.head.bias {
        tensor_map.insert("head.bias".to_string(), b.data());
    }

    save_safetensors(&tensor_map, save_path)?;
    println!("\nSaved trained ViT model weights to {}", save_path);

    let loaded = load_safetensors(save_path)?;
    println!(
        "Successfully reloaded {} weight tensors from SafeTensors!",
        loaded.len()
    );

    println!("\nVision Transformer training on CIFAR-10 completed successfully!");
    Ok(())
}
