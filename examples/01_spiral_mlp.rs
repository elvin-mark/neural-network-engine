//! Example 1: Non-linear 2D Spiral Dataset Classification with Multi-Layer Perceptron (MLP).

use neural_network_engine::prelude::*;

fn main() -> Result<()> {
    println!("============================================================");
    println!("  01_spiral_mlp: Non-linear 2D Spiral Classification Demo   ");
    println!("============================================================\n");

    // 1. Generate 3-class 2D spiral dataset (100 points per spiral = 300 points total)
    let (features_raw, labels) = generate_spiral_dataset(100, 3, 0.2);
    let total_samples = labels.len();
    println!(
        "Generated spiral dataset with {} samples and 3 classes.",
        total_samples
    );

    let x = Tensor::new(features_raw, false);

    // 2. Build Multi-Layer Perceptron: 2 -> 64 (ReLU) -> 64 (ReLU) -> 3 (Logits)
    let model = Sequential::new()
        .add(Linear::new(2, 64))
        .add(ReLU)
        .add(Linear::new(64, 64))
        .add(ReLU)
        .add(Linear::new(64, 3));

    // 3. Setup Adam optimizer with learning rate = 0.01
    let mut optimizer = Adam::new(model.parameters(), 0.01);

    println!("\nStarting training loop (200 epochs)...");
    for epoch in 1..=200 {
        // Forward pass
        let logits = model.forward(&x)?;
        let loss = CrossEntropyLoss::forward_with_indices(&logits, &labels)?;

        // Backward pass
        optimizer.zero_grad();
        loss.backward();

        // Optimizer parameter update step
        optimizer.step()?;

        // Compute classification accuracy
        let logits_data = logits.data();
        let predictions = logits_data.argmax(1)?;
        let correct = predictions
            .iter()
            .zip(labels.iter())
            .filter(|(&pred, &target)| pred == target)
            .count();
        let accuracy = (correct as f32) / (total_samples as f32) * 100.0;

        if epoch % 20 == 0 || epoch == 1 {
            println!(
                "Epoch {:3}/200 | Loss: {:8.5} | Training Accuracy: {:6.2}% ({}/{})",
                epoch,
                loss.item(),
                accuracy,
                correct,
                total_samples
            );
        }
    }

    println!("\nTraining complete! Successfully solved non-linear spiral boundary.");
    Ok(())
}
