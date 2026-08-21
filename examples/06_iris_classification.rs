//! Example 6: Fisher's Iris Dataset Classification using a Multi-Layer Perceptron (MLP).
//!
//! Demonstrates:
//! - Loading the canonical Fisher's Iris dataset (150 samples, 4 features, 3 classes).
//! - Feature standardization and train/test splitting (80% train / 20% test).
//! - Composing an MLP classifier with Sequential, Linear, and ReLU.
//! - Training with Adam optimizer and CrossEntropyLoss.
//! - Evaluating test accuracy, generating a Confusion Matrix, and running sample inference.

use neural_network_engine::prelude::*;

fn compute_accuracy(model: &Sequential, x: &RawTensor, y: &[usize]) -> Result<f32> {
    let _guard = NoGradGuard::new();
    let x_t = Tensor::new(x.clone(), false);
    let logits = model.forward(&x_t)?;
    let slice = logits.data().as_slice().to_vec();
    let num_classes = 3;
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

fn main() -> Result<()> {
    println!("============================================================");
    println!("   06_iris_classification: MLP on Fisher's Iris Dataset     ");
    println!("============================================================\n");

    const CLASS_NAMES: [&str; 3] = ["Iris-Setosa", "Iris-Versicolor", "Iris-Virginica"];

    // 1. Load canonical Iris dataset (150 samples, 4 features)
    let (raw_x, raw_y) = load_iris_dataset();
    println!(
        "Dataset loaded: {} samples, {} features each",
        raw_x.shape()[0],
        raw_x.shape()[1]
    );

    // 2. Feature standardization (z = (x - mean) / std)
    let (std_x, means, stds) = standardize(&raw_x);
    println!(
        "Feature Means:  [{:.2}, {:.2}, {:.2}, {:.2}]",
        means[0], means[1], means[2], means[3]
    );
    println!(
        "Feature Stds:   [{:.2}, {:.2}, {:.2}, {:.2}]",
        stds[0], stds[1], stds[2], stds[3]
    );

    // 3. Train/Test split (80% train = 120, 20% test = 30)
    let (train_x, train_y, test_x, test_y) = train_test_split(&std_x, &raw_y, 0.20, true);
    println!(
        "Train set size: {} samples | Test set size: {} samples\n",
        train_y.len(),
        test_y.len()
    );

    // 4. Construct Multi-Layer Perceptron (MLP)
    // 4 inputs -> 16 hidden -> ReLU -> 16 hidden -> ReLU -> 3 classes
    let model = Sequential::new()
        .add(Linear::new(4, 16))
        .add(ReLU)
        .add(Linear::new(16, 16))
        .add(ReLU)
        .add(Linear::new(16, 3));

    let mut optimizer = Adam::adamw(model.parameters(), 0.02, 1e-4);

    let train_dataset = TensorDataset::new(
        train_x.clone(),
        RawTensor::from_vec(
            train_y.iter().map(|&l| l as f32).collect(),
            vec![train_y.len(), 1],
        ),
    )?;

    println!("Starting training for 100 epochs...\n");
    let batch_size = 16;
    let epochs = 100;

    for epoch in 1..=epochs {
        let mut loader = DataLoader::new(&train_dataset, batch_size, true);
        let mut total_loss = 0.0;
        let mut batches = 0;

        for (batch_x, batch_y) in &mut loader {
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

        if epoch % 20 == 0 || epoch == epochs {
            let avg_loss = total_loss / (batches as f32);
            let train_acc = compute_accuracy(&model, &train_x, &train_y)?;
            let test_acc = compute_accuracy(&model, &test_x, &test_y)?;
            println!(
                "Epoch {:3}/{} | Avg Loss: {:.4} | Train Acc: {:5.1}% | Test Acc: {:5.1}%",
                epoch, epochs, avg_loss, train_acc, test_acc
            );
        }
    }

    // 5. Final Evaluation and Confusion Matrix on Test Set
    println!("\n------------------------------------------------------------");
    println!("               Test Set Evaluation & Matrix                ");
    println!("------------------------------------------------------------");

    let final_test_acc = compute_accuracy(&model, &test_x, &test_y)?;
    println!("Final Test Accuracy: {:.2}%\n", final_test_acc);

    // Compute 3x3 Confusion Matrix
    let _guard = NoGradGuard::new();
    let test_logits = model.forward(&Tensor::new(test_x.clone(), false))?;
    let test_slice = test_logits.data().as_slice().to_vec();
    let mut conf_matrix = [[0usize; 3]; 3]; // [actual][predicted]

    for i in 0..test_y.len() {
        let row = &test_slice[i * 3..(i + 1) * 3];
        let mut pred = 0;
        let mut max_val = f32::NEG_INFINITY;
        for (c, &v) in row.iter().enumerate() {
            if v > max_val {
                max_val = v;
                pred = c;
            }
        }
        conf_matrix[test_y[i]][pred] += 1;
    }

    println!("Confusion Matrix (Rows: Actual, Cols: Predicted):");
    println!("                | Setosa | Versicolor | Virginica |");
    println!("----------------+--------+------------+-----------+");
    for (i, row) in conf_matrix.iter().enumerate() {
        println!(
            "{:15} | {:6} | {:10} | {:9} |",
            CLASS_NAMES[i], row[0], row[1], row[2]
        );
    }
    println!("----------------+--------+------------+-----------+\n");

    // 6. Sample Interactive Predictions
    println!("Sample Test Predictions:");
    for i in 0..5.min(test_y.len()) {
        let row = &test_slice[i * 3..(i + 1) * 3];
        let mut pred = 0;
        let mut max_val = f32::NEG_INFINITY;
        for (c, &v) in row.iter().enumerate() {
            if v > max_val {
                max_val = v;
                pred = c;
            }
        }
        let actual = test_y[i];
        let mark = if pred == actual { "✓" } else { "✗" };
        println!(
            " Sample {:2}: Predicted = {:15} | Actual = {:15} [{}]",
            i + 1,
            CLASS_NAMES[pred],
            CLASS_NAMES[actual],
            mark
        );
    }

    println!("\nIris classification completed successfully!");
    Ok(())
}
