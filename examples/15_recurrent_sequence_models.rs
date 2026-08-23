//! Example 15: Recurrent Neural Networks (RNN, LSTM, GRU) for Sequence Modeling.
//!
//! Demonstrates:
//! 1. Multi-step sequence modeling on a non-linear sine wave temporal forecasting task.
//! 2. Side-by-side training comparison of Elman RNN, LSTM, and GRU architectures.
//! 3. Gradient clipping (`clip_grad_norm`) and Cosine Annealing learning rate scheduling.
//! 4. Bidirectional sequence processing and parameter serialization via SafeTensors.
//!
//! Run with:
//! ```bash
//! cargo run --release --example 15_recurrent_sequence_models
//! ```

use neural_network_engine::prelude::*;
use std::time::Instant;

fn generate_sine_sequence_dataset(num_samples: usize, seq_len: usize) -> (RawTensor, RawTensor) {
    let mut x_vec = Vec::with_capacity(num_samples * seq_len);
    let mut y_vec = Vec::with_capacity(num_samples);

    for i in 0..num_samples {
        let offset = (i as f32) * 0.15;
        for t in 0..seq_len {
            let t_val = offset + (t as f32) * 0.1;
            let val = (t_val).sin() + 0.5 * (2.0 * t_val).cos();
            x_vec.push(val);
        }
        // Predict the very next step in the sequence
        let next_t = offset + (seq_len as f32) * 0.1;
        let target = (next_t).sin() + 0.5 * (2.0 * next_t).cos();
        y_vec.push(target);
    }

    let x = RawTensor::from_vec(x_vec, vec![num_samples, seq_len, 1]);
    let y = RawTensor::from_vec(y_vec, vec![num_samples, 1]);
    (x, y)
}

fn main() -> Result<()> {
    println!("============================================================");
    println!(" 15_recurrent_sequence_models: RNN vs LSTM vs GRU Benchmark ");
    println!("============================================================\n");

    let num_samples = 64;
    let seq_len = 16;
    let input_size = 1;
    let hidden_size = 32;
    let epochs = 30;

    let (x_raw, y_raw) = generate_sine_sequence_dataset(num_samples, seq_len);
    println!(
        "Generated temporal sequence dataset: {} samples, seq_len={}, input_dim={}\n",
        num_samples, seq_len, input_size
    );

    // -------------------------------------------------------------------------
    // 1. Elman RNN Model
    // -------------------------------------------------------------------------
    println!("------------------------------------------------------------");
    println!(" 1. Training Elman RNN (2 Layers, Tanh)");
    println!("------------------------------------------------------------");
    let rnn = RNN::new(input_size, hidden_size, 2, RNNActivation::Tanh, false, 0.0);
    let rnn_fc = Linear::new(hidden_size, 1);
    let mut rnn_params = rnn.parameters();
    rnn_params.extend(rnn_fc.parameters());
    let mut rnn_opt = Adam::new(rnn_params.clone(), 0.02);
    let mut rnn_scheduler = CosineAnnealingLR::new(0.02, epochs, 0.001);

    let start_rnn = Instant::now();
    let mut rnn_final_loss = 0.0f32;
    for epoch in 1..=epochs {
        let x = Tensor::new(x_raw.clone(), false);
        let y = Tensor::new(y_raw.clone(), false);

        let (_out, h_n) = rnn.forward_seq(&x, None)?;
        // h_n is [num_layers, batch_size, hidden_size] -> take last layer
        let last_h = h_n.slice(0, 1, 2)?.squeeze(0)?;
        let pred = rnn_fc.forward(&last_h)?;

        let loss = MSELoss::forward(&pred, &y)?;
        let loss_val = loss.data().item();
        rnn_final_loss = loss_val;

        rnn_opt.zero_grad();
        loss.backward();
        clip_grad_norm(&rnn_params, 1.0);
        rnn_opt.step()?;
        rnn_scheduler.step(&mut rnn_opt);

        if epoch % 10 == 0 || epoch == 1 {
            println!(
                "Epoch {:2}/{} | RNN MSE Loss: {:.6} | LR: {:.5}",
                epoch,
                epochs,
                loss_val,
                rnn_opt.get_lr()
            );
        }
    }
    let rnn_time = start_rnn.elapsed();
    println!("RNN Training Time: {:.2?}\n", rnn_time);

    // -------------------------------------------------------------------------
    // 2. Long Short-Term Memory (LSTM) Model
    // -------------------------------------------------------------------------
    println!("------------------------------------------------------------");
    println!(" 2. Training Long Short-Term Memory (LSTM) (2 Layers)");
    println!("------------------------------------------------------------");
    let lstm = LSTM::new(input_size, hidden_size, 2, false, 0.0);
    let lstm_fc = Linear::new(hidden_size, 1);
    let mut lstm_params = lstm.parameters();
    lstm_params.extend(lstm_fc.parameters());
    let mut lstm_opt = Adam::new(lstm_params.clone(), 0.02);
    let mut lstm_scheduler = CosineAnnealingLR::new(0.02, epochs, 0.001);

    let start_lstm = Instant::now();
    let mut lstm_final_loss = 0.0f32;
    for epoch in 1..=epochs {
        let x = Tensor::new(x_raw.clone(), false);
        let y = Tensor::new(y_raw.clone(), false);

        let (_out, (h_n, _c_n)) = lstm.forward_seq(&x, None)?;
        let last_h = h_n.slice(0, 1, 2)?.squeeze(0)?;
        let pred = lstm_fc.forward(&last_h)?;

        let loss = MSELoss::forward(&pred, &y)?;
        let loss_val = loss.data().item();
        lstm_final_loss = loss_val;

        lstm_opt.zero_grad();
        loss.backward();
        clip_grad_norm(&lstm_params, 1.0);
        lstm_opt.step()?;
        lstm_scheduler.step(&mut lstm_opt);

        if epoch % 10 == 0 || epoch == 1 {
            println!(
                "Epoch {:2}/{} | LSTM MSE Loss: {:.6} | LR: {:.5}",
                epoch,
                epochs,
                loss_val,
                lstm_opt.get_lr()
            );
        }
    }
    let lstm_time = start_lstm.elapsed();
    println!("LSTM Training Time: {:.2?}\n", lstm_time);

    // -------------------------------------------------------------------------
    // 3. Gated Recurrent Unit (GRU) Model
    // -------------------------------------------------------------------------
    println!("------------------------------------------------------------");
    println!(" 3. Training Gated Recurrent Unit (GRU) (2 Layers)");
    println!("------------------------------------------------------------");
    let gru = GRU::new(input_size, hidden_size, 2, false, 0.0);
    let gru_fc = Linear::new(hidden_size, 1);
    let mut gru_params = gru.parameters();
    gru_params.extend(gru_fc.parameters());
    let mut gru_opt = Adam::new(gru_params.clone(), 0.02);
    let mut gru_scheduler = CosineAnnealingLR::new(0.02, epochs, 0.001);

    let start_gru = Instant::now();
    let mut gru_final_loss = 0.0f32;
    for epoch in 1..=epochs {
        let x = Tensor::new(x_raw.clone(), false);
        let y = Tensor::new(y_raw.clone(), false);

        let (_out, h_n) = gru.forward_seq(&x, None)?;
        let last_h = h_n.slice(0, 1, 2)?.squeeze(0)?;
        let pred = gru_fc.forward(&last_h)?;

        let loss = MSELoss::forward(&pred, &y)?;
        let loss_val = loss.data().item();
        gru_final_loss = loss_val;

        gru_opt.zero_grad();
        loss.backward();
        clip_grad_norm(&gru_params, 1.0);
        gru_opt.step()?;
        gru_scheduler.step(&mut gru_opt);

        if epoch % 10 == 0 || epoch == 1 {
            println!(
                "Epoch {:2}/{} | GRU MSE Loss: {:.6} | LR: {:.5}",
                epoch,
                epochs,
                loss_val,
                gru_opt.get_lr()
            );
        }
    }
    let gru_time = start_gru.elapsed();
    println!("GRU Training Time: {:.2?}\n", gru_time);

    // -------------------------------------------------------------------------
    // 4. Comparison Summary
    // -------------------------------------------------------------------------
    println!("============================================================");
    println!("                Recurrent Architecture Benchmark Summary     ");
    println!("============================================================");
    println!("  Model       Final MSE Loss      Training Time");
    println!("------------------------------------------------------------");
    println!(
        "  Elman RNN   {:.6}            {:.2?}",
        rnn_final_loss, rnn_time
    );
    println!(
        "  LSTM        {:.6}            {:.2?}",
        lstm_final_loss, lstm_time
    );
    println!(
        "  GRU         {:.6}            {:.2?}",
        gru_final_loss, gru_time
    );
    println!("============================================================\n");

    // -------------------------------------------------------------------------
    // 5. Bidirectional Sequence Testing & SafeTensors Serialization
    // -------------------------------------------------------------------------
    println!("Testing Bidirectional GRU forward & backward pass...");
    let bi_gru = GRU::new(input_size, 16, 1, true, 0.0);
    let test_x = Tensor::new(x_raw.slice(0, 0, 4)?, false); // [4, 16, 1]
    let (bi_out, bi_hn) = bi_gru.forward_seq(&test_x, None)?;
    println!(
        "Bidirectional GRU output shape: {:?} (2 * hidden_size = 32), h_n shape: {:?}",
        bi_out.shape(),
        bi_hn.shape()
    );

    // Save model parameters to SafeTensors
    let mut tensors = std::collections::HashMap::new();
    for (i, p) in bi_gru.parameters().iter().enumerate() {
        tensors.insert(format!("bi_gru.param_{}", i), p.data());
    }
    save_safetensors(&tensors, "target/bidirectional_gru.safetensors")?;
    println!("Saved Bidirectional GRU model weights to target/bidirectional_gru.safetensors");

    let loaded = load_safetensors("target/bidirectional_gru.safetensors")?;
    println!(
        "Successfully reloaded {} parameter tensors from SafeTensors!\n",
        loaded.len()
    );

    println!("Recurrent sequence modeling example completed successfully!");
    Ok(())
}
