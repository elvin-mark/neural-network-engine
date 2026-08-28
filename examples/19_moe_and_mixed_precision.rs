//! Example 19: Mixture of Experts (MoE) & Automatic Mixed Precision (AMP) Training.
//!
//! Demonstrates:
//! 1. Sparse Mixture of Experts (`MoELayer`, `TopKRouter`, `SparseMoEBlock`) with 8 experts and top-2 routing.
//! 2. Auxiliary load-balancing loss tracking to ensure uniform expert utilization.
//! 3. Dynamic Loss Scaling (`LossScaler`) for numerical stability.
//!
//! Run with:
//! ```bash
//! cargo run --release --example 19_moe_and_mixed_precision
//! ```

use neural_network_engine::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    println!("============================================================");
    println!(" 19: Mixture of Experts (MoE) & Mixed Precision Benchmark   ");
    println!("============================================================\n");

    // =========================================================================
    // 1. Sparse Mixture of Experts (MoE) Architecture Demo
    // =========================================================================
    println!("------------------------------------------------------------");
    println!(" 1. Sparse Mixture of Experts (Mixtral 8x7B-Style Routing)");
    println!("------------------------------------------------------------");

    let d_model = 64;
    let hidden_dim = 128;
    let num_experts = 8;
    let top_k = 2;

    println!("MoE Configuration:");
    println!("  • Model Dimension (d_model): {}", d_model);
    println!("  • Expert Hidden Dim:         {}", hidden_dim);
    println!("  • Total Experts:             {}", num_experts);
    println!(
        "  • Active Experts per Token:  {} (4x compute reduction)",
        top_k
    );
    println!(
        "  • Routing Mechanism:         Top-{} Softmax Router\n",
        top_k
    );

    let moe_config = MoEConfig {
        d_model,
        hidden_dim,
        num_experts,
        top_k,
        aux_loss_coeff: 0.01,
    };

    let moe_block = SparseMoEBlock::new(d_model, 4, moe_config);

    let batch_size = 4;
    let seq_len = 16;
    let tokens = batch_size * seq_len;
    let x = Tensor::randn(&[batch_size, seq_len, d_model], 0.0, 1.0, true);

    println!(
        "Input Tensor: [Batch={}, SeqLen={}, DModel={}] ({} tokens)",
        batch_size, seq_len, d_model, tokens
    );

    let start_moe = Instant::now();
    let (out, aux_loss) = moe_block.forward_with_aux(&x)?;
    let moe_fwd_time = start_moe.elapsed();

    println!("Output Tensor Shape: {:?}", out.shape());
    println!("Auxiliary Load-Balancing Loss: {:.6}", aux_loss.item());
    println!("MoE Block Forward Pass Time: {:.2?}\n", moe_fwd_time);

    // =========================================================================
    // 2. Training Loop with Dynamic Loss Scaling (AMP)
    // =========================================================================
    println!("------------------------------------------------------------");
    println!(" 2. Dynamic Loss Scaling (AMP LossScaler) Training Loop");
    println!("------------------------------------------------------------");

    let mut optimizer = Adam::new(moe_block.parameters(), 1e-3);
    let scaler = LossScaler::new(1024.0);

    println!("Initial Loss Scale: {}", scaler.current_scale());
    println!("Running 20 MoE training steps with AMP LossScaler...\n");

    for step in 1..=20 {
        let x_step = Tensor::randn(&[batch_size, seq_len, d_model], 0.0, 1.0, true);
        let target = Tensor::randn(&[batch_size, seq_len, d_model], 0.0, 1.0, false);

        optimizer.zero_grad();

        let (pred, aux) = moe_block.forward_with_aux(&x_step)?;
        let diff = pred.sub(&target)?;
        let task_loss = diff.mul(&diff)?.mean_all();
        let total_loss = task_loss.add(&aux)?;

        // Scale loss before backward
        let scaled_loss = scaler.scale(&total_loss)?;
        scaled_loss.backward();

        // Step optimizer with gradient unscaling and NaN protection
        let ok = scaler.step(&mut optimizer)?;

        if step % 5 == 0 || step == 1 {
            println!(
                "  Step {:2}/20 | Task Loss: {:.4} | Aux Loss: {:.4} | Scale: {:6.1} | Step OK: {}",
                step,
                task_loss.item(),
                aux.item(),
                scaler.current_scale(),
                ok
            );
        }
    }

    println!("\nMixture of Experts & Mixed Precision example completed successfully!");
    Ok(())
}
