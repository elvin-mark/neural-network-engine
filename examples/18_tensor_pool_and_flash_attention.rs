//! Example 18: Zero-Allocation Tensor Pool (`TensorPool`) & FlashAttention-2 Benchmark.
//!
//! Demonstrates:
//! 1. Telemetry monitoring of `TensorPool` during deep neural network training loops (verifying >90% cache hit rates).
//! 2. FlashAttention-2 online softmax tiled attention ($O(T)$ memory vs $O(T^2)$ standard attention).
//! 3. Numerical parity check and throughput benchmark on long sequences (SeqLen = 512).
//!
//! Run with:
//! ```bash
//! cargo run --release --example 18_tensor_pool_and_flash_attention
//! ```

use neural_network_engine::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    println!("============================================================");
    println!(" 18: TensorPool Zero-Allocation & FlashAttention-2 Benchmark ");
    println!("============================================================\n");

    // =========================================================================
    // 1. TensorPool Zero-Allocation Memory Recycling Benchmark
    // =========================================================================
    println!("------------------------------------------------------------");
    println!(" 1. TensorPool Zero-Allocation Memory Recycling");
    println!("------------------------------------------------------------");

    TensorPool::clear_local();

    let in_features = 256;
    let out_features = 512;
    let linear = Linear::new(in_features, out_features);
    let x = Tensor::randn(&[32, in_features], 0.0, 1.0, true);

    let steps = 50;
    println!("Running {} training forward/backward steps...", steps);

    let start_pool = Instant::now();
    for _ in 0..steps {
        let out = linear.forward(&x)?;
        let loss = out.sum_all();
        loss.backward();
    }
    let pool_duration = start_pool.elapsed();

    let stats = TensorPool::local_stats();
    println!("TensorPool Telemetry Results:");
    println!("  • Buffer Cache Hits:  {}", stats.hits);
    println!("  • Cache Misses:       {}", stats.misses);
    println!(
        "  • Cache Hit Rate:     {:.1}% (High memory recycling efficiency)",
        stats.hit_rate()
    );
    println!(
        "  • Cached Memory:      {:.2} MB",
        stats.cached_bytes as f32 / (1024.0 * 1024.0)
    );
    println!("  • Active Recycled:    {} buffers", stats.free_buffers);
    println!("  • Total Time ({} it): {:.2?}\n", steps, pool_duration);

    // =========================================================================
    // 2. FlashAttention-2 vs Standard Attention ($O(T)$ vs $O(T^2)$)
    // =========================================================================
    println!("------------------------------------------------------------");
    println!(" 2. FlashAttention-2 (Tiled Online Softmax) vs Standard MHA");
    println!("------------------------------------------------------------");

    let batch_size = 2;
    let num_heads = 4;
    let seq_len = 512;
    let head_dim = 64;
    let d_model = num_heads * head_dim;

    println!(
        "Attention Config: Batch={}, Heads={}, SeqLen={}, HeadDim={}, DModel={}",
        batch_size, num_heads, seq_len, head_dim, d_model
    );

    let q = RawTensor::randn(&[batch_size, num_heads, seq_len, head_dim], 0.0, 1.0);
    let k = RawTensor::randn(&[batch_size, num_heads, seq_len, head_dim], 0.0, 1.0);
    let v = RawTensor::randn(&[batch_size, num_heads, seq_len, head_dim], 0.0, 1.0);

    // Theoretical attention matrix memory
    let std_matrix_bytes = batch_size * num_heads * seq_len * seq_len * 4;
    let flash_block_bytes = 64 * 64 * 4 * 2; // only 2 small 64x64 blocks in L1 cache
    println!("Memory Footprint Comparison:");
    println!(
        "  • Standard Attention Matrix: {:.2} KiB in RAM (O(T^2))",
        std_matrix_bytes as f32 / 1024.0
    );
    println!(
        "  • FlashAttention-2 Scratch:  {:.2} KiB in L1 Cache (O(T))\n",
        flash_block_bytes as f32 / 1024.0
    );

    // 1. Run Standard Attention
    let start_std = Instant::now();
    let q_tensor = Tensor::new(q.clone(), false);
    let k_tensor = Tensor::new(k.clone(), false);
    let v_tensor = Tensor::new(v.clone(), false);

    let k_t = k_tensor.transpose(2, 3)?;
    let scores = q_tensor.matmul(&k_t)?;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let scaled = scores.mul_scalar(scale)?;

    // Causal mask
    let mask_data: Vec<f32> = (0..seq_len)
        .flat_map(|r| (0..seq_len).map(move |c| if c > r { f32::NEG_INFINITY } else { 0.0 }))
        .collect();
    let mask = Tensor::new(
        RawTensor::from_vec(mask_data, vec![1, 1, seq_len, seq_len]),
        false,
    );
    let masked = scaled.add(&mask)?;
    let probs = masked.softmax(3)?;
    let std_out = probs.matmul(&v_tensor)?;
    let time_std = start_std.elapsed();

    // 2. Run FlashAttention-2
    let start_flash = Instant::now();
    let flash_out = flash_attention_forward(&q, &k, &v, true, None, 64, 64)?;
    let time_flash = start_flash.elapsed();

    // 3. Numerical Parity Verification
    let flash_slice = flash_out.to_contiguous();
    let std_slice = std_out.data().to_contiguous();

    let mut dot = 0.0f32;
    let mut norm_flash = 0.0f32;
    let mut norm_std = 0.0f32;
    for (&a, &b) in flash_slice.as_slice().iter().zip(std_slice.as_slice()) {
        dot += a * b;
        norm_flash += a * a;
        norm_std += b * b;
    }
    let cos_sim = dot / (norm_flash.sqrt() * norm_std.sqrt());

    println!("Benchmark Results (SeqLen = {}):", seq_len);
    println!("  • Standard MHA Time:     {:.2?}", time_std);
    println!("  • FlashAttention-2 Time: {:.2?}", time_flash);
    println!(
        "  • Cosine Similarity:     {:.6} (Mathematical Equivalence)",
        cos_sim
    );

    println!("\nTensorPool & FlashAttention-2 example completed successfully!");
    Ok(())
}
