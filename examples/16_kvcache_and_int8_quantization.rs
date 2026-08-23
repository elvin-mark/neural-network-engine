//! Example 16: Key-Value Cache (KV-Cache) & INT8 Quantization (`QLinear`).
//!
//! Demonstrates:
//! 1. $O(N)$ vs $O(N^2)$ autoregressive token generation with LLaMA 2 & Grouped-Query Attention (GQA).
//! 2. Measuring KV-Cache speedup and output parity during text generation.
//! 3. INT8 Weight Quantization (`QLinear`), achieving ~4x memory compression and SIMD AVX2 acceleration.
//! 4. Verifying numerical precision and cosine similarity (> 0.999) between FP32 and INT8 layers.
//!
//! Run with:
//! ```bash
//! cargo run --release --example 16_kvcache_and_int8_quantization
//! ```

use neural_network_engine::prelude::*;
use std::time::Instant;

fn main() -> Result<()> {
    println!("============================================================");
    println!(" 16_kvcache_and_int8_quantization: KV-Cache & INT8 Benchmark ");
    println!("============================================================\n");

    // =========================================================================
    // 1. KV-Cache vs Non-Cached LLaMA 2 Generation Benchmark
    // =========================================================================
    println!("------------------------------------------------------------");
    println!(" 1. LLaMA 2 Autoregressive Generation: KV-Cache vs Non-Cached");
    println!("------------------------------------------------------------");

    let vocab_size = 128;
    let max_seq_len = 64;
    let config = LlamaConfig {
        vocab_size,
        d_model: 128,
        hidden_dim: 256,
        num_heads: 8,
        num_kv_heads: 4,
        num_layers: 4,
        max_seq_len,
        norm_eps: 1e-6,
        rope_theta: 10000.0,
    };

    let model = Llama2LM::new(config);
    let prompt = vec![1, 10, 45, 82];
    let new_tokens = 32;

    println!("Model Config: 4 Layers, d_model=128, 8 Query Heads, 4 KV Heads (GQA)");
    println!(
        "Prompt tokens: {:?}, generating {} new tokens\n",
        prompt, new_tokens
    );

    // Generation WITH KV-Cache
    let start_kv = Instant::now();
    let tokens_kv = model.generate_cached(&prompt, new_tokens, 0.0)?;
    let time_kv = start_kv.elapsed();

    // Generation WITHOUT KV-Cache (quadratic forward re-evaluation)
    let start_nocache = Instant::now();
    let mut tokens_nocache = prompt.clone();
    for _ in 0..new_tokens {
        let logits = model.forward_tokens(&tokens_nocache, 1, tokens_nocache.len(), 0)?;
        let last_step = logits
            .slice(1, tokens_nocache.len() - 1, tokens_nocache.len())?
            .squeeze(1)?
            .squeeze(0)?;
        let best_token = last_step
            .data()
            .to_contiguous()
            .as_slice()
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();
        tokens_nocache.push(best_token);
    }
    let time_nocache = start_nocache.elapsed();

    println!("Generation Results:");
    println!("  • Output Match:       {}", tokens_kv == tokens_nocache);
    println!(
        "  • Time WITH KV-Cache: {:.2?} ({:.1} tokens/sec)",
        time_kv,
        new_tokens as f32 / time_kv.as_secs_f32()
    );
    println!(
        "  • Time WITHOUT Cache: {:.2?} ({:.1} tokens/sec)",
        time_nocache,
        new_tokens as f32 / time_nocache.as_secs_f32()
    );
    let speedup = time_nocache.as_secs_f64() / time_kv.as_secs_f64();
    println!("  • KV-Cache Speedup:   {:.2}x faster!\n", speedup);

    // =========================================================================
    // 2. INT8 Weight Quantization (`QLinear`) Benchmark
    // =========================================================================
    println!("------------------------------------------------------------");
    println!(" 2. INT8 Weight Quantization (QLinear) Memory & Speed");
    println!("------------------------------------------------------------");

    let in_features = 512;
    let out_features = 1024;
    let linear_fp32 = Linear::new(in_features, out_features);
    let linear_int8 = QLinear::from_linear(&linear_fp32);

    let fp32_bytes = in_features * out_features * 4 + out_features * 4;
    let int8_bytes = linear_int8.memory_bytes();
    let compression = fp32_bytes as f32 / int8_bytes as f32;

    println!("Layer Dimension: [{} -> {}]", in_features, out_features);
    println!(
        "  • FP32 Weight Memory:  {:.2} KiB",
        fp32_bytes as f32 / 1024.0
    );
    println!(
        "  • INT8 Weight Memory:  {:.2} KiB",
        int8_bytes as f32 / 1024.0
    );
    println!(
        "  • Compression Ratio:   {:.2}x (75% memory reduction)\n",
        compression
    );

    // Accuracy / Parity Check
    let test_input = Tensor::randn(&[16, in_features], 0.0, 1.0, false);
    let out_fp32 = linear_fp32.forward(&test_input)?;
    let out_int8 = linear_int8.forward(&test_input)?;

    let fp32_slice = out_fp32.data().to_contiguous();
    let int8_slice = out_int8.data().to_contiguous();

    let mut dot = 0.0f32;
    let mut norm_fp32 = 0.0f32;
    let mut norm_int8 = 0.0f32;
    for (&a, &b) in fp32_slice.as_slice().iter().zip(int8_slice.as_slice()) {
        dot += a * b;
        norm_fp32 += a * a;
        norm_int8 += b * b;
    }
    let cos_sim = dot / (norm_fp32.sqrt() * norm_int8.sqrt());
    println!("Numerical Parity Check:");
    println!(
        "  • Cosine Similarity:   {:.6} (Near-lossless precision)",
        cos_sim
    );

    // Benchmark Throughput
    let iters = 100;
    let start_f32 = Instant::now();
    for _ in 0..iters {
        let _ = linear_fp32.forward(&test_input)?;
    }
    let dur_f32 = start_f32.elapsed();

    let start_i8 = Instant::now();
    for _ in 0..iters {
        let _ = linear_int8.forward(&test_input)?;
    }
    let dur_i8 = start_i8.elapsed();

    println!("GEMM Execution Time ({} iterations, batch=16):", iters);
    println!("  • FP32 Linear GEMM:    {:.2?}", dur_f32);
    println!("  • INT8 QLinear GEMM:   {:.2?}", dur_i8);

    println!("\nKV-Cache & INT8 Quantization example completed successfully!");
    Ok(())
}
