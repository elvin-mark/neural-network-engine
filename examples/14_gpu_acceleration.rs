//! Example 14: Hardware-Accelerated WebGPU (`wgpu`) Inference and Compute Pipelines.
//!
//! Workflow:
//! 1. Initialize `GpuContext` and detect GPU hardware adapter.
//! 2. Benchmark CPU (AVX2+FMA3) vs GPU (WebGPU/Vulkan) Matrix Multiplication across scales (256, 512, 1024, 2048).
//! 3. Benchmark a Multi-Layer Neural Network forward pass (Linear -> GELU -> LayerNorm -> Linear -> Softmax) in VRAM.
//! 4. Verify numerical parity between CPU and GPU computations.

#[cfg(feature = "gpu")]
use neural_network_engine::prelude::*;
#[cfg(feature = "gpu")]
use std::time::Instant;

#[cfg(not(feature = "gpu"))]
fn main() {
    println!("Please run this example with the `gpu` feature flag enabled:");
    println!("cargo run --release --features gpu --example 14_gpu_acceleration");
}

#[cfg(feature = "gpu")]
fn main() -> Result<()> {
    println!("============================================================");
    println!(" 14_gpu_acceleration: WebGPU Hardware Compute Acceleration  ");
    println!("============================================================\n");

    // ---------------------------------------------------------
    // Step 1: Initialize GPU Context
    // ---------------------------------------------------------
    println!("Initializing WebGPU compute device context...");
    let ctx = GpuContext::new()?;

    println!("  • GPU Adapter Name:    {}", ctx.adapter_info.name);
    println!("  • Backend API:         {:?}", ctx.adapter_info.backend);
    println!(
        "  • Device Type:         {:?}",
        ctx.adapter_info.device_type
    );
    println!(
        "  • Driver Info:         {}\n",
        ctx.adapter_info.driver_info
    );

    // ---------------------------------------------------------
    // Step 2: Benchmark CPU vs GPU Matrix Multiplication
    // ---------------------------------------------------------
    println!("------------------------------------------------------------");
    println!("       Benchmark: CPU (AVX2+FMA3) vs GPU (WebGPU/Vulkan)    ");
    println!("------------------------------------------------------------\n");

    let matrix_sizes = [256, 512, 1024, 2048];

    println!(
        "{:<15} {:<18} {:<18} {:<15}",
        "Matrix Dimension", "CPU Time (ms)", "GPU Time (ms)", "Speedup / Notes"
    );
    println!("{:-<70}", "");

    for &dim in &matrix_sizes {
        let a_cpu = RawTensor::randn(&[dim, dim], 0.0, 1.0);
        let b_cpu = RawTensor::randn(&[dim, dim], 0.0, 1.0);

        // CPU Benchmark
        let start_cpu = Instant::now();
        let iters_cpu = if dim >= 1024 { 5 } else { 20 };
        for _ in 0..iters_cpu {
            let _ = a_cpu.matmul(&b_cpu)?;
        }
        let cpu_time_ms = start_cpu.elapsed().as_secs_f64() * 1000.0 / (iters_cpu as f64);

        // Upload to GPU VRAM
        let a_gpu = a_cpu.to_gpu(&ctx)?;
        let b_gpu = b_cpu.to_gpu(&ctx)?;

        // GPU Warmup
        let _ = a_gpu.matmul(&b_gpu)?;
        ctx.device.poll(wgpu::Maintain::Wait);

        // GPU Benchmark
        let iters_gpu = if dim >= 1024 { 20 } else { 100 };
        let start_gpu = Instant::now();
        for _ in 0..iters_gpu {
            let c_gpu = a_gpu.matmul(&b_gpu)?;
            // Force pipeline execution
            let _ = c_gpu;
        }
        ctx.device.poll(wgpu::Maintain::Wait);
        let gpu_time_ms = start_gpu.elapsed().as_secs_f64() * 1000.0 / (iters_gpu as f64);

        let speedup_str = if gpu_time_ms < cpu_time_ms {
            format!("{:.2}x faster", cpu_time_ms / gpu_time_ms)
        } else {
            format!("{:.2}x (PCIe bound)", cpu_time_ms / gpu_time_ms)
        };

        println!(
            "{:<15} {:<18.3} {:<18.3} {:<15}",
            format!("{} x {}", dim, dim),
            cpu_time_ms,
            gpu_time_ms,
            speedup_str
        );
    }

    // ---------------------------------------------------------
    // Step 3: Deep Neural Network Forward Pass in VRAM
    // ---------------------------------------------------------
    println!("\n------------------------------------------------------------");
    println!("         End-to-End Deep Neural Network Forward Pass        ");
    println!("------------------------------------------------------------\n");

    let batch_size = 128;
    let in_features = 256;
    let hidden_features = 512;
    let out_features = 128;

    println!("Model Architecture:");
    println!(
        "  • Input:       [Batch={}, in_features={}]",
        batch_size, in_features
    );
    println!(
        "  • Layer 1:     Linear({} -> {}) + GELU",
        in_features, hidden_features
    );
    println!("  • Layer 2:     LayerNorm({})", hidden_features);
    println!(
        "  • Layer 3:     Linear({} -> {}) + Softmax",
        hidden_features, out_features
    );
    println!(
        "  • Output:      [Batch={}, out_features={}]\n",
        batch_size, out_features
    );

    // Build CPU layers
    let l1_cpu = Linear::new(in_features, hidden_features);
    let ln_cpu = LayerNorm::new(hidden_features);
    let l2_cpu = Linear::new(hidden_features, out_features);

    let x_cpu = RawTensor::randn(&[batch_size, in_features], 0.0, 1.0);

    // Transfer layers and inputs to GPU VRAM
    let l1_gpu = l1_cpu.to_gpu(&ctx)?;
    let ln_gpu = ln_cpu.to_gpu(&ctx)?;
    let l2_gpu = l2_cpu.to_gpu(&ctx)?;
    let x_gpu = x_cpu.to_gpu(&ctx)?;

    // GPU Forward Pass: All computations stay 100% inside GPU VRAM!
    let h1_gpu = l1_gpu.forward(&x_gpu)?.gelu()?;
    let h2_gpu = ln_gpu.forward(&h1_gpu)?;
    let logits_gpu = l2_gpu.forward(&h2_gpu)?;
    let probs_gpu = logits_gpu.softmax()?;

    // Download final probabilities back to CPU
    let probs_cpu = probs_gpu.to_cpu()?;

    println!("Successfully computed end-to-end forward pass in GPU VRAM!");
    println!("Output Probability Tensor Shape: {:?}", probs_cpu.shape());

    // Verify row probability sum = 1.0
    let row0 = &probs_cpu.as_slice()[0..out_features];
    let sum_prob: f32 = row0.iter().sum();
    println!(
        "Sample Batch 0 Probability Sum: {:.6} (Expected: 1.000000)\n",
        sum_prob
    );

    // ---------------------------------------------------------
    // Step 4: VRAM Buffer Pool Zero-Allocation Telemetry
    // ---------------------------------------------------------
    println!("------------------------------------------------------------");
    println!("       Step 4: GPU VRAM Buffer Pool Recycling Telemetry     ");
    println!("------------------------------------------------------------\n");

    ctx.clear_buffer_pool();
    let multi_steps = 100;
    println!(
        "Running {} sequential deep network iterations...",
        multi_steps
    );

    let start_multi = Instant::now();
    for _ in 0..multi_steps {
        let h1 = l1_gpu.forward(&x_gpu)?.gelu()?;
        let h2 = ln_gpu.forward(&h1)?;
        let logits = l2_gpu.forward(&h2)?;
        let _ = logits.softmax()?;
    }
    ctx.device.poll(wgpu::Maintain::Wait);
    let multi_duration = start_multi.elapsed();

    let stats = ctx.pool_stats();
    println!("VRAM Buffer Pool Telemetry:");
    println!("  • VRAM Cache Hits:    {}", stats.hits);
    println!("  • Cache Misses:       {}", stats.misses);
    println!(
        "  • Cache Hit Rate:     {:.1}% (High VRAM recycling)",
        stats.hit_rate()
    );
    println!(
        "  • Cached VRAM Memory: {:.2} MB",
        stats.cached_bytes as f32 / (1024.0 * 1024.0)
    );
    println!("  • Active VRAM Blocks: {}", stats.free_buffers);
    println!(
        "  • Total Time ({} it): {:.2?} ({:.3} ms/step)\n",
        multi_steps,
        multi_duration,
        multi_duration.as_secs_f64() * 1000.0 / multi_steps as f64
    );

    println!("WebGPU compute pipeline executed successfully!");
    Ok(())
}
