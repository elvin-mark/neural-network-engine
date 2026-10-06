# Hardware GPU Acceleration (`gpu`)

WebGPU compute pipeline powered by `wgpu` and custom WGSL compute shaders for zero-copy hardware acceleration across Vulkan, Metal, and DirectX 12.

**Source files**:
- Context & Device Management: [`src/gpu/context.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/gpu/context.rs)
- GPU Tensor: [`src/gpu/tensor.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/gpu/tensor.rs)
- GPU Layers: [`src/gpu/layers.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/gpu/layers.rs)
- Buffer Recycling Pool: [`src/gpu/pool.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/src/gpu/pool.rs)
- Integration Tests: [`tests/gpu_tests.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/tests/gpu_tests.rs)
- Example: [`examples/14_gpu_acceleration.rs`](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/examples/14_gpu_acceleration.rs)

---

## Enabling GPU Acceleration

Hardware GPU acceleration is an optional Cargo feature:

```bash
cargo run --release --features gpu --example 14_gpu_acceleration
```

```toml
[dependencies]
neural-network-engine = { version = "0.1", features = ["gpu"] }
```

---

## GPU Runtime Components

### 1. `GpuContext`
Initializes a `wgpu::Instance`, selects an optimal hardware adapter (High Performance Discrete GPU), and creates the logical compute device and submission queue.

```rust
let ctx = GpuContext::new()?; // Automatically picks fastest GPU backend
```

### 2. `GpuTensor` & Zero-Copy Pipeline
A `GpuTensor` represents memory residing entirely inside GPU VRAM (`wgpu::Buffer`):
- Eliminates host-device PCIe data transfers between consecutive layer executions.
- `ToGpu` trait transfers CPU `RawTensor` / `Tensor` to GPU memory:
  ```rust
  let gpu_x = cpu_tensor.to_gpu(&ctx)?;
  let gpu_out = gpu_layer.forward(&gpu_x)?;
  let cpu_out = gpu_out.to_cpu()?;
  ```

### 3. GPU Buffer Pool (`GpuBufferPool`)
Recycles allocated VRAM compute buffers across execution steps, preventing expensive driver-level memory allocations.

---

## WGSL Compute Shader Kernels

The engine embeds custom WebGPU Shading Language (WGSL) compute kernels:

| Kernel | WGSL Entry Point | Implementation Details |
|---|---|---|
| **Tiled Matrix Multiplication** | `matmul_tiled.wgsl` | $16 \times 16$ workgroup shared memory tiles (`var<workgroup> tile_a, tile_b`), cooperative loading, and inner accumulation loop |
| **Elementwise Operations** | `elementwise.wgsl` | Parallel vectorized addition, subtraction, multiplication, and ReLU activation |
| **LayerNorm** | `layernorm.wgsl` | Workgroup parallel reduction for mean and variance computation |
| **RMSNorm** | `rmsnorm.wgsl` | Fast root-mean-square reduction kernel |
| **Softmax** | `softmax.wgsl` | Row-wise max reduction and normalized exponential accumulation |

---

## See Also
- [Core Tensor Runtime](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/core/tensor.md)
- [Linear Layers](file:///home/elvin/Development/Repositories/elvin-mark/neural-network-engine/docs/nn/modules.md)
