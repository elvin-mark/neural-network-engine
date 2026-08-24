//! Hardware-accelerated WebGPU compute backend for Linux (Vulkan), macOS (Metal), and Windows (DirectX 12).
//!
//! Features:
//! - Vectorized 4x4 Register-Tiled Matrix Multiplication (`GpuTensor::matmul`)
//! - Zero-Allocation VRAM Buffer Pool (`GpuBufferPool`, `GpuPoolStats`) for high-throughput GPU kernel recycling
//! - Fused Elementwise Arithmetic & Activations (Add, Sub, Mul, Div, ReLU, GELU, SiLU, Tanh, Sigmoid)
//! - Parallel Row-wise Reductions (Softmax, LayerNorm, RMSNorm)
//! - `ToGpu` transfer trait for seamless conversion between host CPU tensors and VRAM GPU buffers

pub mod context;
pub mod layers;
pub mod pool;
pub mod tensor;

pub use context::GpuContext;
pub use layers::{GpuLayerNorm, GpuLinear, GpuRMSNorm, ToGpu};
pub use pool::{GpuBufferPool, GpuPoolStats, PooledGpuBuffer};
pub use tensor::GpuTensor;
