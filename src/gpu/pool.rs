//! VRAM GPU Buffer Pool (Caching Allocator) for WebGPU / Vulkan / Metal / DirectX.
//!
//! Eliminates driver syscall overhead (`vkAllocateMemory` / `device.create_buffer`) and prevents
//! GPU VRAM memory fragmentation during forward and backward passes.

use crate::gpu::context::GpuContext;
use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

/// Minimum bucket power of two (2^8 = 256 bytes, aligned with GPU uniform/storage offsets).
pub const GPU_MIN_BUCKET_POW2: usize = 8;
/// Maximum bucket power of two (2^30 = 1 GiB).
pub const GPU_MAX_BUCKET_POW2: usize = 30;

/// Telemetry statistics for GPU VRAM Buffer Pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GpuPoolStats {
    pub hits: usize,
    pub misses: usize,
    pub cached_bytes: usize,
    pub allocated_bytes: usize,
    pub free_buffers: usize,
}

impl GpuPoolStats {
    pub fn hit_rate(&self) -> f32 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            (self.hits as f32 / total as f32) * 100.0
        }
    }
}

impl fmt::Display for GpuPoolStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GpuPoolStats {{ hits: {}, misses: {}, hit_rate: {:.1}%, cached: {:.2} MB, active_vram_buffers: {} }}",
            self.hits,
            self.misses,
            self.hit_rate(),
            self.cached_bytes as f32 / (1024.0 * 1024.0),
            self.free_buffers
        )
    }
}

/// Size-bucketed VRAM memory pool for `wgpu::Buffer` objects.
pub struct GpuBufferPool {
    buckets: Vec<Vec<wgpu::Buffer>>,
    max_cached_bytes: usize,
    cached_bytes: usize,
    hits: usize,
    misses: usize,
    allocated_bytes: usize,
    enabled: bool,
}

impl GpuBufferPool {
    /// Creates a new GPU buffer pool with the given maximum VRAM cache limit.
    pub fn new(max_cached_bytes: usize) -> Self {
        let num_buckets = GPU_MAX_BUCKET_POW2 - GPU_MIN_BUCKET_POW2 + 1;
        let mut buckets = Vec::with_capacity(num_buckets);
        for _ in 0..num_buckets {
            buckets.push(Vec::new());
        }
        Self {
            buckets,
            max_cached_bytes,
            cached_bytes: 0,
            hits: 0,
            misses: 0,
            allocated_bytes: 0,
            enabled: true,
        }
    }

    #[inline]
    fn bucket_idx(size_bytes: u64) -> usize {
        let size = (size_bytes as usize).max(1 << GPU_MIN_BUCKET_POW2);
        let next_pow2 = size.next_power_of_two();
        let pow2 = next_pow2.trailing_zeros() as usize;
        let clamped_pow2 = pow2.clamp(GPU_MIN_BUCKET_POW2, GPU_MAX_BUCKET_POW2);
        clamped_pow2 - GPU_MIN_BUCKET_POW2
    }

    /// Acquires a VRAM buffer with at least `size_bytes` capacity and specified usage flags.
    pub fn pop(
        &mut self,
        device: &wgpu::Device,
        size_bytes: u64,
        usage: wgpu::BufferUsages,
    ) -> (wgpu::Buffer, u64) {
        let idx = Self::bucket_idx(size_bytes);
        let target_size = (1u64 << (idx + GPU_MIN_BUCKET_POW2)).max(size_bytes);

        if self.enabled {
            if let Some(buf) = self.buckets[idx].pop() {
                self.hits += 1;
                self.cached_bytes = self.cached_bytes.saturating_sub(target_size as usize);
                return (buf, target_size);
            }
        }

        self.misses += 1;
        self.allocated_bytes += target_size as usize;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_pooled_buffer"),
            size: target_size,
            usage,
            mapped_at_creation: false,
        });

        (buffer, target_size)
    }

    /// Recycles a `wgpu::Buffer` back into its VRAM bucket.
    pub fn push(&mut self, buffer: wgpu::Buffer, size_bytes: u64) {
        let bytes = size_bytes as usize;
        if !self.enabled || self.cached_bytes + bytes > self.max_cached_bytes {
            // Drop buffer back to GPU driver / VRAM
            return;
        }

        let idx = Self::bucket_idx(size_bytes);
        self.cached_bytes += bytes;
        self.buckets[idx].push(buffer);
    }

    /// Releases all cached VRAM buffers back to the GPU driver.
    pub fn clear(&mut self) {
        for b in &mut self.buckets {
            b.clear();
        }
        self.cached_bytes = 0;
    }

    /// Returns pool telemetry stats.
    pub fn stats(&self) -> GpuPoolStats {
        let free_buffers: usize = self.buckets.iter().map(|b| b.len()).sum();
        GpuPoolStats {
            hits: self.hits,
            misses: self.misses,
            cached_bytes: self.cached_bytes,
            allocated_bytes: self.allocated_bytes,
            free_buffers,
        }
    }

    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.clear();
        }
    }
}

/// RAII wrapper for a `wgpu::Buffer` that automatically recycles back into `GpuBufferPool` on drop.
pub struct PooledGpuBuffer {
    pub(crate) buffer: Option<wgpu::Buffer>,
    pub(crate) size_bytes: u64,
    pub(crate) ctx: Arc<GpuContext>,
}

impl PooledGpuBuffer {
    pub fn new(buffer: wgpu::Buffer, size_bytes: u64, ctx: Arc<GpuContext>) -> Self {
        Self {
            buffer: Some(buffer),
            size_bytes,
            ctx,
        }
    }
}

impl Deref for PooledGpuBuffer {
    type Target = wgpu::Buffer;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.buffer.as_ref().unwrap()
    }
}

impl DerefMut for PooledGpuBuffer {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.buffer.as_mut().unwrap()
    }
}

impl Drop for PooledGpuBuffer {
    fn drop(&mut self) {
        if let Some(buf) = self.buffer.take() {
            self.ctx.recycle_buffer(buf, self.size_bytes);
        }
    }
}
