//! WebGPU Device Context, Compute Pipeline caching, and VRAM Buffer Pool.

use crate::error::{EngineError, Result};
use crate::gpu::pool::{GpuBufferPool, GpuPoolStats, PooledGpuBuffer};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

/// GPU execution context holding the initialized `wgpu::Device`, `wgpu::Queue`, shader cache, and VRAM pool.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    pipelines: RwLock<HashMap<String, Arc<wgpu::ComputePipeline>>>,
    buffer_pool: Mutex<GpuBufferPool>,
}

impl GpuContext {
    /// Initializes a new GPU context using the highest-performance available GPU adapter (Vulkan/Metal/DX12).
    pub fn new() -> Result<Arc<Self>> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let mut adapters = instance.enumerate_adapters(wgpu::Backends::all());

        // 1. Prioritize Discrete GPU (e.g. NVIDIA RTX/Tesla/A100, AMD dGPU)
        // 2. Then Integrated GPU (e.g. AMD Radeon, Intel Iris, Apple Silicon)
        // 3. Fall back to standard request_adapter / software fallback
        let adapter = adapters
            .iter()
            .position(|a| a.get_info().device_type == wgpu::DeviceType::DiscreteGpu)
            .map(|pos| adapters.remove(pos))
            .or_else(|| {
                adapters
                    .iter()
                    .position(|a| a.get_info().device_type == wgpu::DeviceType::IntegratedGpu)
                    .map(|pos| adapters.remove(pos))
            })
            .or_else(|| adapters.into_iter().next())
            .or_else(|| {
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                }))
            })
            .or_else(|| {
                pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    compatible_surface: None,
                    force_fallback_adapter: true,
                }))
            })
            .ok_or_else(|| {
                EngineError::GpuError(
                    "Failed to find a compatible GPU adapter for WebGPU".to_string(),
                )
            })?;

        let adapter_info = adapter.get_info();

        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: Some("neural_network_engine_gpu_device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
            },
            None,
        ))
        .map_err(|e| EngineError::GpuError(format!("Failed to create GPU device: {:?}", e)))?;

        Ok(Arc::new(Self {
            device,
            queue,
            adapter_info,
            pipelines: RwLock::new(HashMap::new()),
            buffer_pool: Mutex::new(GpuBufferPool::new(1024 * 1024 * 1024)), // 1 GiB VRAM limit
        }))
    }

    /// Retrieves a compiled `wgpu::ComputePipeline` from cache or compiles the WGSL shader source.
    pub fn get_pipeline(
        &self,
        name: &str,
        shader_source: &str,
    ) -> Result<Arc<wgpu::ComputePipeline>> {
        {
            let cache = self.pipelines.read().unwrap();
            if let Some(pipeline) = cache.get(name) {
                return Ok(pipeline.clone());
            }
        }

        // Cache miss: Compile shader module
        let shader = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(name),
                source: wgpu::ShaderSource::Wgsl(shader_source.into()),
            });

        let pipeline = Arc::new(self.device.create_compute_pipeline(
            &wgpu::ComputePipelineDescriptor {
                label: Some(name),
                layout: None,
                module: &shader,
                entry_point: "main",
            },
        ));

        let mut cache = self.pipelines.write().unwrap();
        cache.insert(name.to_string(), pipeline.clone());
        Ok(pipeline)
    }

    /// Acquires a pooled VRAM buffer of at least `size_bytes` size.
    pub fn acquire_buffer(
        self: &Arc<Self>,
        size_bytes: u64,
        usage: wgpu::BufferUsages,
    ) -> Arc<PooledGpuBuffer> {
        let (buffer, target_size) =
            self.buffer_pool
                .lock()
                .unwrap()
                .pop(&self.device, size_bytes, usage);
        Arc::new(PooledGpuBuffer::new(buffer, target_size, self.clone()))
    }

    /// Recycles a `wgpu::Buffer` back into the VRAM pool.
    pub fn recycle_buffer(&self, buffer: wgpu::Buffer, size_bytes: u64) {
        self.buffer_pool.lock().unwrap().push(buffer, size_bytes);
    }

    /// Returns telemetry statistics for the GPU VRAM pool.
    pub fn pool_stats(&self) -> GpuPoolStats {
        self.buffer_pool.lock().unwrap().stats()
    }

    /// Clears all cached VRAM buffers, releasing GPU memory.
    pub fn clear_buffer_pool(&self) {
        self.buffer_pool.lock().unwrap().clear();
    }
}
