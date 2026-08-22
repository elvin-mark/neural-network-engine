//! WebGPU Device Context and Compute Pipeline caching.

use crate::error::{EngineError, Result};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// GPU execution context holding the initialized `wgpu::Device`, `wgpu::Queue`, and compiled shader cache.
pub struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter_info: wgpu::AdapterInfo,
    pipelines: RwLock<HashMap<String, Arc<wgpu::ComputePipeline>>>,
}

impl GpuContext {
    /// Initializes a new GPU context using the highest-performance available GPU adapter (Vulkan/Metal/DX12).
    pub fn new() -> Result<Arc<Self>> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: None,
            force_fallback_adapter: false,
        }))
        .ok_or_else(|| {
            EngineError::GpuError("Failed to find a compatible GPU adapter for WebGPU".to_string())
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
}
