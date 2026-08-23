//! WebGPU-backed Tensor (`GpuTensor`) with asynchronous VRAM storage and WGSL operations.

use crate::error::{EngineError, Result};
use crate::gpu::context::GpuContext;
use crate::tensor::RawTensor;
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// A multi-dimensional tensor stored in GPU VRAM with WebGPU compute acceleration.
#[derive(Clone)]
pub struct GpuTensor {
    pub buffer: Arc<wgpu::Buffer>,
    pub shape: Vec<usize>,
    pub ctx: Arc<GpuContext>,
}

impl GpuTensor {
    /// Creates a new `GpuTensor` from a host slice and shape.
    pub fn from_slice(slice: &[f32], shape: &[usize], ctx: &Arc<GpuContext>) -> Result<Self> {
        let expected_len: usize = shape.iter().product();
        if slice.len() != expected_len {
            return Err(EngineError::IncompatibleShapes {
                op: "GpuTensor::from_slice",
                shapes: vec![vec![slice.len()], shape.to_vec()],
            });
        }

        let buffer = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu_tensor_buffer"),
                contents: bytemuck::cast_slice(slice),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            });

        Ok(Self {
            buffer: Arc::new(buffer),
            shape: shape.to_vec(),
            ctx: ctx.clone(),
        })
    }

    /// Uploads a host `RawTensor` to GPU VRAM.
    pub fn from_raw(raw: &RawTensor, ctx: &Arc<GpuContext>) -> Result<Self> {
        let contig = raw.to_contiguous();
        Self::from_slice(contig.as_slice(), contig.shape(), ctx)
    }

    /// Downloads the VRAM tensor back to host memory as a `RawTensor`.
    pub fn to_cpu(&self) -> Result<RawTensor> {
        let num_elements: usize = self.shape.iter().product();
        let size_bytes = (num_elements * std::mem::size_of::<f32>()) as u64;

        let staging_buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_to_cpu_staging"),
            size: size_bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("to_cpu_encoder"),
            });

        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, size_bytes);
        self.ctx.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = sender.send(res);
        });

        self.ctx.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|e| {
                EngineError::GpuError(format!("GPU buffer mapping channel error: {:?}", e))
            })?
            .map_err(|e| EngineError::GpuError(format!("GPU buffer map async failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let float_slice: &[f32] = bytemuck::cast_slice(&data);
        let result_vec = float_slice.to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(RawTensor::from_vec(result_vec, self.shape.clone()))
    }

    /// Returns the shape of the GPU tensor.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the total number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    // =========================================================================
    // GPU Matrix Multiplication (GEMM)
    // =========================================================================

    /// Computes matrix multiplication `C = A * B` on the GPU.
    pub fn matmul(&self, other: &GpuTensor) -> Result<GpuTensor> {
        if self.shape.len() < 2 || other.shape.len() < 2 {
            return Err(EngineError::IncompatibleShapes {
                op: "GpuTensor::matmul",
                shapes: vec![self.shape.clone(), other.shape.clone()],
            });
        }

        let m = self.shape[self.shape.len() - 2];
        let k_a = self.shape[self.shape.len() - 1];
        let k_b = other.shape[other.shape.len() - 2];
        let n = other.shape[other.shape.len() - 1];

        if k_a != k_b {
            return Err(EngineError::IncompatibleShapes {
                op: "GpuTensor::matmul (inner dimensions mismatch)",
                shapes: vec![self.shape.clone(), other.shape.clone()],
            });
        }
        let k = k_a;

        let c_size_bytes = (m * n * std::mem::size_of::<f32>()) as u64;
        let c_buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_matmul_c_buffer"),
            size: c_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dims_data = [m as u32, k as u32, n as u32, 0u32];
        let dims_buffer = self
            .ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu_matmul_dims"),
                contents: bytemuck::cast_slice(&dims_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let pipeline = self
            .ctx
            .get_pipeline("gemm", include_str!("shaders/gemm.wgsl"))?;

        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpu_matmul_bind_group"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: other.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: c_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: dims_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_matmul_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_matmul_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups_x = (n as u32).div_ceil(64);
            let workgroups_y = (m as u32).div_ceil(64);
            pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }

        self.ctx.queue.submit(Some(encoder.finish()));

        Ok(GpuTensor {
            buffer: Arc::new(c_buffer),
            shape: vec![m, n],
            ctx: self.ctx.clone(),
        })
    }

    // =========================================================================
    // Elementwise Operations & Activations
    // =========================================================================

    fn dispatch_elementwise(
        &self,
        other: Option<&GpuTensor>,
        op: u32,
        scalar: f32,
    ) -> Result<GpuTensor> {
        let len = self.numel();
        let b_len = if let Some(o) = other {
            let o_len = o.numel();
            if self.shape != o.shape
                && o_len != 1
                && (self.shape.is_empty() || o_len != self.shape[self.shape.len() - 1])
            {
                return Err(EngineError::IncompatibleShapes {
                    op: "GpuTensor elementwise",
                    shapes: vec![self.shape.clone(), o.shape.clone()],
                });
            }
            o_len
        } else {
            1
        };

        let c_size_bytes = (len * std::mem::size_of::<f32>()) as u64;
        let c_buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_elem_c_buffer"),
            size: c_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dummy_b = if other.is_none() {
            Some(self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("gpu_dummy_b"),
                size: 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            }))
        } else {
            None
        };

        let b_binding = if let Some(o) = other {
            o.buffer.as_entire_binding()
        } else {
            dummy_b.as_ref().unwrap().as_entire_binding()
        };

        let params_data = [len as u32, op, scalar.to_bits(), b_len as u32];
        let params_buffer = self
            .ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu_elem_params"),
                contents: bytemuck::cast_slice(&params_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let pipeline = self
            .ctx
            .get_pipeline("elementwise", include_str!("shaders/elementwise.wgsl"))?;

        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpu_elem_bind_group"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: b_binding,
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: c_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_elem_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_elem_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (len as u32).div_ceil(256);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.ctx.queue.submit(Some(encoder.finish()));

        Ok(GpuTensor {
            buffer: Arc::new(c_buffer),
            shape: self.shape.clone(),
            ctx: self.ctx.clone(),
        })
    }

    pub fn add(&self, other: &GpuTensor) -> Result<GpuTensor> {
        self.dispatch_elementwise(Some(other), 0, 0.0)
    }

    pub fn sub(&self, other: &GpuTensor) -> Result<GpuTensor> {
        self.dispatch_elementwise(Some(other), 1, 0.0)
    }

    pub fn mul(&self, other: &GpuTensor) -> Result<GpuTensor> {
        self.dispatch_elementwise(Some(other), 2, 0.0)
    }

    pub fn div(&self, other: &GpuTensor) -> Result<GpuTensor> {
        self.dispatch_elementwise(Some(other), 3, 0.0)
    }

    pub fn relu(&self) -> Result<GpuTensor> {
        self.dispatch_elementwise(None, 4, 0.0)
    }

    pub fn gelu(&self) -> Result<GpuTensor> {
        self.dispatch_elementwise(None, 5, 0.0)
    }

    pub fn silu(&self) -> Result<GpuTensor> {
        self.dispatch_elementwise(None, 6, 0.0)
    }

    pub fn tanh(&self) -> Result<GpuTensor> {
        self.dispatch_elementwise(None, 7, 0.0)
    }

    pub fn sigmoid(&self) -> Result<GpuTensor> {
        self.dispatch_elementwise(None, 8, 0.0)
    }

    pub fn scale(&self, scalar: f32) -> Result<GpuTensor> {
        self.dispatch_elementwise(None, 9, scalar)
    }

    // =========================================================================
    // Reductions & Normalizations (Softmax, LayerNorm, RMSNorm)
    // =========================================================================

    /// Computes Softmax over the last dimension on the GPU.
    pub fn softmax(&self) -> Result<GpuTensor> {
        if self.shape.len() < 2 {
            return Err(EngineError::IncompatibleShapes {
                op: "GpuTensor::softmax (expected 2D tensor)",
                shapes: vec![self.shape.clone()],
            });
        }

        let rows = self.shape[0];
        let cols = self.shape[1];

        let c_size_bytes = (rows * cols * std::mem::size_of::<f32>()) as u64;
        let c_buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_softmax_c_buffer"),
            size: c_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let params_data = [rows as u32, cols as u32, 0u32, 0u32];
        let params_buffer = self
            .ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu_softmax_params"),
                contents: bytemuck::cast_slice(&params_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let pipeline = self
            .ctx
            .get_pipeline("softmax", include_str!("shaders/softmax.wgsl"))?;

        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpu_softmax_bind_group"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: c_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_softmax_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_softmax_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (rows as u32).div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.ctx.queue.submit(Some(encoder.finish()));

        Ok(GpuTensor {
            buffer: Arc::new(c_buffer),
            shape: self.shape.clone(),
            ctx: self.ctx.clone(),
        })
    }

    /// Computes LayerNorm across the last dimension on the GPU.
    pub fn layernorm(
        &self,
        gamma: &GpuTensor,
        beta: Option<&GpuTensor>,
        eps: f32,
    ) -> Result<GpuTensor> {
        let rows = self.shape[0];
        let cols = self.shape[1];

        let c_size_bytes = (rows * cols * std::mem::size_of::<f32>()) as u64;
        let c_buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_layernorm_c_buffer"),
            size: c_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dummy_beta = if beta.is_none() {
            Some(
                self.ctx
                    .device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("gpu_dummy_beta"),
                        contents: bytemuck::cast_slice(&vec![0.0f32; cols]),
                        usage: wgpu::BufferUsages::STORAGE,
                    }),
            )
        } else {
            None
        };

        let beta_binding = if let Some(b) = beta {
            b.buffer.as_entire_binding()
        } else {
            dummy_beta.as_ref().unwrap().as_entire_binding()
        };

        let params_data = [rows as u32, cols as u32, eps.to_bits(), 0u32]; // 0 = LayerNorm
        let params_buffer = self
            .ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu_layernorm_params"),
                contents: bytemuck::cast_slice(&params_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let pipeline = self
            .ctx
            .get_pipeline("layernorm", include_str!("shaders/layernorm.wgsl"))?;

        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpu_layernorm_bind_group"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: gamma.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: beta_binding,
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: c_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_layernorm_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_layernorm_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (rows as u32).div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.ctx.queue.submit(Some(encoder.finish()));

        Ok(GpuTensor {
            buffer: Arc::new(c_buffer),
            shape: self.shape.clone(),
            ctx: self.ctx.clone(),
        })
    }

    /// Computes RMSNorm across the last dimension on the GPU.
    pub fn rmsnorm(&self, gamma: &GpuTensor, eps: f32) -> Result<GpuTensor> {
        let rows = self.shape[0];
        let cols = self.shape[1];

        let c_size_bytes = (rows * cols * std::mem::size_of::<f32>()) as u64;
        let c_buffer = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_rmsnorm_c_buffer"),
            size: c_size_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let dummy_beta = self
            .ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu_dummy_beta_rms"),
                contents: bytemuck::cast_slice(&vec![0.0f32; cols]),
                usage: wgpu::BufferUsages::STORAGE,
            });

        let params_data = [rows as u32, cols as u32, eps.to_bits(), 1u32]; // 1 = RMSNorm
        let params_buffer = self
            .ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gpu_rmsnorm_params"),
                contents: bytemuck::cast_slice(&params_data),
                usage: wgpu::BufferUsages::UNIFORM,
            });

        let pipeline = self
            .ctx
            .get_pipeline("layernorm", include_str!("shaders/layernorm.wgsl"))?;

        let bind_group = self
            .ctx
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("gpu_rmsnorm_bind_group"),
                layout: &pipeline.get_bind_group_layout(0),
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: gamma.buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: dummy_beta.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: c_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: params_buffer.as_entire_binding(),
                    },
                ],
            });

        let mut encoder = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("gpu_rmsnorm_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("gpu_rmsnorm_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            let workgroups = (rows as u32).div_ceil(64);
            pass.dispatch_workgroups(workgroups, 1, 1);
        }

        self.ctx.queue.submit(Some(encoder.finish()));

        Ok(GpuTensor {
            buffer: Arc::new(c_buffer),
            shape: self.shape.clone(),
            ctx: self.ctx.clone(),
        })
    }
}
