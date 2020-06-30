use crate::GpuInstance;
use wgpu::{AdapterInfo, Buffer};

pub struct GpuBuffer {
    buffer: Buffer,
    device: AdapterInfo,
    size_bytes: usize,
    staging_output: bool,
}

impl GpuBuffer {
    pub fn layout(&self, binding: usize) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry::new(
            binding as u32,
            wgpu::ShaderStage::COMPUTE,
            wgpu::BindingType::StorageBuffer {
                dynamic: false,
                min_binding_size: wgpu::BufferSize::new(4),
                readonly: false,
            },
        )
    }
}

impl GpuBuffer {
    pub fn raw_buffer(&self) -> &Buffer {
        &self.buffer
    }
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }
    pub fn device(&self) -> &AdapterInfo {
        &self.device
    }
    pub fn staging_output(&self) -> bool {
        self.staging_output
    }
}

impl GpuInstance {
    /// A Buffer which can be COPIED to from other buffers MAPPED to readonly CPU memory
    /// This is needed because we cant read STORAGE buffers directly
    pub fn staging_output_buffer(&self, size: usize) -> GpuBuffer {
        let buffer = self.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size as u64,
            usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        });
        GpuBuffer {
            buffer,
            size_bytes: size,
            staging_output: true,
            device: self.info().clone(),
        }
    }

    /// Storage buffer with given data. Behind the scenes it actually creates a new buffer, maps
    /// it into host-visible memory, copies data from the given slice,
    /// and finally unmaps it, returning a [`Buffer`].
    pub fn gpu_buffer_from_data(&self, input_bytes: &[u8]) -> GpuBuffer {
        let buffer = self.device().create_buffer_with_data(
            input_bytes,
            wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
        );
        GpuBuffer {
            buffer,
            size_bytes: input_bytes.len(),
            staging_output: false,
            device: self.info().clone(),
        }
    }

    /// Creates an empty GPU buffer which can be copied to another buffer.
    /// One used case if to accumulate results of a computation in it and copy them to an
    /// output staging buffer
    pub fn empty_gpu_buffer(&self, size: usize) -> GpuBuffer {
        let buffer = self.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size as u64,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
            mapped_at_creation: false,
        });
        GpuBuffer {
            buffer,
            size_bytes: size,
            staging_output: false,
            device: self.info().clone(),
        }
    }
}
