use crate::gpu_internals::GpuInstance;
use std::convert::TryInto;
use wgpu::util::DeviceExt;
use wgpu::{AdapterInfo, Buffer};

#[derive(Debug)]
pub struct GpuBuffer {
    /// The WebGPU buffer itself
    buffer: Buffer,
    /// Which device this buffer was allocated in
    device_info: AdapterInfo,
    /// The size of this buffer
    size_bytes: usize,
    staging_output: bool,
}


impl GpuBuffer {
    pub fn layout(&self, binding: u32) -> wgpu::BindGroupLayoutEntry {
        wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStage::COMPUTE,
            ty: wgpu::BindingType::StorageBuffer {
                dynamic: false,
                readonly: false,
                min_binding_size: wgpu::BufferSize::new(4),
            },
            count: None
        }
    }
}

impl GpuBuffer {
    pub fn size_bytes(&self) -> usize {
        self.size_bytes
    }
    pub fn device_info(&self) -> &AdapterInfo {
        &self.device_info
    }
    pub fn is_staging_output(&self) -> bool {
        self.staging_output
    }
    pub fn to_bind_resource(&self) -> wgpu::BindingResource {
        wgpu::BindingResource::Buffer(self.buffer.slice(..))
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
            device_info: self.info().clone(),
        }
    }

    /// Storage buffer with given data. Behind the scenes it actually creates a new buffer, maps
    /// it into host-visible memory, copies data from the given slice,
    /// and finally unmaps it, returning a [`Buffer`].
    pub fn gpu_buffer_from_data(&self, input_bytes: &[u8]) -> GpuBuffer {
        let buffer_descriptor = wgpu::util::BufferInitDescriptor {
            label: None,
            contents: input_bytes,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
        };
        let buffer = self.device().create_buffer_init(&buffer_descriptor);
        GpuBuffer {
            buffer,
            size_bytes: input_bytes.len(),
            staging_output: false,
            device_info: self.info().clone(),
        }
    }

    /// Creates an empty GPU buffer which can be copied to another buffer.
    /// One used case if to accumulate results of a computation in it and copy them to an
    /// output staging buffer. Also used to store shader computation results.
    pub fn empty_gpu_buffer(&self, size_bytes: usize) -> GpuBuffer {
        let buffer = self.device().create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: size_bytes as u64,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
            mapped_at_creation: false,
        });
        GpuBuffer {
            buffer,
            size_bytes,
            staging_output: false,
            device_info: self.info().clone(),
        }
    }

    pub fn empty_like(&self, buffer: &GpuBuffer) -> GpuBuffer {
        self.empty_gpu_buffer(buffer.size_bytes)
    }

    pub async fn copy_buffer_to_cpu_mem(&self, src_buffer: &GpuBuffer) -> Vec<f32> {
        let gpu = self;
        let cpu_readable_output_buffer = gpu.staging_output_buffer(src_buffer.size_bytes());

        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(
            &src_buffer.buffer,
            0,
            &cpu_readable_output_buffer.buffer,
            0,
            src_buffer.size_bytes() as u64,
        );

        gpu.queue().submit(Some(encoder.finish()));

        let buffer_slice_a = cpu_readable_output_buffer.buffer.slice(..);
        let buffer_future_a = buffer_slice_a.map_async(wgpu::MapMode::Read);

        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        gpu.device().poll(wgpu::Maintain::Wait);

        if let Ok(()) = buffer_future_a.await {
            let data = buffer_slice_a.get_mapped_range();

            let result: Vec<f32> = data
                .chunks_exact(std::mem::size_of::<f32>())
                .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                .collect();

            // With the current interface, we have to make sure all mapped views are
            // dropped before we unmap the buffer.
            drop(data);
            cpu_readable_output_buffer.buffer.unmap();
            result
        } else {
            panic!("Could not transfer data to CPU!")
        }
    }
}
