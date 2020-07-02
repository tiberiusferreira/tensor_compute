use crate::GpuInstance;
use wgpu::{AdapterInfo, Buffer, Device};
use std::convert::TryInto;

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
    pub fn device_info(&self) -> &AdapterInfo {
        &self.device_info
    }
    pub fn is_staging_output(&self) -> bool {
        self.staging_output
    }
}

impl GpuInstance {
    /// A Buffer which can be COPIED to from other buffers MAPPED to readonly CPU memory
    /// This is needed because we cant read STORAGE buffers directly
    pub fn new_staging_output_buffer(&self, size: usize) -> GpuBuffer {
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
    pub fn new_gpu_buffer_from_data(&self, input_bytes: &[u8]) -> GpuBuffer {
        let buffer = self.device().create_buffer_with_data(
            input_bytes,
            wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
        );
        GpuBuffer {
            buffer,
            size_bytes: input_bytes.len(),
            staging_output: false,
            device_info: self.info().clone(),
        }
    }

    /// Creates an empty GPU buffer which can be copied to another buffer.
    /// One used case if to accumulate results of a computation in it and copy them to an
    /// output staging buffer
    pub fn new_empty_gpu_buffer(&self, size: usize) -> GpuBuffer {
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
            device_info: self.info().clone(),
        }
    }


    pub async fn copy_to_cpu_mem(&self, src_buffer: &GpuBuffer) -> Vec<f32>{
        let gpu = self;
        let cpu_readable_output_buffer = gpu.new_staging_output_buffer(src_buffer.size_bytes());

        let mut encoder = gpu
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(
            src_buffer.raw_buffer(),
            0,
            &cpu_readable_output_buffer.raw_buffer(),
            0,
            src_buffer.size_bytes() as u64,
        );

        gpu.queue().submit(Some(encoder.finish()));

        let buffer_slice_a = cpu_readable_output_buffer.raw_buffer().slice(..);
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
            cpu_readable_output_buffer.raw_buffer().unmap();
            result
        } else {
            panic!("Could not transfer data to CPU!")
        }
    }
}
