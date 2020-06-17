use crate::{GpuBox, CpuTensor2D};
use wgpu::Buffer;
use std::convert::TryInto;

pub struct GpuBuffer{
    buffer: Buffer,
    shape: (u32, u32),
}

impl GpuBuffer{
    pub fn new(buffer: Buffer, shape: (u32, u32)) -> Self{
        Self{
            buffer,
            shape
        }
    }

    pub fn shape(&self) -> (u32, u32){
        self.shape.clone()
    }

    pub fn buffer(&self) -> &Buffer{
        &self.buffer
    }

    pub fn buffer_size_in_bytes(&self) -> u64{
        (std::mem::size_of::<u32>() as u32 * self.shape.0 * self.shape.0) as u64
    }

    pub async fn copy_to_cpu(&self, gpu: &GpuBox) -> CpuTensor2D{
        let cpu_readable_output_buffer = gpu.new_readonly_buffer(self.buffer_size_in_bytes());
        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        encoder.copy_buffer_to_buffer(self.buffer(), 0, &cpu_readable_output_buffer, 0, self.buffer_size_in_bytes());
        gpu.queue.submit(Some(encoder.finish()));
        let buffer_slice_a = cpu_readable_output_buffer.slice(..);
        let buffer_future_a = buffer_slice_a.map_async(wgpu::MapMode::Read);

        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        gpu.device.poll(wgpu::Maintain::Wait);

        if let Ok(()) = buffer_future_a.await {
            let data = buffer_slice_a.get_mapped_range();
            let result: Vec<u32> = data
                .chunks_exact(4)
                .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
                .collect();

            // With the current interface, we have to make sure all mapped views are
            // dropped before we unmap the buffer.
            drop(data);
            cpu_readable_output_buffer.unmap();
            CpuTensor2D{
                data: result,
                shape: self.shape
            }
        }else {
            panic!()
        }

        // unimplemented!()
    }
}

struct BoundGpuBuffer{
    buffer: Buffer,
    readonly: bool,
    binding: u32
}

impl BoundGpuBuffer{
    pub fn layout(&self) -> wgpu::BindGroupLayoutEntry{
        wgpu::BindGroupLayoutEntry {
            binding: self.binding,
            visibility: wgpu::ShaderStage::COMPUTE,
            ty: wgpu::BindingType::StorageBuffer {
                dynamic: false,
                readonly: self.readonly,
            },
            ..Default::default()
        }
    }
}

impl GpuBox {

    // pub fn new_readonly_buffer_new(&self, size: u64) -> GpuBuffer {
    //     GpuBuffer {
    //         buffer: self.device.create_buffer(&wgpu::BufferDescriptor {
    //         label: None,
    //         size,
    //         usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
    //         mapped_at_creation: false,
    //     }),
    //     }
    // }

    /// A Buffer which stores the result of a computation and can be mapped and read from the CPU
    /// This is needed because we cant read STORAGE buffers directly
    pub fn new_readonly_buffer(&self, size: u64) -> Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Storage buffer with given data. Behind the scenes it actually creates a new buffer, maps
    /// it into host-visible memory, copies data from the given slice,
    /// and finally unmaps it, returning a [`Buffer`].
    /// Although it might work, this buffer should not be copied to another
    pub fn new_gpu_storage_buffer_with_data(&self, input_bytes: &[u8]) -> Buffer{
        self.device.create_buffer_with_data(
            input_bytes,
            wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
        )
    }

    /// Same as [`new_input_storage_buffer`] but its contents can be copied to another buffer
    pub fn new_empty_gpu_storage_buffer(&self, size: u64) -> Buffer{
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsage::STORAGE | wgpu::BufferUsage::COPY_SRC,
            mapped_at_creation: false,
        })
    }
}
