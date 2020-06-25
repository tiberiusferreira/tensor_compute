use crate::gpu_buffers::GpuBuffer;
use crate::tensors::CpuTensor2D;
use crate::GpuBox;
use std::convert::TryInto;

pub struct Gpu2DTensor {
    buffer: GpuBuffer,
    shape: (usize, usize),
}

impl Gpu2DTensor {
    pub fn new(gpu: &GpuBox, data: Vec<f32>, shape: (usize, usize)) -> Self {
        assert_eq!(
            data.len(),
            shape.0 * shape.1,
            "Shape is not valid for the size of the data!"
        );
        CpuTensor2D::new(data, shape).to_gpu(gpu)
    }

    pub fn len(&self) -> usize{
        self.shape.0 * self.shape.1
    }

    pub fn from_buffer(buffer: GpuBuffer, shape: (usize, usize)) -> Self {
        Self { buffer, shape }
    }

    pub fn shape(&self) -> (usize, usize) {
        self.shape.clone()
    }

    pub fn reshape(&mut self, shape: (usize, usize)){
        assert_eq!(
            self.len(),
            shape.0 * shape.1,
            "Shape is not valid for the size of the data!"
        );
        self.shape = shape;
    }

    pub fn buffer(&self) -> &GpuBuffer {
        &self.buffer
    }

    pub fn buffer_size_in_bytes(&self) -> usize {
        std::mem::size_of::<f32>() * self.shape.0 * self.shape.1
    }

    pub async fn to_cpu(&self, gpu: &GpuBox) -> CpuTensor2D {
        let cpu_readable_output_buffer = gpu.staging_output_buffer(self.buffer_size_in_bytes());

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(
            self.buffer().raw_buffer(),
            0,
            &cpu_readable_output_buffer.raw_buffer(),
            0,
            self.buffer_size_in_bytes() as u64,
        );

        gpu.queue.submit(Some(encoder.finish()));

        let buffer_slice_a = cpu_readable_output_buffer.raw_buffer().slice(..);
        let buffer_future_a = buffer_slice_a.map_async(wgpu::MapMode::Read);

        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        gpu.device.poll(wgpu::Maintain::Wait);

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
            CpuTensor2D::new(result, self.shape)
        } else {
            panic!("Could not transfer data to CPU!")
        }
    }
}
