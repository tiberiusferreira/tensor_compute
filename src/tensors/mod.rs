use crate::gpu_buffers::GpuBuffer;
use crate::GpuBox;
mod gpu_2d_tensor;
pub use gpu_2d_tensor::Gpu2DTensor;
use std::convert::TryInto;

#[derive(Debug)]
pub struct CpuTensor2D {
    data: Vec<u32>,
    shape: (usize, usize),
}

impl CpuTensor2D {
    pub fn new(data: Vec<u32>, shape: (usize, usize)) -> Self {
        Self { data, shape }
    }
}

impl CpuTensor2D {
    pub fn copy_to_gpu(&self, gpu: &GpuBox) -> Gpu2DTensor {
        Gpu2DTensor::new(
            gpu.gpu_buffer_from_data(bytemuck::cast_slice(&self.data)),
            self.shape,
        )
    }
}
