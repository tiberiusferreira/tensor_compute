use crate::{GpuBox, Gpu2DTensor};

#[derive(Debug)]
pub struct CpuTensor2D {
    data: Vec<f32>,
    shape: (usize, usize),
}

impl CpuTensor2D{
    pub fn new(data: Vec<f32>, shape: (usize, usize)) -> Self {
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
