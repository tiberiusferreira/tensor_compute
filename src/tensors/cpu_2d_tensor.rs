use crate::{Gpu2DTensor, GpuBox};

#[derive(Debug)]
pub struct CpuTensor2D {
    data: Vec<f32>,
    shape: (usize, usize),
}

impl CpuTensor2D {
    pub fn new(data: Vec<f32>, shape: (usize, usize)) -> Self {
        assert_eq!(
            data.len(),
            shape.0 * shape.1,
            "Shape is not valid for the size of the data!"
        );
        Self { data, shape }
    }
}

impl CpuTensor2D {
    pub fn to_gpu(&self, gpu: &GpuBox) -> Gpu2DTensor {
        Gpu2DTensor::from_buffer(
            gpu.gpu_buffer_from_data(bytemuck::cast_slice(&self.data)),
            self.shape,
        )
    }
}
