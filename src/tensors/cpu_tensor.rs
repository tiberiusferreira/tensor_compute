use crate::{GpuInstance, GpuTensor};

#[derive(Debug)]
pub struct CpuTensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl CpuTensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let calc_size = shape.iter().rev().fold(1, |acc: usize, &x| acc * x);
        assert_eq!(
            calc_size,
            data.len(),
            "Shape is not valid for the size of the data!"
        );
        Self { data, shape }
    }
}

impl CpuTensor {
    pub fn to_gpu(&self, gpu: &GpuInstance) -> GpuTensor {
        GpuTensor::from_buffer(
            gpu.gpu_buffer_from_data(bytemuck::cast_slice(&self.data)),
            self.shape.clone(),
        )
    }
    pub fn data_slice(&self) -> &[f32] {
        self.data.as_slice()
    }
}
