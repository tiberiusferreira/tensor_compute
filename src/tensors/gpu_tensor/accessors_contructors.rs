use crate::gpu_internals::gpu_buffers::GpuBuffer;
use crate::gpu_internals::GpuInstance;
use crate::{CpuTensor, GpuTensor, Tensor, GpuStore};
use std::fmt::{Debug, Formatter};
use std::collections::VecDeque;

impl Debug for GpuTensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gpu Tensor")
            .field("shape", &self.shape())
            .field("strides", &self.strides())
            .finish()
    }
}

impl GpuTensor {
    // Accessors
    pub fn storage(&self) -> &GpuBuffer {
        &self.buffer
    }

    pub fn get_gpu(&self) -> &GpuInstance {
        GpuStore::get(self.buffer.device_info())
    }

    pub fn buffer_size_in_bytes(&self) -> usize {
        self.storage().size_bytes()
    }

    pub fn from_data_with_gpu(gpu: &GpuInstance, data: Vec<f32>, shape: Vec<usize>) -> Self {
        let calc_size = shape.iter().rev().fold(1, |acc: usize, &x| acc * x);
        assert_eq!(
            calc_size,
            data.len(),
            "Shape is not valid for the size of the data!"
        );
        CpuTensor::new(data, shape).to_gpu(gpu)
    }

    pub fn from_buffer(buffer: GpuBuffer, shape: VecDeque<usize>) -> Self {
        Self {
            buffer,
            shape: shape.clone(),
            strides: super::utils::strides_from_deque_shape(&shape),
        }
    }

    pub fn from_buffer_with_strides(buffer: GpuBuffer, shape: VecDeque<usize>, strides: VecDeque<usize>) -> Self {
        Self {
            buffer,
            shape: shape.clone(),
            strides
        }
    }


    pub fn from_data(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let gpu = GpuStore::get_default();
        Self::from_data_with_gpu(gpu, data, shape)
    }

    pub fn from_scalar(data: f32) -> Self {
        let gpu = GpuStore::get_default();
        Self::from_data_with_gpu(gpu, vec![data], vec![1])
    }


    pub async fn to_cpu(&self) -> CpuTensor {
        let gpu = self.get_gpu();
        let buffer_in_cpu_mem = gpu.copy_buffer_to_cpu_mem(self.storage()).await;
        CpuTensor::new_with_strides(buffer_in_cpu_mem, self.shape.clone(), self.strides.clone())
    }

}

impl Tensor for GpuTensor {
    fn shape(&self) -> &VecDeque<usize> {
        &self.shape
    }

    fn strides(&self) -> &VecDeque<usize> {
        &self.strides
    }
}

