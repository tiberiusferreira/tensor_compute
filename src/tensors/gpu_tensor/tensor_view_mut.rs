use crate::gpu_internals::gpu_buffers::GpuBuffer;
use crate::gpu_internals::GpuInstance;
use crate::tensors::gpu_tensor::indexing::shape_strides_for_slice_range;
use crate::{CpuTensor, GpuTensor, ShapeStrides, TensorTrait};
use std::collections::VecDeque;

/// Same as GpuTensorView but mutable
pub struct GpuTensorViewMut<'a> {
    original_tensor: &'a mut GpuTensor,
    pub shape_strides: ShapeStrides,
}

impl<'a> GpuTensorViewMut<'a> {
    pub fn from_tensor(gpu_tensor: &'a mut GpuTensor, dim_strides: ShapeStrides) -> Self {
        Self {
            original_tensor: gpu_tensor,
            shape_strides: dim_strides,
        }
    }

    pub fn get_gpu(&self) -> &'static GpuInstance {
        self.original_tensor.get_gpu()
    }

    pub async fn to_cpu(&self) -> CpuTensor {
        let gpu = self.get_gpu();
        let buffer_in_cpu_mem = gpu.copy_buffer_to_cpu_mem(self.internal_gpu_buffer()).await;
        CpuTensor::new_with_strides_and_offset(
            buffer_in_cpu_mem,
            self.shape().clone(),
            self.strides().clone(),
            self.shape_strides.offset,
        )
    }

    pub fn internal_gpu_buffer(&'a self) -> &'a GpuBuffer {
        &self.original_tensor.internal_gpu_buffer()
    }

    pub fn buffer_size_in_bytes(&self) -> usize {
        self.original_tensor.buffer_size_in_bytes()
    }

    pub fn shape(&self) -> &VecDeque<usize> {
        &self.shape_strides.shape
    }

    pub fn strides(&self) -> &VecDeque<usize> {
        &self.shape_strides.strides
    }

    pub fn dim_strides(&self) -> &ShapeStrides {
        &self.shape_strides
    }

    pub fn increase_rank(&mut self) {
        self.shape_strides.increase_rank();
    }
}

impl<'a> TensorTrait for GpuTensorViewMut<'a> {
    fn shape(&self) -> &VecDeque<usize> {
        self.shape()
    }

    fn strides(&self) -> &VecDeque<usize> {
        self.strides()
    }
}
