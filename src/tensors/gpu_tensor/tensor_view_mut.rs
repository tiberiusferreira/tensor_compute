use crate::gpu_internals::gpu_buffers::GpuBuffer;
use crate::gpu_internals::GpuInstance;
pub use crate::GpuAllocated;
use crate::{GpuTensor, ShapeStrideTrait, ShapeStrides, MutShapeStrideTrait};
use async_trait::async_trait;
use std::collections::VecDeque;
/// Same as GpuTensorView but mutable
pub struct GpuTensorViewMut<'a> {
    original_tensor: &'a mut GpuTensor,
    pub shape_strides: ShapeStrides,
}

#[async_trait(?Send)]
impl<'a> GpuAllocated for GpuTensorViewMut<'a> {
    fn get_gpu(&self) -> &'static GpuInstance {
        self.original_tensor.get_gpu()
    }

    fn internal_gpu_buffer(&self) -> &GpuBuffer {
        &self.original_tensor.internal_gpu_buffer()
    }

}

impl<'a> ShapeStrideTrait for GpuTensorViewMut<'a> {
    fn shape(&self) -> &VecDeque<usize> {
        &self.shape_strides.shape
    }

    fn strides(&self) -> &VecDeque<usize> {
        &self.shape_strides.strides
    }

    fn offset(&self) -> usize {
        self.shape_strides.offset
    }
}

impl<'a> GpuTensorViewMut<'a> {
    pub fn from_tensor(gpu_tensor: &'a mut GpuTensor, dim_strides: ShapeStrides) -> Self {
        Self {
            original_tensor: gpu_tensor,
            shape_strides: dim_strides,
        }
    }

}

impl <'a> MutShapeStrideTrait for GpuTensorViewMut<'a>{
    fn increase_rank(&mut self) {
        self.shape_strides.increase_rank();
    }

    fn decrease_rank(&mut self) {
        self.shape_strides.decrease_rank();
    }
}