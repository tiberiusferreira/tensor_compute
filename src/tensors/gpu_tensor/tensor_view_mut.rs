use crate::gpu_internals::gpu_buffers::GpuBuffer;
use crate::gpu_internals::GpuInstance;
use crate::tensors::gpu_tensor::indexing::shape_strides_for_slice_range;
use crate::{CpuTensor, GpuTensor, ShapeStrides, ShapeStrideTrait};
use std::collections::VecDeque;
use async_trait::async_trait;
pub use crate::GpuAllocated;
/// Same as GpuTensorView but mutable
pub struct GpuTensorViewMut<'a> {
    original_tensor: &'a mut GpuTensor,
    pub shape_strides: ShapeStrides,
}


#[async_trait(?Send)]
impl<'a> GpuAllocated for GpuTensorViewMut<'a>{
    fn get_gpu(&self) -> &'static GpuInstance {
        self.original_tensor.get_gpu()
    }

    fn internal_gpu_buffer(&self) -> &GpuBuffer {
        &self.original_tensor.internal_gpu_buffer()
    }

    // async fn to_cpu(&self) -> CpuTensor {
    //     let gpu = self.get_gpu();
    //     let buffer_in_cpu_mem = gpu.copy_buffer_to_cpu_mem(self.internal_gpu_buffer()).await;
    //     CpuTensor::new_with_strides_and_offset(
    //         buffer_in_cpu_mem,
    //         self.shape().clone(),
    //         self.strides().clone(),
    //         self.shape_strides.offset,
    //     )
    // }
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

    pub fn increase_rank(&mut self) {
        self.shape_strides.increase_rank();
    }
}


