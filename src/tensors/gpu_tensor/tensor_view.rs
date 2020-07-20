use crate::gpu_internals::gpu_buffers::GpuBuffer;
use crate::gpu_internals::GpuInstance;
use crate::tensors::gpu_tensor::indexing::shape_strides_for_slice_range;
use crate::{CpuTensor, GpuTensor, ShapeStrides, SliceRangeInfo, TensorTrait};
use std::collections::VecDeque;

/// A GpuTensorView share the same data as the original Tensor,
/// but can have different shapes and strides
/// For example, the original shape could be [2, 2] and the GpuTensorView could be [1, 2, 2]
pub struct GpuTensorView<'a> {
    original_tensor: &'a GpuTensor,
    pub shape_strides: ShapeStrides,
}

/// Used to temporarily modify how the underlying tensor data is interpreted, by changing the
/// tensor shape or strides for example
impl<'a> GpuTensorView<'a> {
    pub fn from_tensor(gpu_tensor: &'a GpuTensor, dim_strides: ShapeStrides) -> Self {
        Self {
            original_tensor: &gpu_tensor,
            shape_strides: dim_strides,
        }
    }

    pub async fn to_tensor(&self) -> GpuTensor {
        self.contiguous().await
    }

    pub fn i<T: Into<SliceRangeInfo>>(&self, bounds: Vec<T>) -> GpuTensorView {
        let bounds: Vec<SliceRangeInfo> = bounds.into_iter().map(|e| e.into()).collect();
        let new_shape_strides = shape_strides_for_slice_range(&self.shape_strides, bounds);
        GpuTensorView::from_tensor(self.original_tensor, new_shape_strides)
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

    pub fn internal_gpu_buffer(&self) -> &'a GpuBuffer {
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

impl<'a> TensorTrait for GpuTensorView<'a> {
    fn shape(&self) -> &VecDeque<usize> {
        self.shape()
    }

    fn strides(&self) -> &VecDeque<usize> {
        self.strides()
    }
}
