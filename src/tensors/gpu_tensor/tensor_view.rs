use crate::gpu_internals::gpu_buffers::GpuBuffer;
use crate::{DimStride, GpuTensor, Tensor};
use std::collections::VecDeque;

/// A GpuTensorView share the same data as the original Tensor,
/// but can have different shapes and strides
/// For example, the original shape could be [2, 2] and the GpuTensorView could be [1, 2, 2]
pub struct GpuTensorView<'a> {
    buffer: &'a GpuBuffer,
    dim_strides: DimStride,
}

/// Used to temporarily modify how the underlying tensor data is interpreted, by changing the
/// tensor shape or strides for example
impl<'a> GpuTensorView<'a> {
    pub fn new(gpu_tensor: &'a GpuTensor, dim_strides: DimStride) -> Self {
        Self {
            buffer: gpu_tensor.storage(),
            dim_strides,
        }
    }

    pub fn buffer(&self) -> &'a GpuBuffer {
        &self.buffer
    }

    pub fn buffer_size_in_bytes(&self) -> usize {
        self.buffer.size_bytes()
    }

    pub fn shape(&self) -> &VecDeque<usize> {
        &self.dim_strides.shape
    }

    pub fn strides(&self) -> &VecDeque<usize> {
        &self.dim_strides.strides
    }

    pub fn dim_strides(&self) -> &DimStride {
        &self.dim_strides
    }

    pub fn increase_rank(&mut self) {
        self.dim_strides.increase_rank();
    }
}

impl<'a> Tensor for GpuTensorView<'a> {
    fn shape(&self) -> &VecDeque<usize> {
        self.shape()
    }

    fn strides(&self) -> &VecDeque<usize> {
        self.strides()
    }
}
