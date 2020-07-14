use crate::gpu_internals::gpu_buffers::GpuBuffer;
use crate::GpuTensor;
use std::collections::VecDeque;

pub struct GpuTensorView<'a> {
    buffer: &'a GpuBuffer,
    shape: VecDeque<usize>,
    strides: VecDeque<usize>,
}

/// Used to temporarily modify how the underlying tensor data is interpreted, by changing the
/// tensor shape or strides for example
impl<'a> GpuTensorView<'a> {
    pub fn new(gpu_tensor: &'a GpuTensor, shape: VecDeque<usize>, strides: VecDeque<usize>) -> Self {
        Self {
            buffer: gpu_tensor.storage(),
            shape,
            strides,
        }
    }

    pub fn buffer(&self) -> &'a GpuBuffer {
        &self.buffer
    }

    pub fn buffer_size_in_bytes(&self) -> usize {
        self.buffer.size_bytes()
    }

    pub fn shape(&self) -> &VecDeque<usize> {
        &self.shape
    }

    pub fn strides(&self) -> &VecDeque<usize> {
        &self.strides
    }
}
