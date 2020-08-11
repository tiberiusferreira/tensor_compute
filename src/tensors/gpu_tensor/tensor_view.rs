use crate::gpu_internals::gpu_buffers::GpuBuffer;
use crate::gpu_internals::GpuInstance;
use crate::tensors::gpu_tensor::indexing::shape_strides_for_slice_range;
use crate::{AsShaderInput, GpuAllocated, GpuTensor, ShapeStrideTrait, ShapeStrides, SliceRangeInfo, MutShapeStrideTrait};
use std::collections::VecDeque;

/// A GpuTensorView share the same data as the original Tensor,
/// but can have different shapes and strides
/// For example, the original shape could be [2, 2] and the GpuTensorView could be [1, 2, 2]
pub struct GpuTensorView<'a> {
    original_tensor: &'a GpuTensor,
    pub shape_strides: ShapeStrides,
}

impl <'a> MutShapeStrideTrait for GpuTensorView<'a>{
    fn increase_rank(&mut self) {
        self.shape_strides.increase_rank();
    }

    fn decrease_rank(&mut self) {
        self.shape_strides.decrease_rank();
    }
}
/// Used to temporarily modify how the underlying tensor data is interpreted, by changing the
/// tensor shape or strides for example
impl<'a> GpuTensorView<'a> {

    pub fn from_tensor(gpu_tensor: &'a GpuTensor, dim_strides: ShapeStrides) -> Self {
        assert!(!gpu_tensor.is_empty(), "cant create a view from an empty Tensor");
        Self {
            original_tensor: &gpu_tensor,
            shape_strides: dim_strides,
        }
    }

    pub async fn to_tensor(&self) -> GpuTensor {
        self.contiguous().await
    }

    pub fn slice<T: Into<SliceRangeInfo>>(&self, bounds: Vec<T>) -> GpuTensorView {
        assert!(!bounds.is_empty(), "Empty bounds for slicing");
        let bounds: Vec<SliceRangeInfo> = bounds.into_iter().map(|e| e.into()).collect();
        let new_shape_strides = shape_strides_for_slice_range(&self.shape_strides, bounds);
        GpuTensorView::from_tensor(self.original_tensor, new_shape_strides)
    }

    pub fn buffer_size_in_bytes(&self) -> usize {
        self.original_tensor.internal_buffer_size_in_bytes()
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

}

impl<'a> GpuAllocated for GpuTensorView<'a> {
    fn get_gpu(&self) -> &'static GpuInstance {
        self.original_tensor.get_gpu()
    }

    fn internal_gpu_buffer(&self) -> &GpuBuffer {
        &self.original_tensor.internal_gpu_buffer()
    }
}

impl<'a> AsShaderInput for GpuTensorView<'a> {}

impl<'a> ShapeStrideTrait for GpuTensorView<'a> {
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
