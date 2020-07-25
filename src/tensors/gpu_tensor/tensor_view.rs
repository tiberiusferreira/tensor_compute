use crate::gpu_internals::gpu_buffers::GpuBuffer;
use crate::gpu_internals::GpuInstance;
use crate::tensors::gpu_tensor::indexing::shape_strides_for_slice_range;
use crate::{GpuTensor, ShapeStrides, SliceRangeInfo, ShapeStrideTrait, GpuAllocated, AsShaderInput};
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
    // pub fn to_shader_inputs(&self, binding_offset: usize) -> Vec<ShaderInput>{
    //     let shape: Vec<u32> = self.shape_strides.shape.iter().map(|&e| e as u32).collect();
    //     let strides: Vec<u32> = self.shape_strides.strides.iter().map(|&e| e as u32).collect();
    //     let shape_strides_len = self.shape_strides.shape.len() as u32;
    //     let offset = self.shape_strides.offset as u32;
    //     let shape_as_uniform = self.get_gpu().new_uniform_buffer(shape.as_slice().as_bytes());
    //     let strides_as_uniform = self.get_gpu().new_uniform_buffer(strides.as_slice().as_bytes());
    //     let shape_strides_len_as_uniform = self.get_gpu().new_uniform_buffer(shape_strides_len.as_bytes());
    //     let offset_as_uniform = self.get_gpu().new_uniform_buffer(offset.as_bytes());
    //     vec![
    //         ShaderInput{
    //             binding_id: binding_offset,
    //             gpu_buffer: BufferType::Storage(self.internal_gpu_buffer()),
    //         },
    //         ShaderInput{
    //             binding_id: binding_offset+1,
    //             gpu_buffer: BufferType::UniformOwned(shape_as_uniform),
    //         },
    //         ShaderInput{
    //             binding_id: binding_offset+2,
    //             gpu_buffer: BufferType::UniformOwned(strides_as_uniform),
    //         },
    //         ShaderInput{
    //             binding_id: binding_offset+3,
    //             gpu_buffer: BufferType::UniformOwned(shape_strides_len_as_uniform),
    //         },
    //         ShaderInput{
    //             binding_id: binding_offset+4,
    //             gpu_buffer: BufferType::UniformOwned(offset_as_uniform),
    //         }
    //     ]
    // }

    pub fn from_tensor(gpu_tensor: &'a GpuTensor, dim_strides: ShapeStrides) -> Self {
        Self {
            original_tensor: &gpu_tensor,
            shape_strides: dim_strides,
        }
    }

    pub async fn to_tensor(&self) -> GpuTensor {
        self.contiguous().await
    }

    pub fn slice<T: Into<SliceRangeInfo>>(&self, bounds: Vec<T>) -> GpuTensorView {
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

    pub fn increase_rank(&mut self) {
        self.shape_strides.increase_rank();
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

impl<'a> AsShaderInput for GpuTensorView<'a>{}

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
