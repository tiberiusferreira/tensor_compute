mod accessors_contructors;
mod external_api;
use crate::gpu_internals::gpu_buffers::GpuBuffer;
use crate::tensors::gpu_tensor::utils::strides_from_deque_shape;
pub use external_api::*;
pub use indexing::SliceRangeInfo;
use std::collections::VecDeque;
mod tensor_view;
pub use tensor_view::*;
mod tensor_view_mut;
use crate::gpu_internals::GpuInstance;
pub use tensor_view_mut::*;

mod gpu_ops;
mod indexing;
mod shape_changing;
pub mod utils;

pub struct GpuTensor {
    buffer: GpuBuffer,
    shape_strides: ShapeStrides,
}

use crate::gpu_internals::shader_runner::{ShaderInput, BufferType};
use zerocopy::AsBytes;

impl GpuTensor{
    pub fn to_shader_inputs(&self, binding_offset: usize) -> Vec<ShaderInput>{
        let shape: Vec<u32> = self.shape_strides.shape.iter().map(|&e| e as u32).collect();
        let strides: Vec<u32> = self.shape_strides.strides.iter().map(|&e| e as u32).collect();
        let shape_strides_len = self.shape_strides.shape.len() as u32;
        let offset = self.shape_strides.offset as u32;
        let shape_as_uniform = self.get_gpu().new_uniform_buffer(shape.as_slice().as_bytes());
        let strides_as_uniform = self.get_gpu().new_uniform_buffer(strides.as_slice().as_bytes());
        let shape_strides_len_as_uniform = self.get_gpu().new_uniform_buffer(shape_strides_len.as_bytes());
        let offset_as_uniform = self.get_gpu().new_uniform_buffer(offset.as_bytes());
        vec![
            ShaderInput{
                binding_id: binding_offset,
                gpu_buffer: BufferType::Storage(self.internal_gpu_buffer()),
            },
            ShaderInput{
                binding_id: binding_offset+1,
                gpu_buffer: BufferType::UniformOwned(shape_as_uniform),
            },
            ShaderInput{
                binding_id: binding_offset+2,
                gpu_buffer: BufferType::UniformOwned(strides_as_uniform),
            },
            ShaderInput{
                binding_id: binding_offset+3,
                gpu_buffer: BufferType::UniformOwned(shape_strides_len_as_uniform),
            },
            ShaderInput{
                binding_id: binding_offset+4,
                gpu_buffer: BufferType::UniformOwned(offset_as_uniform),
            }
        ]
    }
}
#[derive(Debug, Clone)]
pub struct ShapeStrides {
    shape: VecDeque<usize>,
    strides: VecDeque<usize>,
    offset: usize,
}

impl ShapeStrides {
    pub fn from_shape(shape: VecDeque<usize>) -> Self {
        ShapeStrides {
            shape: shape.clone(),
            strides: strides_from_deque_shape(&shape),
            offset: 0,
        }
    }

    pub fn from_shape_and_strides_and_offset(
        shape: VecDeque<usize>,
        strides: VecDeque<usize>,
        offset: usize,
    ) -> Self {
        ShapeStrides {
            shape,
            strides,
            offset,
        }
    }

    pub fn from_shape_vec(shape: Vec<usize>) -> Self {
        let shape = VecDeque::from(shape);
        ShapeStrides {
            shape: shape.clone(),
            strides: strides_from_deque_shape(&shape),
            offset: 0,
        }
    }

    pub fn from_shape_and_strides_vec(shape: Vec<usize>, strides: Vec<usize>) -> Self {
        let shape = VecDeque::from(shape);
        let strides = VecDeque::from(strides);
        ShapeStrides {
            shape: shape.clone(),
            strides,
            offset: 0,
        }
    }

    /// Artificially increases the Tensor rank
    pub fn increase_rank(&mut self) {
        self.shape.push_front(1);
        self.strides.push_front(0);
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn is_scalar(&self) -> bool {
        self.shape.len() == 1 && self.shape[0] == 1
    }
}

#[cfg(test)]
mod tests;
