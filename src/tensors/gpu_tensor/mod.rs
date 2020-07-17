mod accessors_contructors;
mod external_api;
pub use external_api::*;
mod tensor_view;
use crate::gpu_internals::gpu_buffers::GpuBuffer;
use crate::tensors::gpu_tensor::utils::strides_from_deque_shape;
use std::collections::VecDeque;
pub use tensor_view::*;

mod gpu_ops;
mod shape_changing;
pub mod utils;

pub struct GpuTensor {
    buffer: GpuBuffer,
    dim_stride: DimStride,
}


#[derive(Debug, Clone)]
pub struct DimStride {
    shape: VecDeque<usize>,
    strides: VecDeque<usize>,
}

impl DimStride {
    pub fn from_shape(shape: VecDeque<usize>) -> Self {
        DimStride {
            shape: shape.clone(),
            strides: strides_from_deque_shape(&shape),
        }
    }

    pub fn from_shape_and_strides(shape: VecDeque<usize>, strides: VecDeque<usize>) -> Self {
        DimStride { shape, strides }
    }

    pub fn from_shape_vec(shape: Vec<usize>) -> Self {
        let shape = VecDeque::from(shape);
        DimStride {
            shape: shape.clone(),
            strides: strides_from_deque_shape(&shape),
        }
    }

    pub fn from_shape_and_strides_vec(shape: Vec<usize>, strides: Vec<usize>) -> Self {
        let shape = VecDeque::from(shape);
        let strides = VecDeque::from(strides);
        DimStride {
            shape: shape.clone(),
            strides,
        }
    }

    /// Artificially increases the Tensor rank
    pub fn increase_rank(&mut self) {
        self.shape.push_front(1);
        self.strides.push_front(0);
    }

    pub fn rank(&mut self) -> usize {
        self.shape.len()
    }

    pub fn is_scalar(&self) -> bool {
        self.shape.len() == 1 && self.shape[0] == 1
    }
}

#[cfg(test)]
mod tests;
