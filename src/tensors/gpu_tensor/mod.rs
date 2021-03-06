mod accessors_contructors;
use crate::gpu_internals::gpu_buffers::GpuBuffer;
use crate::tensors::gpu_tensor::utils::strides_from_deque_shape;
pub use indexing::SliceRangeInfo;
use std::collections::VecDeque;

pub mod traits;
pub use traits::*;
mod gpu_ops;
mod indexing;
mod shape_changing;
pub mod utils;

pub struct GpuTensor {
    buffer: GpuBuffer,
    shape_strides: ShapeStrides,
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

    /// Increases the Tensor rank
    pub fn increase_rank(&mut self) {
        self.shape.push_front(1);
        self.strides = strides_from_deque_shape(&self.shape);
    }

    /// Decrease the Tensor rank
    pub fn decrease_rank(&mut self) {
        assert_eq!(
            self.shape.pop_front().unwrap(),
            1,
            "Cant decrease rank of Tensor when its leading dimension is not unitary."
        );
        self.strides.pop_front();
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
