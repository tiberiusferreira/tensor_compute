mod cpu_tensor;
mod gpu_tensor;
pub use cpu_tensor::*;
pub use gpu_tensor::*;
use std::collections::VecDeque;

pub trait Tensor {
    fn shape(&self) -> &VecDeque<usize>;
    fn strides(&self) -> &VecDeque<usize>;
    fn rank(&self) -> usize {
        self.shape().len()
    }
    fn numel(&self) -> usize {
        Self::numel_from_shape(self.shape())
    }
    fn numel_from_shape(shape: &VecDeque<usize>) -> usize {
        shape.iter().rev().fold(1, |acc: usize, &x| acc * x)
    }
}
