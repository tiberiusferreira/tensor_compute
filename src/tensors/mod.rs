mod gpu_tensor;
mod cpu_tensor;
pub use cpu_tensor::*;
pub use gpu_tensor::*;

pub trait Tensor{
    fn shape(&self) -> Vec<usize>;
    fn strides(&self) -> Vec<usize>;
    fn numel(&self) -> usize{
        Self::numel_from_shape(self.shape().as_slice())
    }
    fn numel_from_shape(shape: &[usize]) -> usize{
        shape.iter().rev().fold(1, |acc: usize, &x| acc * x)
    }
}