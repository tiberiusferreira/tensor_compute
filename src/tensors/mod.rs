mod cpu_tensor;
mod gpu_tensor;
pub use cpu_tensor::*;
pub use gpu_tensor::*;
use std::collections::VecDeque;

pub trait TensorTrait {
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

pub struct Tensor{
    storage: GpuTensor,
}

impl Tensor{
    pub fn from_vec(vec: Vec<f32>) -> Self{
        let len = vec.len();
        let g_tensor = GpuTensor::from_data(vec, vec![len]);
        Tensor{
            storage: g_tensor
        }
    }
    pub async fn zeros(shape: &[usize]) -> Self{
        Self{
            storage: GpuTensor::new_filled(shape.to_vec(), 0.).await
        }
    }

    pub fn shape(&self) -> Vec<usize>{
        Vec::from(self.storage.shape().clone())
    }

    pub fn strides(&self) -> Vec<usize>{
        Vec::from(self.storage.strides().clone())
    }

    pub async fn zeros_like(other: &Self) -> Self{
        Self::zeros(&other.shape()).await
    }

    pub async fn fill_with(&mut self, value: f32){
        self.storage.fill_with(value).await;
    }
}

pub trait ExternalTensorInterface{
    /* Constructors, there are proxies to these in the Tape */
    fn from_vec(slice: Vec<f32>) -> Self;
    fn zeros(shape: &[usize]) -> Self;
    fn rand(shape: &[usize]) -> Self;
    fn zeros_like(other: &Self) -> Self;
    fn empty() -> Self;

    /* Helper functions */
    fn is_empty(&self) -> bool;
    fn fill_with(&mut self, value: f32);

    /* Shape Changing functions */
    /// Transposes dim 0 and 1, panics if they don't exist
    fn t(&mut self);
    fn reshape(&mut self, shape: &[usize]);
    fn shape(&self) -> &[usize];

    /* Basic Ops */
    fn add(&self, rhs: &Self) -> Self;
    fn sub(&self, rhs: &Self) -> Self;
    fn mul(&self, rhs: &Self) -> Self;

    /* Basic Ops Scalar */
    fn add_scalar(&self, rhs: f32) -> Self;
    fn sub_scalar(&self, rhs: f32) -> Self;
    fn mul_scalar(&self, rhs: f32) -> Self;

    /// sums all elements
    fn sum(&self) -> f32;
    fn matmul2d(&self, rhs: &Self) -> Self;

    // Operating on all elements
    fn map_inplace<F>(&mut self, f: F) where F: FnMut(&mut f32);

    fn index(&self, index: &[usize]) -> f32;
    fn _index_mut(&mut self, index: &[usize]) -> &mut f32;
}
