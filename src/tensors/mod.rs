mod cpu_tensor;
mod gpu_tensor;
use blocking::block_on;
pub use cpu_tensor::*;
pub use gpu_tensor::*;
mod traits;
pub use traits::*;
use std::collections::VecDeque;
use std::fmt::{Debug, Formatter};


pub struct Tensor {
    actual_tensor: GpuTensor,
}

impl Debug for Tensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.actual_tensor.fmt(f)
    }
}

pub struct TensorView<'a> {
    actual_tensor: GpuTensorView<'a>,
}

impl<'a> TensorView<'a> {
    pub async fn make_contiguous_async(&self) -> GpuTensor {
        self.actual_tensor.contiguous().await
    }

    pub async fn make_contiguous(&self) -> GpuTensor {
        block_on(self.actual_tensor.contiguous())
    }

    pub async fn compare_async(&self, other: &Self) -> bool{
        self.actual_tensor.eq(&other.actual_tensor).await
    }

    pub async fn compare(&self, other: &Self) -> bool{
        block_on(self.actual_tensor.eq(&other.actual_tensor))
    }
}

impl Clone for Tensor {
    fn clone(&self) -> Self {
        Self {
            actual_tensor: blocking::block_on(self.actual_tensor.clone()),
        }
    }
}

impl Tensor {
    /*******  Constructors  *******/
    pub fn from_data_1d(vec: Vec<f32>) -> Self {
        assert!(!vec.is_empty(), "Data cant be empty!");
        Tensor {
            actual_tensor: GpuTensor::from_data_1d(vec),
        }
    }

    pub fn from_data_and_shape(vec: Vec<f32>, shape: Vec<usize>) -> Self {
        assert!(!vec.is_empty(), "Data cant be empty!");
        assert!(!shape.is_empty(), "Shape cant be empty!");
        Tensor {
            actual_tensor: GpuTensor::from(vec, shape),
        }
    }

    pub async fn zeros(shape: Vec<usize>) -> Self {
        assert!(!shape.is_empty(), "Shape cant be empty!");
        Self {
            actual_tensor: GpuTensor::new_filled(shape, 0.).await,
        }
    }

    pub async fn rand(_shape: Vec<usize>) -> Self {
        unimplemented!()
    }

    pub async fn zeros_like(other: &Self) -> Self {
        Self::zeros(Vec::from(other.shape().clone())).await
    }

    pub async fn clone_async(&self) -> Self {
        Self {
            actual_tensor: self.actual_tensor.clone().await,
        }
    }


    /*******  Accessors  *******/

    pub fn shape(&self) -> &VecDeque<usize> {
        self.actual_tensor.shape()
    }

    pub fn strides(&self) -> &VecDeque<usize> {
        self.actual_tensor.strides()
    }

    /*******  Ops  *******/
    pub async fn fill_with_async(&mut self, value: f32) {
        self.actual_tensor.fill_with(value).await;
    }

    pub fn fill_with(&mut self, value: f32) {
        block_on(self.actual_tensor.fill_with(value));
    }

    pub async fn matmul_async(&mut self, other: &Self) -> Self {
        Self {
            actual_tensor: self.actual_tensor.matmul(&other.actual_tensor).await,
        }
    }

    pub fn matmul(&self, other: &Self) -> Self {
        Self {
            actual_tensor: block_on(self.actual_tensor.matmul(&other.actual_tensor)),
        }
    }

    pub async fn relu_async(&self, leakage: f32) -> Self {
        Self {
            actual_tensor: self.actual_tensor.leaky_relu(leakage).await,
        }
    }

    pub fn relu(&self, leakage: f32) -> Self {
        Self {
            actual_tensor: block_on(self.actual_tensor.leaky_relu(leakage)),
        }
    }

    pub async fn compare_async(&self, other: &Self) -> bool{
        self.actual_tensor.view().eq(&other.actual_tensor.view()).await
    }

    pub async fn compare(&self, other: &Self) -> bool{
        block_on(self.actual_tensor.view().eq(&other.actual_tensor.view()))
    }

    /*******  Conversions  *******/

    /*******  Indexing Ops  *******/
    pub fn slice<'a, T: Into<SliceRangeInfo>>(&'a self, indices: Vec<T>) -> TensorView<'a> {
        TensorView {
            actual_tensor: self.actual_tensor.slice(indices),
        }
    }

    pub async fn index_async(&self, indices: Vec<usize>) -> f32 {
        self.actual_tensor.to_cpu().await.idx(&indices)
    }

    pub async fn index(&self, indices: Vec<usize>) -> f32 {
        block_on(self.actual_tensor.to_cpu()).idx(&indices)
    }

    pub async fn assign_async<T: Into<SliceRangeInfo>>(&mut self, indices: Vec<T>, value: f32) {
        self.actual_tensor.assign(indices, value).await;
    }

    pub fn assign<T: Into<SliceRangeInfo>>(&mut self, indices: Vec<T>, value: f32) {
        block_on(self.actual_tensor.assign(indices, value));
    }

    /*******  Shape Changing  *******/
    pub async fn transpose_async(&mut self) {
        self.actual_tensor.transpose().await;
    }

    pub async fn transpose(&mut self) {
        block_on(self.actual_tensor.transpose());
    }

    pub fn reshape(&mut self, new_shape: Vec<usize>) {
        self.actual_tensor.reshape(new_shape);
    }
}

pub trait ToImpl {
    /* Constructors, there are proxies to these in the Tape */
    // fn from_vec(slice: Vec<f32>) -> Self;
    // fn zeros(shape: Vec<usize>) -> Self;
    // fn rand(shape: Vec<usize>) -> Self;
    // fn zeros_like(other: &Self) -> Self;
    // fn empty() -> Self;

    /* Helper functions */
    fn is_empty(&self) -> bool;
    fn fill_with(&mut self, value: f32);

    /* Shape Changing functions */
    /// Transposes dim 0 and 1, panics if they don't exist
    fn t(&mut self);
    fn reshape(&mut self, shape: Vec<usize>);
    fn shape(&self) -> Vec<usize>;
    fn strides(&self) -> Vec<usize>;

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
    fn map_inplace<F>(&mut self, f: F)
    where
        F: FnMut(&mut f32);

    fn index(&self, index: &[usize]) -> f32;
    fn _index_mut(&mut self, index: &[usize]) -> &mut f32;
}
