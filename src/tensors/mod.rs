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

#[derive(Debug)]
pub struct Tensor {
    actual_tensor: GpuTensor,
}

// #[derive(Debug)]
pub struct TensorView<'a> {
    actual_tensor: GpuTensorView<'a>,
}

impl Tensor {
    /*******  Constructors  *******/
    pub fn from_vec(vec: Vec<f32>) -> Self {
        assert!(!vec.is_empty(), "Data cant be empty!");
        Tensor {
            actual_tensor: GpuTensor::from_vec(vec),
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
        Self::zeros(other.shape()).await
    }

    pub async fn empty() -> Self {
        Self::zeros(vec![1]).await
    }

    /*******  Accessors  *******/
    // pub async fn is_empty(&self) -> bool {
    //     unimplemented!()
    // }

    pub fn shape(&self) -> Vec<usize> {
        Vec::from(self.actual_tensor.shape().clone())
    }

    pub fn strides(&self) -> Vec<usize> {
        Vec::from(self.actual_tensor.strides().clone())
    }

    /*******  Ops  *******/
    pub async fn fill_with(&mut self, value: f32) {
        self.actual_tensor.fill_with(value).await;
    }

    pub async fn matmul(&mut self, other: &Self) -> Self {
        Self {
            actual_tensor: self.actual_tensor.mm(&other.actual_tensor).await,
        }
    }

    pub async fn relu(&self, leakage: f32) -> Self {
        Self {
            actual_tensor: self.actual_tensor.leaky_relu(leakage).await,
        }
    }

    /*******  Conversions  *******/
    pub async fn copy_to_vec(&self) -> Vec<f32> {
        self.actual_tensor.to_cpu().await.raw_data_slice().to_vec()
    }

    /*******  Indexing Ops  *******/
    pub async fn i<'a, T: Into<SliceRangeInfo>>(&'a self, indices: Vec<T>) -> TensorView<'a> {
        TensorView {
            actual_tensor: self.actual_tensor.i(indices)
        }
    }

    // pub async fn index_mut<'a>(&'a self, _indices: Vec<usize>) -> TensorViewMut<'a>{
    //     TensorView{
    //         actual_tensor: self.actual_tensor.view()
    //     }
    // }

    pub async fn index_move(&self, _indices: Vec<usize>) -> Tensor {
        unimplemented!()
    }

    /*******  Shape Changing  *******/
    pub async fn transpose(&mut self) {
        self.actual_tensor.transpose().await;
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
