mod cpu_tensor;
mod gpu_tensor;
use blocking::block_on;
pub use cpu_tensor::*;
pub use gpu_tensor::*;
pub mod traits;
use std::collections::VecDeque;
use std::fmt::{Debug, Formatter};
pub use traits::*;
pub mod prelude;

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

    pub async fn compare_async(&self, other: &Self) -> bool {
        self.actual_tensor.eq(&other.actual_tensor).await
    }

    pub async fn compare(&self, other: &Self) -> bool {
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

    /// Returns a 1 dimensional [`Tensor`] with the given data.
    ///
    /// # Examples
    ///
    /// ```
    /// use gpu_compute::Tensor;
    /// let tensor = Tensor::from_data_1d(vec![1., 2., 3., 4.]);
    /// assert_eq!(tensor.shape(), &[4]);
    /// assert_eq!(tensor.to_cpu().as_contiguous_vec(), vec![1., 2., 3., 4.]);
    /// ```
    pub fn from_data_1d(vec: Vec<f32>) -> Self {
        assert!(!vec.is_empty(), "Data cant be empty!");
        Tensor {
            actual_tensor: GpuTensor::from_data_1d(vec),
        }
    }

    /// Returns a N dimensional [`Tensor`] with the given data and shape.
    /// Panics if shape does not match the data.
    ///
    /// # Examples
    ///
    /// ```
    /// use gpu_compute::Tensor;
    /// let tensor = Tensor::from_data_and_shape(vec![1., 2., 3., 4.], vec![2, 2]);
    /// assert_eq!(tensor.shape(), &[2, 2]);
    /// assert_eq!(tensor.to_cpu().as_contiguous_vec(), vec![1., 2., 3., 4.]);
    /// ```
    pub fn from_data_and_shape(vec: Vec<f32>, shape: Vec<usize>) -> Self {
        assert!(!vec.is_empty(), "Data cant be empty!");
        assert!(!shape.is_empty(), "Shape cant be empty!");
        Tensor {
            actual_tensor: GpuTensor::from(vec, shape),
        }
    }

    /// Returns a N dimensional [`Tensor`] with the given shape filled with zeros.
    ///
    /// # Examples
    ///
    /// ```
    /// use gpu_compute::Tensor;
    /// let tensor = Tensor::zeros(vec![2, 2]);
    /// assert_eq!(tensor.shape(), &[2, 2]);
    /// assert_eq!(tensor.to_cpu().as_contiguous_vec(), vec![0., 0., 0., 0.]);
    /// ```
    pub fn zeros(shape: Vec<usize>) -> Self {
        block_on(Tensor::zeros_async(shape))
    }

    /// Same as [`Tensor::zeros`], but async.
    pub async fn zeros_async(shape: Vec<usize>) -> Self {
        assert!(!shape.is_empty(), "Shape cant be empty!");
        Self {
            actual_tensor: GpuTensor::new_filled(shape, 0.).await,
        }
    }

    /// Returns a N dimensional [`Tensor`] with the given shape filled with random values
    /// between 0 and 1.
    ///
    /// # Examples
    ///
    /// ```
    /// use gpu_compute::Tensor;
    /// let tensor = Tensor::rand(vec![2, 2]);
    /// assert_eq!(tensor.shape(), &[2, 2]);
    /// let result = tensor.to_cpu().as_contiguous_vec();
    /// for val in &result{
    ///     assert!(*val >= 0.);
    ///     assert!(*val <= 1.);
    /// }
    ///
    /// ```
    pub fn rand(shape: Vec<usize>) -> Self {
        Self {
            actual_tensor: CpuTensor::rand(shape).to_gpu(),
        }
    }

    /// Returns a [`Tensor`] filled with zeros with same shape as the input [`Tensor`]
    ///
    /// # Examples
    ///
    /// ```
    /// use gpu_compute::Tensor;
    /// let original_tensor = Tensor::rand(vec![2, 2]);
    /// let new_tensor = Tensor::zeros_like(&original_tensor);
    /// assert_eq!(new_tensor.shape(), &[2, 2]);
    /// assert_eq!(new_tensor.to_cpu().as_contiguous_vec(), &[0., 0., 0., 0.]);
    ///
    /// ```
    pub fn zeros_like(other: &Self) -> Self {
        Self::zeros(Vec::from(other.shape().clone()))
    }

    /// Same as [`Tensor::zeros_like`], but async.
    pub async fn zeros_like_async(other: &Self) -> Self {
        Self::zeros_async(Vec::from(other.shape().clone())).await
    }

    /// Clones self, returning a new [`Tensor`]
    /// with same shape, data and strides as last one.
    ///
    /// # Examples
    ///
    /// ```
    /// use gpu_compute::Tensor;
    /// let tensor = Tensor::rand(vec![2, 2]);
    /// let new_tensor = tensor.clone();
    /// assert_eq!(tensor.shape(), new_tensor.shape());
    /// assert_eq!(tensor.to_cpu().as_contiguous_vec(), new_tensor.to_cpu().as_contiguous_vec());
    ///
    /// ```
    pub fn clone(&self) -> Self {
        Self {
            actual_tensor: block_on(self.actual_tensor.clone()),
        }
    }

    /// Same as [`Tensor::clone`], but async.
    pub async fn clone_async(&self) -> Self {
        Self {
            actual_tensor: self.actual_tensor.clone().await,
        }
    }

    /*******  Accessors  *******/

    /// Returns the shape of the [`Tensor`]
    ///
    /// # Examples
    ///
    /// ```
    /// use gpu_compute::Tensor;
    /// let tensor = Tensor::rand(vec![2, 2]);
    /// assert_eq!(tensor.shape(), &[2, 2]);
    ///
    /// ```
    pub fn shape(&self) -> &VecDeque<usize> {
        self.actual_tensor.shape()
    }

    /// Returns the strides of the [`Tensor`]
    ///
    /// The strides represent how many elements in the underlying memory one needs to "jump"
    /// to get to the next element in the given dimension.
    ///
    /// If a Tensor has the following data `[1., 2., 3., 4.]`, shape `[2, 2]` and is contiguous,
    /// it can be seen as a "vector of vectors": `[ [1., 2.] , [3., 4.] ]`.
    ///
    /// Each element can be uniquely identified by two indexes: `[x, y]`. For example, the number 3
    /// has indexes: `[1, 0]`. The memory location of the element can be calculated as:
    /// `(1 * 2 + 1 * 0) = 2`. `1 * 2` comes from the first dimension and `1 * 0` from the second one.
    /// So the strides are `[2, 1]`, because for each increase in the first dimension we need to
    /// jump 2 memory positions.
    ///```text
    /// Seeing the original data: [1., 2., 3., 4.], the number 2 is indeed in the memory index 2
    ///         Memory Locations: |0 | 1 | 2 | 3|
    /// ```
    ///
    /// # Examples
    /// ```
    /// use gpu_compute::Tensor;
    /// let tensor = Tensor::rand(vec![2, 2]);
    /// assert_eq!(tensor.strides(), &[2, 1]);
    ///
    /// ```
    pub fn strides(&self) -> &VecDeque<usize> {
        self.actual_tensor.strides()
    }

    /*******  Ops  *******/

    /// Fills `self` with the given value
    ///
    /// # Examples
    ///
    /// ```
    /// use gpu_compute::Tensor;
    /// let mut tensor = Tensor::rand(vec![2, 2]);
    /// tensor.fill_with(5.);
    /// assert_eq!(tensor.to_cpu().as_contiguous_vec(), &[5., 5., 5., 5.]);
    ///
    /// ```
    pub fn fill_with(&mut self, value: f32) {
        block_on(self.actual_tensor.fill_with(value));
    }

    /// Same as [Tensor::fill_with], but async.
    pub async fn fill_with_async(&mut self, value: f32) {
        self.actual_tensor.fill_with(value).await;
    }

    /// Same as [Tensor::matmul], but async.
    pub async fn matmul_async(&mut self, other: &Self) -> Self {
        Self {
            actual_tensor: self.actual_tensor.matmul(&other.actual_tensor).await,
        }
    }

    /// Does a batch 2D matrix multiplication of `self` and the `other`. Inputs must be of rank
    /// 2 or 3.
    ///
    /// The third dimension is broadcast between them (if it exists, othersize 1 is used and later
    /// removed), so they must be broadcastable.
    ///
    /// The first two dimensions must be compatible with Matrix Multiplication, that is:
    /// if `self` has dimensions `[1, 2, 3]`, other must have `[N, 3, M]`, where N and M are any number.
    ///
    /// # Examples
    ///
    /// Matmul with shapes `[2, 2, 2]` and `[2, 2]`. Broadcasting second [`Tensor`] to rank 3.
    /// ```
    /// use gpu_compute::Tensor;
    /// let ma = Tensor::from_data_and_shape(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]);
    /// let mb = Tensor::from_data_and_shape(vec![2., 3., 4., 5.], vec![2, 2]); // will be broadcasted to shape [2, 2, 2]
    /// let result = ma.matmul(&mb);
    /// assert_eq!(result.shape(), &[2, 2, 2]);
    /// assert_eq!(result.to_cpu().as_contiguous_vec(), &[10., 13., 22., 29., 34., 45., 46., 61.]);
    /// ```
    ///
    /// Matmul with shapes `[2, 2]` and `[2, 2]`. Broadcasting both to rank 3.
    /// ```
    /// use gpu_compute::Tensor;
    /// let ma = Tensor::from_data_and_shape(vec![1., 2., 3., 4.], vec![2, 2]); // will be broadcasted to shape [1, 2, 2]
    /// let mb = Tensor::from_data_and_shape(vec![2., 3., 4., 5.], vec![2, 2]); // will be broadcasted to shape [1, 2, 2]
    /// let result = ma.matmul(&mb);
    /// assert_eq!(result.shape(), &[2, 2]);
    /// assert_eq!(result.to_cpu().as_contiguous_vec(), &[10., 13., 22., 29.]);
    /// ```
    pub fn matmul(&self, other: &Self) -> Self {
        Self {
            actual_tensor: block_on(self.actual_tensor.matmul(&other.actual_tensor)),
        }
    }

    /// Same as [Tensor::relu], but async.
    pub async fn relu_async(&self, leakage: f32) -> Self {
        Self {
            actual_tensor: self.actual_tensor.leaky_relu(leakage).await,
        }
    }

    /// Applies the following operation to all elements of the [`Tensor`].
    /// ```Rust
    /// if (element >= 0){
    ///     return element;
    /// }else{
    ///     return element*leakage;
    /// }
    /// ```
    ///
    /// # Examples
    ///
    /// ```
    /// use gpu_compute::Tensor;
    /// let mut tensor = Tensor::from_data_1d(vec![1., 2., 3., -1., -5., 10.]);
    /// let relu_result = tensor.relu(0.1);
    /// assert_eq!(relu_result.to_cpu().as_contiguous_vec(), &[1., 2., 3., -0.1, -0.5, 10.]);
    /// ```
    pub fn relu(&self, leakage: f32) -> Self {
        Self {
            actual_tensor: block_on(self.actual_tensor.leaky_relu(leakage)),
        }
    }

    /// Same as [Tensor::compare], but async.
    pub async fn compare_async(&self, other: &Self) -> bool {
        self.actual_tensor
            .view()
            .eq(&other.actual_tensor.view())
            .await
    }

   /// Returns true if both [`Tensor`]s have the same shape and data.
   ///
   ///
   /// # Examples
   ///
   /// ```
   /// use gpu_compute::Tensor;
   /// let mut tensor = Tensor::from_data_1d(vec![1., 2., 3., -1., -5., 10.]);
   /// let mut tensor_diff_shape = Tensor::from_data_and_shape(vec![1., 2., 3., -1., -5., 10.], vec![2, 3]);
   /// let mut tensor_diff_data = Tensor::from_data_1d(vec![9999., 2., 3., -1., -5., 10.]);
   /// let mut tensor_same_shape_data = Tensor::from_data_1d(vec![1., 2., 3., -1., -5., 10.]);
   /// assert!(tensor.compare(&tensor_same_shape_data));
   /// assert!(!tensor.compare(&tensor_diff_data));
   /// assert!(!tensor.compare(&tensor_diff_shape));
   /// assert!(!tensor.compare(&tensor_diff_shape));
   /// ```
    pub fn compare(&self, other: &Self) -> bool {
        block_on(self.actual_tensor.view().eq(&other.actual_tensor.view()))
    }

    /*******  Conversions  *******/
    pub async fn to_cpu_async(&self) -> CpuTensor {
        self.actual_tensor.to_cpu().await
    }

    pub fn to_cpu(&self) -> CpuTensor {
        block_on(self.actual_tensor.to_cpu())
    }

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
