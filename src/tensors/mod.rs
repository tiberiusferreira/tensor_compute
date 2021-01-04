//! This module wraps the GPU Tensor methods, providing both a blocking and async version of them
//!

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

/// A RawTensor is an N dimensional data structure. This is the entry point for most of the API
/// of this crate. This is normally backed by GPU memory and its device chosen using the current
/// default of the [`crate::GpuStore::get_default()`], which can be changed however, one can NOT
/// do operations using two Tensors from different devices.
pub struct RawTensor {
    actual_tensor: GpuTensor,
}

/// Beware, printing the Tensor forces a copy from GPU memory to CPU memory. For now, the whole
/// Tensor is copied, but could  be optimized in the future. Since this is meant for debugging,
/// optimizing it is not a high priority.
impl Debug for RawTensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.actual_tensor.fmt(f)
    }
}


/// Clones the Tensor data, shape, strides
impl Clone for RawTensor {
    fn clone(&self) -> Self {
        Self {
            actual_tensor: blocking::block_on(self.actual_tensor.clone()),
        }
    }
}

impl RawTensor {
    /*******  Constructors  *******/

    /// Returns an empty 0 dimensional [`Tensor`]. Could be useful as a placeholder.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_compute::RawTensor;
    /// let tensor = RawTensor::empty();
    /// assert_eq!(tensor.shape(), &[]);
    /// ```
    pub fn empty() -> Self {
        RawTensor {
            actual_tensor: GpuTensor::from(vec![], vec![])
        }
    }

    /// Returns a 1 dimensional [`Tensor`] with the given data.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_compute::RawTensor;
    /// let tensor = RawTensor::from_data_1d(vec![1., 2., 3., 4.]);
    /// assert_eq!(tensor.shape(), &[4]);
    /// assert_eq!(tensor.to_cpu().as_contiguous_vec(), vec![1., 2., 3., 4.]);
    /// ```
    pub fn from_data_1d(vec: Vec<f32>) -> Self {
        assert!(!vec.is_empty(), "Data cant be empty!");
        RawTensor {
            actual_tensor: GpuTensor::from_data_1d(vec),
        }
    }

    /// Returns a N dimensional [`Tensor`] with the given data and shape.
    /// Panics if shape does not match the data.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_compute::RawTensor;
    /// let tensor = RawTensor::from_data_and_shape(vec![1., 2., 3., 4.], vec![2, 2]);
    /// assert_eq!(tensor.shape(), &[2, 2]);
    /// assert_eq!(tensor.to_cpu().as_contiguous_vec(), vec![1., 2., 3., 4.]);
    /// ```
    pub fn from_data_and_shape(vec: Vec<f32>, shape: Vec<usize>) -> Self {
        assert!(!vec.is_empty(), "Data cant be empty!");
        assert!(!shape.is_empty(), "Shape cant be empty!");
        RawTensor {
            actual_tensor: GpuTensor::from(vec, shape),
        }
    }

    /// Returns a N dimensional [`Tensor`] with the given shape filled with zeros.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_compute::RawTensor;
    /// let tensor = RawTensor::zeros(vec![2, 2]);
    /// assert_eq!(tensor.shape(), &[2, 2]);
    /// assert_eq!(tensor.to_cpu().as_contiguous_vec(), vec![0., 0., 0., 0.]);
    /// ```
    pub fn zeros(shape: Vec<usize>) -> Self {
        block_on(RawTensor::zeros_async(shape))
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
    /// use tensor_compute::RawTensor;
    /// let tensor = RawTensor::rand(vec![2, 2]);
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
    /// use tensor_compute::RawTensor;
    /// let original_tensor = RawTensor::rand(vec![2, 2]);
    /// let new_tensor = RawTensor::zeros_like(&original_tensor);
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
    /// use tensor_compute::RawTensor;
    /// let tensor = RawTensor::rand(vec![2, 2]);
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


    pub async fn to_vec_async(&self) -> Vec<f32> {
        self.actual_tensor.to_cpu_async().await.as_contiguous_vec()
    }

    pub fn to_vec(&self) -> Vec<f32> {
        self.actual_tensor.to_cpu().as_contiguous_vec()
    }

    /*******  Accessors  *******/

    /// Returns the shape of the [`Tensor`]
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_compute::RawTensor;
    /// let tensor = RawTensor::rand(vec![2, 2]);
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
    /// use tensor_compute::RawTensor;
    /// let tensor = RawTensor::rand(vec![2, 2]);
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
    /// use tensor_compute::RawTensor;
    /// let mut tensor = RawTensor::rand(vec![2, 2]);
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

    /// Adds both [`Tensor`]s returning the result as a new contiguous [`Tensor`].
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_compute::{RawTensor, CpuTransferable};
    /// let tensor_left = RawTensor::from_data_1d(vec![1., 2., 3., 4.]);
    /// let tensor_right = RawTensor::from_data_1d(vec![2., 3., 4., 5.]);
    /// let result = tensor_left.add(&tensor_right);
    /// assert_eq!(
    ///     result.to_cpu().as_contiguous_vec(),
    ///     &[3., 5., 7., 9.]
    /// );
    /// ```
    pub fn add(&self, other: &Self) -> RawTensor {
        RawTensor {
            actual_tensor: block_on(self.actual_tensor.add(&other.actual_tensor))
        }
    }

    /// Adds both [`Tensor`]s returning the result as a new contiguous [`Tensor`].
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_compute::{RawTensor, CpuTransferable};
    /// let tensor_left = RawTensor::from_data_1d(vec![1., 2., 3., 10.]);
    /// let tensor_right = RawTensor::from_data_1d(vec![2., 3., 4., 5.]);
    /// let result = tensor_left.sub(&tensor_right);
    /// assert_eq!(
    ///     result.to_cpu().as_contiguous_vec(),
    ///     &[-1., -1., -1., 5.]
    /// );
    /// ```
    pub fn sub(&self, other: &Self) -> RawTensor {
        RawTensor {
            actual_tensor: block_on(self.actual_tensor.sub(&other.actual_tensor))
        }
    }

    // /// Same as [`TensorView::add`] but async
    // pub async fn add_async(&self, other: &Self) -> Tensor {
    //     Tensor{
    //         actual_tensor: self.actual_tensor.add(&other.actual_tensor).await
    //     }
    // }

    /// Same as [Tensor::matmul], but async.
    pub async fn matmul_async(&mut self, other: &Self) -> Self {
        Self {
            actual_tensor: self.actual_tensor.matmul(&other.actual_tensor).await,
        }
    }

    /// Does a batch 2D matrix multiplication of `self` and the `other`. Inputs must be of rank 3.
    ///
    /// The first two dimensions must be compatible with Matrix Multiplication, that is:
    /// if `self` has dimensions `[1, 2, 3]`, other must have `[1, 3, M]`, where M is any number.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_compute::RawTensor;
    /// let ma = RawTensor::from_data_and_shape(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]);
    /// let mb = RawTensor::from_data_and_shape(vec![2., 3., 4., 5., 2., 3., 4., 5.], vec![2, 2, 2]);
    /// let result = ma.matmul(&mb);
    /// assert_eq!(result.shape(), &[2, 2, 2]);
    /// assert_eq!(result.to_cpu().as_contiguous_vec(), &[10., 13., 22., 29., 34., 45., 46., 61.]);
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
    /// use tensor_compute::RawTensor;
    /// let mut tensor = RawTensor::from_data_1d(vec![1., 2., 3., -1., -5., 10.]);
    /// let relu_result = tensor.relu(0.1);
    /// assert_eq!(relu_result.to_cpu().as_contiguous_vec(), &[1., 2., 3., -0.1, -0.5, 10.]);
    /// ```
    pub fn relu(&self, leakage: f32) -> Self {
        Self {
            actual_tensor: block_on(self.actual_tensor.leaky_relu(leakage)),
        }
    }

    /// Applies the following LogSoftmax operation to all elements of the [`Tensor`].
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// TODO
    /// ```
    // pub fn softmax(&self) -> Self {
    //     Self {
    //         actual_tensor: block_on(self.actual_tensor),
    //     }
    // }

    /// Same as [Tensor::compare], but async.
    pub async fn compare_async(&self, other: &Self) -> bool {
        self.actual_tensor
            .eq(&other.actual_tensor)
            .await
    }

    /// Returns true if both [`Tensor`]s have the same shape and data.
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_compute::RawTensor;
    /// let mut tensor = RawTensor::from_data_1d(vec![1., 2., 3., -1., -5., 10.]);
    /// let mut tensor_diff_shape = RawTensor::from_data_and_shape(vec![1., 2., 3., -1., -5., 10.], vec![2, 3]);
    /// let mut tensor_diff_data = RawTensor::from_data_1d(vec![9999., 2., 3., -1., -5., 10.]);
    /// let mut tensor_same_shape_data = RawTensor::from_data_1d(vec![1., 2., 3., -1., -5., 10.]);
    /// assert!(tensor.compare(&tensor_same_shape_data));
    /// assert!(!tensor.compare(&tensor_diff_data));
    /// assert!(!tensor.compare(&tensor_diff_shape));
    /// assert!(!tensor.compare(&tensor_diff_shape));
    /// ```
    pub fn compare(&self, other: &Self) -> bool {
        block_on(self.actual_tensor.eq(&other.actual_tensor))
    }

    /*******  Conversions  *******/

    /// Same as [Tensor::to_cpu], but async.
    pub async fn to_cpu_async(&self) -> CpuTensor {
        self.actual_tensor.to_cpu_async().await
    }

    /// Copies the [`Tensor`] data from GPU memory to CPU memory.
    ///
    /// Having the Tensor in CPU is necessary for some operations, for example printing the Tensor
    /// data.
    pub fn to_cpu(&self) -> CpuTensor {
        block_on(self.actual_tensor.to_cpu_async())
    }


    /*******  Shape Changing  *******/

    /// Same as [Tensor::transpose], but async.
    pub async fn transpose_async(&self) -> RawTensor {
        RawTensor {
            actual_tensor:self.actual_tensor.transpose().await
        }
    }


    /// Transposes a [`Tensor`] swapping its last two dimensions. For now, this copies the tensor
    /// to a new one to avoid creating a non-contiguous Tensor and all the implications that come from
    /// it.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_compute::{RawTensor, s};
    /// let mut original = RawTensor::from_data_and_shape(
    ///     vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
    ///     vec![1, 2, 2, 3],
    /// );
    /// let transposed = original.transpose();
    /// assert_eq!(
    ///     transposed.shape(),
    ///     &[1, 2, 3, 2]
    /// );
    /// assert_eq!(
    ///     transposed.to_cpu().as_contiguous_vec(),
    ///     &[1., 4., 2., 5., 3., 6., 7., 10., 8., 11., 9., 12.]
    /// );
    /// let transposed_twice = transposed.transpose();
    /// assert_eq!(
    ///     transposed_twice.shape(),
    ///     &[1, 2, 2, 3]
    /// );
    /// assert_eq!(
    ///     transposed_twice.to_cpu().as_contiguous_vec(),
    ///     &[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]
    /// );
    /// ```
    pub fn transpose(&self) -> RawTensor {
        block_on(self.transpose_async())
    }

    /// Reshapes a [`Tensor`] to the given shape. The only restriction is having the same number
    /// of elements as the original shape.
    ///
    /// This does not change the underlying data in any way. It just changes how it is "sliced"
    /// into each dimension.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_compute::{RawTensor, s};
    /// let mut tensor = RawTensor::from_data_and_shape(
    ///     vec![1., 2., 3., 4., 5., 6., 7., 8.],
    ///     vec![2, 4],
    /// );
    /// tensor.reshape(vec![4, 2]);
    /// assert_eq!(
    ///     tensor.shape(),
    ///     &[4, 2]
    /// );
    /// assert_eq!(
    ///     tensor.to_cpu().as_contiguous_vec(),
    ///     &[1., 2., 3., 4., 5., 6., 7., 8.]
    /// );
    /// ```
    pub fn reshape(&mut self, new_shape: Vec<usize>) {
        self.actual_tensor.reshape(new_shape);
    }

    // pub fn view(&self) -> TensorView{
    //     TensorView{
    //         actual_tensor: self.actual_tensor.view()
    //     }
    // }


}
