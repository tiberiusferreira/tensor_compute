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

/// A Tensor is an N dimensional data structure. This is the entry point for most of the API
/// of this crate. This is normally backed by GPU memory and its device chosen using the current
/// default of the [`crate::GpuStore::get_default()`], which can be changed however, one can NOT
/// do operations using two Tensors from different devices.
pub struct Tensor {
    actual_tensor: GpuTensor,
}

/// Beware, printing the Tensor forces a copy from GPU memory to CPU memory. For now, the whole
/// Tensor is copied, but could  be optimized in the future. Since this is meant for debugging,
/// optimizing it is not a high priority.
impl Debug for Tensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.actual_tensor.fmt(f)
    }
}

/// A view into the original [`Tensor`]. A view borrows part of (or even the entirety of) the
/// original Tensor data. It can NOT modify the data itself. It is normally produced by indexing or
/// slicing a Tensor.
///
/// It can be useful for creating a new Tensor from part of another one:
///
/// # Examples
///
/// ```
/// use tensor_compute::{Tensor, TensorView, s};
/// let tensor = Tensor::from_data_and_shape(vec![1., 2., 3., 4.], vec![2, 2]);
/// let first_row: TensorView = tensor.slice(s![1]);
/// assert_eq!(first_row.shape(), &[1, 2]);
/// assert_eq!(first_row.to_cpu().as_contiguous_vec(), &[3., 4.]);
/// let new_tensor = first_row.make_contiguous(); // creates a new owned tensor from the view
/// assert_eq!(new_tensor.shape(), &[1, 2]);
/// assert_eq!(new_tensor.to_cpu().as_contiguous_vec(), &[3., 4.]);
/// ```
pub struct TensorView<'a> {
    actual_tensor: GpuTensorView<'a>,
}

/// Beware, printing the TensorView forces a copy from GPU memory to CPU memory.
impl <'a> Debug for TensorView<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        block_on(self.actual_tensor.to_cpu()).fmt(f)
    }
}

impl<'a> TensorView<'a> {
    /// Same as [`TensorView::make_contiguous`], but async.
    pub async fn make_contiguous_async(&self) -> Tensor {
        Tensor{
            actual_tensor: self.actual_tensor.contiguous().await
        }
    }

    /// Copies the [`TensorView`] into a standalone contiguous [`Tensor`]
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_compute::{Tensor, TensorView, s};
    /// let tensor = Tensor::from_data_and_shape(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]);
    /// let first_row: TensorView = tensor.slice(s![1]);
    /// assert_eq!(first_row.shape(), &[1, 2, 2]);
    /// assert_eq!(first_row.to_cpu().as_contiguous_vec(), &[5., 6., 7., 8.]);
    /// let new_tensor = first_row.make_contiguous(); // creates a new owned tensor from the view
    /// assert_eq!(new_tensor.shape(), &[1, 2, 2]);
    /// assert_eq!(new_tensor.to_cpu().as_contiguous_vec(), &[5., 6., 7., 8.]);
    /// assert!(new_tensor.slice(s![..]).compare(&first_row));
    /// ```
    pub fn make_contiguous(&self) -> Tensor {
        block_on(self.make_contiguous_async())
    }


    /// Same as [`Tensor::compare_async`]
    pub async fn compare_async(&self, other: &Self) -> bool {
        self.actual_tensor.eq(&other.actual_tensor).await
    }

    /// Same as [`Tensor::compare`]
    pub fn compare(&self, other: &Self) -> bool {
        block_on(self.actual_tensor.eq(&other.actual_tensor))
    }

    /// Same as [`Tensor::shape`]
    pub fn shape(&self) -> &VecDeque<usize> {
        self.actual_tensor.shape()
    }

    /// Same as [`Tensor::strides`]
    pub fn strides(&self) -> &VecDeque<usize> {
        self.actual_tensor.strides()
    }

    /// Same as [`Tensor::to_cpu_async`]
    pub async fn to_cpu_async(&self) -> CpuTensor {
        self.actual_tensor.to_cpu().await
    }

    /// Same as [`Tensor::to_cpu`]
    pub fn to_cpu(&self) -> CpuTensor {
        block_on(self.to_cpu_async())
    }


}

/// Clones the Tensor data, shape, strides
impl Clone for Tensor {
    fn clone(&self) -> Self {
        Self {
            actual_tensor: blocking::block_on(self.actual_tensor.clone()),
        }
    }
}

impl Tensor {
    /*******  Constructors  *******/


    // pub fn empty() -> Self {
    //     Tensor {
    //         actual_tensor: GpuTensor::from_data_1d()
    //     }
    // }

    /// Returns a 1 dimensional [`Tensor`] with the given data.
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_compute::Tensor;
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
    /// use tensor_compute::Tensor;
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
    /// use tensor_compute::Tensor;
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
    /// use tensor_compute::Tensor;
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
    /// use tensor_compute::Tensor;
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
    /// use tensor_compute::Tensor;
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
    /// use tensor_compute::Tensor;
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
    /// use tensor_compute::Tensor;
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
    /// use tensor_compute::Tensor;
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
    /// use tensor_compute::Tensor;
    /// let ma = Tensor::from_data_and_shape(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]);
    /// let mb = Tensor::from_data_and_shape(vec![2., 3., 4., 5.], vec![2, 2]); // will be broadcasted to shape [2, 2, 2]
    /// let result = ma.matmul(&mb);
    /// assert_eq!(result.shape(), &[2, 2, 2]);
    /// assert_eq!(result.to_cpu().as_contiguous_vec(), &[10., 13., 22., 29., 34., 45., 46., 61.]);
    /// ```
    ///
    /// Matmul with shapes `[2, 2]` and `[2, 2]`. Broadcasting both to rank 3.
    /// ```
    /// use tensor_compute::Tensor;
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
    /// use tensor_compute::Tensor;
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
    /// use tensor_compute::Tensor;
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

    /// Same as [Tensor::to_cpu], but async.
    pub async fn to_cpu_async(&self) -> CpuTensor {
        self.actual_tensor.to_cpu().await
    }

    /// Copies the [`Tensor`] data from GPU memory to CPU memory.
    ///
    /// Having the Tensor in CPU is necessary for some operations, for example printing the Tensor
    /// data.
    pub fn to_cpu(&self) -> CpuTensor {
        block_on(self.actual_tensor.to_cpu())
    }

    /*******  Indexing Ops  *******/

    /// Slices a [`Tensor`] into a [`TensorView`] using information from [`SliceRangeInfo`]s.
    ///
    /// An ergonomic way of creating [`SliceRangeInfo`]s is the [`s`] macro, which takes either:
    ///
    /// - A single number, representing which `element` of a given dimension to take.
    /// - start:step:exclusive_end representing which range of `elements` of a given dimension to take.
    /// - Normal Rust range representing which range of `elements` of a given dimension to take.
    ///   One can use Rust ranges to represent "take all elements" using (..).
    ///   For example: (0..=10), (0..10) or (..).
    ///
    ///
    /// # Examples
    ///
    /// ```
    /// use tensor_compute::{Tensor, s};
    /// let mut tensor = Tensor::from_data_and_shape(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]);
    /// let tensor_slice = tensor.slice(s![0;..;1]);
    /// assert_eq!(tensor_slice.shape(), &[1, 2, 1]);
    /// assert_eq!(tensor_slice.to_cpu().as_contiguous_vec(), &[2., 4.]);
    /// ```
    pub fn slice<'a, T: Into<SliceRangeInfo>>(&'a self, indices: Vec<T>) -> TensorView<'a> {
        TensorView {
            actual_tensor: self.actual_tensor.slice(indices),
        }
    }

    /// Same as [Tensor::assign], but async.
    pub async fn assign_async<T: Into<SliceRangeInfo>>(&mut self, indices: Vec<T>, value: f32) {
        self.actual_tensor.assign(indices, value).await;
    }


    /// Assigns a single value to all [`SliceRangeInfo`]s.
    /// Not quite sure about this yet. Maybe it is better to create a mutable slice and allow
    /// assigning to it both a regular f32 and also a [`Tensor`] or [`TensorView`] with same shape
    ///
    pub fn assign<T: Into<SliceRangeInfo>>(&mut self, indices: Vec<T>, value: f32) {
        block_on(self.actual_tensor.assign(indices, value));
    }

    /*******  Shape Changing  *******/

    /// Same as [Tensor::transpose], but async.
    pub async fn transpose_async(&self) -> Tensor {
        Tensor{
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
    /// use tensor_compute::{Tensor, s};
    /// let mut original = Tensor::from_data_and_shape(
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
    pub fn transpose(&self) -> Tensor {
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
    /// use tensor_compute::{Tensor, s};
    /// let mut tensor = Tensor::from_data_and_shape(
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
}
