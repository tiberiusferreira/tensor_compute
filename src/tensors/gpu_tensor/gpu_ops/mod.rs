mod bmm;
mod relu;
mod assign;
mod fill_with;
mod make_contiguous;
mod transpose;
mod clone;
mod compare;
mod log_soft_max;
mod tensor_tensor_ops;
use crate::tensors::ShapeStrideTrait;
use crate::{GpuTensor, GpuAllocated};

impl GpuTensor {
    pub async fn eq(&self, other: &Self) -> bool {
        if self.is_empty() || other.is_empty() {
            return self.is_empty() && other.is_empty()
        }
        compare::eq(self.get_gpu(), &self, &other).await
    }

    pub async fn transpose(&self) -> GpuTensor {
        if self.is_empty(){
            return self.clone().await;
        }
        transpose::transpose(self.get_gpu(), &self).await
    }

    pub async fn clone(&self) -> GpuTensor {
        clone::clone(self.get_gpu(), self).await
    }

    pub async fn leaky_relu(&self, leakage: f32) -> GpuTensor {
        if self.is_empty(){
            return self.clone().await;
        }
        relu::leaky_relu(self.get_gpu(), self, leakage).await
    }

    pub async fn fill_with(&mut self, value: f32) {
        if self.is_empty(){
            return;
        }
        fill_with::fill_with(self.get_gpu(), self, value).await;
    }

    pub async fn add(&self, other: &Self) -> Self {
        tensor_tensor_ops::add(self.get_gpu(), self, other).await
    }

    pub async fn sub(&self, other: &Self) -> Self {
        tensor_tensor_ops::sub(self.get_gpu(), self, other).await
    }

    pub async fn log_soft_max(&self) -> Self {
        log_soft_max::log_soft_max(self.get_gpu(), self).await
    }

    pub async fn matmul(&self, other: &Self) -> Self {
        if self.is_empty() || other.is_empty(){
            panic!("Tried to matmul with at least one empty Tensor")
        }
        assert!(
            self.shape().len() == 3 && other.shape().len() == 3,
            "Input to matmul must be of rank 3"
        );
        assert_eq!(
            self.shape()[2],
            other.shape()[1],
            "Shapes do not match for matrix multiply: {:?} and {:?}",
            self.shape(),
            other.shape()
        );
        let gpu = self.get_gpu();
        let mut res = bmm::bmm_kernel(gpu, self, other).await;
        res
    }
}
