mod bmm;
mod relu;
mod assign;
mod fill_with;
mod make_contiguous;
mod transpose;
mod clone;
mod compare;

use crate::tensors::ShapeStrideTrait;
use crate::{GpuAllocated, GpuTensor, GpuTensorView, GpuTensorViewMut, MutShapeStrideTrait};

impl<'a> GpuTensorView<'a> {
    pub async fn contiguous(&self) -> GpuTensor {
        make_contiguous::make_contiguous(self.get_gpu(), self).await
    }
    pub async fn eq(&self, other: &Self) -> bool {
        compare::eq(self.get_gpu(), self, other).await
    }
}

impl<'a> GpuTensorViewMut<'a> {
    pub async fn assign_kernel(&mut self, data: f32) {
        assign::assign(self.get_gpu(), self, data).await;
    }
}
impl GpuTensor {
    pub async fn eq(&self, other: &Self) -> bool {
        if self.is_empty() || other.is_empty() {
            return self.is_empty() && other.is_empty()
        }
        compare::eq(self.get_gpu(), &self.view(), &other.view()).await
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

    pub async fn matmul<'a>(&'a self, other: &'a Self) -> Self {
        if self.is_empty() || other.is_empty(){
            panic!("Tried to matmul with at least one empty Tensor")
        }
        let gpu = self.get_gpu();
        assert!(
            self.shape().len() >= 2 && other.shape().len() >= 2,
            "Input to matmul must be of rank 2 or 3"
        );
        // make sure tensors have rank 3 and same batch size, broadcasting if needed
        let (mut input_data_a_view, mut input_data_b_view) =
            self.broadcast(other, Some(2)).unwrap();
        if input_data_a_view.rank() == 2 {
            input_data_a_view.increase_rank();
            input_data_b_view.increase_rank();
        }
        assert_eq!(input_data_a_view.shape().len(), 3);
        assert_eq!(input_data_b_view.shape().len(), 3);
        assert_eq!(
            input_data_a_view.shape()[2],
            input_data_b_view.shape()[1],
            "Shapes do not match for matrix multiply: {:?} and {:?}",
            input_data_a_view.shape(),
            input_data_b_view.shape()
        );
        let mut res = bmm::bmm_kernel(gpu, &input_data_a_view, &input_data_b_view).await;
        if self.shape().len() == 2 && other.shape().len() == 2 {
            res.decrease_rank();
        }
        res
    }
}
