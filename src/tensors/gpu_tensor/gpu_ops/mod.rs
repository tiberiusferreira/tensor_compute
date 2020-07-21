mod bmm;
pub use bmm::*;
mod relu;
pub use relu::*;
mod assign;
mod fill_with;
mod make_contiguous;
mod transpose;
pub use make_contiguous::*;
mod clone;
pub use clone::*;
mod compare;
pub use compare::*;

use crate::tensors::TensorTrait;
use crate::{GpuTensor, GpuTensorView, GpuTensorViewMut};
pub use assign::*;
pub use fill_with::*;
pub use transpose::transpose;

impl<'a> GpuTensorView<'a> {
    pub async fn contiguous(&self) -> GpuTensor {
        make_contiguous(self.get_gpu(), self).await
    }
    pub async fn eq(&self, other: &Self) -> bool {
        eq(self.get_gpu(), self, other).await
    }
}

impl<'a> GpuTensorViewMut<'a> {
    pub async fn assign_kernel(&mut self, data: f32) {
        assign(self.get_gpu(), self, data).await;
    }
}
impl GpuTensor {
    pub async fn eq(&self, other: &Self) -> bool {
        eq(self.get_gpu(), &self.view(), &other.view()).await
    }

    pub async fn clone(&self) -> GpuTensor {
        clone(self.get_gpu(), self).await
    }

    pub async fn leaky_relu(&self, leakage: f32) -> GpuTensor {
        leaky_relu(self.get_gpu(), self, leakage).await
    }

    pub async fn fill_with(&mut self, value: f32) {
        fill_with(self.get_gpu(), self, value).await;
    }

    pub async fn matmul<'a>(&'a self, other: &'a Self) -> Self {
        let gpu = self.get_gpu();
        // make sure tensors have rank 3 and same batch size, broadcasting if needed
        let (mut input_data_a_view, mut input_data_b_view) =
            self.broadcast(other, Some(2)).unwrap();
        if input_data_a_view.rank() < 3 {
            input_data_a_view.increase_rank();
            input_data_b_view.increase_rank();
        }
        assert_eq!(input_data_a_view.shape().len(), 3);
        assert_eq!(input_data_b_view.shape().len(), 3);

        super::gpu_ops::bmm_kernel(gpu, &input_data_a_view, &input_data_b_view).await
    }
}
