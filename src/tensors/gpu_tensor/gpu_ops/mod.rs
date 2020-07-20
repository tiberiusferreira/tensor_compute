mod bmm;
pub use bmm::*;
mod relu;
pub use relu::*;
mod fill_with;
mod transpose;
mod assign;
mod contiguous;
pub use contiguous::*;
pub use assign::*;
use crate::tensors::TensorTrait;
use crate::{GpuTensor, GpuTensorView, GpuTensorViewMut};
pub use fill_with::*;
pub use transpose::transpose;

impl <'a> GpuTensorView<'a>{
    pub async fn contiguous(&self) -> GpuTensor {
        contiguous(self.get_gpu(), self).await
    }
}

impl <'a> GpuTensorViewMut<'a>{
    pub async fn assign_kernel(&mut self, data: f32) {
        assign(self.get_gpu(), self, data).await;
    }
}
impl GpuTensor {


    pub async fn leaky_relu(&self, leakage: f32) -> GpuTensor {
        leaky_relu(self.get_gpu(), self, leakage).await
    }

    pub async fn fill_with(&mut self, value: f32) {
        fill_with(self.get_gpu(), self, value).await;
    }

    pub async fn mm<'a>(&'a self, other: &'a Self) -> Self {
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


