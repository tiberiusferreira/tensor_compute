mod mm;
pub use mm::*;
mod relu;
pub use relu::*;
mod transpose;
mod fill_with;
pub use fill_with::*;
use crate::tensors::TensorTrait;
use crate::GpuTensor;
pub use transpose::transpose;

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
        let (mut input_data_a_view, mut input_data_b_view) = self.broadcast(other, Some(2)).unwrap();
        if input_data_a_view.rank() < 3 {
            input_data_a_view.increase_rank();
            input_data_b_view.increase_rank();
        }
        assert_eq!(input_data_a_view.shape().len(), 3);
        assert_eq!(input_data_b_view.shape().len(), 3);

        super::gpu_ops::bmm_kernel(gpu, &input_data_a_view, &input_data_b_view).await
    }
}
//
// pub async fn mm_dispatcher(gpu: &GpuInstance,
//                            input_data_a: &GpuTensor,
//                            input_data_b: &GpuTensor){
//
//     if input_data_a.is_scalar() || input_data_b.is_scalar(){
//         // scalar ops
//         unimplemented!();
//     }
//
//     // make sure tensors have rank 3 and same batch size, broadcasting if needed
//     let (mut input_data_a_view, mut input_data_b_view) = input_data_a.broadcast(input_data_b).unwrap();
//     if input_data_a_view.rank() < 3{
//         input_data_a_view.increase_rank();
//         input_data_b_view.increase_rank();
//     }
//     assert_eq!(input_data_a_view.shape().len(), 3);
//     assert_eq!(input_data_b_view.shape().len(), 3);
// }
