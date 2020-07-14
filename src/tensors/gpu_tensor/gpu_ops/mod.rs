mod mm;
pub use mm::*;
mod relu;
pub use relu::*;
mod transpose;
pub use transpose::transpose;
use crate::GpuTensor;


impl GpuTensor{
    pub async fn leaky_relu(&self, leakage: f32) -> GpuTensor{
        leaky_relu(self.get_gpu(), self, leakage).await
    }

    pub async fn mm(&self, other: &Self) -> Self {
        let gpu = self.get_gpu();
        super::gpu_ops::mm(gpu, self, other).await
    }
}

