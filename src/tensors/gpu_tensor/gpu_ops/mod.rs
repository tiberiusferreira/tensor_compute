mod bmm;
mod relu;
mod assign;
mod fill_with;
mod make_contiguous;
mod transpose;
mod compare;
mod log_soft_max;
mod binary_ops;
mod unary_ops;
use crate::{GpuTensor, GpuAllocated};

impl GpuTensor {
    pub async fn eq(&self, other: &Self) -> bool {
        compare::eq(self.gpu(), &self, &other).await
    }

    pub async fn leaky_relu(&self, leakage: f32) -> GpuTensor {
        relu::leaky_relu(self.gpu(), self, leakage).await
    }

    pub async fn fill_with(&mut self, value: f32) {
        fill_with::fill_with(self.gpu(), self, value).await;
    }

    // pub async fn add(&self, other: &Self) -> Self {
    //     binary_ops::add(self.gpu(), self, other).await
    // }

    // pub async fn sub(&self, other: &Self) -> Self {
    //     binary_ops::sub(self.gpu(), self, other).await
    // }

    pub async fn log_soft_max(&self) -> Self {
        // log_soft_max::log_soft_max(self.get_gpu(), self).await
        unimplemented!()
    }

    pub async fn matmul(&self, other: &Self) -> Self {
        let gpu = self.gpu();
        let res = bmm::bmm_kernel(gpu, self, other).await;
        res
    }
}
