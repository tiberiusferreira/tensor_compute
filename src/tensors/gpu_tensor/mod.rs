use std::convert::TryInto;
use crate::gpu_buffers::GpuBuffer;
use crate::{GpuBox, Tensor, GpuStore};
use crate::tensors::CpuTensor;
mod tensor;
pub use tensor::*;
mod tensor_view;
pub use tensor_view::*;
use std::fmt::{Debug, Formatter};

pub struct GpuTensor {
    buffer: GpuBuffer,
    shape: Vec<usize>,
}

pub struct GpuTensor2 {
    buffer: GpuBuffer,
    shape: Vec<usize>,
    device: &'static GpuBox
}

impl GpuTensor{
    pub async fn bmm(&self, other: &Self) -> Self{
        let gpu = GpuStore::get(self.buffer.device());
        gpu.bmm(self, other).await
    }

    pub async fn to_cpu(&self) -> CpuTensor{
        let gpu = GpuStore::get(self.buffer.device());
        self.to_cpu_with_gpu(gpu).await
    }
}
// impl GpuTensor2{
//     pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self{
//             let calc_size = shape.iter().rev().fold(1, |acc: usize, &x| acc * x);
//             assert_eq!(
//                 calc_size,
//                 data.len(),
//                 "Shape is not valid for the size of the data!"
//             );
//             CpuTensor::new(data, shape).to_gpu(gpu)
//
//     }
// }

impl Debug for GpuTensor{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        // self.to_cpu()
        // f.write_fmt()
        unimplemented!()
    }
}

pub struct GpuTensorView<'a> {
    buffer: &'a GpuBuffer,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

#[cfg(test)]
mod tests {
    use crate::{CpuTensor, GpuBox, Tensor, GpuStore};

    #[test]
    fn strides_work() {
        let async_block = async{
            let gpu = GpuStore::current();
            let a = CpuTensor::new(vec![1., 2., 3., 4.], vec![2, 2]);
            let b = a.to_gpu(&gpu);
            assert_eq!(b.strides(), vec![2, 1]);


            let a = CpuTensor::new(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]);
            let b = a.to_gpu(&gpu);
            assert_eq!(b.strides(), vec![4, 2, 1]);
        };
        futures::executor::block_on(async_block);
    }

    #[test]
    fn broadcast_work() {
        let async_block = async{
            let gpu = GpuStore::current();
            let a = CpuTensor::new(vec![1., 2., 3., 4., 1., 2., 3., 4.], vec![2, 2, 2]);
            let a_gpu = a.to_gpu(&gpu);
            let b = CpuTensor::new(vec![1., 2., 3., 4.], vec![2, 2]);
            let b_gpu = b.to_gpu(&gpu);
            a_gpu.broadcast(&b_gpu);
            // assert_eq!(b.strides(), vec![2, 1]);

        };
        futures::executor::block_on(async_block);
    }
}