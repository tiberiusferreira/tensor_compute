use crate::{CpuTensor, GpuStore, Tensor};

#[test]
fn strides_work() {
    let async_block = async {
        let gpu = GpuStore::get_default();
        let a = CpuTensor::new(vec![1., 2., 3., 4.], vec![2, 2]);
        let b = a.to_gpu(&gpu);
        assert_eq!(b.strides(), &[2, 1]);

        let a = CpuTensor::new(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]);
        let b = a.to_gpu(&gpu);
        assert_eq!(b.strides(), &[4, 2, 1]);
    };
    futures::executor::block_on(async_block);
}

#[test]
fn broadcast_work() {
    let async_block = async {
        let gpu = GpuStore::get_default();
        let a = CpuTensor::new(vec![1., 2., 3., 4., 1., 2., 3., 4.], vec![2, 2, 2]);
        let a_gpu = a.to_gpu(&gpu);
        let b = CpuTensor::new(vec![1., 2., 3., 4.], vec![2, 2]);
        let b_gpu = b.to_gpu(&gpu);
        a_gpu.broadcast(&b_gpu);
        // assert_eq!(b.strides(), vec![2, 1]);
    };
    futures::executor::block_on(async_block);
}
