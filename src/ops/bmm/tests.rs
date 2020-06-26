use crate::{GpuBox, GpuTensor, GpuStore};

#[test]
fn bmm_works() {
    let async_block = async {
        let gpu = GpuStore::current();
        let ma = GpuTensor::from_data(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]);
        let mb = GpuTensor::from_data_in_gpu(&gpu, vec![2., 3., 4., 5.], vec![2, 2]);
        let btimes = gpu.bmm(&ma, &mb).await;
        let bcpu_copy = btimes.to_cpu_with_gpu(&gpu).await;
        assert_eq!(bcpu_copy.data_slice(), &[10.0, 13.0, 22.0, 29.0, 34.0, 45.0, 46.0, 61.0]);
    };
    futures::executor::block_on(async_block);
}
