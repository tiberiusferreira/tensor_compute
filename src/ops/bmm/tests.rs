use crate::{GpuBox, Gpu2DTensor, GpuTensor};

#[test]
fn bmm_works() {
    let async_block = async {
        let gpu = GpuBox::new().await;
        let ma = GpuTensor::new(&gpu, vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]);
        let mb = GpuTensor::new(&gpu, vec![2., 3., 4., 5., 6., 7., 8., 9.], vec![2, 2, 2]);
        let btimes = gpu.bmm(&ma, &mb).await;
        let bcpu_copy = btimes.to_cpu(&gpu).await;
        assert_eq!(bcpu_copy.data_slice(), &[10.0, 13.0, 22.0, 29.0, 78.0, 89.0, 106.0, 121.0]);
    };
    futures::executor::block_on(async_block);
}
