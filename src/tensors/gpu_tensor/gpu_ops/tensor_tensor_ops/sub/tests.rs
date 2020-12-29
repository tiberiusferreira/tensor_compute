use crate::prelude::*;
use crate::GpuTensor;

#[test]
fn sub_test() {
    let async_block = async {
        let tensor_a = GpuTensor::from(vec![-1., -2., -3., -4., 5., 6.], vec![3, 2]);
        let tensor_b = GpuTensor::from(vec![-2., -4., -5., -6., 7., -25.], vec![3, 2]);
        let res = tensor_a.sub(&tensor_b).await;
        assert_eq!(
            res.to_cpu().await.raw_data_slice(),
            &[1., 2., 2., 2., -2., 31.]
        );
    };
    futures::executor::block_on(async_block);
}
