use crate::GpuTensor;

#[test]
fn simple_assign() {
    let async_block = async {
        let ma = GpuTensor::from_data_and_shape(vec![1., 2., 3., 4.], vec![2, 2]);
        let mb = GpuTensor::from_data_and_shape(vec![2., 3., 4., 5.], vec![2, 2]);
        let result = &ma.mm(&mb).await;
        assert_eq!(result.to_cpu().await.raw_data_slice(), &[10., 13., 22., 29.]);
    };
    futures::executor::block_on(async_block);
}