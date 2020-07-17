use crate::GpuTensor;

#[test]
fn leaky_relu() {
    let async_block = async {
        let tensor = GpuTensor::from_data_and_shape(vec![-1., -2., -3., -4., 5., 6.], vec![3, 2]);
        let result = tensor.leaky_relu(0.1).await;
        assert_eq!(
            result.to_cpu().await.data_slice(),
            &[-0.1, -0.2, -0.3, -0.4, 5., 6.]
        );
    };
    futures::executor::block_on(async_block);
}
