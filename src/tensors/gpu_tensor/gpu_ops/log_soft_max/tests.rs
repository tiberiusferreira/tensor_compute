use crate::prelude::*;
use crate::GpuTensor;

#[test]
fn log_soft_max() {
    let async_block = async {
        let mut tensor = GpuTensor::from(vec![1., 2., 3.], vec![3]);
        let new = tensor.log_soft_max().await;
        println!("{:?}", new);
        // assert_eq!(
        //     tensor.to_cpu().await.raw_data_slice(),
        //     &[10., 10., 10., 10., 10., 10.]
        // );
    };
    futures::executor::block_on(async_block);
}
