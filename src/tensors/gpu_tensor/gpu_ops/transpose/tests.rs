use crate::prelude::*;
use crate::GpuTensor;

#[test]
pub fn can_transpose() {
    let block = async {
        let original = GpuTensor::from(
            vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.],
            vec![1, 2, 2, 3],
        );
        let transposed = original.transpose().await.to_cpu();

        assert_eq!(
            transposed.raw_data_slice(),
            &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 7.0, 10.0, 8.0, 11.0, 9.0, 12.0]
        );
        assert_eq!(
            transposed.shape(),
            &[1, 2, 3, 2]
        );
    };
    futures::executor::block_on(block);
}
