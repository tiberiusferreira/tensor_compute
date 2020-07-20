use crate::{s, GpuTensor, SliceRangeInfo};

#[test]
fn can_create_contiguous_from_view() {
    let async_block = async {
        let ma =
            GpuTensor::from_data_and_shape(vec![2., 3., 4., 5., 6., 7., 8., 9.], vec![2, 2, 2]);
        let tensor = ma.clone().await;
        assert_eq!(ma.to_cpu().await, tensor.to_cpu().await);
    };
    futures::executor::block_on(async_block);
}
