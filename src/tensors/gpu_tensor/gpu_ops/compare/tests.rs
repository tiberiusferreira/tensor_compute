use crate::{GpuTensor};

#[test]
fn compare_test() {
    let async_block = async {
        let tensor_a = GpuTensor::from((0..6).map(|e| e as f32).collect(), vec![3, 2]);
        let tensor_b = GpuTensor::from((0..6).map(|e| e as f32).collect(), vec![3, 2]);
        let tensor_c = GpuTensor::from((1..7).map(|e| e as f32).collect(), vec![3, 2]);
        assert!(tensor_a.eq(&tensor_b).await);
        assert!(!tensor_a.eq(&tensor_c).await);
    };
    futures::executor::block_on(async_block);
}
