use crate::prelude::*;
use crate::{GpuTensor};


#[test]
fn simple_assign() {
    let async_block = async {
        let mut ma = GpuTensor::from((0..8).into_iter().map(|e|e as f32).collect(), vec![2, 2, 2]);
        ma.assign(s!(0; 1), 10.).await;
        assert_eq!(
            ma.to_cpu().await.raw_data_slice(),
            &[0., 1., 10., 10., 4., 5., 6., 7.]
        );
        println!("{:?}", ma);
    };
    futures::executor::block_on(async_block);
}
