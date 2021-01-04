use crate::{GpuTensor, CpuTransferable};

#[test]
fn simple_rank_2_mm() {
    let async_block = async {
        let ma = GpuTensor::from(vec![1., 2., 3., 4.], vec![1, 2, 2]);
        let mb = GpuTensor::from(vec![2., 3., 4., 5.], vec![1, 2, 2]);
        let result = &ma.matmul(&mb).await;
        assert_eq!(
            result.to_cpu_async().await.raw_data_slice(),
            &[10., 13., 22., 29.]
        );
    };
    futures::executor::block_on(async_block);
}

#[test]
fn simple_rank_2_mm_different_dims() {
    let async_block = async {
        let ma = GpuTensor::from(vec![1., 2., 3., 4., 5., 6.], vec![1, 2, 3]);
        let mb = GpuTensor::from(vec![2., 3., 4., 5., 6., 7.], vec![1, 3, 2]);
        let result = &ma.matmul(&mb).await;
        assert_eq!(
            result.to_cpu_async().await.raw_data_slice(),
            &[28., 34., 64., 79.]
        );
    };
    futures::executor::block_on(async_block);
}

#[test]
fn big_rank_2_mm_different_dims() {
    let async_block = async {
        let ma = GpuTensor::from((0..=19)
                                     .into_iter()
                                     .map(|e| e as f32)
                                     .collect::<Vec<f32>>(), vec![2, 5, 2]);
        let mb = GpuTensor::from((20..=20+19)
                                     .into_iter()
                                     .map(|e| e as f32)
                                     .collect::<Vec<f32>>(), vec![2, 2, 5]);
        let result = &ma.matmul(&mb).await;
        assert_eq!(
            result.to_cpu_async().await.raw_data_slice(),
            &[25.,   26.,   27.,   28.,   29.,
              115.,  120.,  125.,  130.,  135.,
              205.,  214.,  223.,  232.,  241.,
              295.,  308.,  321.,  334.,  347.,
              385.,  402.,  419.,  436.,  453.,
              685.,  706.,  727.,  748.,  769.,
              815.,  840.,  865.,  890.,  915.,
              945.,  974., 1003., 1032., 1061.,
              1075., 1108., 1141., 1174., 1207.,
              1205., 1242., 1279., 1316., 1353.]);
           println!("{:?}", result);
    };
    futures::executor::block_on(async_block);
}

// #[test]
// fn mm_with_broadcasting() {
//     let async_block = async {
//         let ma = GpuTensor::from(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]);
//         let mb = GpuTensor::from(vec![2., 3., 4., 5.], vec![2, 2]);
//         let result = &ma.matmul(&mb).await;
//         assert_eq!(
//             result.to_cpu().await.raw_data_slice(),
//             &[10.0, 13.0, 22.0, 29.0, 34.0, 45.0, 46.0, 61.0]
//         );
//     };
//     futures::executor::block_on(async_block);
// }
//
// #[test]
// fn mm_scalar() {
//     let async_block = async {
//         let ma = GpuTensor::from_data(vec![1., 2., 3., 4.], vec![2, 2]);
//         let mb = GpuTensor::from_data(vec![2.], vec![1]);
//         let result = &ma.mm(&mb).await;
//         println!("{:?}", result);
//         assert_eq!(
//             result.to_cpu().await.data_slice(),
//             &[10.0, 13.0, 22.0, 29.0, 34.0, 45.0, 46.0, 61.0]
//         );
//     };
//     futures::executor::block_on(async_block);
// }
