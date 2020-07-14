//
// #[test]
// fn basic_broadcasting() {
//     let async_block = async {
//         let ma = GpuTensor::from_data(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]);
//         let mb = GpuTensor::from_data(vec![2., 3., 4., 5.], vec![2, 2]);
//         let result = &ma.mm(&mb).await;
//         assert_eq!(
//             result.to_cpu().await.data_slice(),
//             &[10.0, 13.0, 22.0, 29.0, 34.0, 45.0, 46.0, 61.0]
//         );
//     };
//     futures::executor::block_on(async_block);
// }

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