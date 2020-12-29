use crate::prelude::*;
use crate::GpuTensor;

// #[test]
// fn dot_mul() {
//     let async_block = async {
//         let tensor_a = GpuTensor::from(vec![-1., -2., -3., -4., 5., 6.], vec![3, 2]);
//         let tensor_b = GpuTensor::from(vec![-2., -4., -5., -6., 7., -25.], vec![3, 2]);
//         let res = tensor_a.view().dot_div(&tensor_b.view()).await;
//         let expected = &[0.5, 0.5, 0.6, 4./6., 5./7., 6./-25.];
//         let real = res.to_cpu().await;
//         let real = real.raw_data_slice();
//         let sum_diff: f32 = expected
//             .iter()
//             .zip(real.iter())
//             .map(|(l, r)| l-r)
//             .sum();
//         assert!(sum_diff < 0.01);
//     };
//     futures::executor::block_on(async_block);
// }

