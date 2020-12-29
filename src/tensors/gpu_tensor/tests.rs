use crate::{GpuTensor, ShapeStrideTrait};

#[test]
fn strides_work() {
    let a = GpuTensor::from(vec![1., 2., 3., 4.], vec![2, 2]);
    assert_eq!(a.strides(), &[2, 1]);

    let a = GpuTensor::from(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]);
    assert_eq!(a.strides(), &[4, 2, 1]);
}

// #[test]
// fn broadcast_work() {
//     let a = GpuTensor::from(vec![1., 2., 3., 4., 1., 2., 3., 4.], vec![2, 2, 2]);
//     let b = GpuTensor::from(vec![1., 2., 3., 4.], vec![2, 2]);
//     let (a_view, b_view) = a.broadcast(&b, None).unwrap();
//     assert_eq!(a_view.shape(), &[2, 2, 2]);
//     assert_eq!(a_view.strides(), &[4, 2, 1]);
//     assert_eq!(b_view.shape(), &[2, 2, 2]);
//     assert_eq!(b_view.strides(), &[0, 2, 1]);
// }
