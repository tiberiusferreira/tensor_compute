// use super::broadcast_shape_and_stride;
// use crate::ShapeStrides;
// #[test]
// pub fn simple_broadcast_works() {
//     let a = ShapeStrides::from_shape_vec(vec![2, 2]); // -> [2, 2, 2]
//     let b = ShapeStrides::from_shape_vec(vec![2, 2, 2]);
//     let (a, b) = broadcast_shape_and_stride(&a, &b, None).unwrap();
//     assert_eq!(a.shape, [2, 2, 2]);
//     assert_eq!(a.strides, [0, 2, 1]);
//     assert_eq!(b.shape, [2, 2, 2]);
//     assert_eq!(b.strides, [4, 2, 1]);
// }
//
// #[test]
// pub fn simple_broadcast_with_skipping_works() {
//     let a = ShapeStrides::from_shape_vec(vec![2, 2]); // -> [2, 2, 2]
//     let b = ShapeStrides::from_shape_vec(vec![2, 2, 2]);
//     let (a, b) = broadcast_shape_and_stride(&a, &b, Some(2)).unwrap();
//     assert_eq!(a.shape, [2, 2, 2]);
//     assert_eq!(a.strides, [0, 2, 1]);
//     assert_eq!(b.shape, [2, 2, 2]);
//     assert_eq!(b.strides, [4, 2, 1]);
// }
//
// #[test]
// pub fn broadcast_works_with_additional_unit_dim() {
//     let a = ShapeStrides::from_shape_vec(vec![2, 2]);
//     let b = ShapeStrides::from_shape_vec(vec![1, 2, 2]);
//     let (a, b) = broadcast_shape_and_stride(&a, &b, None).unwrap();
//     assert_eq!(a.shape, [1, 2, 2]);
//     assert_eq!(a.strides, [0, 2, 1]);
//     assert_eq!(b.shape, [1, 2, 2]);
//     assert_eq!(b.strides, [4, 2, 1]);
// }
//
// #[test]
// pub fn broadcast_works_from_scalar() {
//     let a = ShapeStrides::from_shape_vec(vec![1]);
//     let b = ShapeStrides::from_shape_vec(vec![100]);
//     let (a, b) = broadcast_shape_and_stride(&a, &b, None).unwrap();
//     assert_eq!(a.shape, [100]);
//     assert_eq!(a.strides, [0]);
//     assert_eq!(b.shape, [100]);
//     assert_eq!(b.strides, [1]);
//
//     let a = ShapeStrides::from_shape_vec(vec![1, 1]);
//     let b = ShapeStrides::from_shape_vec(vec![100]);
//     let (a, b) = broadcast_shape_and_stride(&a, &b, None).unwrap();
//     assert_eq!(a.shape, [1, 100]);
//     assert_eq!(a.strides, [1, 0]);
//     assert_eq!(b.shape, [1, 100]);
//     assert_eq!(b.strides, [0, 1]);
//
//     let a = ShapeStrides::from_shape_vec(vec![1, 2]);
//     let b = ShapeStrides::from_shape_vec(vec![100]);
//     assert!(broadcast_shape_and_stride(&a, &b, None).is_err());
// }
