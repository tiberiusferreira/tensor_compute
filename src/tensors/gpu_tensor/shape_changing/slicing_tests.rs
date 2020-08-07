use crate::{s, GpuTensor};

#[test]
fn slice_works() {
    let a = GpuTensor::from(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]);
    let view = a.slice(s![0;..;..]);
    let expected = GpuTensor::from(vec![1., 2., 3., 4.], vec![1, 2, 2]);
    blocking::block_on(view.eq(&expected.view()));
}
