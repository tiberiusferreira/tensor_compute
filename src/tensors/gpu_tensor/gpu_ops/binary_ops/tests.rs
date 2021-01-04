use crate::prelude::*;
use crate::GpuTensor;

#[test]
fn add_test() {
    let async_block = async {
        let tensor_a = GpuTensor::from(vec![-1., -2., -3., -4., 5., 6.], vec![3, 2]);
        let tensor_b = GpuTensor::from(vec![-2., -4., -5., -6., 7., -25.], vec![3, 2]);
        let res = tensor_a.add(&tensor_b).await;
        assert_eq!(
            res.to_cpu().raw_data_slice(),
            &[-3., -6., -8., -10., 12., -19.]
        );
    };
    futures::executor::block_on(async_block);
}


#[test]
fn sub_test() {
    let async_block = async {
        let tensor_a = GpuTensor::from(vec![-1., -2., -3., -4., 5., 6.], vec![3, 2]);
        let tensor_b = GpuTensor::from(vec![-2., -4., -5., -6., 7., -25.], vec![3, 2]);
        let res = tensor_a.sub(&tensor_b).await;
        assert_eq!(
            res.to_cpu_async().await.raw_data_slice(),
            &[1., 2., 2., 2., -2., 31.]
        );
    };
    futures::executor::block_on(async_block);
}


#[test]
fn dot_mul() {
    let async_block = async {
        let tensor_a = GpuTensor::from(vec![-1., -2., -3., -4., 5., 6.], vec![3, 2]);
        let tensor_b = GpuTensor::from(vec![-2., -4., -5., -6., 7., -25.], vec![3, 2]);
        let c = tensor_a.dot_mul(&tensor_b).await;
        assert_eq!(c.to_cpu().raw_data_slice(), &[2., 8., 15., 24., 35., -150.]);
    };
    futures::executor::block_on(async_block);
}

#[test]
fn dot_div() {
    let async_block = async {
        let tensor_a = GpuTensor::from(vec![-1., -2., -3., 4., 15., 24.], vec![3, 2]);
        let tensor_b = GpuTensor::from(vec![-2., -4., -5., -8., 3., -6.], vec![3, 2]);
        let c = tensor_a.dot_div(&tensor_b).await;
        let err: f32 = c.to_cpu().raw_data_slice().iter()
            .zip([0.5, 0.5, 0.6, -0.5, 5., -4.].iter())
            .map(|(l, r)| l-r)
            .sum();
        assert!(err < 0.01);
    };
    futures::executor::block_on(async_block);
}



#[test]
fn add_scalar_test() {
    let async_block = async {
        let tensor_a = GpuTensor::from(vec![-1., -2., -3., -4., 5., 6.], vec![3, 2]);
        let res = tensor_a.add_scalar(5.).await;
        assert_eq!(
            res.to_cpu().raw_data_slice(),
            &[4., 3., 2., 1., 10., 11.]
        );
    };
    futures::executor::block_on(async_block);
}


#[test]
fn sub_scalar_test() {
    let async_block = async {
        let tensor_a = GpuTensor::from(vec![-1., -2., -3., -4., 5., 6.], vec![3, 2]);
        let res = tensor_a.sub_scalar(1.).await;
        assert_eq!(
            res.to_cpu().raw_data_slice(),
            &[-2., -3., -4., -5., 4., 5.]
        );
    };
    futures::executor::block_on(async_block);
}

#[test]
fn mul_scalar_test() {
    let async_block = async {
        let tensor_a = GpuTensor::from(vec![-1., -2., -3., -4., 5., 6.], vec![3, 2]);
        let res = tensor_a.mul_scalar(3.).await;
        assert_eq!(
            res.to_cpu().raw_data_slice(),
            &[-3., -6., -9., -12., 15., 18.]
        );
    };
    futures::executor::block_on(async_block);
}


#[test]
fn div_scalar_test() {
    let async_block = async {
        let tensor_a = GpuTensor::from(vec![-1., -2., -3., -4., 5., 6.], vec![3, 2]);
        let res = tensor_a.div_scalar(2.).await;

        let err: f32 = res.to_cpu().raw_data_slice().iter()
            .zip([-0.5, -1., -3./2., -2., 5./2., 3.].iter())
            .map(|(l, r)| l-r)
            .sum();
        assert!(err < 0.01);

    };
    futures::executor::block_on(async_block);
}