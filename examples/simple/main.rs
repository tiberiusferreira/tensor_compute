use gpu_compute::{GpuBox, GpuTensor, GpuStore};

fn main() {
    let async_block = async {
        let ma = GpuTensor::from_data(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]);
        let mb = GpuTensor::from_data(vec![2., 3., 4., 5.], vec![2, 2]);
        let result = &ma.bmm(&mb).await;
        let bcpu_copy = result.to_cpu().await;
        println!("{:?}", bcpu_copy);
    };
    futures::executor::block_on(async_block);
}
