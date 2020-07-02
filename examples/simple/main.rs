use gpu_compute::{GpuStore, GpuTensor};

fn main() {
    let async_block = async {
        println!("Running in {:?}", GpuStore::get_default().info());
        let ma = GpuTensor::from_data(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]);
        let mb = GpuTensor::from_data(vec![2., 3., 4., 5.], vec![2, 2]);
        let result = ma.mm(&mb).await;
        println!("{:#?}", result.to_cpu().await);
    };
    futures::executor::block_on(async_block);
}
