use gpu_compute::{Gpu2DTensor, GpuBox};

fn main() {
    let async_block = async {
        let gpu = GpuBox::new().await;
        let ma = Gpu2DTensor::new(&gpu, vec![1., 2., 3., 4.], (2, 2));
        let mb = Gpu2DTensor::new(&gpu, vec![2., 3., 4., 5.], (2, 2));
        let times = gpu.mm(&ma, &mb).await;
        let cpu_copy = times.to_cpu(&gpu).await;
        println!("{:?}", cpu_copy);
    };
    futures::executor::block_on(async_block);
}
