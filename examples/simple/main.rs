use gpu_compute::{Gpu2DTensor, GpuBox, GpuTensor};

fn main() {
    let async_block = async {
        let gpu = GpuBox::new().await;
        let ma = Gpu2DTensor::new(&gpu, vec![1., 2., 3., 4.], (2, 2));
        let mb = Gpu2DTensor::new(&gpu, vec![2., 3., 4., 5.], (2, 2));
        let times = gpu.mm(&ma, &mb).await;

        let ma = GpuTensor::new(&gpu, vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]);
        let mb = GpuTensor::new(&gpu, vec![2., 3., 4., 5.], vec![2, 2]);
        let btimes = gpu.bmm(&ma, &mb).await;

        let cpu_copy = times.to_cpu(&gpu).await;
        let bcpu_copy = btimes.to_cpu(&gpu).await;
        println!("{:?}", cpu_copy);
        println!("{:?}", bcpu_copy);
    };
    futures::executor::block_on(async_block);
}
