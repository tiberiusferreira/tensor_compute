# Rust GPU Compute 

This is a personal project to teach myself WebGPU computing, focused 
on Machine Learning application.

Working Example:
 
```Rust
use gpu_compute::{CpuTensor2D, GpuBox};

fn main() {
    let ma = CpuTensor2D::new(vec![1, 2, 3, 4], (2, 2));
    let mb = CpuTensor2D::new(vec![2, 3, 4, 5], (2, 2));
    let async_block = async {
        let gpu = GpuBox::new().await;
        let ma_gpu = ma.copy_to_gpu(&gpu);
        let mb_gpu = mb.copy_to_gpu(&gpu);
        let times = gpu.mm(&ma_gpu, &mb_gpu).await;
        let cpu_copy = times.copy_to_cpu(&gpu).await;
        println!("{:?}", cpu_copy);
    };
    futures::executor::block_on(async_block);
}
``` 
