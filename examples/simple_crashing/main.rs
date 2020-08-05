use gpu_compute::{GpuTensor, CpuTransferable};

fn main() {
    let mut handles = vec![];
    for _i in 0..2000 {
        let handle = std::thread::spawn(|| {
            let async_block = async {
                let mut tensor =
                    GpuTensor::from(vec![-1., -2., -3., -4., 5., 6.], vec![3, 2]);
                tensor.fill_with(10.).await;
                assert_eq!(
                    tensor.to_cpu().await.raw_data_slice(),
                    &[10., 10., 10., 10., 10., 10.]
                );
            };
            futures::executor::block_on(async_block);
        });
        handles.push(handle);
    }
    for handle in handles{
        handle.join().unwrap();
    }
}
