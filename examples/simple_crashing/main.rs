use gpu_compute::{GpuStore, Tensor, GpuTensor};

fn main() {
    for _i in 0..1000{
        let async_block = async {
            let tensor_a =
                GpuTensor::from((0..6).map(|e| e as f32).collect(), vec![3, 2]);
            let mut tensor_b = tensor_a.clone().await;
            assert!(tensor_a.eq(&tensor_b).await);
            // tensor_b.assign(s!(0 ; 0), 50.).await;
            // println!("{:?}", tensor_a);
            // println!("{:?}", tensor_b);
            // assert!(!tensor_a.eq(&tensor_b).await);
            // tensor_b.assign(s!(0 ; 0), 0.).await;
            // assert!(tensor_a.eq(&tensor_b).await);
            // tensor_b.assign(s!(2 ; 1), 3.).await;
            // assert!(!tensor_a.eq(&tensor_b).await);
            // println!("{:?}", tensor_a);
            // println!("{:?}", tensor_b);
        };
        futures::executor::block_on(async_block);
    }
}
