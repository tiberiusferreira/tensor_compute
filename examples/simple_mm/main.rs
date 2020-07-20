use gpu_compute::{GpuStore, GpuTensor};


fn main() {
    let async_block = async {
        println!("Running in {:?}", GpuStore::get_default().info());
        let ma =
            GpuTensor::from_data_and_shape(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]);
        let mb = GpuTensor::from_data_and_shape(vec![2., 3., 4., 5.], vec![2, 2]);
        let result = ma.mm(&mb).await;
        println!("{}", result.to_cpu().await);
        /*
        Running in AdapterInfo { name: "AMD Radeon Pro 560", vendor: 0, device: 0, device_type: DiscreteGpu, backend: Metal }
        Shape: [2, 2, 2]
        [[[ 10  13 ]
          [ 22  29 ]]

         [[ 34  45 ]
          [ 46  61 ]]]
        */
    };
    futures::executor::block_on(async_block);
}
