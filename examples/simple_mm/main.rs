use gpu_compute::{GpuStore, Tensor};

fn main() {
    println!("Running in {:?}", GpuStore::get_default().info());
    let ma = Tensor::from_data_and_shape(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]);
    let mb = Tensor::from_data_and_shape(vec![2., 3., 4., 5.], vec![2, 2]);
    let result = ma.matmul(&mb);
    println!("{:?}", result);
    /*
    Running in AdapterInfo { name: "AMD Radeon Pro 560", vendor: 0, device: 0, device_type: DiscreteGpu, backend: Metal }
    Shape: [2, 2, 2]
    [[[ 10  13 ]
      [ 22  29 ]]

     [[ 34  45 ]
      [ 46  61 ]]]
    */
}
