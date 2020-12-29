use tensor_compute::{s, GpuStore, RawTensor};

fn main() {
    // println!("Running in {:?}", GpuStore::get_default().info());
    // let mut ma = Tensor::from_data_and_shape(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]);
    // println!("{:?}", ma);
    // ma.assign(s!(1..2; ..; 1..2), 10.);
    // ma.assign(s!(0;0;0), -50.);
    // println!("{:?}", ma);
    /*
    Shape: [2, 2, 2]
    [[[ -50  2 ]
      [ 3  4 ]]

     [[ 5  10 ]
      [ 7  10 ]]]
    */
}
