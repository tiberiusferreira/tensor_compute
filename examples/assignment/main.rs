use gpu_compute::{s, GpuStore, GpuTensor, SliceRangeInfo};


fn main() {
    let async_block = async {
        println!("Running in {:?}", GpuStore::get_default().info());
        let mut ma =
            GpuTensor::from_data_and_shape(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]);
        ma.assign(s!(1..2; ..; 1..2), 10.).await;
        ma.assign(s!(0;0;0), -50.).await;
        println!("{}", ma.to_cpu().await);
        /*
        Shape: [2, 2, 2]
        [[[ -50  2 ]
          [ 3  4 ]]

         [[ 5  10 ]
          [ 7  10 ]]]
        */
    };
    futures::executor::block_on(async_block);
}

