
# Rust Tensor Compute   

This is a personal project to teach myself WebGPU computing, focused   
on Machine Learning application.  

Features for now:

- [X] Select which GPU to use (if more than 1 in system)
- [X] Clone
- [X] (Batch) Matmul
- [X] Relu
- [X] Transpose
- [X] Fill
- [X] Compare
- [X] Make Contiguous
- [X] Slice
- [X] Index
- [X] Create Views Tensor


Working Example:  
  
```Rust  
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
```

## Short-term goals

- Finish public API docs
- More Tests
