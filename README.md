
# Rust GPU Compute   

This is a personal project to teach myself WebGPU computing, focused   
on Machine Learning application.  
  
Working Example:  
  
```Rust  
use tensor_compute::{CpuTensor2D, GpuBox};  
  
  
fn main() {  
	 let async_block = async {  println!("Running in {:?}", GpuStore::get_default().info());  
	 // Running in AdapterInfo { name: "AMD Radeon Pro 560", vendor: 0, device: 0, device_type: DiscreteGpu, backend: Metal } 
	 let ma = GpuTensor::from_data(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]);  
	 let mb = GpuTensor::from_data(vec![2., 3., 4., 5.], vec![2, 2]);  
	 let result = ma.mm(&mb).await;  
	 println!("{:?}", result.to_cpu().await);  
	 // CpuTensor { data: [10.0, 13.0, 22.0, 29.0, 34.0, 45.0, 46.0, 61.0], shape: [2, 2, 2] } }; futures::executor::block_on(async_block);
 }  
  
```

## Short-term goals

### Pretty Print Tensors similar to Pytorch:
- A [2, 2, 2] Tensor should print like:

``` 
CpuTensor {
	data: [[[1., 2.],
	      [3., 4.]],

              [[5., 6.],
              [7., 8.]]]
}
```

- A [2, 2, 2, 2] Tensor should print like:

``` 
CpuTensor {
	data: [[[[0.4756, 0.3498],
	      [0.6470, 0.7661]],

              [[0.9003, 0.2503],
              [0.9068, 0.2080]]],


              [[[0.7718, 0.0772],
              [0.7057, 0.3425]],

              [[0.5234, 0.7733],
              [0.5547, 0.4822]]]]
}
```

## 
