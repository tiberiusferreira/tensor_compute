use crate::{Tensor, GpuTensor, GpuTensorView, GpuBox, CpuTensor, GpuStore};
use crate::gpu_buffers::GpuBuffer;
use std::convert::TryInto;

impl Tensor for GpuTensor{
    fn shape(&self) -> Vec<usize> {
        self.shape.to_vec()
    }

    fn strides(&self) -> Vec<usize> {
        Self::strides_from_shape(&self.shape())
    }
}

impl GpuTensor {
    pub fn storage(&self) -> &GpuBuffer {
        &self.buffer
    }
    pub fn buffer_size_in_bytes(&self) -> usize {
        self.storage().size_bytes()
    }
    pub fn broadcast<'a> (&'a self, other: &'a Self) -> Option<(GpuTensorView<'a>, GpuTensorView<'a>)>{
        let current_shape = self.shape();
        let target_shape = other.shape();
        let mut expanded_current_shape = vec![];
        let mut expanded_target_shape = vec![];
        let diff = (target_shape.len() as i32 - current_shape.len() as i32);
        if diff > 0 {
            for _i in 0..diff{
                expanded_current_shape.push(1);
            }
            expanded_current_shape.extend_from_slice(current_shape.as_slice());
            expanded_target_shape = target_shape.to_vec();
        }else{
            for _i in 0..-diff{
                expanded_target_shape.push(1);
            }
            expanded_target_shape.extend_from_slice(target_shape.as_slice());
            expanded_current_shape = current_shape.to_vec();
        }
        let mut final_curr_strides = GpuTensor::strides_from_shape(&expanded_current_shape);
        let mut final_target_strides = GpuTensor::strides_from_shape(&expanded_target_shape);
        let mut final_current_dims = vec![];
        let mut final_target_dims = vec![];
        for (i, (&curr_dim, &target_dim)) in expanded_current_shape.iter().zip(expanded_target_shape.iter()).enumerate(){
            if curr_dim == target_dim{
                final_current_dims.push(curr_dim);
                final_target_dims.push(target_dim);
            }else if curr_dim == 1{
                final_curr_strides[i] = 0;
                final_current_dims.push(target_dim);
                final_target_dims.push(target_dim);
            }else if target_dim == 1{
                final_target_strides[i] = 0;
                final_current_dims.push(curr_dim);
                final_target_dims.push(curr_dim);
            }else{
                panic!("Cant broadcast between dims {} and {} .", curr_dim, target_dim);
            }
        }
        Some((GpuTensorView{
            buffer: self.storage(),
            shape: final_current_dims,
            strides: final_curr_strides
        },
              GpuTensorView{
                  buffer: other.storage(),
                  shape: final_target_dims,
                  strides: final_target_strides
              }))
    }

    pub fn from_data_in_gpu(gpu: &GpuBox, data: Vec<f32>, shape: Vec<usize>) -> Self {
        let calc_size = shape.iter().rev().fold(1, |acc: usize, &x| acc * x);
        assert_eq!(
            calc_size,
            data.len(),
            "Shape is not valid for the size of the data!"
        );
        CpuTensor::new(data, shape).to_gpu(gpu)
    }

    pub fn from_data(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let gpu = GpuStore::current();
        Self::from_data_in_gpu(gpu, data, shape)
    }


    pub fn from_buffer(buffer: GpuBuffer, shape: Vec<usize>) -> Self {
        Self { buffer, shape }
    }

    pub fn strides_from_shape(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![];
        strides.push(1);
        for dim in shape.iter().skip(1).rev(){
            let biggest_stride = strides.last().unwrap().clone();
            strides.push(dim*biggest_stride)
        }
        strides.reverse();
        strides
    }



    pub fn reshape(&mut self, shape: Vec<usize>){
        let calc_size = shape.iter().fold(0, |acc: usize, &x| acc + x);
        assert_eq!(
            calc_size,
            self.numel(),
            "Shape is not valid for the size of the data!"
        );
        self.shape = shape;
    }


    pub async fn to_cpu_with_gpu(&self, gpu: &GpuBox) -> CpuTensor {
        let gpu = GpuStore::get(self.buffer.device());

        let cpu_readable_output_buffer = gpu.staging_output_buffer(self.buffer_size_in_bytes());

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(
            self.storage().raw_buffer(),
            0,
            &cpu_readable_output_buffer.raw_buffer(),
            0,
            self.buffer_size_in_bytes() as u64,
        );

        gpu.queue.submit(Some(encoder.finish()));

        let buffer_slice_a = cpu_readable_output_buffer.raw_buffer().slice(..);
        let buffer_future_a = buffer_slice_a.map_async(wgpu::MapMode::Read);

        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        gpu.device.poll(wgpu::Maintain::Wait);

        if let Ok(()) = buffer_future_a.await {
            let data = buffer_slice_a.get_mapped_range();

            let result: Vec<f32> = data
                .chunks_exact(std::mem::size_of::<f32>())
                .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                .collect();

            // With the current interface, we have to make sure all mapped views are
            // dropped before we unmap the buffer.
            drop(data);
            cpu_readable_output_buffer.raw_buffer().unmap();
            CpuTensor::new(result, self.shape.clone())
        } else {
            panic!("Could not transfer data to CPU!")
        }
    }
}
