use crate::gpu_buffers::GpuBuffer;
use crate::tensors::CpuTensor2D;
use crate::GpuBox;
use std::convert::TryInto;
use crate::tensors::CpuTensor;

pub struct GpuTensor {
    buffer: GpuBuffer,
    shape: Vec<usize>,
}

pub struct GpuTensorView<'a> {
    buffer: &'a GpuBuffer,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

impl <'a> GpuTensorView<'a>{
    pub fn buffer(&self) -> &'a GpuBuffer {
        &self.buffer
    }

    pub fn buffer_size_in_bytes(&'a self) -> usize {
        std::mem::size_of::<f32>() * self.len()
    }

    pub fn len(&self) -> usize{
        self.shape.iter().fold(1, |acc: usize, &x| acc * x)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> Vec<usize> {
        self.strides.to_vec()
    }

}

impl GpuTensor {
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
            expanded_current_shape.extend_from_slice(current_shape);
            expanded_target_shape = target_shape.to_vec();
        }else{
            for _i in 0..-diff{
                expanded_target_shape.push(1);
            }
            expanded_target_shape.extend_from_slice(target_shape);
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
        println!("Target Shape {:?}", final_target_dims);
        println!("Target Strides {:?}", final_target_strides);
        println!("Current {:?}", final_current_dims);
        println!("Current Strides {:?}", final_curr_strides);
        Some((GpuTensorView{
            buffer: self.buffer(),
            shape: final_current_dims,
            strides: final_curr_strides
        },
        GpuTensorView{
            buffer: other.buffer(),
            shape: final_target_dims,
            strides: final_target_strides
        }))
    }

    pub fn new(gpu: &GpuBox, data: Vec<f32>, shape: Vec<usize>) -> Self {
        let calc_size = shape.iter().rev().fold(1, |acc: usize, &x| acc * x);
        assert_eq!(
            calc_size,
            data.len(),
            "Shape is not valid for the size of the data!"
        );
        CpuTensor::new(data, shape).to_gpu(gpu)
    }

    pub fn len(&self) -> usize{
        self.shape.iter().fold(1, |acc: usize, &x| acc * x)
    }

    pub fn from_buffer(buffer: GpuBuffer, shape: Vec<usize>) -> Self {
        Self { buffer, shape }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
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
    pub fn strides(&self) -> Vec<usize> {
        Self::strides_from_shape(self.shape())
    }

    pub fn reshape(&mut self, shape: Vec<usize>){
        let calc_size = shape.iter().fold(0, |acc: usize, &x| acc + x);
        assert_eq!(
            calc_size,
            self.len(),
            "Shape is not valid for the size of the data!"
        );
        self.shape = shape;
    }

    pub fn buffer(&self) -> &GpuBuffer {
        &self.buffer
    }

    pub fn buffer_size_in_bytes(&self) -> usize {
        std::mem::size_of::<f32>() * self.len()
    }

    pub async fn to_cpu(&self, gpu: &GpuBox) -> CpuTensor {
        let cpu_readable_output_buffer = gpu.staging_output_buffer(self.buffer_size_in_bytes());

        let mut encoder = gpu
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        encoder.copy_buffer_to_buffer(
            self.buffer().raw_buffer(),
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

#[cfg(test)]
mod tests {
    use crate::{CpuTensor, GpuBox};

    #[test]
    fn strides_work() {
        let async_block = async{
            let gpu = GpuBox::new().await;
            let a = CpuTensor::new(vec![1., 2., 3., 4.], vec![2, 2]);
            let b = a.to_gpu(&gpu);
            assert_eq!(b.strides(), vec![2, 1]);


            let a = CpuTensor::new(vec![1., 2., 3., 4., 5., 6., 7., 8.], vec![2, 2, 2]);
            let b = a.to_gpu(&gpu);
            assert_eq!(b.strides(), vec![4, 2, 1]);
        };
        futures::executor::block_on(async_block);
    }

    #[test]
    fn broadcast_work() {
        let async_block = async{
            let gpu = GpuBox::new().await;
            let a = CpuTensor::new(vec![1., 2., 3., 4., 1., 2., 3., 4.], vec![2, 2, 2]);
            let a_gpu = a.to_gpu(&gpu);
            let b = CpuTensor::new(vec![1., 2., 3., 4.], vec![2, 2]);
            let b_gpu = b.to_gpu(&gpu);
            a_gpu.broadcast(&b_gpu);
            // assert_eq!(b.strides(), vec![2, 1]);

        };
        futures::executor::block_on(async_block);
    }
}