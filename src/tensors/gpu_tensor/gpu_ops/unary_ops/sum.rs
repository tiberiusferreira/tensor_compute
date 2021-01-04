use crate::{GpuTensor, GpuAllocated, AsShaderInput};
use crate::gpu_internals::shader_runner::{ThreadGroup};
use std::collections::VecDeque;

impl GpuTensor{
    pub fn sum(&self) -> Self{
        let cs_module = self.gpu().shader_from_file_bytes(wgpu::include_spirv!("sum.spv"));
        let out_buffer_store = self.gpu().empty_gpu_buffer(std::mem::size_of::<f32>());
        let mut shader_inputs = self.to_shader_inputs();
        shader_inputs.append_buffer(&out_buffer_store);
        self.gpu().run_shader(
            &cs_module,
            &shader_inputs,
            ThreadGroup {
                x: 1,
                y: 1,
                z: 1,
            },
        );
        GpuTensor::from_buffer(out_buffer_store, VecDeque::from(vec![1]))
    }
}


#[cfg(test)]
mod test{
    use crate::{GpuTensor, ShapeStrideTrait, CpuTransferable};
    use std::collections::VecDeque;

    #[test]
    fn sum_test_1d(){
        let tensor = GpuTensor::from(vec![2., 3., 4., 5.], vec![4]);
        let sum_tensor = tensor.sum();
        assert_eq!(sum_tensor.shape(), &VecDeque::from(vec![1usize]));
        assert_eq!(sum_tensor.to_cpu().raw_data_slice(), &[14.]);
    }

    #[test]
    fn sum_test_2d(){
        let tensor = GpuTensor::from(vec![2., 3., 4., -5.], vec![2, 2]);
        let sum_tensor = tensor.sum();
        assert_eq!(sum_tensor.shape(), &VecDeque::from(vec![1]));
        assert_eq!(sum_tensor.to_cpu().raw_data_slice(), &[4.]);
    }
}