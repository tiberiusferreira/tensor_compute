mod tests;

use crate::shader_runner::{ShaderInput, ThreadGroup};
use crate::{GpuBox, GpuTensor, Tensor};
use zerocopy::{AsBytes, FromBytes};

#[repr(C)]
#[derive(AsBytes, FromBytes, Clone, Debug)]
struct Shapes {
    batch_size: u32,
    stride_batch_size_a: u32,
    stride_batch_size_b: u32,
    rows_a: u32,
    stride_rows_a: u32,
    cols_a: u32,
    stride_cols_a: u32,
    rows_b: u32,
    stride_rows_b: u32,
    cols_b: u32,
    stride_cols_b: u32,
}

impl GpuBox {
    /// Performs Batch Matrix Multiplication of the input Tensors.
    pub async fn bmm(&self, input_data_a: &GpuTensor, input_data_b: &GpuTensor) -> GpuTensor {
        let (input_data_a_view, input_data_b_view) = input_data_a.broadcast(input_data_b).unwrap();
        assert_eq!(input_data_a_view.shape().len(), 3);
        assert_eq!(input_data_b_view.shape().len(), 3);
        assert_eq!(input_data_a_view.shape()[0], input_data_b_view.shape()[0]);
        let shapes = Shapes{
            batch_size: input_data_a_view.shape()[0] as u32,
            stride_batch_size_a: input_data_a_view.strides()[0] as u32,
            stride_batch_size_b: input_data_b_view.strides()[0] as u32,
            rows_a: input_data_a_view.shape()[1] as u32,
            stride_rows_a: input_data_a_view.strides()[1] as u32,
            cols_a: input_data_a_view.shape()[2] as u32,
            stride_cols_a: input_data_a_view.strides()[2] as u32,
            rows_b: input_data_b_view.shape()[1] as u32,
            stride_rows_b: input_data_b_view.strides()[1] as u32,
            cols_b: input_data_b_view.shape()[2] as u32,
            stride_cols_b: input_data_b_view.strides()[2] as u32
        };
        let cs_module = self.shader_from_file_bytes(wgpu::include_spirv!("bmm.spv"));


        let output_shape = vec![input_data_a_view.shape()[0], input_data_a_view.shape()[2], input_data_b_view.shape()[1]];
        let nb_output_numbers = GpuTensor::numel_from_shape(output_shape.as_slice()); //.iter().rev().fold(1, |acc, &x| acc * x);
        let out_buffer_store =
            self.empty_gpu_buffer(std::mem::size_of::<f32>() * nb_output_numbers);

        let input_structure_data = self.gpu_buffer_from_data(shapes.as_bytes());
        self.run_shader(
            &cs_module,
            vec![
                ShaderInput {
                    binding_id: 0,
                    gpu_buffer: input_data_a_view.buffer(),
                },
                ShaderInput {
                    binding_id: 1,
                    gpu_buffer: input_data_b_view.buffer(),
                },
                ShaderInput {
                    binding_id: 2,
                    gpu_buffer: &out_buffer_store,
                },
                ShaderInput {
                    binding_id: 3,
                    gpu_buffer: &input_structure_data,
                },
            ],
            ThreadGroup {
                x: nb_output_numbers,
                y: 1,
                z: 1,
            },
        );
        GpuTensor::from_buffer(out_buffer_store, output_shape)
    }
}
