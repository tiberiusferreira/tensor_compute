mod tests;

use crate::gpu_internals::shader_runner::{BufferType, ShaderInput, ThreadGroup};
use crate::gpu_internals::GpuInstance;
use crate::{GpuTensor, GpuTensorView, TensorTrait};
use std::collections::VecDeque;
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

/// Performs Batch Matrix Multiplication of the input Tensors.
/// Expects inputs to be of Rank 3, have same batch size and compatible dimensions
pub async fn bmm_kernel<'a>(
    gpu: &GpuInstance,
    input_data_a_view: &GpuTensorView<'a>,
    input_data_b_view: &GpuTensorView<'a>,
) -> GpuTensor {
    // let (input_data_a_view, input_data_b_view) = (input_data_a.view(), input_data_b.view());
    assert_eq!(input_data_a_view.shape().len(), 3);
    assert_eq!(input_data_b_view.shape().len(), 3);
    assert_eq!(input_data_a_view.shape()[0], input_data_b_view.shape()[0]);

    let shapes = Shapes {
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
        stride_cols_b: input_data_b_view.strides()[2] as u32,
    };
    let cs_module = gpu.shader_from_file_bytes(wgpu::include_spirv!("bmm.spv"));

    let output_shape = vec![
        input_data_a_view.shape()[0],
        input_data_a_view.shape()[1],
        input_data_b_view.shape()[2],
    ];
    let nb_output_numbers = GpuTensor::numel_from_shape(&VecDeque::from(output_shape.clone()));
    let out_buffer_store = gpu.new_empty_gpu_buffer(std::mem::size_of::<f32>() * nb_output_numbers);

    let input_structure_data = gpu.new_gpu_buffer_from_data(shapes.as_bytes());
    gpu.run_shader(
        &cs_module,
        vec![
            ShaderInput {
                binding_id: 0,
                gpu_buffer: BufferType::Storage(input_data_a_view.internal_gpu_buffer()),
            },
            ShaderInput {
                binding_id: 1,
                gpu_buffer: BufferType::Storage(input_data_b_view.internal_gpu_buffer()),
            },
            ShaderInput {
                binding_id: 2,
                gpu_buffer: BufferType::Storage(&out_buffer_store),
            },
            ShaderInput {
                binding_id: 3,
                gpu_buffer: BufferType::Storage(&input_structure_data),
            },
        ],
        ThreadGroup {
            x: nb_output_numbers,
            y: 1,
            z: 1,
        },
    );
    GpuTensor::from_buffer(out_buffer_store, VecDeque::from(output_shape))
}
