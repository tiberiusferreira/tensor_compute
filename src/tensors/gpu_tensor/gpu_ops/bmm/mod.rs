#[cfg(test)]
mod tests;

use crate::gpu_internals::shader_runner::{ThreadGroup};
use crate::gpu_internals::GpuInstance;
use crate::{GpuTensor, ShapeStrideTrait, AsShaderInput};
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
pub async fn bmm_kernel(
    gpu: &GpuInstance,
    left: &GpuTensor,
    right: &GpuTensor,
) -> GpuTensor {
    if left.is_empty() || right.is_empty(){
        panic!("Tried to matmul with at least one empty Tensor")
    }
    assert!(
        left.shape().len() == 3 && right.shape().len() == 3,
        "Cant matmul tensor of rank different than 3"
    );
    // the batch sizes need to be the same -> [2x3x4] x [2x4x6] -> [3x4] [4x6]
    assert_eq!(left.shape()[0], right.shape()[0], "matmul batches must be the same");
    assert_eq!(
        left.shape()[2],
        right.shape()[1],
        "Shapes do not match for matrix multiply: {:?} and {:?}",
        left.shape(),
        right.shape()
    );
    assert_eq!(left.shape().len(), 3, "Input to matmul must be of rank 3");

    let mut shader_inputs = left.to_shader_inputs().with_tensor(&right);
    shader_inputs.push_constants.data.clear();
    shader_inputs.push_constants.data.push(left.shape()[0] as u32); // batch_size
    shader_inputs.push_constants.data.push(left.strides()[0] as u32); // stride_batch_size_a
    shader_inputs.push_constants.data.push(right.strides()[0] as u32); // stride_batch_size_b
    shader_inputs.push_constants.data.push(left.shape()[1] as u32); // rows_a
    shader_inputs.push_constants.data.push(left.strides()[1] as u32); // stride_rows_a
    shader_inputs.push_constants.data.push(left.shape()[2] as u32); // cols_a
    shader_inputs.push_constants.data.push(left.strides()[2] as u32); // stride_cols_a
    shader_inputs.push_constants.data.push(right.shape()[1] as u32); // rows_b
    shader_inputs.push_constants.data.push(right.strides()[1] as u32); // stride_rows_b
    shader_inputs.push_constants.data.push(right.shape()[2] as u32); // cols_b
    shader_inputs.push_constants.data.push(right.strides()[2] as u32); // stride_cols_b

    let cs_module = gpu.shader_from_file_bytes(wgpu::include_spirv!("bmm.spv"));
    let output_shape = vec![
        left.shape()[0],
        left.shape()[1],
        right.shape()[2],
    ];
    let nb_output_numbers = GpuTensor::numel_from_shape(&VecDeque::from(output_shape.clone()));
    let out_buffer_store = gpu.empty_gpu_buffer(std::mem::size_of::<f32>() * nb_output_numbers);
    shader_inputs.append_buffer(&out_buffer_store);

    println!("{:?}", shader_inputs);
    gpu.run_shader(
        &cs_module,
            &shader_inputs,
        ThreadGroup {
            x: nb_output_numbers,
            y: 1,
            z: 1,
        },
    );
    GpuTensor::from_buffer(out_buffer_store, VecDeque::from(output_shape))
}
