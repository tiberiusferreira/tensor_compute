#[cfg(test)]
mod tests;

use crate::gpu_internals::shader_runner::{BufferType, ShaderBinding, ThreadGroup};
use crate::gpu_internals::GpuInstance;
use crate::{GpuAllocated, GpuTensor, ShapeStrideTrait, RawTensor};
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
    assert_eq!(left.shape().len(), 3);
    assert_eq!(right.shape().len(), 3);
    // the batch sizes need to be the same -> [2x3x4] x [2x4x6] -> [3x4] [4x6]
    assert_eq!(left.shape()[0], right.shape()[0]);

    assert_eq!(left.shape()[2], right.shape()[1], "Incompatible shapes for matmul");

    let shapes = Shapes {
        batch_size: left.shape()[0] as u32,
        stride_batch_size_a: left.strides()[0] as u32,
        stride_batch_size_b: right.strides()[0] as u32,
        rows_a: left.shape()[1] as u32,
        stride_rows_a: left.strides()[1] as u32,
        cols_a: left.shape()[2] as u32,
        stride_cols_a: left.strides()[2] as u32,
        rows_b: right.shape()[1] as u32,
        stride_rows_b: right.strides()[1] as u32,
        cols_b: right.shape()[2] as u32,
        stride_cols_b: right.strides()[2] as u32,
    };

    let cs_module = gpu.shader_from_file_bytes(wgpu::include_spirv!("bmm.spv"));

    let output_shape = vec![
        left.shape()[0],
        left.shape()[1],
        right.shape()[2],
    ];
    let nb_output_numbers = GpuTensor::numel_from_shape(&VecDeque::from(output_shape.clone()));
    let out_buffer_store = gpu.new_empty_gpu_buffer(std::mem::size_of::<f32>() * nb_output_numbers);

    let input_structure_data = gpu.new_gpu_buffer_from_data(shapes.as_bytes());
    // gpu.run_shader(
    //     &cs_module,
    //     vec![
    //         ShaderBinding {
    //             binding_id: 0,
    //             gpu_buffer: BufferType::Storage(left.internal_gpu_buffer()),
    //         },
    //         ShaderBinding {
    //             binding_id: 1,
    //             gpu_buffer: BufferType::Storage(right.internal_gpu_buffer()),
    //         },
    //         ShaderBinding {
    //             binding_id: 2,
    //             gpu_buffer: BufferType::Storage(&out_buffer_store),
    //         },
    //         ShaderBinding {
    //             binding_id: 3,
    //             gpu_buffer: BufferType::Storage(&input_structure_data),
    //         },
    //     ],
    //     None,
    //     ThreadGroup {
    //         x: nb_output_numbers,
    //         y: 1,
    //         z: 1,
    //     },
    // );
    // GpuTensor::from_buffer(out_buffer_store, VecDeque::from(output_shape))
    unimplemented!()
}
