use crate::gpu_internals::shader_runner::{BufferType, ShaderBinding, ThreadGroup};
use crate::gpu_internals::GpuInstance;
use crate::{GpuAllocated, GpuTensor, ShapeStrideTrait};
use std::collections::VecDeque;
#[cfg(test)]
mod tests;

pub async fn log_soft_max(gpu: &GpuInstance, data: &GpuTensor) -> GpuTensor {
    assert_eq!(data.shape_strides.shape.len(), 1, "Can only apply log_soft_max to one dimensional tensors");
    let cs_module = gpu.shader_from_file_bytes(wgpu::include_spirv!("log_soft_max.spv"));
    let output_buffer = gpu.new_empty_gpu_buffer(data.buffer.size_bytes());
    let nb_output_numbers = data.numel();

    // gpu.run_shader(
    //     &cs_module,
    //     vec![
    //         ShaderBinding {
    //             binding_id: 0,
    //             gpu_buffer: BufferType::Storage(data.internal_gpu_buffer()),
    //         },
    //         ShaderBinding {
    //             binding_id: 1,
    //             gpu_buffer: BufferType::Storage(&output_buffer),
    //         },
    //     ],
    //     None,
    //     ThreadGroup {
    //         x: nb_output_numbers,
    //         y: 1,
    //         z: 1,
    //     },
    // );
    // GpuTensor::from_buffer(output_buffer, data.shape_strides.shape.clone())
    unimplemented!()
}
