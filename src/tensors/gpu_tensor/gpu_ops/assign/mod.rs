use crate::gpu_internals::shader_runner::{BufferType, ShaderInput, ThreadGroup};
use crate::gpu_internals::GpuInstance;
use crate::{GpuAllocated, GpuTensor, GpuTensorViewMut, ShapeStrideTrait};
use zerocopy::AsBytes;

#[cfg(test)]
mod tests;

pub async fn assign<'a>(gpu: &GpuInstance, data: &mut GpuTensorViewMut<'a>, assign_data: f32) {
    let cs_module = gpu.shader_from_file_bytes(wgpu::include_spirv!("assign.spv"));

    let assign_data = GpuTensor::from_scalar(assign_data);
    let nb_output_numbers = data.numel();

    let shape_u32: Vec<u32> = data.shape().iter().map(|s| *s as u32).collect();
    let strides_u32: Vec<u32> = data.strides().iter().map(|s| *s as u32).collect();

    let shapes = gpu.new_uniform_buffer(shape_u32.as_slice().as_bytes());
    let strides = gpu.new_uniform_buffer(strides_u32.as_slice().as_bytes());
    let shapes_strides_len = gpu.new_uniform_buffer(data.shape().len().as_bytes());
    let offset = gpu.new_uniform_buffer(data.shape_strides.offset.as_bytes());

    gpu.run_shader(
        &cs_module,
        vec![
            ShaderInput {
                binding_id: 0,
                gpu_buffer: BufferType::Storage(data.internal_gpu_buffer()),
            },
            ShaderInput {
                binding_id: 1,
                gpu_buffer: BufferType::Uniform(&shapes),
            },
            ShaderInput {
                binding_id: 2,
                gpu_buffer: BufferType::Uniform(&strides),
            },
            ShaderInput {
                binding_id: 3,
                gpu_buffer: BufferType::Uniform(&shapes_strides_len),
            },
            ShaderInput {
                binding_id: 4,
                gpu_buffer: BufferType::Uniform(&offset),
            },
            ShaderInput {
                binding_id: 5,
                gpu_buffer: BufferType::Storage(assign_data.internal_gpu_buffer()),
            },
        ],
        ThreadGroup {
            x: nb_output_numbers,
            y: 1,
            z: 1,
        },
    );
}
