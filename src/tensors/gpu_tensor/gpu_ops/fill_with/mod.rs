use crate::gpu_internals::shader_runner::{ShaderInput, ThreadGroup, BufferType};
use crate::gpu_internals::GpuInstance;
use crate::{GpuTensor, TensorTrait};

#[cfg(test)]
mod tests;

pub async fn fill_with(gpu: &GpuInstance, data: &GpuTensor, fill_with: f32) {
    let cs_module = gpu.shader_from_file_bytes(wgpu::include_spirv!("fill_with.spv"));

    let fill_val = GpuTensor::from_scalar(fill_with);
    let nb_output_numbers = data.numel();

    gpu.run_shader(
        &cs_module,
        vec![
            ShaderInput {
                binding_id: 0,
                gpu_buffer: BufferType::Storage(fill_val.internal_gpu_buffer()),
            },
            ShaderInput {
                binding_id: 1,
                gpu_buffer: BufferType::Storage(data.internal_gpu_buffer()),
            },
        ],
        ThreadGroup {
            x: nb_output_numbers,
            y: 1,
            z: 1,
        },
    );
}
