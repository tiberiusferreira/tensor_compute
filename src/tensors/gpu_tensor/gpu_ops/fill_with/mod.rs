use crate::gpu_internals::shader_runner::{BufferType, ShaderBinding, ThreadGroup};
use crate::gpu_internals::GpuInstance;
use crate::{GpuAllocated, GpuTensor, ShapeStrideTrait, AsShaderInput};

#[cfg(test)]
mod tests;

pub async fn fill_with(gpu: &GpuInstance, data: &GpuTensor, fill_with: f32) {
    let cs_module = gpu.shader_from_file_bytes(wgpu::include_spirv!("fill_with.spv"));

    let mut shader_inputs = data.to_shader_inputs(None);

    shader_inputs.push_constants.data.push(u32::from_ne_bytes(fill_with.to_ne_bytes()));
    let nb_output_numbers = data.numel();
    gpu.run_shader(
        &cs_module,
            &shader_inputs,
        ThreadGroup {
            x: nb_output_numbers,
            y: 1,
            z: 1,
        },
    );
}
