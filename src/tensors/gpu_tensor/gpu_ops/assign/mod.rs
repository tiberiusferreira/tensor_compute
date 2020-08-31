use crate::gpu_internals::shader_runner::{BufferType, ShaderInput, ThreadGroup};
use crate::gpu_internals::GpuInstance;
use crate::{GpuTensorViewMut, ShapeStrideTrait, AsShaderInput};
use zerocopy::AsBytes;

#[cfg(test)]
mod tests;

pub async fn assign<'a>(gpu: &GpuInstance, data: &mut GpuTensorViewMut<'a>, assign_data: f32) {
    let cs_module = gpu.shader_from_file_bytes(wgpu::include_spirv!("assign.spv"));

    let mut shader_inputs = data.to_shader_inputs(0);


    let assign_data_buffer = gpu.new_uniform_buffer(assign_data.as_bytes());

    shader_inputs.push(ShaderInput{
        binding_id: shader_inputs.len(),
        gpu_buffer: BufferType::Uniform(&assign_data_buffer)
    });


    gpu.run_shader(
        &cs_module,
        shader_inputs,
        ThreadGroup {
            x:  data.numel(),
            y: 1,
            z: 1,
        },
    );
}
