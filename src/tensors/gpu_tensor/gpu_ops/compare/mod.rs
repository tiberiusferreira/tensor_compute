use crate::gpu_internals::shader_runner::{BufferType, ShaderInput, ThreadGroup};
use crate::gpu_internals::GpuInstance;
use crate::{GpuTensor, ShapeStrideTrait, GpuTensorView, GpuAllocated, CpuTransferable, AsShaderInput};
use zerocopy::AsBytes;
use std::collections::VecDeque;

#[cfg(test)]
mod tests;

pub async fn eq<'a>(gpu: &GpuInstance, left: &GpuTensorView<'a>, right: &GpuTensorView<'a>) -> bool {
    if left.shape_strides.shape != right.shape_strides.shape{
        return false;
    }
    let cs_module = gpu.shader_from_file_bytes(wgpu::include_spirv!("compare.spv"));
    // uses bindings 0 to 4 (5 items)
    let mut shader_inputs = left.to_shader_inputs(0);
    // uses bindings 5 to 9
    let right_inputs = right.to_shader_inputs(shader_inputs.len());
    shader_inputs.extend(right_inputs);
    let output = gpu.new_gpu_buffer_from_data(0f32.as_bytes());
    shader_inputs.push(ShaderInput{
        binding_id: 10,
        gpu_buffer: BufferType::Storage(&output)
    });
    gpu.run_shader(
        &cs_module,
        shader_inputs,
        ThreadGroup {
            x: left.numel(),
            y: 1,
            z: 1,
        },
    );
    let output = GpuTensor::from_buffer(output, VecDeque::from(vec![1])).to_cpu().await.idx(&vec![0]);
    return output == 0.;
}
