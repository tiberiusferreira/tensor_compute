use crate::gpu_internals::shader_runner::{BufferType, ShaderInput, ThreadGroup};
use crate::gpu_internals::GpuInstance;
use crate::{GpuTensor, GpuTensorView, ShapeStrideTrait, AsShaderInput};

#[cfg(test)]
mod tests;

pub async fn sub<'a>(gpu: &GpuInstance, left_tensor: &GpuTensorView<'a>, right_tensor: &GpuTensorView<'a>) -> GpuTensor {
    assert_eq!(left_tensor.shape(), right_tensor.shape(), "Can't sub tensors with incompatible shapes");
    let cs_module = gpu.shader_from_file_bytes(wgpu::include_spirv!("sub.spv"));

    let mut shader_inputs = left_tensor.to_shader_inputs(0);
    shader_inputs.extend(right_tensor.to_shader_inputs(shader_inputs.len()));
    let nb_output_numbers = left_tensor.numel();
    let output_buffer = gpu.new_empty_gpu_buffer(left_tensor.buffer_size_in_bytes());
    shader_inputs.push(ShaderInput{
        binding_id: shader_inputs.len(),
        gpu_buffer: BufferType::Storage(&output_buffer)
    });
    gpu.run_shader(
        &cs_module,
        shader_inputs,
        ThreadGroup {
            x: nb_output_numbers,
            y: 1,
            z: 1,
        },
    );

    GpuTensor::from_buffer(output_buffer, left_tensor.shape_strides.shape.clone())
}
