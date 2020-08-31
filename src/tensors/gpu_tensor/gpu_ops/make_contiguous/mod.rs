use crate::gpu_internals::shader_runner::{BufferType, ShaderInput, ThreadGroup};
use crate::gpu_internals::GpuInstance;
use crate::{GpuTensor, GpuTensorView, ShapeStrideTrait, ShapeStrides, AsShaderInput};

#[cfg(test)]
mod tests;

pub async fn make_contiguous<'a>(gpu: &GpuInstance, data: &GpuTensorView<'a>) -> GpuTensor {
    let cs_module = gpu.shader_from_file_bytes(wgpu::include_spirv!("make_contiguous.spv"));

    let nb_output_numbers = data.numel();
    let mut shader_inputs = data.to_shader_inputs(0);
    let output = gpu.new_empty_gpu_buffer(std::mem::size_of::<f32>() * nb_output_numbers);
    shader_inputs.push(ShaderInput{
        binding_id: shader_inputs.len(),
        gpu_buffer: BufferType::Storage(&output)
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
    GpuTensor {
        buffer: output,
        shape_strides: ShapeStrides::from_shape(data.shape().clone()),
    }
}
