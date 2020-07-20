use crate::gpu_internals::shader_runner::{BufferType, ShaderInput, ThreadGroup};
use crate::gpu_internals::GpuInstance;
use crate::{GpuTensor, GpuTensorView, ShapeStrides, TensorTrait};
use zerocopy::AsBytes;

#[cfg(test)]
mod tests;

pub async fn make_contiguous<'a>(gpu: &GpuInstance, data: &GpuTensorView<'a>) -> GpuTensor {
    let cs_module = gpu.shader_from_file_bytes(wgpu::include_spirv!("make_contiguous.spv"));

    let nb_output_numbers = data.numel();

    let shape_u32: Vec<u32> = data.shape().iter().map(|s| *s as u32).collect();
    let strides_u32: Vec<u32> = data.strides().iter().map(|s| *s as u32).collect();

    let shapes = gpu.new_uniform_buffer(shape_u32.as_slice().as_bytes());
    let strides = gpu.new_uniform_buffer(strides_u32.as_slice().as_bytes());
    let shapes_strides_len = gpu.new_uniform_buffer(data.shape().len().as_bytes());
    let offset = gpu.new_uniform_buffer(data.shape_strides.offset.as_bytes());

    let output = gpu.new_empty_gpu_buffer(std::mem::size_of::<f32>() * nb_output_numbers);
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
                gpu_buffer: BufferType::Storage(&output),
            },
        ],
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
