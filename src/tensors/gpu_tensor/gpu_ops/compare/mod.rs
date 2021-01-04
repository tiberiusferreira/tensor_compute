use crate::gpu_internals::shader_runner::{ThreadGroup};
use crate::gpu_internals::GpuInstance;
use crate::{AsShaderInput, CpuTransferable, GpuTensor, ShapeStrideTrait};
use std::collections::VecDeque;
use zerocopy::AsBytes;

#[cfg(test)]
mod tests;

pub async fn eq(
    gpu: &GpuInstance,
    left: &GpuTensor,
    right: &GpuTensor,
) -> bool {
    if left.is_empty() || right.is_empty() {
        return left.is_empty() && right.is_empty()
    }
    if left.shape_strides.shape != right.shape_strides.shape {
        return false;
    }
    let cs_module = gpu.shader_from_file_bytes(wgpu::include_spirv!("compare.spv"));
    // uses bindings 0
    let mut shader_inputs = left.to_shader_inputs().with_tensor(right);
    // uses bindings 1
    let output = gpu.gpu_buffer_from_data(0f32.as_bytes());
    shader_inputs.append_buffer(&output);
    gpu.run_shader(
        &cs_module,
        &shader_inputs,
        ThreadGroup {
            x: left.numel(),
            y: 1,
            z: 1,
        },
    );
    let output = GpuTensor::from_buffer(output, VecDeque::from(vec![1]))
        .to_cpu_async()
        .await
        .idx(&vec![0]);
    return output == 0.;
}
