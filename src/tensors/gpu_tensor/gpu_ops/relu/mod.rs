use crate::gpu_internals::GpuInstance;
use crate::{GpuTensor};
//
// #[cfg(test)]
// mod tests;
//
pub async fn leaky_relu(_gpu: &GpuInstance, data: &GpuTensor, _leakage: f32) -> GpuTensor {
    if data.is_empty(){
        return data.clone().await;
    }
    // let cs_module = gpu.shader_from_file_bytes(wgpu::include_spirv!("relu.spv"));
    //
    // let leakage_as_tensor = GpuTensor::from_scalar(leakage);
    // let nb_output_numbers = data.numel();
    // let out_buffer_store = gpu.new_empty_gpu_buffer(std::mem::size_of::<f32>() * nb_output_numbers);
    //
    // gpu.run_shader(
    //     &cs_module,
    //     vec![
    //         ShaderBinding {
    //             binding_id: 0,
    //             gpu_buffer: BufferType::Storage(data.internal_gpu_buffer()),
    //         },
    //         ShaderBinding {
    //             binding_id: 1,
    //             gpu_buffer: BufferType::Storage(leakage_as_tensor.internal_gpu_buffer()),
    //         },
    //         ShaderBinding {
    //             binding_id: 2,
    //             gpu_buffer: BufferType::Storage(&out_buffer_store),
    //         },
    //     ],
    //     None,
    //     ThreadGroup {
    //         x: nb_output_numbers,
    //         y: 1,
    //         z: 1,
    //     },
    // );
    // GpuTensor::from_buffer(out_buffer_store, data.shape().clone())
    unimplemented!()
}
