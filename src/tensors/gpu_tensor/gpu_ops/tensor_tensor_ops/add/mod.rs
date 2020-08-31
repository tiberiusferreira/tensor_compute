// use crate::gpu_internals::shader_runner::{BufferType, ShaderInput, ThreadGroup};
// use crate::gpu_internals::GpuInstance;
// use crate::{GpuAllocated, GpuTensor, GpuTensorView, ShapeStrideTrait};
//
// #[cfg(test)]
// mod tests;
//
// pub async fn add(gpu: &GpuInstance, left_tensor: &GpuTensorView, right_tensor: &GpuTensorView) -> GpuTensor {
//     assert_eq!(left_tensor.shape(), right_tensor.shape(), "Can't add tensors with incompatible shapes");
//     let cs_module = gpu.shader_from_file_bytes(wgpu::include_spirv!("add.spv"));
//
//     let nb_output_numbers = data_left.numel();
//     let output_buffer = gpu.new_empty_gpu_buffer(left_tensor.buffer_size_in_bytes());
//
//     gpu.run_shader(
//         &cs_module,
//         vec![
//             ShaderInput {
//                 binding_id: 0,
//                 gpu_buffer: BufferType::Storage(left_tensor.internal_gpu_buffer()),
//             },
//             ShaderInput {
//                 binding_id: 1,
//                 gpu_buffer: BufferType::Storage(right_tensor.internal_gpu_buffer()),
//             },
//             ShaderInput {
//                 binding_id: 2,
//                 gpu_buffer: BufferType::Storage(&output_buffer),
//             },
//         ],
//         ThreadGroup {
//             x: nb_output_numbers,
//             y: 1,
//             z: 1,
//         },
//     );
//
//     GpuTensor::from_buffer(output_buffer, left_tensor.shape_strides.shape.clone())
// }
