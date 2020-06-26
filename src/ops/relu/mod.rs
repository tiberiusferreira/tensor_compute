use crate::shader_runner::{ShaderInput, ThreadGroup};
use crate::{GpuBox, GpuTensor, Tensor};
use zerocopy::{AsBytes, FromBytes};
#[cfg(test)]
mod tests;

impl GpuBox {
    pub async fn relu(&self, data: &GpuTensor) -> GpuTensor {
        let cs_module = self.shader_from_file_bytes(wgpu::include_spirv!("relu.spv"));

        let nb_output_numbers = data.numel();
        let out_buffer_store =
            self.empty_gpu_buffer(std::mem::size_of::<u32>() * nb_output_numbers);

        self.run_shader(
            &cs_module,
            vec![
                ShaderInput {
                    binding_id: 0,
                    gpu_buffer: data.storage(),
                },
                ShaderInput {
                    binding_id: 1,
                    gpu_buffer: &out_buffer_store,
                },
            ],
            ThreadGroup {
                x: nb_output_numbers,
                y: 1,
                z: 1,
            },
        );
        GpuTensor::from_buffer(out_buffer_store, data.shape())
    }
}
