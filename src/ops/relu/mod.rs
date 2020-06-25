use crate::shader_runner::{ShaderInput, ThreadGroup};
use crate::{Gpu2DTensor, GpuBox};
use zerocopy::{AsBytes, FromBytes};
#[cfg(test)]
mod tests;

impl GpuBox {
    pub async fn relu(&self, data: &Gpu2DTensor) -> Gpu2DTensor {
        let cs_module = self.shader_from_file_bytes(wgpu::include_spirv!("relu.spv"));

        let nb_output_numbers = data.len();
        let out_buffer_store =
            self.empty_gpu_buffer(std::mem::size_of::<u32>() * nb_output_numbers);

        self.run_shader(
            &cs_module,
            vec![
                ShaderInput {
                    binding_id: 0,
                    gpu_buffer: data.buffer(),
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
        Gpu2DTensor::from_buffer(out_buffer_store, data.shape())
    }
}
