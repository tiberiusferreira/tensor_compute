use crate::shader_runner::{ShaderInput, ThreadGroup};
use crate::{Gpu2DTensor, GpuBox};
use zerocopy::{AsBytes, FromBytes};

#[repr(C)]
#[derive(AsBytes, FromBytes, Clone, Debug)]
struct MatricesData {
    rows_a: u32,
    cols_a: u32,
    rows_b: u32,
    cols_b: u32,
}

impl GpuBox {
    pub async fn mm(&self, input_data_a: &Gpu2DTensor, input_data_b: &Gpu2DTensor) -> Gpu2DTensor {
        assert_eq!(
            input_data_a.shape().1,
            input_data_b.shape().0,
            "Matrices shapes dont match for multiplication"
        );
        let matrices_data = MatricesData {
            rows_a: input_data_a.shape().0 as u32,
            cols_a: input_data_a.shape().1 as u32,
            rows_b: input_data_b.shape().0 as u32,
            cols_b: input_data_b.shape().1 as u32,
        };

        let cs_module = self.shader_from_file_bytes(wgpu::include_spirv!("mm.spv"));

        let output_shape = (input_data_a.shape().0, input_data_b.shape().1);
        let nb_output_numbers = output_shape.0 * output_shape.1;
        let out_buffer_store =
            self.empty_gpu_buffer(std::mem::size_of::<f32>() * nb_output_numbers);

        let input_structure_data = self.gpu_buffer_from_data(matrices_data.as_bytes());
        self.run_shader(
            &cs_module,
            vec![
                ShaderInput {
                    binding_id: 0,
                    gpu_buffer: input_data_a.buffer(),
                },
                ShaderInput {
                    binding_id: 1,
                    gpu_buffer: input_data_b.buffer(),
                },
                ShaderInput {
                    binding_id: 2,
                    gpu_buffer: &out_buffer_store,
                },
                ShaderInput {
                    binding_id: 3,
                    gpu_buffer: &input_structure_data,
                },
            ],
            ThreadGroup {
                x: nb_output_numbers,
                y: 1,
                z: 1,
            },
        );
        Gpu2DTensor::from_buffer(out_buffer_store, output_shape)
    }
}
