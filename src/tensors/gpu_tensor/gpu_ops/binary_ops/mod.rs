use crate::gpu_internals::GpuInstance;
use crate::{GpuTensor, ShapeStrideTrait, GpuAllocated, AsShaderInput};
use crate::gpu_internals::shader_runner::{ShaderBinding, ThreadGroup, BufferType};
#[cfg(test)]
mod tests;

macro_rules! bin_element_wise_op {
    ($operation_name:literal, $fun_name:ident, $shader_path:literal) => {
        impl GpuTensor{
            pub async fn $fun_name(&self, right_tensor: &GpuTensor) -> GpuTensor {
                assert_eq!(self.shape(), right_tensor.shape(), "Can't {} tensors with incompatible shapes", $operation_name);
                let cs_module = self.gpu().shader_from_file_bytes(wgpu::include_spirv!($shader_path));
                let nb_output_numbers = self.numel();
                let output_buffer = self.gpu().empty_like(self.buffer());
                let mut shader_inputs = self.to_shader_inputs()
                    .with_tensor(right_tensor)
                    .with_buffer(&output_buffer);
                self.gpu().run_shader(
                    &cs_module,
                    &shader_inputs,
                    ThreadGroup {
                        x: nb_output_numbers,
                        y: 1,
                        z: 1,
                    },
                );
                GpuTensor::from_buffer(output_buffer, self.shape_strides.shape.clone())
            }
        }
    }
}

bin_element_wise_op!("add", add, "add.spv");
bin_element_wise_op!("sub", sub, "sub.spv");
bin_element_wise_op!("dot_div", dot_div, "dot_div.spv");
bin_element_wise_op!("dot_mul", dot_mul, "dot_mul.spv");



macro_rules! bin_element_wise_scalar_op {
    ($operation_name:literal, $fun_name:ident, $shader_path:literal) => {
        impl GpuTensor{
            pub async fn $fun_name(&self, scalar: f32) -> GpuTensor {
                let cs_module = self.gpu().shader_from_file_bytes(wgpu::include_spirv!($shader_path));
                let nb_output_numbers = self.numel();
                let output_buffer = self.gpu().empty_like(self.buffer());
                let mut shader_inputs = self.to_shader_inputs()
                    .with_buffer(&output_buffer);
                shader_inputs.push_constants.data.push(u32::from_ne_bytes(scalar.to_ne_bytes()));
                self.gpu().run_shader(
                    &cs_module,
                    &shader_inputs,
                    ThreadGroup {
                        x: nb_output_numbers,
                        y: 1,
                        z: 1,
                    },
                );
                GpuTensor::from_buffer(output_buffer, self.shape_strides.shape.clone())
            }
        }
    }
}

bin_element_wise_scalar_op!("add_scalar", add_scalar, "add_scalar.spv");
bin_element_wise_scalar_op!("sub_scalar", sub_scalar, "sub_scalar.spv");
bin_element_wise_scalar_op!("mul_scalar", mul_scalar, "mul_scalar.spv");
bin_element_wise_scalar_op!("div_scalar", div_scalar, "div_scalar.spv");
bin_element_wise_scalar_op!("pow_scalar", pow_scalar, "pow_scalar.spv");