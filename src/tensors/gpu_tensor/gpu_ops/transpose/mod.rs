use crate::{GpuTensor, ShapeStrideTrait, GpuAllocated, AsShaderInput};
use crate::gpu_internals::shader_runner::ThreadGroup;

#[cfg(test)]
mod tests;

impl GpuTensor{
    pub async fn transpose(&self) -> GpuTensor {
        if self.is_empty(){
            return self.clone().await;
        }
        let cs_module = self.gpu().shader_from_file_bytes(wgpu::include_spirv!("transpose.spv"));
        let out_buffer = self.gpu().empty_like(self.buffer());
        let shader_inputs = self.to_shader_inputs().with_buffer(&out_buffer);

        self.gpu().run_shader(
            &cs_module,
            &shader_inputs,
            ThreadGroup {
                x: self.numel(),
                y: 1,
                z: 1,
            },
        );
        let mut shape = self.shape().clone();
        shape.swap(shape.len() - 2, shape.len() - 1);
        GpuTensor::from_buffer(out_buffer, shape.clone())
    }

}