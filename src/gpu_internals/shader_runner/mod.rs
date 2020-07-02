mod shader_runner;
use crate::GpuInstance;
pub use shader_runner::*;
use wgpu::{ShaderModule, ShaderModuleSource};

impl GpuInstance {
    pub fn shader_from_file_bytes(&self, shader_module: ShaderModuleSource) -> ShaderModule {
        let cs_module = self.device().create_shader_module(shader_module);
        cs_module
    }
}
