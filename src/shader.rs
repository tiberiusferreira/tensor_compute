use crate::GpuBox;
use wgpu::ShaderModule;

impl GpuBox {
    pub fn shader(&self, file: &[u8]) -> ShaderModule {
        let cs_module = self
            .device
            .create_shader_module(&wgpu::read_spirv(std::io::Cursor::new(&file[..])).unwrap());
        cs_module
    }
}
