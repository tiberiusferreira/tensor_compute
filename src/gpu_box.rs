use crate::tensors::Gpu2DTensor;
use crate::{GpuBox};
use zerocopy::AsBytes;
use crate::shader_runner::{ShaderInput, ThreadGroup};

impl GpuBox {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::new();
        let adapter = instance
            .request_adapter(
                &wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::Default,
                    compatible_surface: None,
                },
                wgpu::UnsafeExtensions::disallow(),
                wgpu::BackendBit::PRIMARY,
            )
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    extensions: wgpu::Extensions::empty(),
                    limits: wgpu::Limits::default(),
                    shader_validation: true,
                },
                None,
            )
            .await
            .unwrap();
        Self { device, queue }
    }
}
