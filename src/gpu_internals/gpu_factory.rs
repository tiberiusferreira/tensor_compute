use crate::gpu_internals::{GpuInfo, GpuInstance};

/// Should be used for querying the available GPUs and instantiating a [GpuInstance] in
/// order to be able to interact with them. Does not need to be kept alive after a [GpuInstance].
/// ** Having multiple GpuInstances referring to the same physical device might have unexpected
/// consequences! **
pub struct GpuFactory {
    adapters: Vec<wgpu::Adapter>,
}

impl GpuFactory {
    pub async fn new() -> Self {
        let wgpu_instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
        let adapters: Vec<wgpu::Adapter> = wgpu_instance
            .enumerate_adapters(wgpu::UnsafeFeatures::disallow(), wgpu::BackendBit::PRIMARY)
            .collect();
        GpuFactory {
            adapters,
        }
    }

    pub async fn list_gpus(&self) -> Vec<GpuInfo> {
        self.adapters
            .iter()
            .map(|adapter| adapter.get_info())
            .collect()
    }

    /// ** Having multiple GpuInstances referring to the same physical device might have unexpected
    /// consequences! **
    pub async fn request_gpu(&self, gpu_info: &GpuInfo) -> GpuInstance {
        let adapter = self
            .adapters
            .iter()
            .find(|adapter| &adapter.get_info() == gpu_info)
            .expect("Adapter does not exist for given GpuInfo");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                    shader_validation: true,
                },
                None,
            )
            .await
            .unwrap();
        GpuInstance {
            device,
            queue,
            info: gpu_info.clone(),
        }
    }
}
