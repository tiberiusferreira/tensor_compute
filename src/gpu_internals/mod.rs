use wgpu::{AdapterInfo, Device, Queue};

pub mod gpu_buffers;
pub mod shader_runner;
mod gpu_factory;

pub type GpuInfo = AdapterInfo;

/// Represents a physical GPU, main entry to the API.
/// Can instantiated by a [GpuFactory].
pub struct GpuInstance {
    device: Device,
    queue: Queue,
    info: AdapterInfo,
}


impl GpuInstance {
     fn device(&self) -> &Device {
        &self.device
    }

    fn queue(&self) -> &Queue {
        &self.queue
    }

    pub fn info(&self) -> &AdapterInfo {
        &self.info
    }
}
