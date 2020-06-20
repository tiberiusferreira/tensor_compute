mod gpu_box;
mod gpu_buffers;
mod tensors;
pub use tensors::*;
use wgpu::{Device, Queue};
mod ops;
mod shader_runner;
pub use ops::*;

pub struct GpuBox {
    device: Device,
    queue: Queue,
}
