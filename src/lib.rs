mod gpu_box;
mod gpu_buffers;
mod tensors;
pub use tensors::*;
use wgpu::{Device, Queue};
mod shader_runner;
mod ops;
pub use ops::*;
use zerocopy::{AsBytes, FromBytes};

pub struct GpuBox {
    device: Device,
    queue: Queue,
}


