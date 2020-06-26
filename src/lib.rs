mod gpu_buffers;
mod tensors;
pub use tensors::*;
use wgpu::{Device, Queue, AdapterInfo};
mod ops;
mod shader_runner;
pub use ops::*;
use once_cell::sync::{OnceCell, Lazy};

mod gpu_store;
pub use gpu_store::*;



pub struct GpuBox {
    device: Device,
    queue: Queue,
    info: AdapterInfo
}
