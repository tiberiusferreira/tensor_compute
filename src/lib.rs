mod gpu_box;
mod gpu_buffers;
mod shader;
use gpu_box::*;
mod tensors;
use std::convert::TryInto;
pub use tensors::*;
use wgpu::{Device, Queue};
use zerocopy::{AsBytes, FromBytes};

pub struct GpuBox {
    device: Device,
    queue: Queue,
}

#[repr(C)]
#[derive(AsBytes, FromBytes, Clone, Debug)]
struct MatricesData {
    rows_a: u32,
    cols_a: u32,
    rows_b: u32,
    cols_b: u32,
}
