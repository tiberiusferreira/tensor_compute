mod gpu_buffers;
mod gpu_box;
mod shader;
use gpu_box::*;
use std::convert::TryInto;
use wgpu::{Device, Queue};
use zerocopy::{AsBytes, FromBytes};
use crate::gpu_buffers::GpuBuffer;

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

#[derive(Debug)]
pub struct CpuTensor2D {
    data: Vec<u32>,
    shape: (u32, u32),
}

impl CpuTensor2D{
    pub fn new(data: Vec<u32>, shape: (u32, u32)) -> Self{
        Self{
            data,
            shape
        }
    }
}

impl CpuTensor2D{
    pub fn copy_to_gpu(&self, gpu: &GpuBox) -> GpuBuffer{
        GpuBuffer::new(gpu.new_gpu_storage_buffer_with_data(bytemuck::cast_slice(&self.data)), self.shape)
    }
}


