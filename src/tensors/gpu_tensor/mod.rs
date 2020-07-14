mod external_api;
mod accessors_contructors;
pub use external_api::*;
mod tensor_view;
use crate::gpu_internals::gpu_buffers::GpuBuffer;
pub use tensor_view::*;
use std::collections::VecDeque;

mod gpu_ops;
mod shape_changing;
pub mod utils;

pub struct GpuTensor {
    buffer: GpuBuffer,
    shape: VecDeque<usize>,
    strides: VecDeque<usize>,
}

#[cfg(test)]
mod tests;
