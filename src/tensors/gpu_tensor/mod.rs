mod external_api;
mod tensor;
pub use external_api::*;
mod tensor_view;
use crate::gpu_internals::gpu_buffers::GpuBuffer;
pub use tensor_view::*;
mod gpu_ops;

pub struct GpuTensor {
    buffer: GpuBuffer,
    shape: Vec<usize>,
    strides: Vec<usize>,
}

#[cfg(test)]
mod tests;
