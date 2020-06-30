use crate::gpu_buffers::GpuBuffer;
use crate::tensors::CpuTensor;
use crate::{GpuInstance, GpuStore, Tensor};
mod tensor;
mod external_api;
pub use external_api::*;
mod tensor_view;
pub use tensor_view::*;
use std::fmt::{Debug, Formatter};

pub struct GpuTensor {
    buffer: GpuBuffer,
    shape: Vec<usize>,
    strides: Vec<usize>,
}


#[cfg(test)]
mod tests;

