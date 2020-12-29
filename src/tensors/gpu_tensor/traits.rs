use crate::gpu_internals::gpu_buffers::GpuBuffer;
use crate::gpu_internals::shader_runner::{BufferType, ShaderBinding, ShaderInputs, PushConstants};
use crate::gpu_internals::GpuInstance;
use crate::{CpuTensor, ShapeStrideTrait, GpuTensor};
use async_trait::async_trait;
use zerocopy::AsBytes;
use std::process::exit;

#[async_trait(?Send)]
pub trait GpuAllocated {
    fn get_gpu(&self) -> &'static GpuInstance;
    fn internal_gpu_buffer(&self) -> &GpuBuffer;
    fn buffer_size_in_bytes(&self) -> usize {
        self.internal_gpu_buffer().size_bytes()
    }
}

impl<'a> AsShaderInput for GpuTensor {}


#[async_trait(?Send)]
pub trait CpuTransferable {
    async fn to_cpu(&self) -> CpuTensor;
}

#[async_trait(?Send)]
impl<T> CpuTransferable for T
where
    T: GpuAllocated + ShapeStrideTrait,
{
    async fn to_cpu(&self) -> CpuTensor {
        let gpu = self.get_gpu();
        let buffer_in_cpu_mem = gpu.copy_buffer_to_cpu_mem(self.internal_gpu_buffer()).await;
        CpuTensor::new_with_strides_and_offset(
            buffer_in_cpu_mem,
            self.shape().clone(),
            self.strides().clone(),
            self.offset(),
        )
    }
}

pub trait AsShaderInput: GpuAllocated + ShapeStrideTrait {
    fn to_shader_inputs<'a>(&'a self, extend_from: Option<ShaderInputs<'a>>) -> ShaderInputs<'a> {
        let ShaderInputs{
            mut bindings,
            mut push_constants
        } = extend_from.unwrap_or_default();

        let mut shape: Vec<u32> = self.shape().iter().map(|&e| e as u32).collect();
        let mut strides: Vec<u32> = self.strides().iter().map(|&e| e as u32).collect();
        while shape.len() < 8{
            shape.push(0);
        }
        while strides.len() < 8{
            strides.push(0);
        }
        let shape_strides_len = self.shape().len() as u32;
        push_constants.data.push(shape_strides_len);
        push_constants.data.extend_from_slice(shape.as_slice());
        push_constants.data.extend_from_slice(strides.as_slice());
        // let offset = self.offset() as u32;
        // let offset_as_buffer = self.get_gpu().new_gpu_buffer_from_data(offset.as_bytes());

        bindings.push(
            ShaderBinding {
                binding_id: bindings.len() as u32,
                gpu_buffer_type: BufferType::Storage(self.internal_gpu_buffer()),
            });
        ShaderInputs{
            bindings,
            push_constants
        }
    }
}
