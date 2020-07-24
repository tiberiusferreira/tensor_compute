use crate::gpu_internals::GpuInstance;
use crate::gpu_internals::gpu_buffers::GpuBuffer;
use crate::{CpuTensor, ShapeStrideTrait};
use async_trait::async_trait;
use crate::gpu_internals::shader_runner::{ShaderInput, BufferType};
use zerocopy::AsBytes;

#[async_trait(?Send)]
pub trait GpuAllocated{
    fn get_gpu(&self) -> &'static GpuInstance;
    fn internal_gpu_buffer(&self) -> &GpuBuffer;
    fn internal_buffer_size_in_bytes(&self) -> usize{
        self.internal_gpu_buffer().size_bytes()
    }
}

#[async_trait(?Send)]
pub trait CpuTransferable{
    async fn to_cpu(&self) -> CpuTensor;
}

#[async_trait(?Send)]
impl <T> CpuTransferable for T where T: GpuAllocated + ShapeStrideTrait {
    async fn to_cpu(&self) -> CpuTensor{
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

pub trait AsShaderInput: GpuAllocated + ShapeStrideTrait{
    fn to_shader_inputs(&self, binding_offset: usize) -> Vec<ShaderInput>{
        let mut shape: Vec<u128> = self.shape().iter().map(|&e| e as u128).collect();
        let mut strides: Vec<u128> = self.strides().iter().map(|&e| e as u128).collect();
        // Uniform Buffer elements need to be 128bits each:
        // see https://www.khronos.org/registry/OpenGL/specs/gl/glspec46.core.pdf page 146 (pdf page 168)
        assert!(shape.len() <= 20, "Shape cant have more than 20 elements");
        assert!(strides.len() <= 20, "Strides cant have more than 20 elements");
        while shape.len() < 20{
            shape.push(0);
        }
        while strides.len() < 20{
            strides.push(0);
        }
        let shape_strides_len = self.shape().len() as u32;
        let offset = self.offset() as u32;
        let shape_as_uniform = self.get_gpu().new_uniform_buffer(shape.as_slice().as_bytes());
        let strides_as_uniform = self.get_gpu().new_uniform_buffer(strides.as_slice().as_bytes());
        let shape_strides_len_as_uniform = self.get_gpu().new_uniform_buffer(shape_strides_len.as_bytes());
        let offset_as_uniform = self.get_gpu().new_uniform_buffer(offset.as_bytes());
        vec![
            ShaderInput{
                binding_id: binding_offset,
                gpu_buffer: BufferType::Storage(self.internal_gpu_buffer()),
            },
            ShaderInput{
                binding_id: binding_offset+1,
                gpu_buffer: BufferType::UniformOwned(shape_as_uniform),
            },
            ShaderInput{
                binding_id: binding_offset+2,
                gpu_buffer: BufferType::UniformOwned(strides_as_uniform),
            },
            ShaderInput{
                binding_id: binding_offset+3,
                gpu_buffer: BufferType::UniformOwned(shape_strides_len_as_uniform),
            },
            ShaderInput{
                binding_id: binding_offset+4,
                gpu_buffer: BufferType::UniformOwned(offset_as_uniform),
            }
        ]
    }
}