use crate::gpu_internals::gpu_buffers::GpuBuffer;
use crate::gpu_internals::GpuInstance;
use crate::{CpuTensor, DimStride, GpuStore, GpuTensor, TensorTrait};
use std::collections::VecDeque;
use std::fmt::{Debug, Formatter};

impl Debug for GpuTensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gpu Tensor")
            .field("shape", &self.shape())
            .field("strides", &self.strides())
            .finish()
    }
}

impl GpuTensor {
    // Accessors
    pub fn internal_gpu_buffer(&self) -> &GpuBuffer {
        &self.buffer
    }

    pub fn get_gpu(&self) -> &GpuInstance {
        GpuStore::get(self.buffer.device_info())
    }

    pub fn buffer_size_in_bytes(&self) -> usize {
        self.internal_gpu_buffer().size_bytes()
    }

    pub async fn uninitialized(shape: Vec<usize>) -> GpuTensor{
        Self::new_filled(shape, 0.).await
    }

    pub async fn new_filled(shape: Vec<usize>, fill_val: f32) -> GpuTensor {
        let gpu = GpuStore::get_default();
        let numel: usize = GpuTensor::numel_from_shape(&VecDeque::from(shape.clone()));
        let buffer = gpu.new_empty_gpu_buffer(numel * std::mem::size_of::<f32>());
        let mut tensor = GpuTensor{
            buffer,
            dim_stride: DimStride::from_shape_vec(shape)
        };
        tensor.fill_with(fill_val).await;
        tensor
    }

    pub fn from_data_with_gpu(gpu: &GpuInstance, data: Vec<f32>, shape: Vec<usize>) -> Self {
        let calc_size = shape.iter().rev().fold(1, |acc: usize, &x| acc * x);
        assert_eq!(
            calc_size,
            data.len(),
            "Shape is not valid for the size of the data!"
        );
        CpuTensor::new(data, shape).to_gpu(gpu)
    }

    pub fn from_buffer(buffer: GpuBuffer, shape: VecDeque<usize>) -> Self {
        Self {
            buffer,
            dim_stride: DimStride::from_shape(shape),
        }
    }

    pub fn from_buffer_with_strides(
        buffer: GpuBuffer,
        shape: VecDeque<usize>,
        strides: VecDeque<usize>,
    ) -> Self {
        Self {
            buffer,
            dim_stride: DimStride::from_shape_and_strides(shape, strides),
        }
    }

    pub fn from_data(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let gpu = GpuStore::get_default();
        Self::from_data_with_gpu(gpu, data, shape)
    }

    pub fn from_scalar(data: f32) -> Self {
        let gpu = GpuStore::get_default();
        Self::from_data_with_gpu(gpu, vec![data], vec![1])
    }

    pub async fn to_cpu(&self) -> CpuTensor {
        let gpu = self.get_gpu();
        let buffer_in_cpu_mem = gpu.copy_buffer_to_cpu_mem(self.internal_gpu_buffer()).await;
        CpuTensor::new_with_strides(
            buffer_in_cpu_mem,
            self.shape().clone(),
            self.strides().clone(),
        )
    }

    pub fn dim_strides(&self) -> &DimStride {
        &self.dim_stride
    }

    pub fn is_scalar(&self) -> bool {
        self.dim_stride.is_scalar()
    }
}

impl TensorTrait for GpuTensor {
    fn shape(&self) -> &VecDeque<usize> {
        &self.dim_stride.shape
    }

    fn strides(&self) -> &VecDeque<usize> {
        &self.dim_stride.strides
    }
}
