use crate::gpu_internals::gpu_buffers::GpuBuffer;
use crate::gpu_internals::GpuInstance;
use crate::tensors::gpu_tensor::indexing::{shape_strides_for_slice_range, SliceRangeInfo};
use crate::tensors::gpu_tensor::utils::strides_from_deque_shape;
use crate::{
    CpuTransferable, GpuAllocated, GpuStore, GpuTensor, GpuTensorView, GpuTensorViewMut,
    ShapeStrideTrait, ShapeStrides,
};
use async_trait::async_trait;
use std::collections::VecDeque;
use std::fmt::{Debug, Display, Formatter};
impl Debug for GpuTensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let k = blocking::block_on(self.to_cpu());
        Display::fmt(&k, f)
    }
}

#[async_trait(?Send)]
impl GpuAllocated for GpuTensor {
    fn get_gpu(&self) -> &'static GpuInstance {
        GpuStore::get(self.buffer.device_info())
    }

    fn internal_gpu_buffer(&self) -> &GpuBuffer {
        &self.buffer
    }
}

impl ShapeStrideTrait for GpuTensor {
    fn shape(&self) -> &VecDeque<usize> {
        &self.shape_strides.shape
    }

    fn strides(&self) -> &VecDeque<usize> {
        &self.shape_strides.strides
    }

    fn offset(&self) -> usize {
        self.shape_strides.offset
    }
}

impl GpuTensor {
    pub async fn uninitialized(shape: Vec<usize>) -> GpuTensor {
        Self::new_filled(shape, 0.).await
    }

    pub async fn new_filled(shape: Vec<usize>, fill_val: f32) -> GpuTensor {
        let gpu = GpuStore::get_default();
        let numel: usize = GpuTensor::numel_from_shape(&VecDeque::from(shape.clone()));
        let buffer = gpu.new_empty_gpu_buffer(numel * std::mem::size_of::<f32>());
        let mut tensor = GpuTensor {
            buffer,
            shape_strides: ShapeStrides::from_shape_vec(shape),
        };
        tensor.fill_with(fill_val).await;
        tensor
    }

    pub fn from_data_with_gpu(gpu: &GpuInstance, data: Vec<f32>, shape: Vec<usize>) -> Self {
        let shape = &VecDeque::from(shape);
        let calc_size = GpuTensor::numel_from_shape(&shape);
        assert_eq!(
            calc_size,
            data.len(),
            "Shape is not valid for the size of the data!"
        );
        let strides = strides_from_deque_shape(&shape);
        GpuTensor::from_buffer_with_strides_and_offset(
            gpu.new_gpu_buffer_from_data(bytemuck::cast_slice(&data)),
            shape.clone(),
            strides.clone(),
            0,
        )
    }

    pub fn from_buffer(buffer: GpuBuffer, shape: VecDeque<usize>) -> Self {
        Self {
            buffer,
            shape_strides: ShapeStrides::from_shape(shape),
        }
    }

    pub fn from_buffer_with_strides_and_offset(
        buffer: GpuBuffer,
        shape: VecDeque<usize>,
        strides: VecDeque<usize>,
        offset: usize,
    ) -> Self {
        Self {
            buffer,
            shape_strides: ShapeStrides::from_shape_and_strides_and_offset(shape, strides, offset),
        }
    }

    pub fn from(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let gpu = GpuStore::get_default();
        Self::from_data_with_gpu(gpu, data, shape)
    }

    pub fn from_data_1d(data: Vec<f32>) -> Self {
        let len = data.len();
        assert!(len >= 1, "Can't create 1D Tensor with empty data!");
        let gpu = GpuStore::get_default();
        Self::from_data_with_gpu(gpu, data, vec![len])
    }

    pub fn from_scalar(data: f32) -> Self {
        let gpu = GpuStore::get_default();
        Self::from_data_with_gpu(gpu, vec![data], vec![1])
    }

    pub fn dim_strides(&self) -> &ShapeStrides {
        &self.shape_strides
    }

    pub fn is_scalar(&self) -> bool {
        self.shape_strides.is_scalar()
    }

    pub fn is_empty(&self) -> bool {
        self.shape_strides.shape.len() == 0
    }

    pub fn slice<T: Into<SliceRangeInfo>>(&self, bounds: Vec<T>) -> GpuTensorView {
        let bounds: Vec<SliceRangeInfo> = bounds.into_iter().map(|e| e.into()).collect();
        let new_shape_strides = shape_strides_for_slice_range(&self.shape_strides, bounds);
        GpuTensorView::from_tensor(self, new_shape_strides)
    }

    pub async fn assign<T: Into<SliceRangeInfo>>(&mut self, bounds: Vec<T>, value: f32) {
        let bounds: Vec<SliceRangeInfo> = bounds.into_iter().map(|e| e.into()).collect();
        let new_shape_strides = shape_strides_for_slice_range(&self.shape_strides, bounds);
        let mut to_be_changed = GpuTensorViewMut::from_tensor(self, new_shape_strides);
        to_be_changed.assign_kernel(value).await;
    }
}
