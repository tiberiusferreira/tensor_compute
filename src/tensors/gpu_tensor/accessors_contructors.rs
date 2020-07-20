use crate::gpu_internals::gpu_buffers::GpuBuffer;
use crate::gpu_internals::GpuInstance;
use crate::tensors::gpu_tensor::indexing::{shape_strides_for_slice_range, SliceRangeInfo};
use crate::{CpuTensor, GpuStore, GpuTensor, GpuTensorView, ShapeStrides, TensorTrait};
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

    pub fn get_gpu(&self) -> &'static GpuInstance {
        GpuStore::get(self.buffer.device_info())
    }

    pub fn buffer_size_in_bytes(&self) -> usize {
        self.internal_gpu_buffer().size_bytes()
    }

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

    pub fn from_vec(data: Vec<f32>) -> Self {
        let len = data.len();
        let gpu = GpuStore::get_default();
        Self::from_data_with_gpu(gpu, data, vec![len])
    }

    pub fn from_data_and_shape(data: Vec<f32>, shape: Vec<usize>) -> Self {
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
        CpuTensor::new_with_strides_and_offset(
            buffer_in_cpu_mem,
            self.shape().clone(),
            self.strides().clone(),
            self.shape_strides.offset,
        )
    }

    pub fn dim_strides(&self) -> &ShapeStrides {
        &self.shape_strides
    }

    pub fn is_scalar(&self) -> bool {
        self.shape_strides.is_scalar()
    }

    pub fn i<T: Into<SliceRangeInfo>>(&self, bounds: Vec<T>) -> GpuTensorView {
        let bounds: Vec<SliceRangeInfo> = bounds.into_iter().map(|e| e.into()).collect();
        let new_shape_strides = shape_strides_for_slice_range(&self.shape_strides, bounds);
        GpuTensorView::new(self, new_shape_strides)
    }

    pub async fn assign<T: Into<SliceRangeInfo>>(&mut self, bounds: Vec<T>, value: f32){
        let bounds: Vec<SliceRangeInfo> = bounds.into_iter().map(|e| e.into()).collect();
        let new_shape_strides = shape_strides_for_slice_range(&self.shape_strides, bounds);
        let mut to_be_changed = GpuTensorView::new(self, new_shape_strides);
        to_be_changed.assign_kernel(value).await;
    }
}




// #[test]
// fn test_bounds() {
//     let block = async {
//         let a = GpuTensor::uninitialized(vec![3, 3]).await;
//         a.index(s!(2; (2,2,2); 3));
//         a.index(s!(2; (2,2,2)));
//     };
//     futures::executor::block_on(block);
// }

impl TensorTrait for GpuTensor {
    fn shape(&self) -> &VecDeque<usize> {
        &self.shape_strides.shape
    }

    fn strides(&self) -> &VecDeque<usize> {
        &self.shape_strides.strides
    }
}
