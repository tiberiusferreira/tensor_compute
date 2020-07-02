use crate::gpu_buffers::GpuBuffer;
use crate::{CpuTensor, GpuInstance, GpuStore, GpuTensor, GpuTensorView, Tensor};
use std::convert::TryInto;
use std::fmt::{Debug, Formatter};

impl Debug for GpuTensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Gpu Tensor")
            .field("shape", &self.shape())
            .field("strides", &self.strides())
            .finish()
    }
}

// Internal use impls
impl GpuTensor {
    pub fn storage(&self) -> &GpuBuffer {
        &self.buffer
    }

    pub fn buffer_size_in_bytes(&self) -> usize {
        self.storage().size_bytes()
    }

    pub fn broadcast<'a>(
        &'a self,
        other: &'a Self,
    ) -> Option<(GpuTensorView<'a>, GpuTensorView<'a>)> {
        let current_shape = self.shape();
        let target_shape = other.shape();
        let mut expanded_current_shape = vec![];
        let mut expanded_target_shape = vec![];
        let diff = target_shape.len() as i32 - current_shape.len() as i32;
        if diff > 0 {
            for _i in 0..diff {
                expanded_current_shape.push(1);
            }
            expanded_current_shape.extend_from_slice(current_shape.as_slice());
            expanded_target_shape = target_shape.to_vec();
        } else {
            for _i in 0..-diff {
                expanded_target_shape.push(1);
            }
            expanded_target_shape.extend_from_slice(target_shape.as_slice());
            expanded_current_shape = current_shape.to_vec();
        }
        let mut final_curr_strides = GpuTensor::strides_from_shape(&expanded_current_shape);
        let mut final_target_strides = GpuTensor::strides_from_shape(&expanded_target_shape);
        let mut final_current_dims = vec![];
        let mut final_target_dims = vec![];
        for (i, (&curr_dim, &target_dim)) in expanded_current_shape
            .iter()
            .zip(expanded_target_shape.iter())
            .enumerate()
        {
            if curr_dim == target_dim {
                final_current_dims.push(curr_dim);
                final_target_dims.push(target_dim);
            } else if curr_dim == 1 {
                final_curr_strides[i] = 0;
                final_current_dims.push(target_dim);
                final_target_dims.push(target_dim);
            } else if target_dim == 1 {
                final_target_strides[i] = 0;
                final_current_dims.push(curr_dim);
                final_target_dims.push(curr_dim);
            } else {
                panic!(
                    "Cant broadcast between dims {} and {} .",
                    curr_dim, target_dim
                );
            }
        }
        Some((
            GpuTensorView::new(self, final_current_dims, final_curr_strides),
            GpuTensorView::new(other, final_target_dims, final_target_strides),
        ))
    }

    pub(super) fn from_data_with_gpu(gpu: &GpuInstance, data: Vec<f32>, shape: Vec<usize>) -> Self {
        let calc_size = shape.iter().rev().fold(1, |acc: usize, &x| acc * x);
        assert_eq!(
            calc_size,
            data.len(),
            "Shape is not valid for the size of the data!"
        );
        CpuTensor::new(data, shape).to_gpu(gpu)
    }

    pub fn from_buffer(buffer: GpuBuffer, shape: Vec<usize>) -> Self {
        Self {
            buffer,
            shape: shape.clone(),
            strides: Self::strides_from_shape(shape.as_slice()),
        }
    }

    pub fn strides_from_shape(shape: &[usize]) -> Vec<usize> {
        let mut strides = vec![];
        strides.push(1);
        for dim in shape.iter().skip(1).rev() {
            let biggest_stride = strides.last().unwrap().clone();
            strides.push(dim * biggest_stride)
        }
        strides.reverse();
        strides
    }

    pub async fn to_cpu_with_gpu(&self, gpu: &GpuInstance) -> CpuTensor {
        let buffer_in_cpu_mem = gpu.copy_to_cpu_mem(self.storage()).await;
        CpuTensor::new(buffer_in_cpu_mem, self.shape.clone())
    }
}
