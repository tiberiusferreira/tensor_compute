pub use super::gpu_tensor::traits::*;
use async_trait::async_trait;
use std::collections::VecDeque;
use crate::tensors::gpu_tensor::utils::strides_from_deque_shape;

pub trait ShapeStrideTrait {
    fn shape(&self) -> &VecDeque<usize>;
    fn strides(&self) -> &VecDeque<usize>;
    fn offset(&self) -> usize;
    fn rank(&self) -> usize {
        self.shape().len()
    }
    fn numel(&self) -> usize {
        Self::numel_from_shape(self.shape())
    }
    fn numel_from_shape(shape: &VecDeque<usize>) -> usize {
        if shape.len() == 0{
            return 0;
        }
        shape.iter().rev().fold(1, |acc: usize, &x| acc * x)
    }

    /// In order to be contiguous, each one of the strides need to be either 0 (fake dimension)
    /// or equal to the stride a contiguous tensor of the same shape would have
    fn is_contiguous(&self) -> bool{
        let curr_strides = self.strides();
        let contiguous_strides = strides_from_deque_shape(self.shape());
        for (curr_stride, contiguous_stride) in curr_strides.iter().zip(contiguous_strides.iter()){
            if *curr_stride != 0 && curr_stride != contiguous_stride{
                return false;
            }
        }
        return true;
    }
}

pub trait MutShapeStrideTrait: ShapeStrideTrait{
    /// Artificially increase the rank, setting the new dimension shape to 1
    fn increase_rank(&mut self);
    /// Decrease the rank if the leading shape dimension is 1, otherwise panics
    fn decrease_rank(&mut self);
}

#[async_trait(?Send)]
pub trait Constructable: Sized + ShapeStrideTrait {
    fn from_data(vec: Vec<f32>) -> Self;

    fn from_data_and_shape(vec: Vec<f32>, shape: Vec<usize>);

    async fn zeros(shape: Vec<usize>) -> Self;

    async fn rand(shape: Vec<usize>) -> Self;

    async fn zeros_like(other: &Self) -> Self {
        Self::zeros(Vec::from(other.shape().clone())).await
    }

    async fn clone_async(&self) -> Self;
}
