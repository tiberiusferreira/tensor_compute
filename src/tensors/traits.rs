pub use super::gpu_tensor::traits::*;
use async_trait::async_trait;
use std::collections::VecDeque;

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
        shape.iter().rev().fold(1, |acc: usize, &x| acc * x)
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
