use crate::{GpuTensor, ShapeStrides, MutShapeStrideTrait, ShapeStrideTrait};
#[cfg(test)]
mod broadcast_tests;
#[cfg(test)]
mod slicing_tests;

impl GpuTensor {
    pub fn reshape(&mut self, shape: Vec<usize>) {
        for stride in self.strides() {
            if *stride == 0 {
                panic!("Cant reshape tensor with stride 0");
            }
        }
        let shape = std::collections::VecDeque::from(shape);
        let numel = GpuTensor::numel_from_shape(&shape);
        assert_eq!(
            numel,
            self.numel(),
            "Shape is not valid for the size of the data!"
        );
        self.shape_strides = ShapeStrides::from_shape(shape);
    }

}
