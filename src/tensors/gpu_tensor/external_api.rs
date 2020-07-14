use crate::{GpuTensor, Tensor};
use std::collections::VecDeque;

impl GpuTensor {

    //
    // pub fn expand(&mut self, shape: Vec<usize>) {
    //     assert!(
    //         shape.len() >= self.shape().len(),
    //         "Target shape needs to have the same number of \
    //     dimensions of current shape or more."
    //     );
    //     for (index, dim) in self.shape().iter().enumerate() {
    //         if self.shape()[index] == 1 {
    //             self.shape[index] = *dim;
    //             self.strides[index] = 0;
    //         } else if (*dim) != self.shape()[index] {
    //             panic!(
    //                 "Cant expand original non-unitary dimension {} to {}",
    //                 self.shape()[index],
    //                 *dim
    //             );
    //         }
    //     }
    // }

    // TODO! We need to check if we have 0 strides, if we do, we cant avoid copying
    pub fn reshape(&mut self, shape: Vec<usize>) {
        for stride in &self.strides {
            if *stride == 0 {
                panic!("Cant reshape tensor with stride 0");
            }
        }
        let calc_size = shape.iter().fold(0, |acc: usize, &x| acc + x);
        assert_eq!(
            calc_size,
            self.numel(),
            "Shape is not valid for the size of the data!"
        );
        self.shape = VecDeque::from(shape);
    }
}

