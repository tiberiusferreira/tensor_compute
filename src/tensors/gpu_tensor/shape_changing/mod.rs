use crate::{GpuTensor, GpuTensorView, ShapeStrides, GpuAllocated};
mod broadcast_tests;
mod slicing_tests;
impl GpuTensor {
    /// Increases Tensor rank artificially by appending adding one dimension to it
    pub fn increase_rank(&self) -> GpuTensorView {
        let mut new_dims = self.shape_strides.clone();
        new_dims.increase_rank();
        GpuTensorView::from_tensor(self, new_dims)
    }

    pub fn view(&self) -> GpuTensorView {
        GpuTensorView::from_tensor(&self, self.dim_strides().clone())
    }

    /// Tensor are broadcastable if:
    /// - Each tensor has at least one dimension.
    /// - When iterating over the dimension sizes, starting at the trailing dimension,
    /// the dimension sizes must either be equal, one of them is 1, or one of them does not exist.
    ///
    /// The broadcasting rules are:
    /// If the number of dimensions of x and y are not equal,
    /// prepend 1 to the dimensions of the tensor with fewer dimensions to make them equal length.
    /// Then, for each dimension size, the resulting dimension size is the max of the sizes of x and y along that dimension.
    pub fn broadcast<'a>(
        &'a self,
        other: &'a Self,
        skipping_dims: Option<usize>,
    ) -> Option<(GpuTensorView<'a>, GpuTensorView<'a>)> {
        let current_shape = self.dim_strides();
        let target_shape = other.dim_strides();
        let (broadcasted_curr, broadcasted_target) =
            broadcast_shape_and_stride(&current_shape, &target_shape, skipping_dims).unwrap();
        Some((
            GpuTensorView::from_tensor(self, broadcasted_curr),
            GpuTensorView::from_tensor(other, broadcasted_target),
        ))
    }

    pub async fn transpose(&self) -> GpuTensor {
        use crate::tensors::gpu_tensor::gpu_ops::transpose;
        transpose(self.get_gpu(), &self).await
    }
}

fn broadcast_shape_and_stride(
    current: &ShapeStrides,
    target: &ShapeStrides,
    skipping_dims: Option<usize>,
) -> Result<(ShapeStrides, ShapeStrides), String> {
    let mut current = current.clone();
    let mut target = target.clone();
    let (larger_rank, smaller_rank) = if current.shape.len() > target.shape.len() {
        (&mut current, &mut target)
    } else {
        (&mut target, &mut current)
    };
    let skipping_dims = skipping_dims.unwrap_or(0);
    assert!(
        skipping_dims <= smaller_rank.rank(),
        "Number of dims to skip is bigger than dims of the tensor itself!"
    );
    // make sure shapes have the same rank by adding 1s to the front of the shorter one
    let rank_diff = larger_rank.shape.len() - smaller_rank.shape.len();
    for _i in 0..rank_diff {
        smaller_rank.increase_rank();
    }
    assert_eq!(smaller_rank.shape.len(), larger_rank.shape.len());

    let final_rank = smaller_rank.rank();
    // Now for each dimension:
    let (curr_shape, curr_stride) = (&mut current.shape, &mut current.strides);
    let (target_shape, target_stride) = (&mut target.shape, &mut target.strides);
    for (id, (current_dim, target_dim)) in curr_shape
        .iter_mut()
        .zip(target_shape.iter_mut())
        .enumerate()
    {
        if final_rank - id <= skipping_dims {
            continue;
        }
        // id is used to modify the strides, but we need to take care of the skipped dims
        // let id = id + skipping_dims;
        // IF they are equal -> do nothing
        if current_dim == target_dim {
            continue;
        }
        // IF one of them is 1 and the other is DIM (DIM!=1) -> change dimension 1 into DIM, and
        // change strides to 0. This is OK since a stride of 0 means we will keep hitting the same
        // memory location when indexing into the expanded "fake" dimension increase
        match (&current_dim, &target_dim) {
            (1, targ) => {
                *current_dim = **targ;
                curr_stride[id] = 0;
            }
            (curr, 1) => {
                *target_dim = **curr;
                target_stride[id] = 0;
            }
            // IF they are different and neither is 1, they are not broadcastable
            _ => {
                return Err(format!(
                    "Cant broadcast between dims {} and {} .",
                    current_dim, target_dim
                ));
            }
        }
    }
    Ok((current, target))
}
