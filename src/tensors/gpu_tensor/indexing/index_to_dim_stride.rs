use crate::tensors::gpu_tensor::indexing::SliceRangeInfo;
use crate::{s, ShapeStrides};

pub fn shape_strides_for_slice_range<T: Into<SliceRangeInfo>>(
    original: &ShapeStrides,
    bounds: Vec<T>,
) -> ShapeStrides {
    let bounds: Vec<SliceRangeInfo> = bounds.into_iter().map(|e| e.into()).collect();
    let mut new_dims = original.clone();
    let new_shape = &mut new_dims.shape;
    let new_strides = &mut new_dims.strides;
    let offset = &mut new_dims.offset;
    assert!(
        bounds.len() <= original.rank(),
        "Tried to index non existing dimension"
    );
    for (slice_range, (strides, shape)) in bounds
        .iter()
        .zip(new_strides.iter_mut().zip(new_shape.iter_mut()))
    {
        let start = slice_range.start;
        assert!(
            start <= *shape - 1,
            format!(
                "Indexing out of range! Tried to get the element \
                        {:?} (zero indexed) of a dimension of size: {:?}",
                start, *shape
            )
        );
        if start != 0 {
            *offset += *strides * start;
        }
        *strides = (*strides) * slice_range.step;
        let step = slice_range.step as f32;
        let numel;
        if let Some(inclusive_end) = slice_range.inclusive_end {
            assert!(
                inclusive_end <= *shape - 1,
                format!(
                    "Indexing out of range! Tried to get the element \
                        {:?} (zero indexed) of a dimension of size: {:?}",
                    inclusive_end, *shape
                )
            );
            // (end - slice_range.start + 1) = number of elements between start and end
            // for example: [0 1 2] and start = 0, end = 2 => 2 - 0 + 1 = 3 = number of elements
            // and: [3 4 5 6] and start = 0, end = 3 => 3 - 0 + 1 = 4 = number of elements
            numel = (inclusive_end - slice_range.start + 1) as f32;
        } else {
            // If there is no end, the number of elements is the same as before minus the number
            // of elements we skip because of the custom start
            numel = (*shape - slice_range.start) as f32;
        }
        *shape = (numel as f32 / step).ceil() as usize;
    }
    new_dims
}

#[test]
fn can_index() {
    let dim_stride = ShapeStrides::from_shape_vec(vec![4]);
    let slices_info = s![0..4];
    let shape_strides = shape_strides_for_slice_range(&dim_stride, slices_info);
    assert_eq!(shape_strides.shape, [4]);
    assert_eq!(shape_strides.strides, [1]);
    assert_eq!(shape_strides.offset, 0);

    let slices_info = s![1..4];
    let shape_strides = shape_strides_for_slice_range(&dim_stride, slices_info);
    assert_eq!(shape_strides.shape, [3]);
    assert_eq!(shape_strides.strides, [1]);
    assert_eq!(shape_strides.offset, 1);
}

#[test]
fn can_index_edge_cases() {
    // index whole dimension
    let dim_stride = ShapeStrides::from_shape_vec(vec![4]);
    let slices_info = s![0..4];
    let shape_strides = shape_strides_for_slice_range(&dim_stride, slices_info);
    assert_eq!(shape_strides.shape, [4]);
    assert_eq!(shape_strides.strides, [1]);
    assert_eq!(shape_strides.offset, 0);

    // index part of a dimension
    let dim_stride = ShapeStrides::from_shape_vec(vec![4]);
    let slices_info = s![1..3];
    let shape_strides = shape_strides_for_slice_range(&dim_stride, slices_info);
    assert_eq!(shape_strides.shape, [2]);
    assert_eq!(shape_strides.strides, [1]);
    assert_eq!(shape_strides.offset, 1);

    // index last dimension
    let dim_stride = ShapeStrides::from_shape_vec(vec![4]);
    let slices_info = s![3..=3];
    let shape_strides = shape_strides_for_slice_range(&dim_stride, slices_info);
    assert_eq!(shape_strides.shape, [1]);
    assert_eq!(shape_strides.strides, [1]);
    assert_eq!(shape_strides.offset, 3);

    // index part of the dimensions
    let dim_stride = ShapeStrides::from_shape_vec(vec![4, 4, 4]);
    let slices_info = s![..; 1..4];
    let shape_strides = shape_strides_for_slice_range(&dim_stride, slices_info);
    assert_eq!(shape_strides.shape, [4, 3, 4]);
    assert_eq!(shape_strides.strides, [16, 4, 1]);
    assert_eq!(shape_strides.offset, 4);

    // index part of the dimensions
    let dim_stride = ShapeStrides::from_shape_vec(vec![4, 4, 4]);
    let slices_info = s![1..2; 1..4];
    let shape_strides = shape_strides_for_slice_range(&dim_stride, slices_info);
    assert_eq!(shape_strides.shape, [1, 3, 4]);
    assert_eq!(shape_strides.strides, [16, 4, 1]);
    assert_eq!(shape_strides.offset, 20);
}

#[test]
#[ignore]
#[should_panic]
fn doesnt_allow_end_smaller_than_start() {
    let slices_info = s![0..0];
    println!("{:?}", slices_info); // make sure compiler doesnt remove instructions
}

#[test]
fn allows_start_equals_end() {
    let dim_stride = ShapeStrides::from_shape_vec(vec![4]);
    let slices_info = s![0..=0];
    let shape_stride = shape_strides_for_slice_range(&dim_stride, slices_info);
    assert_eq!(shape_stride.shape, [1]);
    assert_eq!(shape_stride.strides, [1]);
    assert_eq!(shape_stride.offset, 0);
}

#[test]
#[ignore]
#[should_panic]
fn cant_index_more_elements_than_dimension_has() {
    let dim_stride = ShapeStrides::from_shape_vec(vec![4]);
    shape_strides_for_slice_range(&dim_stride, s![0..5]);
}
