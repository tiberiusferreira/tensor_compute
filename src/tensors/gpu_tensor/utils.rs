use std::collections::VecDeque;

pub fn strides_from_deque_shape(shape: &VecDeque<usize>) -> VecDeque<usize> {
    // If shape is empty, so should be the strides
    if shape.is_empty(){
        return VecDeque::new();
    }
    let mut strides = VecDeque::new();
    strides.push_back(1);
    for dim in shape.iter().skip(1).rev() {
        let biggest_stride = strides[0];
        strides.push_front(dim * biggest_stride)
    }
    strides
}

#[test]
pub fn can_calc_strides_from_shape() {
    assert_eq!(
        strides_from_deque_shape(&VecDeque::new()),
        []
    );
    assert_eq!(
        strides_from_deque_shape(&VecDeque::from(vec![2, 2])),
        [2, 1]
    );
    assert_eq!(
        strides_from_deque_shape(&VecDeque::from(vec![1, 2])),
        [2, 1]
    );
    assert_eq!(
        strides_from_deque_shape(&VecDeque::from(vec![3, 1, 2])),
        [2, 2, 1]
    );
    assert_eq!(
        strides_from_deque_shape(&VecDeque::from(vec![1, 1])),
        [1, 1]
    );
    assert_eq!(
        strides_from_deque_shape(&VecDeque::from(vec![4, 3, 2, 2])),
        [12, 4, 2, 1]
    );
}
