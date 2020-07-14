use std::collections::VecDeque;

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


pub fn strides_from_deque_shape(shape: &VecDeque<usize>) -> VecDeque<usize> {
    let mut strides = VecDeque::new();
    strides.push_back(1);
    for dim in shape.iter().skip(1).rev() {
        let biggest_stride = strides[0];
        strides.push_front(dim * biggest_stride)
    }
    strides
}

#[test]
pub fn can_calc_strides_from_shape(){
    assert_eq!(strides_from_deque_shape(&VecDeque::from(vec![2, 2])), [2, 1]);
    assert_eq!(strides_from_deque_shape(&VecDeque::from(vec![1, 2])), [2, 1]);
    assert_eq!(strides_from_deque_shape(&VecDeque::from(vec![3, 1, 2])), [2, 2, 1]);
    assert_eq!(strides_from_deque_shape(&VecDeque::from(vec![1, 1])), [1, 1]);
    assert_eq!(strides_from_deque_shape(&VecDeque::from(vec![4, 3, 2, 2])), [12, 4, 2, 1]);
}

