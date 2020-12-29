use crate::autograd::Tensor;
use crate::RawTensor;

pub fn set_matmul_grad(left: &Tensor, right: &Tensor, child_grad: &RawTensor){
    let right_t = right.read_lock().tensor.transpose();
    let new_left_grad = child_grad.matmul(&right_t);
    let new_grad = if let Some(existing) = &left.write_lock().grad{
        existing.add(&new_left_grad)
    }else{
        new_left_grad
    };
    left.write_lock().grad = Some(new_grad);


    let left_t = left.read_lock().tensor.transpose();
    let new_right_grad = left_t.matmul(&child_grad);
    let new_grad = if let Some(existing) = &right.write_lock().grad{
        existing.add(&new_right_grad)
    }else{
        new_right_grad
    };
    right.write_lock().grad = Some(new_grad);
}