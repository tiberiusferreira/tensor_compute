
use super::RawTensor;
use std::sync::{RwLock, Arc, RwLockReadGuard, RwLockWriteGuard};
use crate::autograd::ops::{set_matmul_grad, set_exp_grad, set_sum_grad};

mod ops;
type Shared<T> = Arc<RwLock<T>>;

#[derive(Debug)]
enum Op{
    MatMul(Tensor, Tensor),
    Exp(Tensor),
    Sum(Tensor),
}

impl Op{
    pub fn propagate_grad(&self, child_grad: &RawTensor){
        match self{
            Op::MatMul(left, right) => {
                set_matmul_grad(left, right, child_grad);
                left.backward();
                right.backward();
            }
            Op::Exp(input) => {
                set_exp_grad(input, child_grad);
                input.backward();
            }
            Op::Sum(input) => {
                set_sum_grad(input, child_grad);
                input.backward();
            }
        }
    }
}

#[derive(Debug)]
pub struct Tensor {
    inner: Shared<VariableData>
}

#[derive(Debug)]
pub struct VariableData{
    parent_op: Option<Op>,
    tensor: RawTensor,
    grad: Option<RawTensor>
}

impl Tensor {
    /// Makes a shallow clone of the Tensor, creating a new reference to the underlying data
    /// without copying it
    pub fn shallow_clone(&self) -> Self{
        Self{
            inner: self.inner.clone()
        }
    }

    pub fn to_vec(&self) -> Vec<f32>{
        self.inner.read().expect("Error acquiring read lock").tensor.to_vec()
    }

    /// Creates a read lock on it, panicking if that was not successful
    pub fn read_lock(&self) -> RwLockReadGuard<VariableData>{
        self.inner.read().expect("Error acquiring read lock")
    }

    /// Creates a write lock on it, panicking if that was not successful
    pub fn write_lock(&self) -> RwLockWriteGuard<VariableData>{
        self.inner.write().expect("Error acquiring write lock")
    }

    // pub fn shape(&self) -> &[usize]{
    //
    // }

    /// Manually sets the gradient of this Tensor.
    /// This function deep clones the input tensor
    pub fn set_grad(&mut self, grad: Tensor){
        let tensor = grad.write_lock().tensor.clone();
        self.write_lock().grad = Some(tensor);
    }

    /// Creates a new Tensor from the provided data and shape
    pub fn from_data_and_shape(data: Vec<f32>, shape: Vec<usize>) -> Self{
        Self{
            inner: Arc::new(RwLock::new(VariableData{
                parent_op: None,
                tensor: RawTensor::from_data_and_shape(data, shape),
                grad: None
            }))
        }
    }

    /// Simple Batch Matrix Multiplication. Requires both inputs to have rank 3 and same batch size
    pub fn matmul(&self, other_var: &Tensor) -> Self{
        let inner = self.read_lock();
        let self_tensor = &inner.tensor;
        let other = other_var.read_lock();
        let other_tensor = &other.tensor;
        let dot_mul_res = self_tensor.matmul(&other_tensor);
        Tensor {
            inner: Arc::new(RwLock::new(VariableData{
                parent_op: Some(Op::MatMul(self.shallow_clone(), other_var.shallow_clone())),
                tensor: dot_mul_res,
                grad: None
            }))
        }
    }

    pub fn exp(&self) -> Self{
        let inner = self.read_lock();
        let res = inner.tensor.exp();
        Tensor {
            inner: Arc::new(RwLock::new(VariableData{
                parent_op: Some(Op::Exp(self.shallow_clone())),
                tensor: res,
                grad: None
            }))
        }
    }

    pub fn sum(&self) -> Self{
        let inner = self.read_lock();
        let res = inner.tensor.sum();
        Tensor {
            inner: Arc::new(RwLock::new(VariableData{
                parent_op: Some(Op::Sum(self.shallow_clone())),
                tensor: res,
                grad: None
            }))
        }
    }

    /// Back propagates the gradients from itself into parent Tensors
    pub fn backward(&self){
        let default_grad = RawTensor::from_data_and_shape(vec![1.], vec![1]);
        let read_guard = self.read_lock();
        let self_grad = if self.read_lock().tensor.numel() == 1 && self.read_lock().grad.is_none(){
            &default_grad
        }else{
            read_guard.grad.as_ref().expect("Can't call backwards without grad")
        };
        if let Some(parent_op) = &read_guard.parent_op{
            parent_op.propagate_grad(&self_grad)
        }
    }
}

#[test]
fn matmul_grad_works(){
    let left = Tensor::from_data_and_shape(vec![1., 2., 3., 4.], vec![1, 2, 2]);
    let right = Tensor::from_data_and_shape(vec![1., 2., 3., 4.], vec![1, 2, 2]);

    let mut c = left.matmul(&right);
    let c_grad = Tensor::from_data_and_shape(vec![1., 1., 1., 1.], vec![1, 2, 2]);
    c.set_grad(c_grad);
    c.backward();

    assert_eq!(right.read_lock().grad.as_ref().unwrap().to_vec(), &[4., 4., 6., 6.]);
}

#[test]
fn exp_grad_works(){
    let left = Tensor::from_data_and_shape(vec![1., 2., 3., 4.], vec![1, 2, 2]);
    let sum = left.exp().sum();
    sum.backward();
    assert_eq!(left.read_lock().grad.as_ref().unwrap().to_vec(), &[1., 2., 3., 4.]);
}

// #[test]
// fn softmax_grad_works(){
//     let left = Tensor::from_data_and_shape(vec![1., 2., 3., 4.], vec![1, 1, 4]);
//     let exp = left.exp();
//     let exp_sum = exp.sum();
//     sum.backward();
//     assert_eq!(left.read_lock().grad.as_ref().unwrap().to_vec(), &[1., 2., 3., 4.]);
// }


#[test]
fn sum_grad_works(){
    let left = Tensor::from_data_and_shape(vec![1., 2., 3., 4.], vec![1, 2, 2]);
    let mut c = left.sum();
    c.backward();
    assert_eq!(left.read_lock().grad.as_ref().unwrap().to_vec(), &[1., 1., 1., 1.]);


    let left = Tensor::from_data_and_shape(vec![1., 2., 3., 4.], vec![1, 2, 2]);
    let mut c = left.sum();
    let c_grad = Tensor::from_data_and_shape(vec![2.], vec![1]);
    c.set_grad(c_grad);
    c.backward();
    assert_eq!(left.read_lock().grad.as_ref().unwrap().to_vec(), &[2., 2., 2., 2.]);
}