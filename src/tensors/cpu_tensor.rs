use crate::gpu_internals::GpuInstance;
use crate::{GpuTensor, Tensor};
use std::fmt::{Display, Formatter, Write, Debug};
use std::collections::VecDeque;
use crate::utils::strides_from_deque_shape;

#[derive(Debug)]
pub struct CpuTensor {
    data: Vec<f32>,
    shape: VecDeque<usize>,
    strides: VecDeque<usize>,
}

impl Display for CpuTensor{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut stri = String::new();
        for i in 0..self.shape[0]-1{
            self.index(i).print(&mut stri, false);
        }
        f.write_str("Shape: ").unwrap();
        self.shape.fmt(f).unwrap();
        f.write_str(" Strides: ").unwrap();
        self.strides.fmt(f).unwrap();
        f.write_str("\n").unwrap();
        let mut stri = String::new();
        for i in 0..self.shape[0]-1{
            self.index(i).print(&mut stri, false);
        }
        self.index(self.shape[0]-1).print(&mut stri, true);
        f.write_str(&stri).unwrap();
        Ok(())
    }
}
impl Tensor for CpuTensor{
    fn shape(&self) -> &VecDeque<usize> {
        &self.shape
    }

    fn strides(&self) -> &VecDeque<usize> {
        &self.strides
    }
}

impl Indexable for CpuTensor{
    fn index(&self, idx: usize) -> CpuTensorView {
        self.index(idx)
    }

    fn data_slice(&self) -> &[f32] {
        self.data_slice()
    }
}

impl <'a> Indexable for CpuTensorView<'a>{
    fn index(&self, idx: usize) -> CpuTensorView {
        self.index(idx)
    }

    fn data_slice(&self) -> &[f32] {
        self.data
    }
}

trait Indexable{
    fn index(&self, idx: usize) -> CpuTensorView;
    fn data_slice(&self) -> &[f32];
}
trait Printable{
    fn print(&self, buff: &mut String, last: bool);
}
impl <T> Printable for T where T : Tensor + Indexable{
    fn print(&self, buff: &mut String, last: bool) {
        if self.shape().len() == 1{
            if last{
                buff.write_str(&format!("{:?}", self.data_slice())).unwrap();
            }else{
                buff.write_str(&format!("{:?}\n", self.data_slice())).unwrap();
            }
        }else{
            let last_shape = self.shape()[0]-1;
            for idx in 0..last_shape{
                self.index(idx).print(buff, false);
            }
            if last{
                self.index(last_shape).print(buff, true);
            }else{
                self.index(last_shape).print(buff, false);
                buff.write_str("\n").unwrap();
            }
        }

    }
}

#[derive(Debug)]
pub struct CpuTensorView<'a> {
    data: &'a [f32],
    shape: VecDeque<usize>,
    strides: VecDeque<usize>,
}

impl <'a> Tensor for CpuTensorView<'a>{
    fn shape(&self) -> &VecDeque<usize> {
        &self.shape
    }

    fn strides(&self) -> &VecDeque<usize> {
        &self.strides
    }
}
impl <'a> CpuTensorView<'a>{
    pub fn index(&self, idx: usize) -> CpuTensorView{
        let mut new_shape = self.shape.clone();
        new_shape.pop_front();
        let mut new_strides = self.strides.clone();
        new_strides.pop_front();
        CpuTensorView{
            data: &self.data[self.strides[0]*idx..self.strides[0]*(idx+1)],
            shape: new_shape,
            strides: new_strides
        }
    }

}

impl CpuTensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let calc_size = shape.iter().rev().fold(1, |acc: usize, &x| acc * x);
        assert_eq!(
            calc_size,
            data.len(),
            "Shape is not valid for the size of the data!"
        );
        let shape = VecDeque::from(shape);
        let strides = strides_from_deque_shape(&shape);
        Self { data, shape, strides }
    }
    pub fn new_with_strides(data: Vec<f32>, shape: VecDeque<usize>, strides: VecDeque<usize>) -> Self {
        let calc_size = shape.iter().rev().fold(1, |acc: usize, &x| acc * x);
        assert_eq!(
            calc_size,
            data.len(),
            "Shape is not valid for the size of the data!"
        );
        Self { data, shape, strides }
    }
}

impl CpuTensor {
    pub fn to_gpu(&self, gpu: &GpuInstance) -> GpuTensor {
        GpuTensor::from_buffer_with_strides(
            gpu.new_gpu_buffer_from_data(bytemuck::cast_slice(&self.data)),
            self.shape.clone(),
            self.strides.clone()
        )
    }
    pub fn data_slice(&self) -> &[f32] {
        self.data.as_slice()
    }

    pub fn index(&self, idx: usize) -> CpuTensorView{
        let mut new_shape = self.shape.clone();
        new_shape.pop_front();
        let mut new_strides = self.strides.clone();
        new_strides.pop_front();
        CpuTensorView{
            data: &self.data[self.strides[0]*idx..self.strides[0]*(idx+1)],
            shape: new_shape,
            strides: new_strides
        }
    }

}

// #[test]
// fn some(){
//     let tensor = CpuTensor::new(vec![1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.], vec![2, 2, 3, 2]);
//     println!("{}", tensor);
//     println!("a");
// }