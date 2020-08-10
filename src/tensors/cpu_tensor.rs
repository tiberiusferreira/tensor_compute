use crate::utils::strides_from_deque_shape;
use crate::{GpuStore, GpuTensor, ShapeStrideTrait};
use std::collections::VecDeque;
use std::fmt::{Debug, Display, Formatter};

#[derive(Debug)]
pub struct CpuTensor {
    data: Vec<f32>,
    shape: VecDeque<usize>,
    strides: VecDeque<usize>,
    offset: usize,
}

impl PartialEq for CpuTensor {
    fn eq(&self, other: &Self) -> bool {
        if other.shape != self.shape {
            return false;
        }
        let mut self_indexer = LinearIndexer::from_shape(self.shape());
        while let Some((idx, _nb_dim_closed)) = self_indexer.next() {
            if self.idx(idx) != other.idx(idx) {
                return false;
            }
        }
        true
    }
}

impl Display for CpuTensor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str("Shape: ").unwrap();
        self.shape.fmt(f).unwrap();
        f.write_str(" Strides: ").unwrap();
        self.strides.fmt(f).unwrap();
        f.write_str("\n").unwrap();
        let mut indexer = LinearIndexer::from_shape_vec(&Vec::from(self.shape.clone()));
        for _i in 0..self.shape.len() {
            f.write_str("[").unwrap();
        }
        let ident = self.shape.len();
        while let Some((idx, dims_closed)) = indexer.next() {
            for _i in 0..dims_closed {
                f.write_str("]").unwrap();
            }
            for _i in 0..dims_closed {
                f.write_str("\n").unwrap();
                // keep indentation
                for _i in 0..ident - dims_closed {
                    f.write_str(" ").unwrap();
                }
            }
            for _i in 0..dims_closed {
                f.write_str("[").unwrap();
            }
            f.write_str(" ").unwrap();
            std::fmt::Display::fmt(&self.idx(&idx), f).unwrap();
            f.write_str(" ").unwrap();
        }
        for _i in 0..self.shape.len() {
            f.write_str("]").unwrap();
        }
        f.write_str("\n").unwrap();
        Ok(())
    }
}
impl ShapeStrideTrait for CpuTensor {
    fn shape(&self) -> &VecDeque<usize> {
        &self.shape
    }

    fn strides(&self) -> &VecDeque<usize> {
        &self.strides
    }

    fn offset(&self) -> usize {
        self.offset
    }
}

impl CpuTensor {
    pub fn from_data_and_shape(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let shape = VecDeque::from(shape);
        let numel = GpuTensor::numel_from_shape(&shape);
        assert_eq!(
            numel,
            data.len(),
            "Shape is not valid for the size of the data!"
        );
        let strides = strides_from_deque_shape(&shape);
        Self {
            data,
            shape,
            strides,
            offset: 0,
        }
    }

    pub fn rand(shape: Vec<usize>) -> Self {
        use rand::Rng;
        let shape = VecDeque::from(shape);
        let numel = GpuTensor::numel_from_shape(&shape);
        let mut rng = rand::thread_rng();
        let mut data = Vec::with_capacity(numel);
        for _i in 0..numel {
            data.push(rng.gen());
        }
        let strides = strides_from_deque_shape(&shape);
        Self {
            data,
            shape,
            strides,
            offset: 0,
        }
    }

    pub fn new_with_strides_and_offset(
        data: Vec<f32>,
        shape: VecDeque<usize>,
        strides: VecDeque<usize>,
        offset: usize,
    ) -> Self {
        let numel = GpuTensor::numel_from_shape(&shape);
        assert!(numel <= data.len(), "Data is too small for given shape");
        Self {
            data,
            shape,
            strides,
            offset,
        }
    }
}

impl CpuTensor {
    pub fn to_gpu(&self) -> GpuTensor {
        let gpu = GpuStore::get_default();
        GpuTensor::from_buffer_with_strides_and_offset(
            gpu.new_gpu_buffer_from_data(bytemuck::cast_slice(&self.data)),
            self.shape.clone(),
            self.strides.clone(),
            self.offset,
        )
    }
    pub fn raw_data_slice(&self) -> &[f32] {
        &self.data.as_slice()
    }

    pub fn as_contiguous_vec(&self) -> Vec<f32> {
        let mut indexer = LinearIndexer::from_shape(self.shape());
        let mut output = Vec::with_capacity(self.numel());
        while let Some((val, _)) = indexer.next() {
            output.push(self.idx(val));
        }
        output
    }

    pub fn idx(&self, idx: &Vec<usize>) -> f32 {
        assert_eq!(
            idx.len(),
            self.shape.len(),
            "Tried to index shape: {:?} with index: {:?}",
            self.shape,
            idx
        );
        let strides_iter = self.strides.iter();
        let shape_iter = self.shape.iter();
        let idx_iter = idx.iter();
        let mut linearized_idx = self.offset;
        for (stride, (idx, shape)) in strides_iter.zip(idx_iter.zip(shape_iter)) {
            assert!(
                *idx + 1 <= *shape,
                format!(
                    "tried indexing element {} when dimension length was: {}",
                    *idx + 1,
                    shape
                )
            );
            linearized_idx += idx * stride;
        }
        self.data[linearized_idx]
    }
}

pub struct LinearIndexer {
    curr_index: Vec<usize>,
    max_index: Vec<usize>,
    started: bool,
}
impl LinearIndexer {
    pub fn from_shape_vec(shape: &Vec<usize>) -> Self {
        let new_vec: Vec<usize> = shape.iter().map(|_e| 0).collect();
        Self {
            curr_index: new_vec,
            max_index: shape.clone(),
            started: false,
        }
    }

    pub fn from_shape(shape: &VecDeque<usize>) -> Self {
        let new_vec: Vec<usize> = shape.iter().map(|_e| 0).collect();
        Self {
            curr_index: new_vec,
            max_index: Vec::from(shape.clone()),
            started: false,
        }
    }

    /// Returns the next index and the number of dimensions "closed" in this iteration
    pub fn next(&mut self) -> Option<(&Vec<usize>, usize)> {
        let mut nb_dims_closed: usize = 0;
        if !self.started {
            nb_dims_closed = 0;
            self.started = true;
        } else {
            let mut found = false;
            let max_iter = self.max_index.iter().rev();
            let curr_iter = self.curr_index.iter_mut().rev();
            for (cur, max) in curr_iter.zip(max_iter) {
                if *cur < *max - 1 {
                    *cur += 1;
                    found = true;
                    break;
                } else {
                    *cur = 0;
                    nb_dims_closed += 1;
                }
            }
            if !found {
                return None;
            }
        }
        return Some((&self.curr_index, nb_dims_closed));
    }
}

// #[test]
// fn indexer_works(){
//     let shape = vec![2, 3, 4];
//     let mut indexer = LinearIndexer::new(&shape);
//     while let Some(idx) = indexer.next(){
//         println!("{:?}", idx);
//     }
// }
