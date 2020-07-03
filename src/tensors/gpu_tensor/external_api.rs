use crate::gpu_internals::GpuInstance;
use crate::{CpuTensor, GpuStore, GpuTensor, Tensor};

impl GpuTensor {
    fn get_gpu(&self) -> &GpuInstance {
        GpuStore::get(self.buffer.device_info())
    }

    pub async fn mm(&self, other: &Self) -> Self {
        let gpu = self.get_gpu();
        super::gpu_ops::mm(gpu, self, other).await
    }

    pub async fn to_cpu(&self) -> CpuTensor {
        let gpu = self.get_gpu();
        self.to_cpu_with_gpu(gpu).await
    }

    pub fn from_data(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let gpu = GpuStore::get_default();
        Self::from_data_with_gpu(gpu, data, shape)
    }

    pub fn expand(&mut self, shape: Vec<usize>) {
        assert!(
            shape.len() >= self.shape().len(),
            "Target shape needs to have the same number of \
        dimensions of current shape or more."
        );
        for (index, dim) in self.shape().iter().enumerate() {
            if self.shape()[index] == 1 {
                self.shape[index] = *dim;
                self.strides[index] = 0;
            } else if (*dim) != self.shape()[index] {
                panic!(
                    "Cant expand original non-unitary dimension {} to {}",
                    self.shape()[index],
                    *dim
                );
            }
        }
    }

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
        self.shape = shape;
    }
}

impl Tensor for GpuTensor {
    fn shape(&self) -> Vec<usize> {
        self.shape.to_vec()
    }

    fn strides(&self) -> Vec<usize> {
        self.strides.clone()
    }
}
