use crate::gpu_buffers::GpuBuffer;
use crate::GpuTensorView;

impl <'a> GpuTensorView<'a>{
    pub fn buffer(&self) -> &'a GpuBuffer {
        &self.buffer
    }

    pub fn buffer_size_in_bytes(&self) -> usize {
        self.buffer.size_bytes()
    }


    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> Vec<usize> {
        self.strides.to_vec()
    }

}