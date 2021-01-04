mod sum;
use crate::{GpuTensor, GpuAllocated, AsShaderInput};
use crate::gpu_internals::shader_runner::{ThreadGroup};
use crate::tensors::traits::ShapeStrideTrait;

macro_rules! bin_element_wise_unary_op {
    ($operation_name:literal, $fun_name:ident, $shader_path:literal) => {
        impl GpuTensor{
            pub async fn $fun_name(&self) -> GpuTensor {
                let cs_module = self.gpu().shader_from_file_bytes(wgpu::include_spirv!($shader_path));
                let nb_output_numbers = self.numel();
                let output_buffer = self.gpu().empty_like(self.buffer());
                let shader_inputs = self.to_shader_inputs()
                    .with_buffer(&output_buffer);
                self.gpu().run_shader(
                    &cs_module,
                    &shader_inputs,
                    ThreadGroup {
                        x: nb_output_numbers,
                        y: 1,
                        z: 1,
                    },
                );
                GpuTensor::from_buffer(output_buffer, self.shape_strides.shape.clone())
            }
        }
    }
}

bin_element_wise_unary_op!("exp", exp, "exp.spv");
bin_element_wise_unary_op!("ln", ln, "ln.spv");
bin_element_wise_unary_op!("clone", clone, "clone.spv");

mod test {
    use crate::{GpuTensor, CpuTransferable};

    #[test]
    fn exp_test() {
        let async_block = async {
            let tensor_a = GpuTensor::from(vec![-1., -2., -3., -4., 5., 6.], vec![3, 2]);
            let res = tensor_a.exp().await;
            let expected: Vec<f32> = vec![-1f32, -2., -3., -4., 5., 6.].into_iter().map(|e| e.exp()).collect();
            let err: f32 = res.to_cpu().raw_data_slice().iter()
                .zip(expected.iter())
                .map(|(l, r)| l - r)
                .sum();
            assert!(err < 0.01);
        };
        futures::executor::block_on(async_block);
    }

    #[test]
    fn ln_test() {
        let async_block = async {
            let tensor_a = GpuTensor::from(vec![1., 2., 3., 4., 5., 6.], vec![3, 2]);
            let res = tensor_a.ln().await;
            let expected: Vec<f32> = vec![1f32, 2., 3., 4., 5., 6.].into_iter().map(|e| e.ln()).collect();
            let err: f32 = res.to_cpu().raw_data_slice().iter()
                .zip(expected.iter())
                .map(|(l, r)| l - r)
                .sum();
            assert!(err < 0.01, err);
        };
        futures::executor::block_on(async_block);
    }

    #[test]
    fn can_clone() {
        let async_block = async {
            let ma = GpuTensor::from(vec![2., 3., 4., 5., 6., 7., 8., 9.], vec![2, 2, 2]);
            let tensor = ma.clone().await;
            assert_eq!(ma.to_cpu(), tensor.to_cpu());
        };
        futures::executor::block_on(async_block);
    }

    #[test]
    fn can_clone_non_uniform_shape() {
        let async_block = async {
            let ma = GpuTensor::from(vec![2., 3., 4., 5., 6., 7.], vec![2, 3]);
            let tensor = ma.clone().await;
            assert_eq!(ma.to_cpu(), tensor.to_cpu());
        };
        futures::executor::block_on(async_block);
    }

}

