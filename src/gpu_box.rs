use crate::gpu_buffers::GpuBuffer;
use crate::tensors::Gpu2DTensor;
use crate::{GpuBox, MatricesData};
use wgpu::{BindGroupLayoutEntry, ShaderModule};
use zerocopy::AsBytes;

impl GpuBox {
    pub async fn new() -> Self {
        let instance = wgpu::Instance::new();
        let adapter = instance
            .request_adapter(
                &wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::Default,
                    compatible_surface: None,
                },
                wgpu::UnsafeExtensions::disallow(),
                wgpu::BackendBit::PRIMARY,
            )
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    extensions: wgpu::Extensions::empty(),
                    limits: wgpu::Limits::default(),
                    shader_validation: true,
                },
                None,
            )
            .await
            .unwrap();
        Self { device, queue }
    }
}

impl GpuBox {
    pub async fn mm(&self, input_data_a: &Gpu2DTensor, input_data_b: &Gpu2DTensor) -> Gpu2DTensor {
        let matrices_data = MatricesData {
            rows_a: input_data_a.shape().0 as u32,
            cols_a: input_data_a.shape().1 as u32,
            rows_b: input_data_b.shape().0 as u32,
            cols_b: input_data_b.shape().1 as u32,
        };

        let cs_module = self.shader(include_bytes!("shader.spv"));

        let output_shape = (input_data_a.shape().0, input_data_b.shape().1);
        let nb_output_numbers = output_shape.0 * output_shape.1;
        let out_buffer_store =
            self.empty_gpu_buffer(std::mem::size_of::<u32>() * nb_output_numbers);

        let input_structure_data = self.gpu_buffer_from_data(matrices_data.as_bytes());
        self.run_shader(
            &cs_module,
            vec![
                (0, input_data_a.buffer()),
                (1, input_data_b.buffer()),
                (2, &out_buffer_store),
                (3, &input_structure_data),
            ],
            nb_output_numbers,
        );
        Gpu2DTensor::new(out_buffer_store, output_shape)
    }
}

impl GpuBox {
    pub fn run_shader(
        &self,
        shader: &ShaderModule,
        shader_inputs: Vec<(usize, &GpuBuffer)>,
        threads: usize,
    ) {
        let bindings_layouts: Vec<BindGroupLayoutEntry> = shader_inputs
            .iter()
            .map(|(id, inputs)| inputs.layout(*id))
            .collect();
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    bindings: bindings_layouts.as_slice(),
                });

        let bindings: Vec<wgpu::Binding> = shader_inputs
            .iter()
            .map(|(id, inputs)| wgpu::Binding {
                binding: *id as u32,
                resource: wgpu::BindingResource::Buffer(inputs.raw_buffer().slice(..)),
            })
            .collect();
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            bindings: bindings.as_slice(),
        });

        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                bind_group_layouts: &[&bind_group_layout],
            });

        let compute_pipeline =
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    layout: &pipeline_layout,
                    compute_stage: wgpu::ProgrammableStageDescriptor {
                        module: &shader,
                        entry_point: "main",
                    },
                });
        //
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass();
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch(threads as u32, 1, 1);
        }
        self.queue.submit(Some(encoder.finish()));
    }
}
