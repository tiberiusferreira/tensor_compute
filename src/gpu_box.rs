use crate::{GpuBox, MatricesData};
use zerocopy::AsBytes;
use crate::gpu_buffers::GpuBuffer;

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
    pub async fn mm(&self, input_data_a: &GpuBuffer, input_data_b: &GpuBuffer) -> GpuBuffer {

        let matrices_data = MatricesData {
            rows_a: input_data_a.shape().0,
            cols_a: input_data_a.shape().1,
            rows_b: input_data_b.shape().0,
            cols_b: input_data_b.shape().1,
        };

        let cs_module = self.shader(include_bytes!("shader.spv"));

        let output_shape = (input_data_a.shape().0, input_data_b.shape().1);
        let nb_output_numbers =  output_shape.0 * output_shape.1;
        let out_buffer_store = self.new_empty_gpu_storage_buffer(std::mem::size_of::<u32>() as u64 * nb_output_numbers as u64);

        let input_structure_data = self.new_gpu_storage_buffer_with_data(matrices_data.as_bytes());



        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    bindings: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            ty: wgpu::BindingType::StorageBuffer {
                                dynamic: false,
                                readonly: false,
                            },
                            ..Default::default()
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            ty: wgpu::BindingType::StorageBuffer {
                                dynamic: false,
                                readonly: false,
                            },
                            ..Default::default()
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            ty: wgpu::BindingType::StorageBuffer {
                                dynamic: false,
                                readonly: false,
                            },
                            ..Default::default()
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStage::COMPUTE,
                            ty: wgpu::BindingType::StorageBuffer {
                                dynamic: false,
                                readonly: false,
                            },
                            ..Default::default()
                        },
                    ],
                });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            bindings: &[
                wgpu::Binding {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(input_data_a.buffer().slice(..)),
                },
                wgpu::Binding {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(input_data_b.buffer().slice(..)),
                },
                wgpu::Binding {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(out_buffer_store.slice(..)),
                },
                wgpu::Binding {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(input_structure_data.slice(..)),
                },
            ],
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
                        module: &cs_module,
                        entry_point: "main",
                    },
                });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass();
            cpass.set_pipeline(&compute_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch(nb_output_numbers as u32, 1, 1);
        }
        // encoder.copy_buffer_to_buffer(&out_buffer_store, 0, &out_buffer_staging, 0, input_size_b);

        self.queue.submit(Some(encoder.finish()));

        GpuBuffer::new(out_buffer_store, output_shape)

        // Note that we're not calling `.await` here.
        // let buffer_slice_a = out_buffer_staging.slice(..);
        // let buffer_future_a = buffer_slice_a.map_async(wgpu::MapMode::Read);

        // Poll the device in a blocking manner so that our future resolves.
        // In an actual application, `device.poll(...)` should
        // be called in an event loop or on another thread.
        // self.device.poll(wgpu::Maintain::Wait);

        // if let Ok(()) = buffer_future_a.await {
            // let data = buffer_slice_a.get_mapped_range();
            // let result = data
            //     .chunks_exact(4)
            //     .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
            //     .collect();

            // With the current interface, we have to make sure all mapped views are
            // dropped before we unmap the buffer.
            // drop(data);
            // out_buffer_staging.unmap();

            // result
            // unimplemented!()
        // } else {
        //     panic!("failed to run compute on gpu!")
        // }
    }
}
