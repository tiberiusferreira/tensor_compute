use crate::gpu_buffers::GpuBuffer;
use crate::GpuBox;
use wgpu::{BindGroupLayoutEntry, Binding, ShaderModule};

pub struct ShaderInput<'a> {
    pub binding_id: usize,
    pub gpu_buffer: &'a GpuBuffer,
}

impl<'a> ShaderInput<'a> {
    pub fn to_bind_group_layout(&self) -> BindGroupLayoutEntry {
        self.gpu_buffer.layout(self.binding_id)
    }

    pub fn to_bind_group(&self) -> Binding {
        Binding {
            binding: self.binding_id as u32,
            resource: wgpu::BindingResource::Buffer(self.gpu_buffer.raw_buffer().slice(..)),
        }
    }
}

pub struct ThreadGroup {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl GpuBox {
    pub fn run_shader(
        &self,
        shader: &ShaderModule,
        shader_inputs: Vec<ShaderInput>,
        threads: ThreadGroup,
    ) {
        let bindings_layouts: Vec<BindGroupLayoutEntry> = shader_inputs
            .iter()
            .map(ShaderInput::to_bind_group_layout)
            .collect();
        let bind_group_layout =
            self.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    bindings: bindings_layouts.as_slice(),
                });
        let bindings: Vec<Binding> = shader_inputs
            .iter()
            .map(ShaderInput::to_bind_group)
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
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut compute_pass = encoder.begin_compute_pass();
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch(threads.x as u32, threads.y as u32, threads.z as u32);
        }
        self.queue.submit(Some(encoder.finish()));
    }
}
