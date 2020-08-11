use crate::gpu_internals::gpu_buffers::{GpuBuffer, GpuUniformBuffer};
use crate::gpu_internals::GpuInstance;
use std::borrow::Cow::Borrowed;
use wgpu::{BindGroupEntry, BindGroupLayoutEntry, BindingResource, ShaderModule};

pub enum BufferType<'a> {
    Storage(&'a GpuBuffer),
    Uniform(&'a GpuUniformBuffer),
    UniformOwned(GpuUniformBuffer),
}

impl<'a> BufferType<'a> {
    pub fn layout(&self, id: usize) -> BindGroupLayoutEntry {
        match self {
            BufferType::Storage(a) => a.layout(id),
            BufferType::Uniform(a) => a.layout(id),
            BufferType::UniformOwned(a) => a.layout(id),
        }
    }
    pub fn to_bind_resource(&self) -> BindingResource {
        match self {
            BufferType::Storage(a) => a.to_bind_resource(),
            BufferType::Uniform(a) => a.to_bind_resource(),
            BufferType::UniformOwned(a) => a.to_bind_resource(),
        }
    }
}

pub struct ShaderInput<'a> {
    pub binding_id: usize,
    pub gpu_buffer: BufferType<'a>,
}

impl<'a> ShaderInput<'a> {
    pub fn to_bind_group_layout(&self) -> BindGroupLayoutEntry {
        self.gpu_buffer.layout(self.binding_id)
    }

    pub fn to_bind_group(&self) -> BindGroupEntry {
        BindGroupEntry {
            binding: self.binding_id as u32,
            resource: self.gpu_buffer.to_bind_resource(),
        }
    }
}

pub struct ThreadGroup {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

impl GpuInstance {
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
            self.device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: Borrowed(bindings_layouts.as_slice()),
                });
        let bindings: Vec<BindGroupEntry> = shader_inputs
            .iter()
            .map(ShaderInput::to_bind_group)
            .collect();
        let bind_group = self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: Borrowed(bindings.as_slice()),
        });
        let pipeline_layout =
            self.device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    bind_group_layouts: Borrowed(&[&bind_group_layout]),
                    push_constant_ranges: Borrowed(&[]),
                });
        let compute_pipeline =
            self.device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    layout: Some(&pipeline_layout),
                    compute_stage: wgpu::ProgrammableStageDescriptor {
                        module: &shader,
                        entry_point: Borrowed("main"),
                    },
                });
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut compute_pass = encoder.begin_compute_pass();
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch(threads.x as u32, threads.y as u32, threads.z as u32);
        }
        self.queue().submit(Some(encoder.finish()));
    }
}
