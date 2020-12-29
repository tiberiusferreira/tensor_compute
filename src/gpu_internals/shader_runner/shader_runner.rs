use crate::gpu_internals::gpu_buffers::{GpuBuffer, GpuUniformBuffer};
use crate::gpu_internals::GpuInstance;
use wgpu::{BindGroupEntry, BindGroupLayoutEntry, BindingResource, ShaderModule};

#[derive(Debug)]
pub enum BufferType<'a> {
    Storage(&'a GpuBuffer),
    StorageOwned(GpuBuffer),
}

impl<'a> BufferType<'a> {
    pub fn layout(&self, id: u32) -> BindGroupLayoutEntry {
        match self {
            BufferType::Storage(a) => a.layout(id),
            BufferType::StorageOwned(a) => a.layout(id),
        }
    }
    pub fn to_bind_resource(&self) -> BindingResource {
        match self {
            BufferType::Storage(a) => a.to_bind_resource(),
            BufferType::StorageOwned(a) => a.to_bind_resource(),
        }
    }
}

#[derive(Debug, Default)]
pub struct ShaderInputs<'a>{
    pub bindings: Vec<ShaderBinding<'a>>,
    pub push_constants: PushConstants
}

impl <'a> ShaderInputs<'a>{
    pub fn append_buffer(&mut self, gpu_buffer: &'a GpuBuffer){
        self.bindings.push(ShaderBinding{
            binding_id: self.bindings.len() as u32,
            gpu_buffer_type: BufferType::Storage(gpu_buffer)
        });
    }
}

#[derive(Debug)]
pub struct ShaderBinding<'a> {
    pub binding_id: u32,
    pub gpu_buffer_type: BufferType<'a>,
}

impl<'a> ShaderBinding<'a> {
    pub fn to_bind_group_layout(&self) -> BindGroupLayoutEntry {
        self.gpu_buffer_type.layout(self.binding_id)
    }

    pub fn to_bind_group(&self) -> BindGroupEntry {
        BindGroupEntry {
            binding: self.binding_id as u32,
            resource: self.gpu_buffer_type.to_bind_resource(),
        }
    }
}

pub struct ThreadGroup {
    pub x: usize,
    pub y: usize,
    pub z: usize,
}

#[derive(Debug, Default)]
pub struct PushConstants{
    pub offset: u32,
    pub data: Vec<u32>
}


impl GpuInstance {
    pub fn run_shader(
        &self,
        shader: &ShaderModule,
        shader_inputs: &ShaderInputs,
        threads: ThreadGroup,
    ) {
        let bindings_layouts: Vec<BindGroupLayoutEntry> = shader_inputs.bindings
            .iter()
            .map(ShaderBinding::to_bind_group_layout)
            .collect();
        let bind_group_layout =
            self.device()
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: None,
                    entries: bindings_layouts.as_slice(),
                });
        let bindings: Vec<BindGroupEntry> = shader_inputs.bindings
            .iter()
            .map(ShaderBinding::to_bind_group)
            .collect();
        let bind_group = self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &bind_group_layout,
            entries: bindings.as_slice(),
        });
        let pipeline_layout =
            self.device()
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[wgpu::PushConstantRange{
                        stages: wgpu::ShaderStage::COMPUTE,
                        range: shader_inputs.push_constants.offset..4*shader_inputs.push_constants.data.len() as u32
                    }],
                });
        let compute_pipeline =
            self.device()
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: None,
                    layout: Some(&pipeline_layout),
                    compute_stage: wgpu::ProgrammableStageDescriptor {
                        module: &shader,
                        entry_point: &"main",
                    },
                });
        let mut encoder = self
            .device()
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut compute_pass = encoder.begin_compute_pass();
            compute_pass.set_pipeline(&compute_pipeline);
            compute_pass.set_push_constants(shader_inputs.push_constants.offset, shader_inputs.push_constants.data.as_slice());
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch(threads.x as u32, threads.y as u32, threads.z as u32);
        }
        self.queue().submit(Some(encoder.finish()));
    }
}
