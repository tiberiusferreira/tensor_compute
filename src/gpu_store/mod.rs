use wgpu::{Device, Queue, AdapterInfo};
use once_cell::sync::Lazy;
use crate::GpuBox;
use std::collections::HashMap;

static DEVICES: Lazy<GpuStore> = Lazy::new(|| {
    let (s,r) = std::sync::mpsc::channel();
    std::thread::spawn(move ||{
        s.send(futures::executor::block_on(GpuStore::new()))
    });
    r.recv().unwrap()
});

impl GpuBox {
    pub fn device(&self) -> &Device{
        &self.device
    }

    pub fn queue(&self) -> &Queue{
        &self.queue
    }
}

pub struct GpuStore {
    current: usize,
    available_devices: Vec<GpuBox>
}

impl GpuStore {
    pub fn current() -> &'static GpuBox {
        &DEVICES.available_devices[DEVICES.current]
    }

    pub fn get(id: &AdapterInfo) -> &GpuBox{
        &DEVICES.available_devices.iter().find(|&gpu| &gpu.info == id).unwrap()
    }

    async fn new() -> Self {
        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
        let mut available_devices: Vec<GpuBox> = vec![];
        for adapter in instance.enumerate_adapters(
            wgpu::UnsafeExtensions::disallow(),
            wgpu::BackendBit::PRIMARY
        ){

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
            available_devices.push(GpuBox {
                device,
                queue,
                info: adapter.get_info()
            });
        }
        GpuStore {
            current: 0,
            available_devices
        }
    }
}