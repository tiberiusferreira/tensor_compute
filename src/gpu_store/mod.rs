use crate::GpuInstance;
use once_cell::sync::Lazy;
use std::sync::RwLock;
use wgpu::{AdapterInfo, Device, Queue};

static DEVICES: Lazy<GpuStore> = Lazy::new(|| {
    let (s, r) = std::sync::mpsc::channel();
    std::thread::spawn(move || s.send(futures::executor::block_on(GpuStore::new())));
    r.recv().unwrap()
});

pub struct GpuStore {
    // Yes, I could use atomics here, but its just a prototype for now
    current_default: RwLock<usize>,
    available_devices: Vec<GpuInstance>,
}

impl GpuStore {
    pub fn get_default() -> &'static GpuInstance {
        &DEVICES.available_devices[*DEVICES.current_default.read().unwrap()]
    }

    pub fn set_default(id: &AdapterInfo) {
        let new_val = (&DEVICES)
            .available_devices
            .iter()
            .position(|gpu| gpu.info() == id)
            .unwrap();
        let mut lock = DEVICES.current_default.write().unwrap();
        *lock = new_val;
    }

    pub fn get(id: &AdapterInfo) -> &GpuInstance {
        &DEVICES
            .available_devices
            .iter()
            .find(|&gpu| gpu.info() == id)
            .unwrap()
    }

    pub fn list_gpus() -> Vec<&'static AdapterInfo> {
        (&DEVICES.available_devices)
            .iter()
            .map(|gpu| gpu.info())
            .collect()
    }

    async fn new() -> Self {
        // let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
        // let mut available_devices: Vec<GpuInstance> = vec![];
        // for adapter in
        //     instance.enumerate_adapters(wgpu::UnsafeFeatures::disallow(), wgpu::BackendBit::PRIMARY)
        // {
        //     let (device, queue) = adapter
        //         .request_device(
        //             &wgpu::DeviceDescriptor {
        //                 features: wgpu::Features::empty(),
        //                 limits: wgpu::Limits::default(),
        //                 shader_validation: true,
        //             },
        //             None,
        //         )
        //         .await
        //         .unwrap();
        //     available_devices.push(GpuInstance {
        //         device,
        //         queue,
        //         info: adapter.get_info(),
        //     });
        // }
        // GpuStore {
        //     current_default: RwLock::new(0),
        //     available_devices,
        // }
        unimplemented!()
    }
}
