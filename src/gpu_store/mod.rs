use crate::gpu_internals::{GpuInfo, GpuInstance};
use once_cell::sync::Lazy;
use std::sync::RwLock;

static DEVICES: Lazy<GpuStore> = Lazy::new(|| {
    let (s, r) = std::sync::mpsc::channel();
    std::thread::spawn(move || s.send(futures::executor::block_on(GpuStore::new())));
    r.recv().unwrap()
});

pub struct GpuStore {
    current: RwLock<usize>,
    available_devices: Vec<GpuInstance>,
}

impl GpuStore {
    pub fn get_default() -> &'static GpuInstance {
        let current_idx = DEVICES.current.read().unwrap();
        &DEVICES.available_devices[*current_idx]
    }

    pub fn get(gpu_info: &GpuInfo) -> &'static GpuInstance {
        DEVICES.available_devices.iter().find(|dev| dev.info() == gpu_info).unwrap()
    }

    pub async fn select_gpu(gpu_info: &GpuInfo) {
        let idx = DEVICES.available_devices.iter().position(|dev| dev.info() == gpu_info).unwrap();
        *(&DEVICES.current).write().unwrap() = idx;
    }

    pub fn list_gpus() -> Vec<&'static GpuInfo> {
        (&DEVICES).available_devices.iter().map(|dev| dev.info()).collect()
    }

    async fn new() -> Self {
        let gpu_factory = crate::gpu_internals::gpu_factory::GpuFactory::new().await;

        let gpu_list = gpu_factory
            .list_gpus()
            .await;

        let mut gpu_instances = vec![];
        for gpu_info in &gpu_list{
            gpu_instances.push(gpu_factory.request_gpu(&gpu_info).await);
        }
        assert!(!gpu_instances.is_empty(), "No GPU detected!");
        Self {
            current: RwLock::new(0),
            available_devices: gpu_instances,
        }
    }
}
