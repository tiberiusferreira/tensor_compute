//! # A WebGPU based Tensor Library
//!
//! This library is intended as a pure Rust `Tensor` library which runs on the GPU everywhere.
//! In order to achieve it, it build upon the [`wgpu`] crate which can target any platform supporting
//! either `Vulkan`, `Metal` or `DX12`. It can run even on a *Raspberry Pi 4 GPU*.
//!
//! The API entry point is the [`Tensor`] structure.
mod tensors;
pub use tensors::*;
pub use gpu_store::*;

mod gpu_internals;
mod gpu_store;
