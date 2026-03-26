//! AURORA GPU Engine
//!
//! GPU compute engine supporting CUDA, ROCm, Vulkan, and OpenCL backends.

#![warn(missing_docs)]

pub mod cuda;
pub mod rocm;
pub mod vulkan;
pub mod opencl;

use aurora_core::device::DeviceId;
use aurora_core::error::Result;
use aurora_core::tensor::Tensor;

/// GPU engine for compute operations
#[derive(Debug)]
pub struct GpuEngine {
    device_id: DeviceId,
}

impl GpuEngine {
    /// Create a new GPU engine
    pub fn new(device_id: DeviceId) -> Result<Self> {
        Ok(Self { device_id })
    }
    
    /// Get device ID
    pub fn device_id(&self) -> DeviceId {
        self.device_id
    }
}

/// GPU memory allocator
trait GpuAllocator {
    /// Allocate device memory
    fn allocate(&self, size: usize) -> Result<u64>;
    
    /// Free device memory
    fn free(&self, ptr: u64) -> Result<()>;
    
    /// Copy host to device
    fn copy_h2d(&self, host: *const u8, device: u64, size: usize) -> Result<()>;
    
    /// Copy device to host
    fn copy_d2h(&self, device: u64, host: *mut u8, size: usize) -> Result<()>;
    
    /// Copy device to device
    fn copy_d2d(&self, src: u64, dst: u64, size: usize) -> Result<()>;
}
