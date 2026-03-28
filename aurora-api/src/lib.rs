//! AURORA Public API
//!
//! High-level API for application integration.

#![warn(missing_docs)]

use aurora_core::device::DeviceId;
use aurora_profiler::profile::HardwareProfile;

/// AURORA runtime handle
pub struct AuroraRuntime {
    profile: HardwareProfile,
}

impl AuroraRuntime {
    /// Initialize the AURORA runtime
    pub fn initialize() -> Result<Self> {
        aurora_core::initialize()?;
        
        // Detect hardware
        let mut profiler = aurora_profiler::HardwareProfiler::new();
        let profile = profiler.detect()?;
        
        Ok(Self { profile })
    }
    
    /// Create a tensor
    pub fn create_tensor(&self, shape: TensorShape, dtype: DataType) -> Result<Tensor> {
        let id = 1; // Would be generated
        Ok(Tensor::new(id, shape, dtype, DeviceId::CPU))
    }
    
    /// Execute a kernel
    pub fn execute_kernel(
        &self,
        _kernel_type: KernelType,
        _inputs: &[&Tensor],
        _output: &mut Tensor,
    ) -> Result<()> {
        // Placeholder - would execute kernel
        Ok(())
    }
    
    /// Execute a compute graph
    pub fn execute_graph(&self, _graph: &ComputeGraph) -> Result<Vec<Tensor>> {
        // Placeholder - would execute graph
        Ok(vec![])
    }
    
    /// Get hardware profile
    pub fn hardware_profile(&self) -> &HardwareProfile {
        &self.profile
    }
    
    /// Shutdown the runtime
    pub fn shutdown(&self) -> Result<()> {
        aurora_core::shutdown()
    }
}

/// Version information
pub fn version() -> &'static str {
    aurora_core::VERSION
}

/// Build information
pub mod build_info {
    pub use aurora_core::build_info::*;
}

// Re-export core types
pub use aurora_core::{
    device::{Device, DeviceId, DeviceType, SimdLevel},
    error::{AuroraError, Result},
    graph::{ComputeGraph, NodeId, OpType},
    kernel::{Kernel, KernelType, LaunchConfig},
    memory::{Allocation, MemoryPool, MemoryType},
    tensor::{DataType, Layout, Tensor, TensorShape},
    types::Scalar,
};
