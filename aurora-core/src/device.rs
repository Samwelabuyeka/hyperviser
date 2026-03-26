//! Device abstraction for CPU and GPU devices

use serde::{Deserialize, Serialize};
use std::fmt;

/// Unique device identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DeviceId(pub u32);

impl DeviceId {
    /// CPU device ID (always 0)
    pub const CPU: DeviceId = DeviceId(0);
    
    /// Create a new device ID
    pub const fn new(id: u32) -> Self {
        Self(id)
    }
    
    /// Check if this is the CPU device
    pub const fn is_cpu(&self) -> bool {
        self.0 == 0
    }
    
    /// Check if this is a GPU device
    pub const fn is_gpu(&self) -> bool {
        self.0 > 0
    }
    
    /// Get GPU index (0 for first GPU, etc.)
    pub const fn gpu_index(&self) -> Option<u32> {
        if self.is_gpu() {
            Some(self.0 - 1)
        } else {
            None
        }
    }
}

impl fmt::Display for DeviceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_cpu() {
            write!(f, "CPU")
        } else {
            write!(f, "GPU{}", self.0 - 1)
        }
    }
}

impl Default for DeviceId {
    fn default() -> Self {
        Self::CPU
    }
}

/// Type of compute device
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DeviceType {
    /// CPU device
    Cpu,
    /// NVIDIA GPU via CUDA
    Cuda,
    /// AMD GPU via ROCm
    Rocm,
    /// Generic GPU via Vulkan
    Vulkan,
    /// Generic GPU via OpenCL
    OpenCL,
    /// Intel GPU via Level Zero
    LevelZero,
}

impl DeviceType {
    /// Check if this is a CPU device
    pub const fn is_cpu(&self) -> bool {
        matches!(self, DeviceType::Cpu)
    }
    
    /// Check if this is a GPU device
    pub const fn is_gpu(&self) -> bool {
        !self.is_cpu()
    }
    
    /// Get the name of the device type
    pub const fn name(&self) -> &'static str {
        match self {
            DeviceType::Cpu => "CPU",
            DeviceType::Cuda => "CUDA",
            DeviceType::Rocm => "ROCm",
            DeviceType::Vulkan => "Vulkan",
            DeviceType::OpenCL => "OpenCL",
            DeviceType::LevelZero => "Level Zero",
        }
    }
    
    /// Get preferred memory type for this device
    pub const fn preferred_memory_type(&self) -> &'static str {
        match self {
            DeviceType::Cpu => "host",
            DeviceType::Cuda => "cuda",
            DeviceType::Rocm => "rocm",
            DeviceType::Vulkan => "vulkan",
            DeviceType::OpenCL => "opencl",
            DeviceType::LevelZero => "level_zero",
        }
    }
}

impl fmt::Display for DeviceType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Compute capability for a device
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ComputeCapability {
    /// Major version
    pub major: u32,
    /// Minor version
    pub minor: u32,
}

impl ComputeCapability {
    /// Create a new compute capability
    pub const fn new(major: u32, minor: u32) -> Self {
        Self { major, minor }
    }
    
    /// Check if this capability supports a minimum version
    pub const fn supports(&self, major: u32, minor: u32) -> bool {
        self.major > major || (self.major == major && self.minor >= minor)
    }
    
    /// Get version as tuple
    pub const fn as_tuple(&self) -> (u32, u32) {
        (self.major, self.minor)
    }
}

impl fmt::Display for ComputeCapability {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

/// SIMD instruction set level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SimdLevel {
    /// No SIMD (scalar only)
    Scalar,
    /// SSE2
    Sse2,
    /// SSE4.2
    Sse4_2,
    /// AVX
    Avx,
    /// AVX2
    Avx2,
    /// AVX-512
    Avx512,
    /// ARM NEON
    Neon,
    /// ARM SVE
    Sve,
}

impl SimdLevel {
    /// Get the vector width in bytes
    pub const fn vector_width(&self) -> usize {
        match self {
            SimdLevel::Scalar => 8,
            SimdLevel::Sse2 | SimdLevel::Sse4_2 => 16,
            SimdLevel::Avx | SimdLevel::Avx2 => 32,
            SimdLevel::Avx512 => 64,
            SimdLevel::Neon => 16,
            SimdLevel::Sve => 16,
        }
    }
    
    /// Get the vector width in f32 elements
    pub const fn f32_width(&self) -> usize {
        self.vector_width() / 4
    }
    
    /// Get the vector width in f64 elements
    pub const fn f64_width(&self) -> usize {
        self.vector_width() / 8
    }
    
    /// Get the vector width in i32 elements
    pub const fn i32_width(&self) -> usize {
        self.vector_width() / 4
    }
    
    /// Get the name of the SIMD level
    pub const fn name(&self) -> &'static str {
        match self {
            SimdLevel::Scalar => "Scalar",
            SimdLevel::Sse2 => "SSE2",
            SimdLevel::Sse4_2 => "SSE4.2",
            SimdLevel::Avx => "AVX",
            SimdLevel::Avx2 => "AVX2",
            SimdLevel::Avx512 => "AVX-512",
            SimdLevel::Neon => "NEON",
            SimdLevel::Sve => "SVE",
        }
    }
    
    /// Check if this level supports another level
    pub const fn supports(&self, other: SimdLevel) -> bool {
        let self_rank = self.rank();
        let other_rank = other.rank();
        self_rank >= other_rank
    }
    
    /// Get rank for comparison (higher = more capable)
    const fn rank(&self) -> u8 {
        match self {
            SimdLevel::Scalar => 0,
            SimdLevel::Sse2 => 1,
            SimdLevel::Sse4_2 => 2,
            SimdLevel::Avx => 3,
            SimdLevel::Avx2 => 4,
            SimdLevel::Avx512 => 5,
            SimdLevel::Neon => 3,
            SimdLevel::Sve => 4,
        }
    }
}

impl Default for SimdLevel {
    fn default() -> Self {
        SimdLevel::Scalar
    }
}

impl fmt::Display for SimdLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// CPU-specific properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuProperties {
    /// SIMD level supported
    pub simd_level: SimdLevel,
    /// L1 cache size per core in bytes
    pub l1_cache: usize,
    /// L2 cache size per core in bytes
    pub l2_cache: usize,
    /// L3 cache size in bytes
    pub l3_cache: usize,
    /// Number of NUMA nodes
    pub numa_nodes: usize,
    /// Base frequency in MHz
    pub base_frequency: u32,
    /// Boost frequency in MHz
    pub boost_frequency: u32,
    /// CPU features
    pub features: Vec<String>,
}

impl CpuProperties {
    /// Create default CPU properties
    pub fn new(simd_level: SimdLevel) -> Self {
        Self {
            simd_level,
            l1_cache: 32 * 1024,
            l2_cache: 256 * 1024,
            l3_cache: 8 * 1024 * 1024,
            numa_nodes: 1,
            base_frequency: 2000,
            boost_frequency: 2000,
            features: Vec::new(),
        }
    }
    
    /// Get total cache size
    pub fn total_cache(&self, num_cores: usize) -> usize {
        self.l1_cache * num_cores + self.l2_cache * num_cores + self.l3_cache
    }
    
    /// Estimate peak FP32 performance (GFLOPS)
    pub fn estimate_peak_gflops(&self, num_cores: usize) -> f64 {
        let ops_per_cycle = match self.simd_level {
            SimdLevel::Avx512 => 32.0,
            SimdLevel::Avx2 => 16.0,
            SimdLevel::Avx => 8.0,
            SimdLevel::Sse4_2 | SimdLevel::Sse2 => 4.0,
            _ => 1.0,
        };
        
        let freq_ghz = self.base_frequency as f64 / 1000.0;
        num_cores as f64 * freq_ghz * ops_per_cycle
    }
}

impl Default for CpuProperties {
    fn default() -> Self {
        Self::new(SimdLevel::Scalar)
    }
}

/// GPU-specific properties
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuProperties {
    /// Warp/wavefront size
    pub warp_size: u32,
    /// Maximum threads per block
    pub max_threads_per_block: u32,
    /// Maximum shared memory per block in bytes
    pub max_shared_memory: usize,
    /// Number of registers per block
    pub registers_per_block: u32,
    /// PCIe bandwidth in GB/s
    pub pcie_bandwidth: f32,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth: f32,
    /// Peak compute in TFLOPS (FP32)
    pub peak_compute_fp32: f32,
    /// Peak compute in TFLOPS (FP16)
    pub peak_compute_fp16: f32,
    /// Driver version
    pub driver_version: String,
}

impl Default for GpuProperties {
    fn default() -> Self {
        Self {
            warp_size: 32,
            max_threads_per_block: 1024,
            max_shared_memory: 48 * 1024,
            registers_per_block: 65536,
            pcie_bandwidth: 16.0,
            memory_bandwidth: 256.0,
            peak_compute_fp32: 10.0,
            peak_compute_fp16: 20.0,
            driver_version: "unknown".to_string(),
        }
    }
}

/// Device information and capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Device {
    /// Device ID
    pub id: DeviceId,
    /// Device type
    pub device_type: DeviceType,
    /// Device name
    pub name: String,
    /// Compute capability (for GPUs)
    pub compute_capability: Option<ComputeCapability>,
    /// Total memory in bytes
    pub total_memory: usize,
    /// Available memory in bytes
    pub available_memory: usize,
    /// Number of compute units (cores/SMs)
    pub compute_units: usize,
    /// Maximum work group size
    pub max_work_group_size: usize,
    /// Maximum memory allocation size
    pub max_allocation_size: usize,
    /// Whether the device supports unified memory
    pub unified_memory: bool,
    /// Whether the device supports memory pooling
    pub memory_pools: bool,
    /// CPU-specific properties
    pub cpu_properties: Option<CpuProperties>,
    /// GPU-specific properties
    pub gpu_properties: Option<GpuProperties>,
}

impl Device {
    /// Create a new CPU device
    pub fn new_cpu(name: String, cores: usize, memory: usize) -> Self {
        let cpu_props = CpuProperties::default();
        
        Self {
            id: DeviceId::CPU,
            device_type: DeviceType::Cpu,
            name,
            compute_capability: None,
            total_memory: memory,
            available_memory: memory,
            compute_units: cores,
            max_work_group_size: cores,
            max_allocation_size: memory,
            unified_memory: true,
            memory_pools: true,
            cpu_properties: Some(cpu_props),
            gpu_properties: None,
        }
    }
    
    /// Create a new GPU device
    pub fn new_gpu(
        id: DeviceId,
        device_type: DeviceType,
        name: String,
        compute_capability: Option<ComputeCapability>,
        memory: usize,
        compute_units: usize,
    ) -> Self {
        let gpu_props = GpuProperties::default();
        
        Self {
            id,
            device_type,
            name,
            compute_capability,
            total_memory: memory,
            available_memory: memory,
            compute_units,
            max_work_group_size: 1024,
            max_allocation_size: memory / 4,
            unified_memory: false,
            memory_pools: device_type == DeviceType::Cuda,
            cpu_properties: None,
            gpu_properties: Some(gpu_props),
        }
    }
    
    /// Check if this is a CPU device
    pub const fn is_cpu(&self) -> bool {
        self.device_type.is_cpu()
    }
    
    /// Check if this is a GPU device
    pub const fn is_gpu(&self) -> bool {
        self.device_type.is_gpu()
    }
    
    /// Get memory usage percentage
    pub fn memory_usage_percent(&self) -> f64 {
        if self.total_memory == 0 {
            return 0.0;
        }
        let used = self.total_memory - self.available_memory;
        (used as f64 / self.total_memory as f64) * 100.0
    }
    
    /// Get SIMD level (for CPU devices)
    pub fn simd_level(&self) -> Option<SimdLevel> {
        self.cpu_properties.as_ref().map(|p| p.simd_level)
    }
    
    /// Estimate peak performance (GFLOPS for CPU, TFLOPS for GPU)
    pub fn estimate_peak_performance(&self) -> f64 {
        if let Some(ref cpu) = self.cpu_properties {
            cpu.estimate_peak_gflops(self.compute_units)
        } else if let Some(ref gpu) = self.gpu_properties {
            gpu.peak_compute_fp32 as f64 * 1000.0 // Convert TFLOPS to GFLOPS
        } else {
            0.0
        }
    }
    
    /// Get memory bandwidth (GB/s)
    pub fn memory_bandwidth_gbps(&self) -> f64 {
        if let Some(ref gpu) = self.gpu_properties {
            gpu.memory_bandwidth as f64
        } else {
            // Estimate for CPU
            25.0
        }
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} [{}] - {} ({} compute units, {} memory)",
            self.id,
            self.device_type,
            self.name,
            self.compute_units,
            crate::format_bytes(self.total_memory as u64)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_id() {
        assert!(DeviceId::CPU.is_cpu());
        assert!(!DeviceId::CPU.is_gpu());
        
        let gpu_id = DeviceId::new(1);
        assert!(gpu_id.is_gpu());
        assert_eq!(gpu_id.gpu_index(), Some(0));
    }

    #[test]
    fn test_simd_level() {
        assert_eq!(SimdLevel::Avx512.vector_width(), 64);
        assert_eq!(SimdLevel::Avx2.f32_width(), 8);
        assert!(SimdLevel::Avx512.supports(SimdLevel::Avx2));
        assert!(!SimdLevel::Avx2.supports(SimdLevel::Avx512));
    }

    #[test]
    fn test_compute_capability() {
        let cc = ComputeCapability::new(8, 6);
        assert!(cc.supports(8, 0));
        assert!(cc.supports(7, 5));
        assert!(!cc.supports(9, 0));
    }

    #[test]
    fn test_cpu_device() {
        let device = Device::new_cpu("Test CPU".to_string(), 8, 16 * 1024 * 1024 * 1024);
        assert!(device.is_cpu());
        assert_eq!(device.compute_units, 8);
        assert!(device.simd_level().is_some());
    }

    #[test]
    fn test_gpu_device() {
        let device = Device::new_gpu(
            DeviceId::new(1),
            DeviceType::Cuda,
            "RTX 3080".to_string(),
            Some(ComputeCapability::new(8, 6)),
            10 * 1024 * 1024 * 1024,
            68,
        );
        assert!(device.is_gpu());
        assert_eq!(device.compute_units, 68);
        assert!(device.gpu_properties.is_some());
    }

    #[test]
    fn test_cpu_properties() {
        let props = CpuProperties::new(SimdLevel::Avx2);
        assert_eq!(props.simd_level, SimdLevel::Avx2);
        assert!(props.estimate_peak_gflops(8) > 0.0);
    }
}
