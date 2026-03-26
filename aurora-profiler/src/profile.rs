//! Hardware profile and caching

use aurora_core::device::{Device, DeviceId, DeviceType, SimdLevel};
use aurora_core::error::{AuroraError, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;

use crate::cpu::CpuInfo;
use crate::gpu::GpuInfo;
use crate::memory::MemoryInfo;

/// Complete hardware profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    /// Profile version
    pub version: String,
    /// Timestamp when profile was created
    pub timestamp: u64,
    /// CPU information
    pub cpu: CpuInfo,
    /// GPU information
    pub gpus: Vec<GpuInfo>,
    /// Memory information
    pub memory: MemoryInfo,
    /// Optimal configuration
    pub optimal_config: OptimalConfig,
}

impl HardwareProfile {
    /// Create a new hardware profile
    pub fn new(cpu: CpuInfo, gpus: Vec<GpuInfo>, memory: MemoryInfo) -> Self {
        let optimal_config = compute_optimal_config(&cpu, &gpus, &memory);
        
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
            cpu,
            gpus,
            memory,
            optimal_config,
        }
    }
    
    /// Get total compute capability
    pub fn total_compute_units(&self) -> usize {
        let cpu_cus = self.cpu.physical_cores;
        let gpu_cus: u32 = self.gpus.iter().map(|g| g.compute_units).sum();
        cpu_cus + gpu_cus as usize
    }
    
    /// Get total memory
    pub fn total_memory_mb(&self) -> u64 {
        self.memory.total_mb + self.gpus.iter().map(|g| g.vram_mb).sum::<u64>()
    }
    
    /// Check if GPU is available
    pub fn has_gpu(&self) -> bool {
        !self.gpus.is_empty()
    }
    
    /// Get primary GPU (if any)
    pub fn primary_gpu(&self) -> Option<&GpuInfo> {
        self.gpus.first()
    }
    
    /// Get best compute device for a workload
    pub fn best_device_for(&self, workload: &WorkloadCharacteristics) -> DeviceId {
        // Simple heuristic - can be made more sophisticated
        if workload.requires_gpu && self.has_gpu() {
            // Check if GPU is faster for this workload
            if workload.data_size_mb > self.optimal_config.gpu_min_size_mb {
                return DeviceId::new(1); // First GPU
            }
        }
        
        DeviceId::CPU
    }
    
    /// Convert to AURORA devices
    pub fn to_devices(&self) -> Vec<Device> {
        let mut devices = Vec::new();
        
        // Add CPU device
        let cpu_device = Device::new_cpu(
            self.cpu.model_name.clone(),
            self.cpu.physical_cores,
            (self.memory.total_mb * 1024 * 1024) as usize,
        );
        devices.push(cpu_device);
        
        // Add GPU devices
        for (i, gpu) in self.gpus.iter().enumerate() {
            let device_type = match gpu.gpu_type {
                crate::gpu::GpuType::Nvidia => DeviceType::Cuda,
                crate::gpu::GpuType::Amd => DeviceType::Rocm,
                crate::gpu::GpuType::Intel => DeviceType::LevelZero,
                crate::gpu::GpuType::Vulkan => DeviceType::Vulkan,
                crate::gpu::GpuType::Unknown => DeviceType::OpenCL,
            };
            
            let compute_cap = gpu.compute_capability.map(|(major, minor)| {
                aurora_core::device::ComputeCapability::new(major as u32, minor as u32)
            });
            
            let gpu_device = Device::new_gpu(
                DeviceId::new((i + 1) as u32),
                device_type,
                gpu.name.clone(),
                compute_cap,
                (gpu.vram_mb * 1024 * 1024) as usize,
                gpu.compute_units as usize,
            );
            
            devices.push(gpu_device);
        }
        
        devices
    }
}

/// Workload characteristics for device selection
#[derive(Debug, Clone)]
pub struct WorkloadCharacteristics {
    /// Whether workload requires GPU features
    pub requires_gpu: bool,
    /// Data size in MB
    pub data_size_mb: u64,
    /// Compute intensity (FLOPs per byte)
    pub compute_intensity: f64,
    /// Memory bandwidth required (GB/s)
    pub memory_bandwidth_gbps: f64,
    /// Latency sensitivity
    pub latency_sensitive: bool,
}

impl WorkloadCharacteristics {
    /// Create a new workload description
    pub fn new() -> Self {
        Self {
            requires_gpu: false,
            data_size_mb: 0,
            compute_intensity: 0.0,
            memory_bandwidth_gbps: 0.0,
            latency_sensitive: false,
        }
    }
    
    /// Set GPU requirement
    pub fn with_gpu(mut self, requires: bool) -> Self {
        self.requires_gpu = requires;
        self
    }
    
    /// Set data size
    pub fn with_data_size(mut self, mb: u64) -> Self {
        self.data_size_mb = mb;
        self
    }
    
    /// Set compute intensity
    pub fn with_compute_intensity(mut self, intensity: f64) -> Self {
        self.compute_intensity = intensity;
        self
    }
}

impl Default for WorkloadCharacteristics {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimal configuration computed from hardware profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimalConfig {
    /// Optimal thread pool size for CPU
    pub cpu_thread_pool_size: usize,
    /// Optimal work group size for CPU kernels
    pub cpu_work_group_size: usize,
    /// Optimal work group size for GPU kernels
    pub gpu_work_group_size: u32,
    /// Minimum data size to use GPU (MB)
    pub gpu_min_size_mb: u64,
    /// Matrix size threshold for GPU
    pub gpu_matmul_threshold: usize,
    /// Batch size for operations
    pub optimal_batch_size: usize,
    /// Use NUMA-aware allocation
    pub use_numa: bool,
    /// Use HugePages
    pub use_hugepages: bool,
    /// Preferred memory type
    pub preferred_memory_type: String,
}

/// Compute optimal configuration from hardware profile
fn compute_optimal_config(cpu: &CpuInfo, gpus: &[GpuInfo], memory: &MemoryInfo) -> OptimalConfig {
    // Thread pool: physical cores * 2 for I/O overlap, or just physical cores for compute
    let cpu_thread_pool_size = cpu.physical_cores;
    
    // Work group size based on SIMD level
    let cpu_work_group_size = match cpu.simd_level {
        SimdLevel::Avx512 => 512,
        SimdLevel::Avx2 => 256,
        SimdLevel::Avx => 128,
        _ => 64,
    };
    
    // GPU work group size
    let gpu_work_group_size = if let Some(gpu) = gpus.first() {
        match gpu.gpu_type {
            crate::gpu::GpuType::Nvidia => 256,
            crate::gpu::GpuType::Amd => 256,
            _ => 128,
        }
    } else {
        128
    };
    
    // GPU threshold based on PCIe bandwidth
    let gpu_min_size_mb = 16; // Minimum 16MB to justify GPU transfer overhead
    
    // Matrix multiplication threshold
    let gpu_matmul_threshold = if gpus.is_empty() {
        usize::MAX // Never use GPU
    } else {
        512 // 512x512 matrices and larger go to GPU
    };
    
    // Batch size
    let optimal_batch_size = cpu_work_group_size * 4;
    
    // NUMA and HugePages
    let use_numa = memory.numa.len() > 1;
    let use_hugepages = memory.hugepages_available > 0;
    
    OptimalConfig {
        cpu_thread_pool_size,
        cpu_work_group_size,
        gpu_work_group_size,
        gpu_min_size_mb,
        gpu_matmul_threshold,
        optimal_batch_size,
        use_numa,
        use_hugepages,
        preferred_memory_type: if use_hugepages {
            "hugepage".to_string()
        } else {
            "pinned".to_string()
        },
    }
}

/// Profile cache for storing/loading hardware profiles
#[derive(Debug)]
pub struct ProfileCache {
    /// Cache file path
    path: String,
}

impl ProfileCache {
    /// Create a new profile cache
    pub fn new(path: &str) -> Self {
        Self {
            path: path.to_string(),
        }
    }
    
    /// Load profile from cache
    pub fn load(&self) -> Result<HardwareProfile> {
        if !Path::new(&self.path).exists() {
            return Err(AuroraError::invalid_arg("Cache file does not exist"));
        }
        
        let content = fs::read_to_string(&self.path)?;
        let profile: HardwareProfile = toml::from_str(&content)
            .map_err(|e| AuroraError::invalid_arg(format!("Failed to parse profile: {}", e)))?;
        
        // Check if profile is stale (older than 30 days)
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        
        if now - profile.timestamp > 30 * 24 * 60 * 60 {
            return Err(AuroraError::invalid_arg("Profile is stale"));
        }
        
        Ok(profile)
    }
    
    /// Save profile to cache
    pub fn save(&self, profile: &HardwareProfile) -> Result<()> {
        // Create directory if needed
        if let Some(parent) = Path::new(&self.path).parent() {
            fs::create_dir_all(parent)?;
        }
        
        let content = toml::to_string_pretty(profile)
            .map_err(|e| AuroraError::invalid_arg(format!("Failed to serialize profile: {}", e)))?;
        
        fs::write(&self.path, content)?;
        
        Ok(())
    }
    
    /// Clear the cache
    pub fn clear(&self) -> Result<()> {
        if Path::new(&self.path).exists() {
            fs::remove_file(&self.path)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_workload_characteristics() {
        let workload = WorkloadCharacteristics::new()
            .with_gpu(true)
            .with_data_size(100)
            .with_compute_intensity(10.0);
        
        assert!(workload.requires_gpu);
        assert_eq!(workload.data_size_mb, 100);
    }
}
