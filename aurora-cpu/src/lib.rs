//! AURORA CPU Engine
//!
//! High-performance CPU compute engine with SIMD multi-versioning,
//! work-stealing scheduler, and NUMA-aware memory allocation.

#![warn(missing_docs)]

pub mod scheduler;
pub mod simd;
pub mod numa;
pub mod thread_pool;

use aurora_core::device::{Device, SimdLevel as CoreSimdLevel};
use aurora_core::error::Result;
use aurora_profiler::cpu::CpuInfo;
use tracing::info;

pub use aurora_core::device::SimdLevel;
pub use simd::{SimdDispatcher, VectorOps, ScalarOps, Sse2Ops, AvxOps, Avx2Ops, Avx512Ops};
pub use thread_pool::{ThreadPool, Task, TaskId};

/// CPU compute engine
pub struct CpuEngine {
    /// Device info
    device: Device,
    /// CPU information
    cpu_info: CpuInfo,
    /// Thread pool for parallel execution
    thread_pool: ThreadPool,
    /// SIMD dispatcher
    simd_dispatcher: SimdDispatcher,
}

impl CpuEngine {
    /// Create a new CPU engine
    pub fn new(cpu_info: CpuInfo) -> Result<Self> {
        let device = Device::new_cpu(
            cpu_info.model_name.clone(),
            cpu_info.physical_cores,
            cpu_info.l3_cache * cpu_info.physical_cores + cpu_info.l3_cache,
        );
        
        // Create thread pool
        let thread_pool = ThreadPool::new(cpu_info.physical_cores)?;
        
        // Create SIMD dispatcher
        let simd_level = detect_simd_level(&cpu_info);
        let simd_dispatcher = SimdDispatcher::new(simd_level);
        
        info!("CPU Engine initialized: {} cores, {:?}", 
            cpu_info.physical_cores, 
            simd_level);
        
        Ok(Self {
            device,
            cpu_info,
            thread_pool,
            simd_dispatcher,
        })
    }
    
    /// Get the device
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    /// Get CPU info
    pub fn cpu_info(&self) -> &CpuInfo {
        &self.cpu_info
    }
    
    /// Get thread pool
    pub fn thread_pool(&self) -> &ThreadPool {
        &self.thread_pool
    }
    
    /// Get SIMD dispatcher
    pub fn simd_dispatcher(&self) -> &SimdDispatcher {
        &self.simd_dispatcher
    }
    
    /// Get optimal SIMD implementation
    pub fn vector_ops(&self) -> Box<dyn VectorOps> {
        self.simd_dispatcher.get_impl()
    }
    
    /// Execute a function in parallel
    pub fn parallel_for<F>(&self, range: std::ops::Range<usize>, func: F) -> Result<()>
    where
        F: Fn(usize) + Send + Sync + 'static,
    {
        self.thread_pool.parallel_for(range, func)
    }
    
    /// Get optimal SIMD level
    pub fn simd_level(&self) -> SimdLevel {
        self.simd_dispatcher.level()
    }
    
    /// Shutdown the engine
    pub fn shutdown(&mut self) -> Result<()> {
        info!("Shutting down CPU Engine");
        self.thread_pool.shutdown()?;
        Ok(())
    }
}

impl std::fmt::Debug for CpuEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CpuEngine")
            .field("device", &self.device)
            .field("simd_level", &self.simd_dispatcher.level())
            .field("threads", &self.thread_pool.num_workers())
            .finish()
    }
}

/// Detect SIMD level from CPU info
fn detect_simd_level(cpu_info: &CpuInfo) -> SimdLevel {
    match cpu_info.simd_level {
        CoreSimdLevel::Avx512 => SimdLevel::Avx512,
        CoreSimdLevel::Avx2 => SimdLevel::Avx2,
        CoreSimdLevel::Avx => SimdLevel::Avx,
        CoreSimdLevel::Sse4_2 | CoreSimdLevel::Sse2 => SimdLevel::Sse2,
        _ => SimdLevel::Scalar,
    }
}

/// CPU performance metrics
#[derive(Debug, Clone, Default)]
pub struct CpuMetrics {
    /// Total tasks executed
    pub tasks_executed: u64,
    /// Total FLOPs performed
    pub total_flops: u64,
    /// Average task latency (microseconds)
    pub avg_latency_us: f64,
    /// Cache hit rate (percentage)
    pub cache_hit_rate: f64,
    /// Memory bandwidth utilization (GB/s)
    pub memory_bandwidth_gbps: f64,
}

impl CpuMetrics {
    /// Create new metrics
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Record task execution
    pub fn record_task(&mut self, flops: u64, latency_us: f64) {
        self.tasks_executed += 1;
        self.total_flops += flops;
        
        // Update running average
        let n = self.tasks_executed as f64;
        self.avg_latency_us = (self.avg_latency_us * (n - 1.0) + latency_us) / n;
    }
    
    /// Get average GFLOPS
    pub fn avg_gflops(&self) -> f64 {
        if self.avg_latency_us == 0.0 {
            return 0.0;
        }
        (self.total_flops as f64 / self.avg_latency_us) / 1e3
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_level() {
        assert_eq!(SimdLevel::Avx512.vector_width(), 64);
        assert_eq!(SimdLevel::Avx2.vector_width(), 32);
        assert_eq!(SimdLevel::Avx2.f32_width(), 8);
    }

    #[test]
    fn test_cpu_metrics() {
        let mut metrics = CpuMetrics::new();
        metrics.record_task(1_000_000, 1000.0);
        metrics.record_task(2_000_000, 2000.0);
        
        assert_eq!(metrics.tasks_executed, 2);
        assert_eq!(metrics.total_flops, 3_000_000);
    }
}
