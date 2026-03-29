//! AURORA Hardware Profiler
//!
//! Detects and profiles CPU, GPU, and memory hardware for optimal compute scheduling.

#![warn(missing_docs)]

pub mod cpu;
pub mod gpu;
pub mod memory;
pub mod profile;
pub mod benchmark;

pub use cpu::{CpuProfiler, CpuInfo, CpuFeatures};
pub use gpu::{GpuProfiler, GpuInfo, GpuType};
pub use memory::{MemoryProfiler, MemoryInfo, NumaInfo};
pub use profile::{HardwareProfile, ProfileCache};
pub use benchmark::{MicroBenchmark, BenchmarkResults};

use aurora_core::error::Result;
use tracing::info;

/// Main hardware profiler
#[derive(Debug)]
pub struct HardwareProfiler {
    /// CPU profiler
    cpu: CpuProfiler,
    /// GPU profiler
    gpu: GpuProfiler,
    /// Memory profiler
    memory: MemoryProfiler,
    /// Profile cache
    cache: Option<ProfileCache>,
}

impl HardwareProfiler {
    /// Create a new hardware profiler
    pub fn new() -> Self {
        Self {
            cpu: CpuProfiler::new(),
            gpu: GpuProfiler::new(),
            memory: MemoryProfiler::new(),
            cache: None,
        }
    }
    
    /// Enable profile caching
    pub fn with_cache(mut self, cache_path: &str) -> Self {
        self.cache = Some(ProfileCache::new(cache_path));
        self
    }
    
    /// Detect all hardware
    pub fn detect(&mut self) -> Result<HardwareProfile> {
        info!("Starting hardware detection...");
        
        // Try to load from cache first
        if let Some(ref cache) = self.cache {
            if let Ok(profile) = cache.load() {
                info!("Loaded hardware profile from cache");
                return Ok(profile);
            }
        }
        
        // Detect CPU
        info!("Detecting CPU...");
        let cpu_info = self.cpu.detect()?;
        
        // Detect GPUs
        info!("Detecting GPUs...");
        let gpu_infos = self.gpu.detect_all()?;
        
        // Detect memory
        info!("Detecting memory...");
        let memory_info = self.memory.detect()?;
        
        // Build hardware profile
        let profile = HardwareProfile::new(cpu_info, gpu_infos, memory_info);
        
        // Save to cache
        if let Some(ref cache) = self.cache {
            if let Err(e) = cache.save(&profile) {
                tracing::warn!("Failed to save profile cache: {}", e);
            }
        }
        
        info!("Hardware detection complete");
        Ok(profile)
    }
    
    /// Run microbenchmarks to characterize performance
    pub fn benchmark(&mut self) -> Result<BenchmarkResults> {
        info!("Running microbenchmarks...");
        
        let mut results = BenchmarkResults::new();
        
        // CPU benchmarks
        results.cpu = self.cpu.benchmark()?;
        
        // Memory benchmarks
        results.memory = self.memory.benchmark()?;
        
        // GPU benchmarks (if available)
        if self.gpu.has_gpus() {
            results.gpu = Some(self.gpu.benchmark()?);
        }
        
        info!("Microbenchmarks complete");
        Ok(results)
    }
    
    /// Get CPU profiler
    pub fn cpu(&self) -> &CpuProfiler {
        &self.cpu
    }
    
    /// Get GPU profiler
    pub fn gpu(&self) -> &GpuProfiler {
        &self.gpu
    }
    
    /// Get memory profiler
    pub fn memory(&self) -> &MemoryProfiler {
        &self.memory
    }
}

impl Default for HardwareProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Print a summary of the hardware profile
pub fn print_profile(profile: &HardwareProfile) {
    println!("╔════════════════════════════════════════════════════════════╗");
    println!("║              AURORA Hardware Profile                       ║");
    println!("╚════════════════════════════════════════════════════════════╝");
    println!();
    
    // CPU section
    println!("┌─ CPU ─────────────────────────────────────────────────────┐");
    println!("│ Model:        {}", profile.cpu.model_name);
    println!("│ Cores:        {} physical, {} logical", 
        profile.cpu.physical_cores, profile.cpu.logical_cores);
    println!("│ SIMD Level:   {}", profile.cpu.simd_level.name());
    println!("│ Frequency:    {} MHz (base) / {} MHz (boost)",
        profile.cpu.base_frequency, profile.cpu.boost_frequency);
    println!("│ Cache:        L1={}KB, L2={}KB, L3={}KB",
        profile.cpu.l1_cache / 1024,
        profile.cpu.l2_cache / 1024,
        profile.cpu.l3_cache / 1024);
    println!("│ NUMA Nodes:   {}", profile.cpu.numa_nodes);
    println!("└───────────────────────────────────────────────────────────┘");
    println!();
    
    // GPU section
    if profile.gpus.is_empty() {
        println!("┌─ GPU ─────────────────────────────────────────────────────┐");
        println!("│ No GPUs detected");
        println!("└───────────────────────────────────────────────────────────┘");
    } else {
        for (i, gpu) in profile.gpus.iter().enumerate() {
            println!("┌─ GPU {} ──────────────────────────────────────────────────┐", i);
            println!("│ Name:         {}", gpu.name);
            println!("│ Type:         {:?}", gpu.gpu_type);
            println!("│ VRAM:         {} MB", gpu.vram_mb);
            println!("│ Compute:      {} units", gpu.compute_units);
            println!("│ Memory BW:    {:.1} GB/s", gpu.memory_bandwidth_gbps);
            println!("└───────────────────────────────────────────────────────────┘");
        }
    }
    println!();
    
    // Memory section
    println!("┌─ Memory ──────────────────────────────────────────────────┐");
    println!("│ Total:        {} MB", profile.memory.total_mb);
    println!("│ Available:    {} MB", profile.memory.available_mb);
    println!("│ HugePages:    {} ({} MB)", 
        profile.memory.hugepages_available,
        profile.memory.hugepage_size_mb);
    println!("└───────────────────────────────────────────────────────────┘");
    println!();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_creation() {
        let profiler = HardwareProfiler::new();
        let features = profiler.cpu().features();
        assert!(
            features.sse2
                || features.sse3
                || features.ssse3
                || features.sse4_1
                || features.sse4_2
                || features.avx
                || features.avx2
                || features.has_avx512()
        );
    }
}
