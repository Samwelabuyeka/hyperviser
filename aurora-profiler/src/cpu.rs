//! CPU detection and profiling

use aurora_core::device::SimdLevel;
use aurora_core::error::{AuroraError, Result};
use raw_cpuid::CpuId;
use serde::{Deserialize, Serialize};
use std::fs;

/// CPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpuInfo {
    /// CPU model name
    pub model_name: String,
    /// Vendor ID
    pub vendor_id: String,
    /// Physical cores
    pub physical_cores: usize,
    /// Logical cores (including hyperthreads)
    pub logical_cores: usize,
    /// SIMD instruction level
    pub simd_level: SimdLevel,
    /// CPU features
    pub features: CpuFeatures,
    /// L1 cache size per core (bytes)
    pub l1_cache: usize,
    /// L2 cache size per core (bytes)
    pub l2_cache: usize,
    /// L3 cache size (bytes)
    pub l3_cache: usize,
    /// Number of NUMA nodes
    pub numa_nodes: usize,
    /// Base frequency (MHz)
    pub base_frequency: u32,
    /// Boost frequency (MHz)
    pub boost_frequency: u32,
}

impl CpuInfo {
    /// Create default CPU info
    pub fn default() -> Self {
        Self {
            model_name: "Unknown".to_string(),
            vendor_id: "Unknown".to_string(),
            physical_cores: 1,
            logical_cores: 1,
            simd_level: SimdLevel::Scalar,
            features: CpuFeatures::default(),
            l1_cache: 32 * 1024,
            l2_cache: 256 * 1024,
            l3_cache: 8 * 1024 * 1024,
            numa_nodes: 1,
            base_frequency: 2000,
            boost_frequency: 2000,
        }
    }
    
    /// Check if CPU supports AVX-512
    pub fn has_avx512(&self) -> bool {
        matches!(self.simd_level, SimdLevel::Avx512)
    }
    
    /// Check if CPU supports AVX2
    pub fn has_avx2(&self) -> bool {
        matches!(self.simd_level, SimdLevel::Avx2 | SimdLevel::Avx512)
    }
    
    /// Check if CPU supports AVX
    pub fn has_avx(&self) -> bool {
        matches!(self.simd_level, SimdLevel::Avx | SimdLevel::Avx2 | SimdLevel::Avx512)
    }
    
    /// Get total cache size
    pub fn total_cache(&self) -> usize {
        self.l1_cache * self.physical_cores +
        self.l2_cache * self.physical_cores +
        self.l3_cache
    }
    
    /// Estimate peak FP32 performance (GFLOPS)
    pub fn estimate_peak_gflops(&self) -> f64 {
        // Rough estimate: cores * frequency * ops per cycle
        // AVX2: 16 FLOPS/cycle (8-wide * 2 FMA)
        // AVX-512: 32 FLOPS/cycle (16-wide * 2 FMA)
        let ops_per_cycle = match self.simd_level {
            SimdLevel::Avx512 => 32.0,
            SimdLevel::Avx2 => 16.0,
            SimdLevel::Avx => 8.0,
            SimdLevel::Sse4_2 | SimdLevel::Sse2 => 4.0,
            _ => 1.0,
        };
        
        let freq_ghz = self.base_frequency as f64 / 1000.0;
        self.physical_cores as f64 * freq_ghz * ops_per_cycle
    }
}

/// CPU feature flags
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CpuFeatures {
    /// SSE2 support
    pub sse2: bool,
    /// SSE3 support
    pub sse3: bool,
    /// SSSE3 support
    pub ssse3: bool,
    /// SSE4.1 support
    pub sse4_1: bool,
    /// SSE4.2 support
    pub sse4_2: bool,
    /// AVX support
    pub avx: bool,
    /// AVX2 support
    pub avx2: bool,
    /// AVX-512F support
    pub avx512f: bool,
    /// AVX-512VL support
    pub avx512vl: bool,
    /// AVX-512BW support
    pub avx512bw: bool,
    /// AVX-512DQ support
    pub avx512dq: bool,
    /// FMA3 support
    pub fma: bool,
    /// BMI1 support
    pub bmi1: bool,
    /// BMI2 support
    pub bmi2: bool,
    /// LZCNT support
    pub lzcnt: bool,
    /// POPCNT support
    pub popcnt: bool,
    /// AES-NI support
    pub aes: bool,
    /// SHA support
    pub sha: bool,
    /// RDRAND support
    pub rdrand: bool,
    /// RDSEED support
    pub rdseed: bool,
}

impl CpuFeatures {
    /// Check if any AVX-512 extensions are supported
    pub fn has_avx512(&self) -> bool {
        self.avx512f || self.avx512vl || self.avx512bw || self.avx512dq
    }
}

/// CPU profiler
#[derive(Debug)]
pub struct CpuProfiler {
    features: CpuFeatures,
}

impl CpuProfiler {
    /// Create a new CPU profiler
    pub fn new() -> Self {
        Self {
            features: detect_features(),
        }
    }
    
    /// Get detected features
    pub fn features(&self) -> &CpuFeatures {
        &self.features
    }
    
    /// Detect CPU information
    pub fn detect(&self) -> Result<CpuInfo> {
        let cpuid = CpuId::new();
        
        // Get vendor and brand
        let vendor = cpuid.get_vendor_info()
            .map(|v| v.as_str().to_string())
            .unwrap_or_else(|| "Unknown".to_string());
        
        let brand = cpuid.get_processor_brand_string()
            .map(|b| b.as_str().to_string())
            .unwrap_or_else(|| "Unknown".to_string());
        
        // Get core counts
        let (physical_cores, logical_cores) = get_core_counts(&cpuid);
        
        // Get cache info
        let (l1_cache, l2_cache, l3_cache) = get_cache_info(&cpuid);
        
        // Get frequency info
        let (base_freq, boost_freq) = get_frequency_info(&cpuid);
        
        // Determine SIMD level
        let simd_level = determine_simd_level(&self.features);
        
        // Get NUMA info
        let numa_nodes = get_numa_nodes();
        
        Ok(CpuInfo {
            model_name: brand,
            vendor_id: vendor,
            physical_cores,
            logical_cores,
            simd_level,
            features: self.features.clone(),
            l1_cache,
            l2_cache,
            l3_cache,
            numa_nodes,
            base_frequency: base_freq,
            boost_frequency: boost_freq,
        })
    }
    
    /// Run CPU benchmarks
    pub fn benchmark(&self) -> Result<CpuBenchmarkResults> {
        use std::time::Instant;
        
        let mut results = CpuBenchmarkResults::default();
        
        // Memory bandwidth benchmark
        results.memory_bandwidth_gbps = benchmark_memory_bandwidth()?;
        
        // Matrix multiplication benchmark
        results.matmul_gflops = benchmark_matmul(&self.features)?;
        
        // Vector operation benchmark
        results.vector_gflops = benchmark_vector_ops(&self.features)?;
        
        Ok(results)
    }
}

impl Default for CpuProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Detect CPU features
fn detect_features() -> CpuFeatures {
    let cpuid = CpuId::new();
    let mut features = CpuFeatures::default();
    
    if let Some(finfo) = cpuid.get_feature_info() {
        features.sse2 = finfo.has_sse2();
        features.sse3 = finfo.has_sse3();
        features.ssse3 = finfo.has_ssse3();
        features.sse4_1 = finfo.has_sse41();
        features.sse4_2 = finfo.has_sse42();
        features.avx = finfo.has_avx();
        features.fma = finfo.has_fma();
        features.popcnt = finfo.has_popcnt();
        features.aes = finfo.has_aes();
    }
    
    if let Some(efinfo) = cpuid.get_extended_feature_info() {
        features.avx2 = efinfo.has_avx2();
        features.bmi1 = efinfo.has_bmi1();
        features.bmi2 = efinfo.has_bmi2();
        features.lzcnt = efinfo.has_lzcnt();
        features.sha = efinfo.has_sha();
    }
    
    if let Some(sinfo) = cpuid.get_extended_state_info() {
        features.avx512f = sinfo.has_avx512f();
        features.avx512vl = sinfo.has_avx512vl();
        features.avx512bw = sinfo.has_avx512bw();
        features.avx512dq = sinfo.has_avx512dq();
    }
    
    features
}

/// Determine SIMD level from features
fn determine_simd_level(features: &CpuFeatures) -> SimdLevel {
    if features.avx512f {
        SimdLevel::Avx512
    } else if features.avx2 {
        SimdLevel::Avx2
    } else if features.avx {
        SimdLevel::Avx
    } else if features.sse4_2 {
        SimdLevel::Sse4_2
    } else if features.sse2 {
        SimdLevel::Sse2
    } else {
        SimdLevel::Scalar
    }
}

/// Get core counts from CPUID
fn get_core_counts(cpuid: &CpuId) -> (usize, usize) {
    let physical = cpuid.get_cpu_identification()
        .map(|id| id.pkg_nr_cores() as usize)
        .or_else(|| cpuid.get_feature_info().map(|f| f.max_logical_cpus() as usize))
        .unwrap_or(1);
    
    let logical = cpuid.get_feature_info()
        .map(|f| f.max_logical_cpus() as usize)
        .unwrap_or(physical);
    
    (physical, logical)
}

/// Get cache information
fn get_cache_info(cpuid: &CpuId) -> (usize, usize, usize) {
    let mut l1 = 32 * 1024; // Default 32KB L1
    let mut l2 = 256 * 1024; // Default 256KB L2
    let mut l3 = 8 * 1024 * 1024; // Default 8MB L3
    
    if let Some(cache_info) = cpuid.get_cache_parameters() {
        for cache in cache_info {
            match cache.level() {
                1 => l1 = cache.sets() * cache.associativity() * cache.coherency_line_size(),
                2 => l2 = cache.sets() * cache.associativity() * cache.coherency_line_size(),
                3 => l3 = cache.sets() * cache.associativity() * cache.coherency_line_size(),
                _ => {}
            }
        }
    }
    
    (l1, l2, l3)
}

/// Get frequency information
fn get_frequency_info(cpuid: &CpuId) -> (u32, u32) {
    let base = cpuid.get_processor_frequency_info()
        .map(|f| f.processor_base_frequency())
        .unwrap_or(2000);
    
    let max = cpuid.get_processor_frequency_info()
        .map(|f| f.processor_max_frequency())
        .unwrap_or(base);
    
    (base, max)
}

/// Get NUMA node count
fn get_numa_nodes() -> usize {
    // Try to read from /sys/devices/system/node/
    match fs::read_dir("/sys/devices/system/node/") {
        Ok(entries) => {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.file_name()
                        .to_str()
                        .map(|s| s.starts_with("node"))
                        .unwrap_or(false)
                })
                .count()
                .max(1)
        }
        Err(_) => 1,
    }
}

/// CPU benchmark results
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CpuBenchmarkResults {
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f64,
    /// Matrix multiplication performance in GFLOPS
    pub matmul_gflops: f64,
    /// Vector operation performance in GFLOPS
    pub vector_gflops: f64,
    /// Latency in nanoseconds
    pub latency_ns: f64,
}

/// Benchmark memory bandwidth
fn benchmark_memory_bandwidth() -> Result<f64> {
    use std::time::Instant;
    
    const SIZE: usize = 64 * 1024 * 1024; // 64MB
    const ITERATIONS: usize = 10;
    
    let mut data = vec![0u8; SIZE];
    let mut sum: u64 = 0;
    
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        for chunk in data.chunks_exact(64) {
            sum = sum.wrapping_add(chunk[0] as u64);
        }
        // Prevent optimization
        std::hint::black_box(sum);
    }
    let elapsed = start.elapsed();
    
    let seconds = elapsed.as_secs_f64();
    let bytes_processed = (SIZE * ITERATIONS) as f64;
    let bandwidth_gbps = bytes_processed / seconds / 1e9;
    
    Ok(bandwidth_gbps)
}

/// Benchmark matrix multiplication
fn benchmark_matmul(features: &CpuFeatures) -> Result<f64> {
    use std::time::Instant;
    
    const N: usize = 512;
    const ITERATIONS: usize = 5;
    
    // Simple matrix multiplication benchmark
    let a = vec![1.0f32; N * N];
    let b = vec![1.0f32; N * N];
    let mut c = vec![0.0f32; N * N];
    
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        for i in 0..N {
            for j in 0..N {
                let mut sum = 0.0f32;
                for k in 0..N {
                    sum += a[i * N + k] * b[k * N + j];
                }
                c[i * N + j] = sum;
            }
        }
        std::hint::black_box(&c);
    }
    let elapsed = start.elapsed();
    
    let seconds = elapsed.as_secs_f64();
    let flops = 2.0 * (N as f64).powi(3) * ITERATIONS as f64;
    let gflops = flops / seconds / 1e9;
    
    Ok(gflops)
}

/// Benchmark vector operations
fn benchmark_vector_ops(features: &CpuFeatures) -> Result<f64> {
    use std::time::Instant;
    
    const SIZE: usize = 16 * 1024 * 1024; // 16M elements
    const ITERATIONS: usize = 10;
    
    let a = vec![1.0f32; SIZE];
    let b = vec![2.0f32; SIZE];
    let mut c = vec![0.0f32; SIZE];
    
    let start = Instant::now();
    for _ in 0..ITERATIONS {
        for i in 0..SIZE {
            c[i] = a[i] + b[i];
        }
        std::hint::black_box(&c);
    }
    let elapsed = start.elapsed();
    
    let seconds = elapsed.as_secs_f64();
    let flops = SIZE as f64 * ITERATIONS as f64;
    let gflops = flops / seconds / 1e9;
    
    Ok(gflops)
}
