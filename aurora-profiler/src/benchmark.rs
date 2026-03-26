//! Microbenchmarks for hardware characterization

use aurora_core::error::Result;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};

use crate::cpu::CpuBenchmarkResults;
use crate::gpu::GpuBenchmarkResults;
use crate::memory::MemoryBenchmarkResults;

/// Microbenchmark trait
pub trait MicroBenchmark {
    /// Benchmark name
    fn name(&self) -> &str;
    
    /// Run the benchmark
    fn run(&self) -> Result<BenchmarkResult>;
    
    /// Get estimated duration
    fn estimated_duration(&self) -> Duration;
}

/// Single benchmark result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    /// Benchmark name
    pub name: String,
    /// Metric name (e.g., "bandwidth", "latency")
    pub metric: String,
    /// Value
    pub value: f64,
    /// Unit
    pub unit: String,
    /// Execution time
    pub duration_ms: f64,
    /// Number of iterations
    pub iterations: usize,
}

impl BenchmarkResult {
    /// Create a new benchmark result
    pub fn new(name: impl Into<String>, metric: impl Into<String>, value: f64, unit: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            metric: metric.into(),
            value,
            unit: unit.into(),
            duration_ms: 0.0,
            iterations: 1,
        }
    }
    
    /// Set duration
    pub fn with_duration(mut self, ms: f64) -> Self {
        self.duration_ms = ms;
        self
    }
    
    /// Set iterations
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }
}

/// Complete benchmark results
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct BenchmarkResults {
    /// CPU benchmark results
    pub cpu: CpuBenchmarkResults,
    /// GPU benchmark results
    pub gpu: Option<GpuBenchmarkResults>,
    /// Memory benchmark results
    pub memory: MemoryBenchmarkResults,
    /// Individual benchmark results
    pub individual: Vec<BenchmarkResult>,
    /// Total benchmark time
    pub total_duration_ms: f64,
}

impl BenchmarkResults {
    /// Create new benchmark results
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Add an individual result
    pub fn add_result(&mut self, result: BenchmarkResult) {
        self.individual.push(result);
    }
    
    /// Get result by name
    pub fn get(&self, name: &str) -> Option<&BenchmarkResult> {
        self.individual.iter().find(|r| r.name == name)
    }
    
    /// Get CPU memory bandwidth
    pub fn cpu_memory_bandwidth_gbps(&self) -> f64 {
        self.memory.sequential_read_gbps
    }
    
    /// Get CPU compute performance
    pub fn cpu_compute_gflops(&self) -> f64 {
        self.cpu.matmul_gflops
    }
    
    /// Get GPU compute performance (if available)
    pub fn gpu_compute_tflops(&self) -> Option<f64> {
        self.gpu.as_ref().map(|g| {
            g.per_gpu.first().map(|p| p.compute_fp32_tflops).unwrap_or(0.0)
        })
    }
}

/// Memory bandwidth benchmark
pub struct MemoryBandwidthBenchmark {
    /// Buffer size in bytes
    buffer_size: usize,
    /// Number of iterations
    iterations: usize,
}

impl MemoryBandwidthBenchmark {
    /// Create a new memory bandwidth benchmark
    pub fn new() -> Self {
        Self {
            buffer_size: 256 * 1024 * 1024, // 256MB
            iterations: 10,
        }
    }
    
    /// Set buffer size
    pub fn with_buffer_size(mut self, bytes: usize) -> Self {
        self.buffer_size = bytes;
        self
    }
    
    /// Set iterations
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }
}

impl MicroBenchmark for MemoryBandwidthBenchmark {
    fn name(&self) -> &str {
        "memory_bandwidth"
    }
    
    fn run(&self) -> Result<BenchmarkResult> {
        let mut buffer = vec![0u8; self.buffer_size];
        
        // Initialize with pattern
        for i in 0..self.buffer_size {
            buffer[i] = (i % 256) as u8;
        }
        
        let mut sum: u64 = 0;
        let start = Instant::now();
        
        for _ in 0..self.iterations {
            // Sequential read
            for chunk in buffer.chunks_exact(64) {
                sum = sum.wrapping_add(chunk[0] as u64);
            }
            std::hint::black_box(sum);
        }
        
        let elapsed = start.elapsed();
        let seconds = elapsed.as_secs_f64();
        let bytes_processed = (self.buffer_size * self.iterations) as f64;
        let bandwidth_gbps = bytes_processed / seconds / 1e9;
        
        Ok(BenchmarkResult::new(
            "memory_bandwidth",
            "bandwidth",
            bandwidth_gbps,
            "GB/s"
        ).with_duration(elapsed.as_millis() as f64)
         .with_iterations(self.iterations))
    }
    
    fn estimated_duration(&self) -> Duration {
        // Rough estimate: 10GB/s bandwidth
        let bytes = self.buffer_size * self.iterations;
        let seconds = bytes as f64 / 10e9;
        Duration::from_secs_f64(seconds * 1.5) // Add 50% margin
    }
}

/// Matrix multiplication benchmark
pub struct MatmulBenchmark {
    /// Matrix size (N x N)
    size: usize,
    /// Number of iterations
    iterations: usize,
}

impl MatmulBenchmark {
    /// Create a new matrix multiplication benchmark
    pub fn new() -> Self {
        Self {
            size: 512,
            iterations: 5,
        }
    }
    
    /// Set matrix size
    pub fn with_size(mut self, size: usize) -> Self {
        self.size = size;
        self
    }
    
    /// Set iterations
    pub fn with_iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }
}

impl MicroBenchmark for MicroBenchmark for MatmulBenchmark {
    fn name(&self) -> &str {
        "matmul"
    }
    
    fn run(&self) -> Result<BenchmarkResult> {
        let n = self.size;
        let a = vec![1.0f32; n * n];
        let b = vec![1.0f32; n * n];
        let mut c = vec![0.0f32; n * n];
        
        let start = Instant::now();
        
        for _ in 0..self.iterations {
            // Naive matrix multiplication
            for i in 0..n {
                for j in 0..n {
                    let mut sum = 0.0f32;
                    for k in 0..n {
                        sum += a[i * n + k] * b[k * n + j];
                    }
                    c[i * n + j] = sum;
                }
            }
            std::hint::black_box(&c);
        }
        
        let elapsed = start.elapsed();
        let seconds = elapsed.as_secs_f64();
        let flops = 2.0 * (n as f64).powi(3) * self.iterations as f64;
        let gflops = flops / seconds / 1e9;
        
        Ok(BenchmarkResult::new(
            "matmul",
            "compute",
            gflops,
            "GFLOPS"
        ).with_duration(elapsed.as_millis() as f64)
         .with_iterations(self.iterations))
    }
    
    fn estimated_duration(&self) -> Duration {
        // Rough estimate: 10 GFLOPS
        let flops = 2.0 * (self.size as f64).powi(3) * self.iterations as f64;
        let seconds = flops / 10e9;
        Duration::from_secs_f64(seconds * 1.5)
    }
}

/// Latency benchmark
pub struct LatencyBenchmark {
    /// Number of random accesses
    num_accesses: usize,
}

impl LatencyBenchmark {
    /// Create a new latency benchmark
    pub fn new() -> Self {
        Self {
            num_accesses: 1_000_000,
        }
    }
    
    /// Set number of accesses
    pub fn with_accesses(mut self, accesses: usize) -> Self {
        self.num_accesses = accesses;
        self
    }
}

impl MicroBenchmark for LatencyBenchmark {
    fn name(&self) -> &str {
        "memory_latency"
    }
    
    fn run(&self) -> Result<BenchmarkResult> {
        const BUFFER_SIZE: usize = 64 * 1024 * 1024; // 64MB
        
        // Create pointer-chasing buffer
        let mut buffer = vec![0u32; BUFFER_SIZE / 4];
        let num_pointers = buffer.len();
        
        // Initialize with random pattern
        for i in 0..num_pointers {
            buffer[i] = ((i + 1) % num_pointers) as u32;
        }
        
        // Shuffle
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        for i in 0..num_pointers {
            let mut hasher = DefaultHasher::new();
            i.hash(&mut hasher);
            let j = (hasher.finish() as usize) % num_pointers;
            buffer.swap(i, j);
        }
        
        // Chase pointers
        let mut idx = 0usize;
        let start = Instant::now();
        
        for _ in 0..self.num_accesses {
            idx = buffer[idx] as usize;
        }
        
        let elapsed = start.elapsed();
        std::hint::black_box(idx);
        
        let nanos = elapsed.as_nanos() as f64;
        let latency_ns = nanos / self.num_accesses as f64;
        
        Ok(BenchmarkResult::new(
            "memory_latency",
            "latency",
            latency_ns,
            "ns"
        ).with_duration(elapsed.as_millis() as f64)
         .with_iterations(self.num_accesses))
    }
    
    fn estimated_duration(&self) -> Duration {
        // Rough estimate: 100ns per access
        let nanos = self.num_accesses as f64 * 100.0;
        Duration::from_nanos(nanos as u64)
    }
}

/// Run all benchmarks
pub fn run_all_benchmarks() -> Result<BenchmarkResults> {
    let start = Instant::now();
    let mut results = BenchmarkResults::new();
    
    // Memory bandwidth
    let mem_bw = MemoryBandwidthBenchmark::new();
    results.add_result(mem_bw.run()?);
    
    // Matrix multiplication
    let matmul = MatmulBenchmark::new();
    results.add_result(matmul.run()?);
    
    // Latency
    let latency = LatencyBenchmark::new();
    results.add_result(latency.run()?);
    
    results.total_duration_ms = start.elapsed().as_millis() as f64;
    
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_benchmark_result() {
        let result = BenchmarkResult::new("test", "metric", 100.0, "units")
            .with_duration(10.0)
            .with_iterations(5);
        
        assert_eq!(result.name, "test");
        assert_eq!(result.value, 100.0);
    }
}
