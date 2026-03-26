//! Memory and NUMA detection

use aurora_core::error::Result;
use serde::{Deserialize, Serialize};
use std::fs;

/// Memory information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryInfo {
    /// Total physical memory (MB)
    pub total_mb: u64,
    /// Available memory (MB)
    pub available_mb: u64,
    /// Free memory (MB)
    pub free_mb: u64,
    /// Buffers/cache (MB)
    pub buffers_mb: u64,
    /// Swap total (MB)
    pub swap_total_mb: u64,
    /// Swap free (MB)
    pub swap_free_mb: u64,
    /// HugePages available
    pub hugepages_available: u64,
    /// HugePage size (MB)
    pub hugepage_size_mb: u64,
    /// Transparent HugePages enabled
    pub thp_enabled: bool,
    /// NUMA information
    pub numa: Vec<NumaInfo>,
    /// Memory bandwidth estimate (GB/s)
    pub estimated_bandwidth_gbps: f64,
}

impl MemoryInfo {
    /// Create default memory info
    pub fn default() -> Self {
        Self {
            total_mb: 8192,
            available_mb: 4096,
            free_mb: 2048,
            buffers_mb: 1024,
            swap_total_mb: 4096,
            swap_free_mb: 4096,
            hugepages_available: 0,
            hugepage_size_mb: 2,
            thp_enabled: false,
            numa: vec![NumaInfo::default()],
            estimated_bandwidth_gbps: 25.0,
        }
    }
    
    /// Get used memory
    pub fn used_mb(&self) -> u64 {
        self.total_mb - self.available_mb
    }
    
    /// Get memory usage percentage
    pub fn usage_percent(&self) -> f64 {
        if self.total_mb == 0 {
            return 0.0;
        }
        (self.used_mb() as f64 / self.total_mb as f64) * 100.0
    }
    
    /// Check if system has enough memory for workload
    pub fn has_enough_memory(&self, required_mb: u64) -> bool {
        self.available_mb >= required_mb
    }
    
    /// Get total NUMA-local memory
    pub fn numa_local_memory_mb(&self) -> u64 {
        self.numa.iter().map(|n| n.total_mb).sum()
    }
}

/// NUMA node information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NumaInfo {
    /// Node ID
    pub node_id: i32,
    /// Total memory on this node (MB)
    pub total_mb: u64,
    /// Free memory on this node (MB)
    pub free_mb: u64,
    /// CPU cores on this node
    pub cpus: Vec<usize>,
    /// Distance to other nodes
    pub distances: Vec<u32>,
}

impl NumaInfo {
    /// Create default NUMA info
    pub fn default() -> Self {
        Self {
            node_id: 0,
            total_mb: 8192,
            free_mb: 4096,
            cpus: (0..num_cpus::get()).collect(),
            distances: vec![10],
        }
    }
}

/// Memory profiler
#[derive(Debug, Default)]
pub struct MemoryProfiler;

impl MemoryProfiler {
    /// Create a new memory profiler
    pub fn new() -> Self {
        Self
    }
    
    /// Detect memory configuration
    pub fn detect(&self) -> Result<MemoryInfo> {
        let mut info = MemoryInfo::default();
        
        // Read /proc/meminfo
        if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if parts.len() >= 2 {
                    let value_kb = parts[1].parse::<u64>().unwrap_or(0);
                    let value_mb = value_kb / 1024;
                    
                    match parts[0] {
                        "MemTotal:" => info.total_mb = value_mb,
                        "MemAvailable:" => info.available_mb = value_mb,
                        "MemFree:" => info.free_mb = value_mb,
                        "Buffers:" => info.buffers_mb = value_mb,
                        "Cached:" => info.buffers_mb += value_mb,
                        "SwapTotal:" => info.swap_total_mb = value_mb,
                        "SwapFree:" => info.swap_free_mb = value_mb,
                        "HugePages_Total:" => info.hugepages_available = parts[1].parse().unwrap_or(0),
                        "Hugepagesize:" => info.hugepage_size_mb = value_kb / 1024,
                        _ => {}
                    }
                }
            }
        }
        
        // Detect NUMA topology
        info.numa = detect_numa_topology()?;
        
        // Check Transparent HugePages
        info.thp_enabled = check_transparent_hugepages();
        
        // Estimate memory bandwidth
        info.estimated_bandwidth_gbps = estimate_memory_bandwidth(&info);
        
        Ok(info)
    }
    
    /// Run memory benchmarks
    pub fn benchmark(&self) -> Result<MemoryBenchmarkResults> {
        use std::time::Instant;
        
        let mut results = MemoryBenchmarkResults::default();
        
        // Sequential read bandwidth
        results.sequential_read_gbps = benchmark_sequential_read()?;
        
        // Random read latency
        results.random_read_latency_ns = benchmark_random_latency()?;
        
        // NUMA bandwidth
        results.numa_bandwidth_gbps = benchmark_numa_bandwidth()?;
        
        Ok(results)
    }
}

/// Detect NUMA topology
fn detect_numa_topology() -> Result<Vec<NumaInfo>> {
    let mut nodes = Vec::new();
    
    // Check if NUMA is available
    let numa_path = "/sys/devices/system/node/";
    if let Ok(entries) = fs::read_dir(numa_path) {
        for entry in entries.filter_map(|e| e.ok()) {
            let name = entry.file_name();
            let name_str = name.to_string_lossy();
            
            if name_str.starts_with("node") {
                if let Ok(node_id) = name_str[4..].parse::<i32>() {
                    let mut node = NumaInfo {
                        node_id,
                        ..NumaInfo::default()
                    };
                    
                    // Read memory info for this node
                    let meminfo_path = format!("{}/node{}/meminfo", numa_path, node_id);
                    if let Ok(meminfo) = fs::read_to_string(&meminfo_path) {
                        for line in meminfo.lines() {
                            let parts: Vec<&str> = line.split_whitespace().collect();
                            if parts.len() >= 4 {
                                let value_kb = parts[3].parse::<u64>().unwrap_or(0);
                                match parts[2] {
                                    "MemTotal:" => node.total_mb = value_kb / 1024,
                                    "MemFree:" => node.free_mb = value_kb / 1024,
                                    _ => {}
                                }
                            }
                        }
                    }
                    
                    // Read CPU list for this node
                    let cpulist_path = format!("{}/node{}/cpulist", numa_path, node_id);
                    if let Ok(cpulist) = fs::read_to_string(&cpulist_path) {
                        node.cpus = parse_cpulist(&cpulist);
                    }
                    
                    // Read distance matrix
                    let distance_path = format!("{}/node{}/distance", numa_path, node_id);
                    if let Ok(distance) = fs::read_to_string(&distance_path) {
                        node.distances = distance
                            .split_whitespace()
                            .filter_map(|s| s.parse().ok())
                            .collect();
                    }
                    
                    nodes.push(node);
                }
            }
        }
    }
    
    if nodes.is_empty() {
        nodes.push(NumaInfo::default());
    }
    
    Ok(nodes)
}

/// Parse CPU list (e.g., "0-3,5,7-9")
fn parse_cpulist(cpulist: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    
    for part in cpulist.trim().split(',') {
        if part.contains('-') {
            let range: Vec<&str> = part.split('-').collect();
            if range.len() == 2 {
                if let (Ok(start), Ok(end)) = (range[0].parse::<usize>(), range[1].parse::<usize>()) {
                    cpus.extend(start..=end);
                }
            }
        } else if let Ok(cpu) = part.parse::<usize>() {
            cpus.push(cpu);
        }
    }
    
    cpus
}

/// Check if Transparent HugePages is enabled
fn check_transparent_hugepages() -> bool {
    fs::read_to_string("/sys/kernel/mm/transparent_hugepage/enabled")
        .map(|s| s.contains("[always]") || s.contains("[madvise]"))
        .unwrap_or(false)
}

/// Estimate memory bandwidth based on system info
fn estimate_memory_bandwidth(info: &MemoryInfo) -> f64 {
    // Estimate based on total memory and NUMA configuration
    let channels = (info.total_mb / 4096).max(1).min(8) as f64;
    let ddr_version = if info.total_mb > 65536 { 5.0 } else { 4.0 };
    let base_bw_per_channel = if ddr_version >= 5.0 { 38.4 } else { 25.6 };
    
    channels * base_bw_per_channel
}

/// Memory benchmark results
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryBenchmarkResults {
    /// Sequential read bandwidth (GB/s)
    pub sequential_read_gbps: f64,
    /// Sequential write bandwidth (GB/s)
    pub sequential_write_gbps: f64,
    /// Random read latency (ns)
    pub random_read_latency_ns: f64,
    /// NUMA local bandwidth (GB/s)
    pub numa_bandwidth_gbps: f64,
    /// NUMA remote bandwidth (GB/s)
    pub numa_remote_bandwidth_gbps: f64,
}

/// Benchmark sequential read bandwidth
fn benchmark_sequential_read() -> Result<f64> {
    use std::time::Instant;
    
    const SIZE: usize = 256 * 1024 * 1024; // 256MB
    const ITERATIONS: usize = 5;
    
    // Allocate and initialize buffer
    let mut buffer = vec![0u8; SIZE];
    for i in 0..SIZE {
        buffer[i] = (i % 256) as u8;
    }
    
    let mut sum: u64 = 0;
    let start = Instant::now();
    
    for _ in 0..ITERATIONS {
        for chunk in buffer.chunks_exact(64) {
            sum = sum.wrapping_add(chunk[0] as u64);
        }
        std::hint::black_box(sum);
    }
    
    let elapsed = start.elapsed();
    let seconds = elapsed.as_secs_f64();
    let bytes = (SIZE * ITERATIONS) as f64;
    
    Ok(bytes / seconds / 1e9)
}

/// Benchmark random read latency
fn benchmark_random_latency() -> Result<f64> {
    use std::time::Instant;
    
    const SIZE: usize = 64 * 1024 * 1024; // 64MB
    const ACCESSES: usize = 1000000;
    
    // Create a linked list pattern through buffer
    let mut buffer = vec![0u32; SIZE / 4];
    let num_pointers = buffer.len();
    
    // Create random access pattern
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
    
    for _ in 0..ACCESSES {
        idx = buffer[idx] as usize;
    }
    
    let elapsed = start.elapsed();
    std::hint::black_box(idx);
    
    let nanos = elapsed.as_nanos() as f64;
    Ok(nanos / ACCESSES as f64)
}

/// Benchmark NUMA bandwidth
fn benchmark_numa_bandwidth() -> Result<f64> {
    // Placeholder - real implementation would test NUMA-local vs remote
    Ok(25.0)
}
