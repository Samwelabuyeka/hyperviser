//! GPU detection and profiling

use aurora_core::error::Result;
use serde::{Deserialize, Serialize};
use std::process::Command;

/// GPU type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GpuType {
    /// NVIDIA GPU (CUDA)
    Nvidia,
    /// AMD GPU (ROCm)
    Amd,
    /// Intel GPU (Level Zero/oneAPI)
    Intel,
    /// Generic GPU (Vulkan)
    Vulkan,
    /// Unknown GPU
    Unknown,
}

impl GpuType {
    /// Get the runtime name for this GPU type
    pub fn runtime_name(&self) -> &'static str {
        match self {
            GpuType::Nvidia => "CUDA",
            GpuType::Amd => "ROCm",
            GpuType::Intel => "Level Zero",
            GpuType::Vulkan => "Vulkan",
            GpuType::Unknown => "Unknown",
        }
    }
}

/// GPU information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuInfo {
    /// GPU index
    pub index: u32,
    /// GPU type
    pub gpu_type: GpuType,
    /// GPU name
    pub name: String,
    /// VRAM size in MB
    pub vram_mb: u64,
    /// Number of compute units (SMs/CUs)
    pub compute_units: u32,
    /// Maximum clock frequency (MHz)
    pub max_clock_mhz: u32,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gbps: f32,
    /// PCIe generation
    pub pcie_gen: u8,
    /// PCIe lanes
    pub pcie_lanes: u8,
    /// Compute capability (for CUDA)
    pub compute_capability: Option<(i32, i32)>,
    /// Driver version
    pub driver_version: String,
}

impl GpuInfo {
    /// Create unknown GPU info
    pub fn unknown() -> Self {
        Self {
            index: 0,
            gpu_type: GpuType::Unknown,
            name: "Unknown GPU".to_string(),
            vram_mb: 0,
            compute_units: 0,
            max_clock_mhz: 0,
            memory_bandwidth_gbps: 0.0,
            pcie_gen: 0,
            pcie_lanes: 0,
            compute_capability: None,
            driver_version: "Unknown".to_string(),
        }
    }
    
    /// Estimate peak FP32 performance (TFLOPS)
    pub fn estimate_peak_tflops(&self) -> f32 {
        // Rough estimate: CUs * clock * ops per cycle
        // NVIDIA: 128 FLOPS/cycle per SM
        // AMD: 128 FLOPS/cycle per CU
        let ops_per_cycle = match self.gpu_type {
            GpuType::Nvidia => 128.0,
            GpuType::Amd => 128.0,
            GpuType::Intel => 128.0,
            _ => 64.0,
        };
        
        let clock_ghz = self.max_clock_mhz as f32 / 1000.0;
        self.compute_units as f32 * clock_ghz * ops_per_cycle / 1000.0
    }
    
    /// Check if this GPU is suitable for compute
    pub fn is_compute_capable(&self) -> bool {
        self.vram_mb >= 1024 && self.compute_units > 0
    }
}

/// GPU profiler
#[derive(Debug, Default)]
pub struct GpuProfiler {
    gpus: Vec<GpuInfo>,
    has_cuda: bool,
    has_rocm: bool,
    has_vulkan: bool,
}

impl GpuProfiler {
    /// Create a new GPU profiler
    pub fn new() -> Self {
        Self {
            gpus: Vec::new(),
            has_cuda: false,
            has_rocm: false,
            has_vulkan: false,
        }
    }
    
    /// Check if any GPUs are available
    pub fn has_gpus(&self) -> bool {
        !self.gpus.is_empty()
    }
    
    /// Get detected GPUs
    pub fn gpus(&self) -> &[GpuInfo] {
        &self.gpus
    }
    
    /// Detect all GPUs
    pub fn detect_all(&mut self) -> Result<Vec<GpuInfo>> {
        self.gpus.clear();
        
        // Try NVIDIA first
        if let Ok(nvidia_gpus) = detect_nvidia_gpus() {
            self.gpus.extend(nvidia_gpus);
            self.has_cuda = true;
        }
        
        // Try AMD
        if let Ok(amd_gpus) = detect_amd_gpus() {
            self.gpus.extend(amd_gpus);
            self.has_rocm = true;
        }
        
        // Try Intel
        if let Ok(intel_gpus) = detect_intel_gpus() {
            self.gpus.extend(intel_gpus);
        }
        
        // Check for Vulkan
        self.has_vulkan = check_vulkan_support();
        
        // Assign indices
        for (i, gpu) in self.gpus.iter_mut().enumerate() {
            gpu.index = i as u32;
        }
        
        Ok(self.gpus.clone())
    }
    
    /// Run GPU benchmarks
    pub fn benchmark(&self) -> Result<GpuBenchmarkResults> {
        let mut results = GpuBenchmarkResults::default();
        
        for gpu in &self.gpus {
            let gpu_result = benchmark_gpu(gpu)?;
            results.per_gpu.push(gpu_result);
        }
        
        Ok(results)
    }
}

/// Detect NVIDIA GPUs using nvidia-smi
fn detect_nvidia_gpus() -> Result<Vec<GpuInfo>> {
    let output = Command::new("nvidia-smi")
        .args(&["--query-gpu=name,memory.total,clocks.max.sm,compute_cap", 
                "--format=csv,noheader"])
        .output();
    
    match output {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut gpus = Vec::new();
            
            for (i, line) in stdout.lines().enumerate() {
                let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
                if parts.len() >= 4 {
                    let name = parts[0].to_string();
                    let vram_str = parts[1].replace(" MiB", "").replace(" MB", "");
                    let vram_mb = vram_str.parse::<u64>().unwrap_or(0);
                    let clock_str = parts[2].replace(" MHz", "");
                    let max_clock = clock_str.parse::<u32>().unwrap_or(0);
                    let cc_parts: Vec<i32> = parts[3].split('.')
                        .filter_map(|s| s.parse().ok())
                        .collect();
                    let compute_cap = if cc_parts.len() >= 2 {
                        Some((cc_parts[0], cc_parts[1]))
                    } else {
                        None
                    };
                    
                    // Estimate compute units from compute capability
                    let compute_units = estimate_nvidia_cus(&name, compute_cap);
                    
                    gpus.push(GpuInfo {
                        index: i as u32,
                        gpu_type: GpuType::Nvidia,
                        name,
                        vram_mb,
                        compute_units,
                        max_clock_mhz: max_clock,
                        memory_bandwidth_gbps: estimate_memory_bw(vram_mb),
                        pcie_gen: 4,
                        pcie_lanes: 16,
                        compute_capability: compute_cap,
                        driver_version: get_nvidia_driver_version(),
                    });
                }
            }
            
            Ok(gpus)
        }
        _ => Ok(Vec::new()),
    }
}

/// Estimate NVIDIA compute units from name and compute capability
fn estimate_nvidia_cus(name: &str, cc: Option<(i32, i32)>) -> u32 {
    // Rough estimates based on common GPUs
    if name.contains("4090") || name.contains("3090") {
        16384 / 128 // 128 CUDA cores per SM
    } else if name.contains("4080") {
        9728 / 128
    } else if name.contains("4070") {
        5888 / 128
    } else if name.contains("A100") {
        6912 / 64 // Tensor cores
    } else if name.contains("H100") {
        16896 / 128
    } else {
        // Default estimate
        match cc {
            Some((major, _)) if major >= 8 => 80,
            Some((major, _)) if major >= 7 => 64,
            _ => 32,
        }
    }
}

/// Get NVIDIA driver version
fn get_nvidia_driver_version() -> String {
    Command::new("nvidia-smi")
        .args(&["--query-gpu=driver_version", "--format=csv,noheader"])
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_else(|| "Unknown".to_string())
}

/// Detect AMD GPUs using rocminfo
fn detect_amd_gpus() -> Result<Vec<GpuInfo>> {
    // Try rocminfo first
    let output = Command::new("rocminfo")
        .output();
    
    match output {
        Ok(output) if output.status.success() => {
            // Parse rocminfo output
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut gpus = Vec::new();
            
            // Simple parsing - in production would be more robust
            if stdout.contains("AMD") || stdout.contains("gfx") {
                // Extract GPU info from rocminfo
                gpus.push(GpuInfo {
                    index: 0,
                    gpu_type: GpuType::Amd,
                    name: "AMD GPU".to_string(),
                    vram_mb: 16384, // Default estimate
                    compute_units: 60,
                    max_clock_mhz: 2000,
                    memory_bandwidth_gbps: 512.0,
                    pcie_gen: 4,
                    pcie_lanes: 16,
                    compute_capability: None,
                    driver_version: "ROCm".to_string(),
                });
            }
            
            Ok(gpus)
        }
        _ => {
            // Try lspci as fallback
            detect_amd_via_lspci()
        }
    }
}

/// Detect AMD GPUs via lspci
fn detect_amd_via_lspci() -> Result<Vec<GpuInfo>> {
    let output = Command::new("lspci")
        .args(&["-nn"])
        .output();
    
    match output {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut gpus = Vec::new();
            
            for line in stdout.lines() {
                if line.contains("VGA") && line.contains("AMD") {
                    gpus.push(GpuInfo {
                        index: gpus.len() as u32,
                        gpu_type: GpuType::Amd,
                        name: "AMD GPU".to_string(),
                        vram_mb: 8192,
                        compute_units: 40,
                        max_clock_mhz: 1800,
                        memory_bandwidth_gbps: 256.0,
                        pcie_gen: 4,
                        pcie_lanes: 16,
                        compute_capability: None,
                        driver_version: "Unknown".to_string(),
                    });
                }
            }
            
            Ok(gpus)
        }
        _ => Ok(Vec::new()),
    }
}

/// Detect Intel GPUs
fn detect_intel_gpus() -> Result<Vec<GpuInfo>> {
    let output = Command::new("lspci")
        .args(&["-nn"])
        .output();
    
    match output {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let mut gpus = Vec::new();
            
            for line in stdout.lines() {
                if line.contains("VGA") && line.contains("Intel") {
                    // Check if it's a discrete GPU (Arc)
                    let is_arc = line.contains("Arc") || line.contains("DG2");
                    
                    gpus.push(GpuInfo {
                        index: gpus.len() as u32,
                        gpu_type: GpuType::Intel,
                        name: if is_arc { "Intel Arc".to_string() } else { "Intel Integrated".to_string() },
                        vram_mb: if is_arc { 8192 } else { 1024 },
                        compute_units: if is_arc { 32 } else { 16 },
                        max_clock_mhz: if is_arc { 2200 } else { 1200 },
                        memory_bandwidth_gbps: if is_arc { 256.0 } else { 64.0 },
                        pcie_gen: 4,
                        pcie_lanes: if is_arc { 16 } else { 4 },
                        compute_capability: None,
                        driver_version: "Unknown".to_string(),
                    });
                }
            }
            
            Ok(gpus)
        }
        _ => Ok(Vec::new()),
    }
}

/// Check Vulkan support
fn check_vulkan_support() -> bool {
    Command::new("vulkaninfo")
        .arg("--summary")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Estimate memory bandwidth from VRAM size
fn estimate_memory_bw(vram_mb: u64) -> f32 {
    // Rough estimates based on VRAM size
    match vram_mb {
        0..=4096 => 128.0,
        4097..=8192 => 256.0,
        8193..=16384 => 512.0,
        16385..=24576 => 768.0,
        _ => 1000.0,
    }
}

/// GPU benchmark results
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GpuBenchmarkResults {
    /// Results per GPU
    pub per_gpu: Vec<PerGpuBenchmarkResults>,
    /// PCIe bandwidth
    pub pcie_bandwidth_gbps: f64,
}

/// Per-GPU benchmark results
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PerGpuBenchmarkResults {
    /// GPU index
    pub gpu_index: u32,
    /// Memory bandwidth achieved
    pub memory_bandwidth_gbps: f64,
    /// Compute performance (FP32)
    pub compute_fp32_tflops: f64,
    /// Compute performance (FP16)
    pub compute_fp16_tflops: f64,
    /// Kernel launch overhead
    pub launch_overhead_us: f64,
}

/// Benchmark a specific GPU
fn benchmark_gpu(gpu: &GpuInfo) -> Result<PerGpuBenchmarkResults> {
    // Placeholder - real implementation would run actual GPU benchmarks
    Ok(PerGpuBenchmarkResults {
        gpu_index: gpu.index,
        memory_bandwidth_gbps: gpu.memory_bandwidth_gbps as f64 * 0.8,
        compute_fp32_tflops: gpu.estimate_peak_tflops() as f64 * 0.7,
        compute_fp16_tflops: gpu.estimate_peak_tflops() as f64 * 1.4,
        launch_overhead_us: 5.0,
    })
}
