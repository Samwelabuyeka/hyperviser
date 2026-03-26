//! AURORA Orchestrator
//!
//! Hybrid execution orchestrator that decides whether to run workloads
//! on CPU, GPU, or both based on hardware capabilities and workload characteristics.

#![warn(missing_docs)]

use aurora_core::device::DeviceId;
use aurora_core::error::Result;
use aurora_core::graph::ComputeGraph;
use aurora_core::kernel::KernelType;
use aurora_core::tensor::Tensor;
use aurora_profiler::profile::{HardwareProfile, WorkloadCharacteristics};
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use tracing::{info, debug, warn};

/// Execution strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionStrategy {
    /// Run on CPU only
    CpuOnly,
    /// Run on GPU only
    GpuOnly,
    /// Split work between CPU and GPU
    Hybrid,
    /// Auto-detect best strategy
    Auto,
}

/// Workload split configuration
#[derive(Debug, Clone)]
pub struct WorkloadSplit {
    /// Percentage of work for CPU (0-100)
    pub cpu_percent: u8,
    /// Percentage of work for GPU (0-100)
    pub gpu_percent: u8,
    /// Split dimension (for tensor operations)
    pub split_dim: Option<usize>,
}

impl WorkloadSplit {
    /// Create a CPU-only split
    pub fn cpu_only() -> Self {
        Self {
            cpu_percent: 100,
            gpu_percent: 0,
            split_dim: None,
        }
    }
    
    /// Create a GPU-only split
    pub fn gpu_only() -> Self {
        Self {
            cpu_percent: 0,
            gpu_percent: 100,
            split_dim: None,
        }
    }
    
    /// Create a 50/50 hybrid split
    pub fn hybrid_equal() -> Self {
        Self {
            cpu_percent: 50,
            gpu_percent: 50,
            split_dim: Some(0),
        }
    }
    
    /// Create a custom split
    pub fn new(cpu_percent: u8, gpu_percent: u8) -> Self {
        assert_eq!(cpu_percent + gpu_percent, 100, "Percentages must sum to 100");
        Self {
            cpu_percent,
            gpu_percent,
            split_dim: Some(0),
        }
    }
    
    /// Validate the split
    pub fn is_valid(&self) -> bool {
        self.cpu_percent + self.gpu_percent == 100
    }
}

/// Execution plan for a workload
#[derive(Debug, Clone)]
pub struct ExecutionPlan {
    /// Target strategy
    pub strategy: ExecutionStrategy,
    /// Workload split
    pub split: WorkloadSplit,
    /// Target device for each operation
    pub device_assignments: HashMap<u64, DeviceId>,
    /// Estimated execution time (microseconds)
    pub estimated_time_us: u64,
    /// Estimated memory usage (bytes)
    pub estimated_memory_bytes: usize,
}

impl ExecutionPlan {
    /// Create a new execution plan
    pub fn new(strategy: ExecutionStrategy) -> Self {
        Self {
            strategy,
            split: WorkloadSplit::cpu_only(),
            device_assignments: HashMap::new(),
            estimated_time_us: 0,
            estimated_memory_bytes: 0,
        }
    }
    
    /// Set workload split
    pub fn with_split(mut self, split: WorkloadSplit) -> Self {
        self.split = split;
        self
    }
    
    /// Add device assignment
    pub fn assign_device(mut self, op_id: u64, device: DeviceId) -> Self {
        self.device_assignments.insert(op_id, device);
        self
    }
    
    /// Estimate execution time
    pub fn with_estimated_time(mut self, time_us: u64) -> Self {
        self.estimated_time_us = time_us;
        self
    }
}

/// Performance model for device selection
#[derive(Debug, Clone)]
pub struct PerformanceModel {
    /// CPU compute performance (GFLOPS)
    pub cpu_gflops: f64,
    /// GPU compute performance (TFLOPS)
    pub gpu_tflops: f64,
    /// CPU memory bandwidth (GB/s)
    pub cpu_bw_gbps: f64,
    /// GPU memory bandwidth (GB/s)
    pub gpu_bw_gbps: f64,
    /// PCIe bandwidth (GB/s)
    pub pcie_bw_gbps: f64,
    /// GPU kernel launch overhead (microseconds)
    pub gpu_launch_overhead_us: f64,
    /// Data transfer overhead per MB (microseconds)
    pub transfer_overhead_us_per_mb: f64,
}

impl PerformanceModel {
    /// Create a performance model from hardware profile
    pub fn from_profile(profile: &HardwareProfile) -> Self {
        let cpu_gflops = profile.cpu.estimate_peak_gflops();
        
        let (gpu_tflops, gpu_bw) = profile.primary_gpu()
            .map(|g| (g.estimate_peak_tflops() as f64, g.memory_bandwidth_gbps as f64))
            .unwrap_or((0.0, 0.0));
        
        Self {
            cpu_gflops,
            gpu_tflops,
            cpu_bw_gbps: profile.memory.estimated_bandwidth_gbps,
            gpu_bw_gbps: gpu_bw,
            pcie_bw_gbps: 16.0, // PCIe 4.0 x16
            gpu_launch_overhead_us: 5.0,
            transfer_overhead_us_per_mb: 10.0,
        }
    }
    
    /// Estimate time for CPU execution
    pub fn estimate_cpu_time(&self, flops: u64, data_mb: f64) -> f64 {
        let compute_time_us = (flops as f64 / self.cpu_gflops) / 1e3;
        let memory_time_us = (data_mb / self.cpu_bw_gbps) * 1e6;
        compute_time_us.max(memory_time_us)
    }
    
    /// Estimate time for GPU execution
    pub fn estimate_gpu_time(&self, flops: u64, data_mb: f64) -> f64 {
        if self.gpu_tflops == 0.0 {
            return f64::INFINITY;
        }
        
        let transfer_time_us = data_mb * self.transfer_overhead_us_per_mb;
        let compute_time_us = (flops as f64 / (self.gpu_tflops * 1e3)) / 1e3;
        let memory_time_us = (data_mb / self.gpu_bw_gbps) * 1e6;
        
        transfer_time_us + self.gpu_launch_overhead_us + compute_time_us.max(memory_time_us)
    }
    
    /// Estimate time for hybrid execution
    pub fn estimate_hybrid_time(&self, flops: u64, data_mb: f64, cpu_percent: u8) -> f64 {
        let cpu_ratio = cpu_percent as f64 / 100.0;
        let gpu_ratio = 1.0 - cpu_ratio;
        
        let cpu_flops = (flops as f64 * cpu_ratio) as u64;
        let gpu_flops = (flops as f64 * gpu_ratio) as u64;
        let cpu_data = data_mb * cpu_ratio;
        let gpu_data = data_mb * gpu_ratio;
        
        let cpu_time = self.estimate_cpu_time(cpu_flops, cpu_data);
        let gpu_time = self.estimate_gpu_time(gpu_flops, gpu_data);
        
        // Parallel execution - take the max
        cpu_time.max(gpu_time)
    }
}

/// Hybrid orchestrator
pub struct Orchestrator {
    /// Hardware profile
    profile: Arc<RwLock<HardwareProfile>>,
    /// Performance model
    perf_model: Arc<RwLock<PerformanceModel>>,
    /// Execution history for adaptive tuning
    execution_history: Arc<RwLock<Vec<ExecutionRecord>>>,
    /// Adaptive tuning enabled
    adaptive: bool,
}

/// Execution record for adaptive tuning
#[derive(Debug, Clone)]
struct ExecutionRecord {
    /// Workload characteristics
    workload: WorkloadCharacteristics,
    /// Strategy used
    strategy: ExecutionStrategy,
    /// Actual execution time
    actual_time_us: u64,
    /// Predicted execution time
    predicted_time_us: u64,
}

impl Orchestrator {
    /// Create a new orchestrator
    pub fn new(profile: HardwareProfile) -> Self {
        let perf_model = PerformanceModel::from_profile(&profile);
        
        info!("Orchestrator initialized:");
        info!("  CPU: {:.1} GFLOPS, {:.1} GB/s", perf_model.cpu_gflops, perf_model.cpu_bw_gbps);
        if profile.has_gpu() {
            info!("  GPU: {:.1} TFLOPS, {:.1} GB/s", perf_model.gpu_tflops, perf_model.gpu_bw_gbps);
        }
        
        Self {
            profile: Arc::new(RwLock::new(profile)),
            perf_model: Arc::new(RwLock::new(perf_model)),
            execution_history: Arc::new(RwLock::new(Vec::new())),
            adaptive: true,
        }
    }
    
    /// Disable adaptive tuning
    pub fn disable_adaptive(mut self) -> Self {
        self.adaptive = false;
        self
    }
    
    /// Update hardware profile
    pub fn update_profile(&self, profile: HardwareProfile) {
        let mut p = self.profile.write();
        let mut pm = self.perf_model.write();
        
        *pm = PerformanceModel::from_profile(&profile);
        *p = profile;
    }
    
    /// Decide execution strategy for a workload
    pub fn decide_strategy(&self, workload: &WorkloadCharacteristics) -> ExecutionPlan {
        let profile = self.profile.read();
        let perf_model = self.perf_model.read();
        
        // Check if GPU is available
        if !profile.has_gpu() || workload.data_size_mb < profile.optimal_config.gpu_min_size_mb {
            return ExecutionPlan::new(ExecutionStrategy::CpuOnly)
                .with_split(WorkloadSplit::cpu_only())
                .with_estimated_time(perf_model.estimate_cpu_time(
                    workload.compute_intensity as u64 * workload.data_size_mb as u64,
                    workload.data_size_mb as f64
                ) as u64);
        }
        
        // Estimate times for different strategies
        let flops = (workload.compute_intensity * workload.data_size_mb as f64) as u64;
        let data_mb = workload.data_size_mb as f64;
        
        let cpu_time = perf_model.estimate_cpu_time(flops, data_mb);
        let gpu_time = perf_model.estimate_gpu_time(flops, data_mb);
        let hybrid_time = perf_model.estimate_hybrid_time(flops, data_mb, 30); // 30% CPU
        
        debug!("Estimated times - CPU: {:.1}us, GPU: {:.1}us, Hybrid: {:.1}us",
            cpu_time, gpu_time, hybrid_time);
        
        // Choose best strategy
        let (strategy, split, time) = if gpu_time < cpu_time * 0.8 && gpu_time < hybrid_time {
            (ExecutionStrategy::GpuOnly, WorkloadSplit::gpu_only(), gpu_time)
        } else if hybrid_time < cpu_time && hybrid_time < gpu_time {
            (ExecutionStrategy::Hybrid, WorkloadSplit::new(30, 70), hybrid_time)
        } else {
            (ExecutionStrategy::CpuOnly, WorkloadSplit::cpu_only(), cpu_time)
        };
        
        ExecutionPlan::new(strategy)
            .with_split(split)
            .with_estimated_time(time as u64)
    }
    
    /// Decide execution strategy for a compute graph
    pub fn plan_graph_execution(&self, graph: &ComputeGraph) -> ExecutionPlan {
        let profile = self.profile.read();
        
        // Estimate graph characteristics
        let mut total_flops: u64 = 0;
        let mut total_data_mb: u64 = 0;
        
        for (_, node) in graph.nodes() {
            total_flops += node.compute_cost;
            total_data_mb += node.memory_cost as u64 / (1024 * 1024);
        }
        
        let workload = WorkloadCharacteristics::new()
            .with_gpu(true)
            .with_data_size(total_data_mb)
            .with_compute_intensity(total_flops as f64 / total_data_mb.max(1) as f64);
        
        self.decide_strategy(&workload)
    }
    
    /// Record execution result for adaptive tuning
    pub fn record_execution(
        &self,
        workload: &WorkloadCharacteristics,
        strategy: ExecutionStrategy,
        predicted_time_us: u64,
        actual_time_us: u64,
    ) {
        if !self.adaptive {
            return;
        }
        
        let record = ExecutionRecord {
            workload: workload.clone(),
            strategy,
            actual_time_us,
            predicted_time_us,
        };
        
        let mut history = self.execution_history.write();
        history.push(record);
        
        // Keep history bounded
        if history.len() > 1000 {
            history.remove(0);
        }
        
        // Check if model needs adjustment
        let error = (actual_time_us as f64 - predicted_time_us as f64).abs() 
            / predicted_time_us as f64;
        
        if error > 0.3 {
            warn!("Performance model error: {:.1}%, consider re-tuning", error * 100.0);
        }
    }
    
    /// Get execution statistics
    pub fn get_statistics(&self) -> OrchestratorStats {
        let history = self.execution_history.read();
        let profile = self.profile.read();
        
        let total_executions = history.len();
        let cpu_executions = history.iter().filter(|r| r.strategy == ExecutionStrategy::CpuOnly).count();
        let gpu_executions = history.iter().filter(|r| r.strategy == ExecutionStrategy::GpuOnly).count();
        let hybrid_executions = history.iter().filter(|r| r.strategy == ExecutionStrategy::Hybrid).count();
        
        let avg_prediction_error = if total_executions > 0 {
            history.iter()
                .map(|r| (r.actual_time_us as f64 - r.predicted_time_us as f64).abs() 
                    / r.predicted_time_us as f64)
                .sum::<f64>() / total_executions as f64
        } else {
            0.0
        };
        
        OrchestratorStats {
            total_executions,
            cpu_executions,
            gpu_executions,
            hybrid_executions,
            avg_prediction_error,
            has_gpu: profile.has_gpu(),
        }
    }
}

/// Orchestrator statistics
#[derive(Debug, Clone)]
pub struct OrchestratorStats {
    /// Total number of executions
    pub total_executions: usize,
    /// CPU-only executions
    pub cpu_executions: usize,
    /// GPU-only executions
    pub gpu_executions: usize,
    /// Hybrid executions
    pub hybrid_executions: usize,
    /// Average prediction error (0-1)
    pub avg_prediction_error: f64,
    /// Whether GPU is available
    pub has_gpu: bool,
}

impl OrchestratorStats {
    /// Get CPU usage percentage
    pub fn cpu_percent(&self) -> f64 {
        if self.total_executions == 0 {
            return 0.0;
        }
        (self.cpu_executions as f64 / self.total_executions as f64) * 100.0
    }
    
    /// Get GPU usage percentage
    pub fn gpu_percent(&self) -> f64 {
        if self.total_executions == 0 {
            return 0.0;
        }
        (self.gpu_executions as f64 / self.total_executions as f64) * 100.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aurora_profiler::cpu::CpuInfo;
    use aurora_profiler::memory::MemoryInfo;

    fn create_test_profile() -> HardwareProfile {
        HardwareProfile::new(
            CpuInfo::default(),
            vec![],
            MemoryInfo::default(),
        )
    }

    #[test]
    fn test_workload_split() {
        let split = WorkloadSplit::new(30, 70);
        assert_eq!(split.cpu_percent, 30);
        assert_eq!(split.gpu_percent, 70);
        assert!(split.is_valid());
    }

    #[test]
    fn test_decide_strategy() {
        let profile = create_test_profile();
        let orchestrator = Orchestrator::new(profile);
        
        let workload = WorkloadCharacteristics::new()
            .with_data_size(100)
            .with_compute_intensity(1000.0);
        
        let plan = orchestrator.decide_strategy(&workload);
        assert_eq!(plan.strategy, ExecutionStrategy::CpuOnly);
    }
}
