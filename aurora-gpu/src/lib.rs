//! AURORA GPU Engine
//!
//! GPU compute engine supporting CUDA, ROCm, Vulkan, and OpenCL backends.

#![warn(missing_docs)]

pub mod cuda;
pub mod rocm;
pub mod vulkan;
pub mod opencl;

use aurora_core::device::DeviceId;
use aurora_core::error::{AuroraError, Result};
use std::process::Command;

/// Known GPU backend kinds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendKind {
    /// NVIDIA CUDA backend.
    Cuda,
    /// AMD ROCm backend.
    Rocm,
    /// Vulkan compute backend.
    Vulkan,
    /// OpenCL backend.
    OpenCL,
}

/// Practical GPU posture for a backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendProfile {
    /// Discrete or workstation-class target.
    HighPerformance,
    /// Often available on integrated or lower-power systems.
    LowPower,
}

impl BackendKind {
    /// Stable backend display name.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Cuda => "CUDA",
            Self::Rocm => "ROCm",
            Self::Vulkan => "Vulkan",
            Self::OpenCL => "OpenCL",
        }
    }

    fn probe_command(self) -> (&'static str, &'static [&'static str]) {
        match self {
            Self::Cuda => ("nvidia-smi", &["--query-gpu=name", "--format=csv,noheader"]),
            Self::Rocm => ("rocminfo", &[]),
            Self::Vulkan => ("vulkaninfo", &["--summary"]),
            Self::OpenCL => ("clinfo", &[]),
        }
    }

    /// Whether this backend is a practical target for integrated or lower-end GPUs.
    pub const fn profile(self) -> BackendProfile {
        match self {
            Self::Cuda | Self::Rocm => BackendProfile::HighPerformance,
            Self::Vulkan | Self::OpenCL => BackendProfile::LowPower,
        }
    }
}

/// Status of a probed GPU backend on the current host.
#[derive(Debug, Clone)]
pub struct BackendStatus {
    /// Backend kind.
    pub kind: BackendKind,
    /// Whether the host exposes a usable probe command.
    pub available: bool,
    /// Human-readable diagnostic.
    pub detail: String,
}

impl BackendStatus {
    fn unavailable(kind: BackendKind, detail: impl Into<String>) -> Self {
        Self {
            kind,
            available: false,
            detail: detail.into(),
        }
    }

    fn available(kind: BackendKind, detail: impl Into<String>) -> Self {
        Self {
            kind,
            available: true,
            detail: detail.into(),
        }
    }
}

/// GPU engine for compute operations
#[derive(Debug)]
pub struct GpuEngine {
    device_id: DeviceId,
}

impl GpuEngine {
    /// Create a new GPU engine
    pub fn new(device_id: DeviceId) -> Result<Self> {
        if !device_id.is_gpu() {
            return Err(AuroraError::invalid_arg("GPU engine requires a GPU device id"));
        }

        let available = available_backends();
        if available.is_empty() {
            let details = probe_backends()
                .into_iter()
                .map(|status| format!("{}: {}", status.kind.as_str(), status.detail))
                .collect::<Vec<_>>()
                .join("; ");
            return Err(AuroraError::not_found(format!(
                "no GPU backend is available on this host ({details})"
            )));
        }

        Ok(Self { device_id })
    }
    
    /// Get device ID
    pub fn device_id(&self) -> DeviceId {
        self.device_id
    }
}

/// Probe all supported GPU backends on the current host.
pub fn probe_backends() -> Vec<BackendStatus> {
    [
        BackendKind::Cuda,
        BackendKind::Rocm,
        BackendKind::Vulkan,
        BackendKind::OpenCL,
    ]
    .into_iter()
    .map(probe_backend)
    .collect()
}

/// Return only the GPU backends that are currently available.
pub fn available_backends() -> Vec<BackendKind> {
    probe_backends()
        .into_iter()
        .filter(|status| status.available)
        .map(|status| status.kind)
        .collect()
}

/// Return available backends suitable for integrated or lower-power GPUs.
pub fn available_low_power_backends() -> Vec<BackendKind> {
    probe_backends()
        .into_iter()
        .filter(|status| status.available && status.kind.profile() == BackendProfile::LowPower)
        .map(|status| status.kind)
        .collect()
}

fn probe_backend(kind: BackendKind) -> BackendStatus {
    let (program, args) = kind.probe_command();
    match Command::new(program).args(args).output() {
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let summary = stdout
                .lines()
                .find(|line| !line.trim().is_empty())
                .map(|line| line.trim().to_string())
                .unwrap_or_else(|| "probe command succeeded".to_string());
            BackendStatus::available(kind, summary)
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let detail = if stderr.trim().is_empty() {
                format!("probe exited with status {}", output.status)
            } else {
                stderr.trim().to_string()
            };
            BackendStatus::unavailable(kind, detail)
        }
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => {
            BackendStatus::unavailable(kind, format!("{program} not installed"))
        }
        Err(err) => BackendStatus::unavailable(kind, err.to_string()),
    }
}

/// GPU memory allocator
trait GpuAllocator {
    /// Allocate device memory
    fn allocate(&self, size: usize) -> Result<u64>;
    
    /// Free device memory
    fn free(&self, ptr: u64) -> Result<()>;
    
    /// Copy host to device
    fn copy_h2d(&self, host: *const u8, device: u64, size: usize) -> Result<()>;
    
    /// Copy device to host
    fn copy_d2h(&self, device: u64, host: *mut u8, size: usize) -> Result<()>;
    
    /// Copy device to device
    fn copy_d2d(&self, src: u64, dst: u64, size: usize) -> Result<()>;
}
