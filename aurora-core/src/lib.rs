//! AURORA Core - Foundation Types and Abstractions
//! 
//! Production-ready core types for the AURORA compute runtime.

#![warn(missing_docs)]
#![deny(unsafe_code)]

pub mod device;
pub mod error;
pub mod tensor;
pub mod types;
pub mod kernel;
pub mod graph;
pub mod memory;

pub use device::{Device, DeviceId, DeviceType, ComputeCapability, SimdLevel};
pub use error::{AuroraError, Result};
pub use tensor::{Tensor, TensorShape, DataType, Layout};
pub use types::{Scalar, Dim, Range, Padding, ConvParams, PoolParams};
pub use tensor::Stride;
pub use kernel::{Kernel, KernelId, KernelSignature, LaunchConfig, KernelType, BinaryOp, UnaryOp, ReduceOp, PoolType};
pub use graph::{ComputeGraph, NodeId, OpType, GraphExecutor, ExecutionProfile};
pub use memory::{MemoryPool, Allocation, MemoryType, MemoryStats};

use std::sync::atomic::{AtomicBool, Ordering};

/// Version information for AURORA
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Build information
pub mod build_info {
    /// Git commit hash
    pub const GIT_COMMIT: &str = match option_env!("GIT_COMMIT") {
        Some(value) => value,
        None => "unknown",
    };
    /// Build timestamp
    pub const BUILD_TIME: &str = match option_env!("BUILD_TIME") {
        Some(value) => value,
        None => "unknown",
    };
    /// Target architecture
    pub const TARGET: &str = match option_env!("TARGET") {
        Some(value) => value,
        None => "unknown",
    };
    /// Build profile
    pub const PROFILE: &str = match option_env!("PROFILE") {
        Some(value) => value,
        None => "unknown",
    };
    /// Rust version
    pub const RUST_VERSION: &str = match option_env!("RUSTC_VERSION") {
        Some(value) => value,
        None => "unknown",
    };
}

static INITIALIZED: AtomicBool = AtomicBool::new(false);

/// Initialize the AURORA runtime
/// 
/// # Errors
/// Returns error if already initialized or if setup fails
pub fn initialize() -> Result<()> {
    if INITIALIZED.swap(true, Ordering::SeqCst) {
        return Err(AuroraError::AlreadyInitialized);
    }
    
    tracing::info!("Initializing AURORA Runtime v{}", VERSION);
    
    // Set up panic handler
    std::panic::set_hook(Box::new(|info| {
        tracing::error!("AURORA Panic: {}", info);
        std::process::exit(1);
    }));
    
    // Log build info
    tracing::debug!("Build: {} @ {}", build_info::GIT_COMMIT, build_info::BUILD_TIME);
    tracing::debug!("Target: {} ({})", build_info::TARGET, build_info::PROFILE);
    
    Ok(())
}

/// Check if AURORA is initialized
pub fn is_initialized() -> bool {
    INITIALIZED.load(Ordering::SeqCst)
}

/// Shutdown the AURORA runtime gracefully
/// 
/// # Errors
/// Returns error if not initialized
pub fn shutdown() -> Result<()> {
    if !INITIALIZED.swap(false, Ordering::SeqCst) {
        return Err(AuroraError::NotInitialized);
    }
    
    tracing::info!("Shutting down AURORA Runtime");
    Ok(())
}

/// Format bytes to human-readable string
pub fn format_bytes(bytes: u64) -> String {
    const UNITS: &[&str] = &["B", "KB", "MB", "GB", "TB", "PB"];
    let mut size = bytes as f64;
    let mut unit_idx = 0;
    
    while size >= 1024.0 && unit_idx < UNITS.len() - 1 {
        size /= 1024.0;
        unit_idx += 1;
    }
    
    format!("{:.2} {}", size, UNITS[unit_idx])
}

/// Format FLOPS to human-readable string
pub fn format_flops(flops: f64) -> String {
    if flops >= 1e12 {
        format!("{:.2} TFLOPS", flops / 1e12)
    } else if flops >= 1e9 {
        format!("{:.2} GFLOPS", flops / 1e9)
    } else if flops >= 1e6 {
        format!("{:.2} MFLOPS", flops / 1e6)
    } else {
        format!("{:.2} FLOPS", flops)
    }
}

/// Round up to alignment
pub const fn align_up(size: usize, align: usize) -> usize {
    (size + align - 1) & !(align - 1)
}

/// Check if value is power of 2
pub const fn is_power_of_2(n: usize) -> bool {
    n != 0 && (n & (n - 1)) == 0
}

/// Get next power of 2
pub const fn next_power_of_2(n: usize) -> usize {
    if n <= 1 {
        return 1;
    }
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(512), "512.00 B");
        assert_eq!(format_bytes(1024), "1.00 KB");
        assert_eq!(format_bytes(1024 * 1024), "1.00 MB");
        assert_eq!(format_bytes(1024 * 1024 * 1024), "1.00 GB");
    }

    #[test]
    fn test_format_flops() {
        assert_eq!(format_flops(1e6), "1.00 MFLOPS");
        assert_eq!(format_flops(1e9), "1.00 GFLOPS");
        assert_eq!(format_flops(1e12), "1.00 TFLOPS");
    }

    #[test]
    fn test_align_up() {
        assert_eq!(align_up(5, 8), 8);
        assert_eq!(align_up(8, 8), 8);
        assert_eq!(align_up(9, 8), 16);
        assert_eq!(align_up(64, 64), 64);
    }

    #[test]
    fn test_power_of_2() {
        assert!(is_power_of_2(1));
        assert!(is_power_of_2(2));
        assert!(is_power_of_2(4));
        assert!(is_power_of_2(1024));
        assert!(!is_power_of_2(3));
        assert!(!is_power_of_2(100));
    }

    #[test]
    fn test_next_power_of_2() {
        assert_eq!(next_power_of_2(0), 1);
        assert_eq!(next_power_of_2(1), 1);
        assert_eq!(next_power_of_2(5), 8);
        assert_eq!(next_power_of_2(100), 128);
    }
}
