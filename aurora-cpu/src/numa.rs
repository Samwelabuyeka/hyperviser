//! Minimal NUMA helpers used by the CPU engine.

use aurora_core::error::Result;

/// Snapshot of NUMA-related tuning information.
#[derive(Debug, Clone, Default)]
pub struct NumaTopology {
    /// Number of NUMA nodes detected.
    pub nodes: usize,
    /// Whether NUMA-aware placement should be enabled.
    pub enabled: bool,
}

/// Detect NUMA topology from the host.
pub fn detect_topology() -> Result<NumaTopology> {
    #[cfg(target_os = "linux")]
    {
        let path = std::path::Path::new("/sys/devices/system/node");
        if path.exists() {
            let nodes = std::fs::read_dir(path)?
                .filter_map(|entry| entry.ok())
                .filter(|entry| {
                    entry
                        .file_name()
                        .to_string_lossy()
                        .starts_with("node")
                })
                .count()
                .max(1);
            return Ok(NumaTopology {
                nodes,
                enabled: nodes > 1,
            });
        }
    }

    Ok(NumaTopology {
        nodes: 1,
        enabled: false,
    })
}
