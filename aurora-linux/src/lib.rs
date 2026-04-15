//! AURORA Linux Integration
//!
//! Linux kernel integration including CPU affinity, scheduler control,
//! HugePages, NUMA binding, and performance governor management.

#![warn(missing_docs)]

use aurora_core::error::{AuroraError, Result};
use std::fs;
use std::path::Path;

/// Set CPU affinity for the current thread.
pub fn set_cpu_affinity(cpus: &[usize]) -> Result<()> {
    if cpus.is_empty() {
        return Err(AuroraError::invalid_arg("cpu affinity list cannot be empty"));
    }

    // SAFETY: `cpu_set_t` is zero-initialized and passed to the libc CPU_* APIs
    // using the correct size for the current platform.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        libc::CPU_ZERO(&mut set);
        for &cpu in cpus {
            libc::CPU_SET(cpu, &mut set);
        }

        let result = libc::sched_setaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &set);
        if result != 0 {
            return Err(AuroraError::IoError(std::io::Error::last_os_error().to_string()));
        }
    }

    Ok(())
}

/// Get current CPU affinity.
pub fn get_cpu_affinity() -> Result<Vec<usize>> {
    // SAFETY: `cpu_set_t` is zero-initialized and the libc call writes into it
    // using the correct size.
    unsafe {
        let mut set: libc::cpu_set_t = std::mem::zeroed();
        let result = libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set);
        if result != 0 {
            return Err(AuroraError::IoError(std::io::Error::last_os_error().to_string()));
        }

        let mut cpus = Vec::new();
        let limit = libc::CPU_SETSIZE as usize;
        for cpu in 0..limit {
            if libc::CPU_ISSET(cpu, &set) {
                cpus.push(cpu);
            }
        }
        Ok(cpus)
    }
}

/// Set process priority (nice value).
pub fn set_priority(priority: i32) -> Result<()> {
    // SAFETY: `setpriority` is called for the current process only.
    let result = unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, priority) };
    if result != 0 {
        return Err(AuroraError::IoError(std::io::Error::last_os_error().to_string()));
    }
    Ok(())
}

/// Set CPU frequency governor across online CPUs.
pub fn set_governor(governor: &str) -> Result<()> {
    let cpu_root = Path::new("/sys/devices/system/cpu");
    let mut changed = false;

    for entry in fs::read_dir(cpu_root)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("cpu") || name[3..].chars().any(|c| !c.is_ascii_digit()) {
            continue;
        }

        let path = entry.path().join("cpufreq/scaling_governor");
        if path.exists() {
            write_path(&path, governor)?;
            changed = true;
        }
    }

    if !changed {
        return Err(AuroraError::not_found(
            "no writable CPU governor controls were found",
        ));
    }

    Ok(())
}

/// Enable or disable CPU idle states on Intel idle-managed systems.
pub fn set_idle_states(enable: bool) -> Result<()> {
    let path = Path::new("/sys/module/intel_idle/parameters/max_cstate");
    if !path.exists() {
        return Err(AuroraError::not_found(
            "intel_idle max_cstate control is not available on this host",
        ));
    }

    let value = if enable { "9" } else { "1" };
    write_path(path, value)
}

/// Configure Transparent HugePages.
pub fn configure_thp(mode: &str) -> Result<()> {
    let enabled = Path::new("/sys/kernel/mm/transparent_hugepage/enabled");
    let defrag = Path::new("/sys/kernel/mm/transparent_hugepage/defrag");

    if !enabled.exists() {
        return Err(AuroraError::not_found(
            "transparent hugepage controls are not available",
        ));
    }

    write_path(enabled, mode)?;
    if defrag.exists() {
        let defrag_mode = if mode == "always" { "always" } else { "madvise" };
        write_path(defrag, defrag_mode)?;
    }
    Ok(())
}

/// Reserve HugePages.
pub fn reserve_hugepages(count: usize) -> Result<()> {
    write_path(Path::new("/proc/sys/vm/nr_hugepages"), &count.to_string())
}

/// Set vm.swappiness.
pub fn set_swappiness(value: u32) -> Result<()> {
    write_path(Path::new("/proc/sys/vm/swappiness"), &value.to_string())
}

/// Set a block-device read ahead value in kilobytes.
pub fn set_readahead_kb(value: u32) -> Result<()> {
    let block_root = Path::new("/sys/block");
    let mut changed = false;

    for entry in fs::read_dir(block_root)? {
        let entry = entry?;
        let path = entry.path().join("queue/read_ahead_kb");
        if path.exists() {
            write_path(&path, &value.to_string())?;
            changed = true;
        }
    }

    if !changed {
        return Err(AuroraError::not_found(
            "no writable block read_ahead_kb controls were found",
        ));
    }

    Ok(())
}

/// Set I/O scheduler across block devices when supported.
pub fn set_io_scheduler(scheduler: &str) -> Result<()> {
    let block_root = Path::new("/sys/block");
    let mut changed = false;

    for entry in fs::read_dir(block_root)? {
        let entry = entry?;
        let path = entry.path().join("queue/scheduler");
        if !path.exists() {
            continue;
        }

        let contents = fs::read_to_string(&path)?;
        if contents.split_whitespace().any(|token| token.trim_matches(['[', ']']) == scheduler) {
            write_path(&path, scheduler)?;
            changed = true;
        }
    }

    if !changed {
        return Err(AuroraError::not_found(format!(
            "no block device accepted the '{}' scheduler",
            scheduler
        )));
    }

    Ok(())
}

/// Get system load average.
pub fn load_average() -> Result<(f64, f64, f64)> {
    let mut values = [0f64; 3];
    // SAFETY: `values` points to valid writable memory for three f64 values.
    let count = unsafe { libc::getloadavg(values.as_mut_ptr(), 3) };
    if count != 3 {
        return Err(AuroraError::IoError(std::io::Error::last_os_error().to_string()));
    }
    Ok((values[0], values[1], values[2]))
}

/// Get memory statistics from `/proc/meminfo`.
pub fn memory_stats() -> Result<MemoryStats> {
    let meminfo = fs::read_to_string("/proc/meminfo")?;
    let mut stats = MemoryStats::default();

    for line in meminfo.lines() {
        let mut parts = line.split_whitespace();
        let key = parts.next().unwrap_or_default().trim_end_matches(':');
        let value_kb = parts
            .next()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(0);
        let value = value_kb * 1024;

        match key {
            "MemTotal" => stats.total = value,
            "MemFree" => stats.free = value,
            "MemAvailable" => stats.available = value,
            "Buffers" => stats.buffers = value,
            "Cached" => stats.cached = value,
            "SwapTotal" => stats.swap_total = value,
            "SwapFree" => stats.swap_free = value,
            _ => {}
        }
    }

    Ok(stats)
}

/// NUMA topology snapshot.
#[derive(Debug, Clone, Default)]
pub struct NumaTopology {
    /// Detected NUMA node ids.
    pub nodes: Vec<NumaNode>,
}

/// NUMA node description.
#[derive(Debug, Clone, Default)]
pub struct NumaNode {
    /// Node id.
    pub id: usize,
    /// CPUs that belong to this node.
    pub cpus: Vec<usize>,
}

/// Discover NUMA nodes from sysfs.
pub fn numa_topology() -> Result<NumaTopology> {
    let root = Path::new("/sys/devices/system/node");
    if !root.exists() {
        return Ok(NumaTopology::default());
    }

    let mut nodes = Vec::new();
    for entry in fs::read_dir(root)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if !name.starts_with("node") {
            continue;
        }
        let node_id = name[4..]
            .parse::<usize>()
            .map_err(|err| AuroraError::invalid_arg(format!("invalid NUMA node '{}': {}", name, err)))?;
        let cpulist = entry.path().join("cpulist");
        let cpus = if cpulist.exists() {
            parse_cpu_list(&fs::read_to_string(cpulist)?)
        } else {
            Vec::new()
        };
        nodes.push(NumaNode { id: node_id, cpus });
    }
    nodes.sort_by_key(|node| node.id);
    Ok(NumaTopology { nodes })
}

/// Return online CPUs, ordered by NUMA node when available.
pub fn preferred_cpu_order() -> Result<Vec<usize>> {
    let topo = numa_topology()?;
    if topo.nodes.is_empty() {
        return get_cpu_affinity();
    }

    let mut ordered = Vec::new();
    let max_len = topo.nodes.iter().map(|node| node.cpus.len()).max().unwrap_or(0);
    for idx in 0..max_len {
        for node in &topo.nodes {
            if let Some(&cpu) = node.cpus.get(idx) {
                ordered.push(cpu);
            }
        }
    }
    if ordered.is_empty() {
        get_cpu_affinity()
    } else {
        Ok(ordered)
    }
}

fn write_path(path: &Path, value: &str) -> Result<()> {
    fs::write(path, value).map_err(|err| AuroraError::IoError(format!("{}: {}", path.display(), err)))
}

fn parse_cpu_list(raw: &str) -> Vec<usize> {
    let mut cpus = Vec::new();
    for part in raw.trim().split(',').filter(|part| !part.is_empty()) {
        if let Some((start, end)) = part.split_once('-') {
            let start = start.trim().parse::<usize>().ok();
            let end = end.trim().parse::<usize>().ok();
            if let (Some(start), Some(end)) = (start, end) {
                cpus.extend(start..=end);
            }
        } else if let Ok(cpu) = part.trim().parse::<usize>() {
            cpus.push(cpu);
        }
    }
    cpus
}

/// Memory statistics.
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total memory in bytes.
    pub total: u64,
    /// Free memory in bytes.
    pub free: u64,
    /// Available memory in bytes.
    pub available: u64,
    /// Buffers in bytes.
    pub buffers: u64,
    /// Cached memory in bytes.
    pub cached: u64,
    /// Swap total in bytes.
    pub swap_total: u64,
    /// Swap free in bytes.
    pub swap_free: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn memory_stats_reads_proc_meminfo() {
        let stats = memory_stats().unwrap();
        assert!(stats.total > 0);
        assert!(stats.available > 0);
    }

    #[test]
    fn load_average_reads_values() {
        let (one, five, fifteen) = load_average().unwrap();
        assert!(one >= 0.0);
        assert!(five >= 0.0);
        assert!(fifteen >= 0.0);
    }
}
