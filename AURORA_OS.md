# AURORA OS - High-Performance Compute Distribution

## Overview

AURORA OS is a Linux-based High-Performance Compute Distribution designed for maximum performance extraction from modern hardware. It combines a custom-tuned Linux kernel, a Rust-based compute runtime, and system-level optimizations.

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Applications Layer                        │
│         AI / HPC / Simulation / Rendering                   │
├─────────────────────────────────────────────────────────────┤
│                   AURORA Runtime                             │
│    CPU Engine + GPU Engine + Hybrid Orchestrator            │
├─────────────────────────────────────────────────────────────┤
│              Kernel Acceleration Layer                       │
│    aurora_compute.ko - Custom kernel module                 │
├─────────────────────────────────────────────────────────────┤
│              Linux Kernel (Tuned)                            │
│    PREEMPT_NONE / NUMA / HugePages / Tickless               │
├─────────────────────────────────────────────────────────────┤
│                    Hardware                                  │
│    CPU / GPU / Memory / NUMA / PCIe                         │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. Custom Linux Kernel

**Configuration:**
```
CONFIG_HZ_1000=y              # 1000Hz timer
CONFIG_NO_HZ_FULL=y           # Full tickless
CONFIG_PREEMPT_NONE=y         # Throughput mode
CONFIG_NUMA=y                 # NUMA support
CONFIG_TRANSPARENT_HUGEPAGE_ALWAYS=y
CONFIG_CPU_FREQ_DEFAULT_GOV_PERFORMANCE=y
```

**Features:**
- Tickless operation for compute workloads
- NUMA-aware memory allocation
- Transparent HugePages always enabled
- Performance governor default
- Disabled power saving states

### 2. Kernel Module (aurora_compute.ko)

**IOCTL Interface:**
- `SET_AFFINITY` - CPU affinity control
- `LOCK_MEMORY` - Prevent swapping
- `RESERVE_HUGEPAGE` - HugePage allocation
- `SET_NUMA_POLICY` - NUMA binding
- `GET_PERF_COUNTERS` - Hardware counters
- `SET_CPU_GOV` - Governor control

**Build:**
```bash
cd kernel-module
make
sudo make install
sudo make load
```

### 3. System Tuning Script

**Applies:**
- Kernel parameters (sysctl)
- CPU governor settings
- Memory optimizations
- NUMA configuration
- I/O scheduler tuning
- Resource limits
- IRQ affinity

**Usage:**
```bash
sudo ./scripts/aurora-tune.sh
```

### 4. AURORA Runtime

**12 Rust Crates:**
- `aurora-core` - Foundation types
- `aurora-profiler` - Hardware detection
- `aurora-cpu` - CPU compute engine
- `aurora-gpu` - GPU compute engine
- `aurora-orchestrator` - Hybrid execution
- `aurora-tensor` - Tensor operations
- `aurora-memory` - Memory management
- `aurora-linux` - Linux integration
- `aurora-autotune` - Auto-tuning
- `aurora-api` - Public API
- `aurora-cli` - Command-line interface
- `aurora-distro` - Ubuntu 24 remaster, USB writer, partition planner, boot repair
- Docker-assisted Ubuntu 24 remaster path for more repeatable ISO builds
- first-boot autosetup services for zram, service trimming, storage tuning, and boot responsiveness

**SIMD Support:**
- Scalar (fallback)
- SSE2 (128-bit)
- AVX (256-bit)
- AVX2 (256-bit + FMA)
- AVX-512 (512-bit)

## Installation

### Prerequisites

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    linux-headers-$(uname -r) \
    libhwloc-dev \
    numactl \
    cpufrequtils

# Fedora/RHEL
sudo dnf install -y \
    kernel-devel \
    hwloc-devel \
    numactl \
    kernel-headers
```

### Quick Install

```bash
# 1. Clone repository
git clone https://github.com/aurora-runtime/aurora.git
cd aurora

# 2. Build runtime
./build.sh release

# 3. Install kernel module
cd kernel-module
make && sudo make install && sudo make load
cd ..

# 4. Tune system
sudo ./scripts/aurora-tune.sh

# 5. Install runtime
sudo ./install/install.sh
```

### Full OS Install (Advanced)

```bash
# 1. Build custom kernel
./scripts/build-kernel.sh

# 2. Create boot entry
sudo ./scripts/install-kernel.sh

# 3. Reboot into AURORA kernel
sudo reboot

# 4. Complete installation
sudo ./scripts/aurora-tune.sh
sudo ./install/install.sh --kernel --tune
```

## Remastering Ubuntu 24 Into AURORA OS

Use the Rust-native distro tool:

```bash
cargo run -p aurora-distro -- prepare-host
cargo run -p aurora-distro -- init-tree --out distro --distro-name "Aurora Neon" --desktop gnome
sudo cargo run -p aurora-distro -- build-iso --tree distro --prompt-usb
```

Capabilities included in the distro workflow:

- BIOS legacy support
- UEFI support
- desktop presets for GNOME, KDE, and minimal shells
- custom branding and logo path configuration
- Kali-inspired gaming-themed GRUB/Plymouth/desktop overlays
- hardware scan and partition planning
- USB writer flow after ISO build
- boot repair for BIOS and UEFI systems

## Usage

### Command Line

```bash
# Detect hardware
aurora detect

# Run benchmarks
aurora benchmark

# Monitor system
aurora status --watch

# Run compute kernel
aurora run --kernel matmul --size 4096
```

### Library API

```rust
use aurora_api::AuroraRuntime;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize
    let runtime = AuroraRuntime::initialize()?;
    
    // Create tensors
    let a = runtime.create_tensor(shape![1024, 1024], F32)?;
    let b = runtime.create_tensor(shape![1024, 1024], F32)?;
    let mut c = runtime.create_tensor(shape![1024, 1024], F32)?;
    
    // Execute
    runtime.execute_kernel(KernelType::Matmul, &[&a, &b], &mut c)?;
    
    // Cleanup
    runtime.shutdown()?;
    Ok(())
}
```

## Performance Expectations

| Hardware | Improvement |
|----------|-------------|
| Weak CPU | 3-6x vs naive |
| Strong CPU | 80-95% peak utilization |
| With GPU | 10-30% over unoptimized |
| Hybrid | Better utilization |

## CI/CD Pipeline

```bash
# Run full pipeline
./scripts/ci-pipeline.sh

# Skip benchmarks
./scripts/ci-pipeline.sh --skip-benchmarks

# Skip audit
./scripts/ci-pipeline.sh --skip-audit
```

**Pipeline Stages:**
1. Format check (`cargo fmt`)
2. Build check (`cargo check`)
3. Lint check (`cargo clippy`)
4. Unsafe code check
5. TODO/placeholder check
6. Unit tests (`cargo test`)
7. Benchmarks (`cargo bench`)
8. Release build
9. Documentation
10. Security audit
11. Module completeness
12. Performance validation

## Kernel Configuration

### Recommended Kernel Parameters

Add to `/etc/default/grub`:
```
GRUB_CMDLINE_LINUX_DEFAULT="quiet isolcpus=2-15 nohz_full=2-15 rcu_nocbs=2-15 intel_pstate=disable processor.max_cstate=1 idle=poll"
```

### Sysctl Parameters

Set in `/etc/sysctl.d/99-aurora.conf`:
```
vm.swappiness = 10
vm.dirty_ratio = 5
vm.nr_hugepages = 512
kernel.numa_balancing = 1
```

## Monitoring

### System Metrics

```bash
# CPU usage
watch -n 1 'cat /proc/stat | head -1'

# Memory
watch -n 1 'cat /proc/meminfo | grep -E "MemTotal|MemFree|Huge"'

# NUMA
numastat

# Performance counters
perf stat -a sleep 1
```

### AURORA Metrics

```bash
# Runtime statistics
aurora status

# Hardware profile
aurora detect --format json

# Benchmark results
aurora benchmark --format json
```

## Troubleshooting

### Kernel Module Won't Load

```bash
# Check kernel version
uname -r

# Install headers
sudo apt-get install linux-headers-$(uname -r)

# Rebuild module
cd kernel-module
make clean && make
sudo make load
```

### Performance Issues

```bash
# Verify tuning
sudo ./scripts/aurora-tune.sh --verify

# Check CPU governor
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Check HugePages
cat /proc/sys/vm/nr_hugepages

# Check THP
cat /sys/kernel/mm/transparent_hugepage/enabled
```

### Build Failures

```bash
# Clean build
cargo clean

# Update Rust
rustup update

# Check dependencies
./scripts/ci-pipeline.sh
```

## Development

### Adding a New Kernel

1. Define in `aurora-core/src/kernel.rs`
2. Implement in `aurora-cpu/src/kernels/`
3. Add tests
4. Register in kernel registry
5. Update benchmarks

### Adding SIMD Support

1. Add to `aurora-cpu/src/simd.rs`
2. Implement trait methods
3. Add to dispatcher
4. Benchmark vs scalar

### Contributing

1. Fork repository
2. Create feature branch
3. Run CI pipeline
4. Submit PR

## License

MIT OR Apache-2.0

## Resources

- Documentation: `docs/`
- Examples: `examples/`
- Tests: `*/src/*.rs` (#[cfg(test)])
- Benchmarks: `*/benches/`

---

**AURORA OS** - Maximum Performance Computing
