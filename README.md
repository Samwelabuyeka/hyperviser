# AURORA

**A**daptive **U**nified **R**untime & **O**rchestration for **R**esource **A**cceleration

Linux-native. Rust-based. Hardware-adaptive.

---

## Overview

AURORA is a hardware-adaptive compute runtime that extracts near-maximum performance from CPU + GPU + Linux kernel. It is NOT a fake GPU, silicon replacement, or magic performance multiplier. It IS a unified compute layer that intelligently distributes workloads across available hardware.

## Core Philosophy

- **NOT** a fake GPU
- **NOT** a silicon replacement  
- **NOT** a magic performance multiplier
- **IS** a hardware-adaptive compute runtime
- **IS** Linux-native and Rust-based
- **IS** designed for real-world performance extraction

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│              User Applications                          │
│     (AI, Physics, Rendering, HPC, Simulation)          │
├─────────────────────────────────────────────────────────┤
│              AURORA Public API                          │
│        Kernel Launch / Tensor Ops / Graph Exec         │
├─────────────────────────────────────────────────────────┤
│            Hardware Profiler & Mutator                  │
│  CPU | GPU | Memory | NUMA | Cache | Bandwidth        │
├─────────────────────────────────────────────────────────┤
│               Adaptive Orchestrator                     │
│   Decides: CPU / GPU / Hybrid / Split Execution       │
├─────────────────────┬───────────────────────────────────┤
│     CPU Engine      │         GPU Engine                │
│  SIMD + Threads     │    CUDA/ROCm/Vulkan/OpenCL        │
├─────────────────────┴───────────────────────────────────┤
│             Linux Kernel Integration                    │
│  Affinity | HugePages | NUMA | mlock | Scheduler      │
└─────────────────────────────────────────────────────────┘
```

## Features

### Hardware Profiler
- CPU feature detection (SSE, AVX, AVX2, AVX-512)
- GPU detection (NVIDIA CUDA, AMD ROCm, Intel, Vulkan)
- Memory topology and NUMA detection
- Microbenchmarks for performance characterization

### CPU Engine
- Lock-free work-stealing thread pool
- SIMD multi-version kernels (SSE2, AVX, AVX2, AVX-512)
- Cache-tiled matrix operations
- NUMA-aware memory allocation

### GPU Engine
- CUDA backend for NVIDIA GPUs
- ROCm backend for AMD GPUs
- Vulkan compute backend
- OpenCL fallback

### Hybrid Orchestrator
- Runtime decision making for CPU vs GPU
- Workload splitting across devices
- Adaptive tuning based on execution history
- Performance model-based scheduling

### Linux Integration
- CPU affinity control
- HugePages support
- NUMA memory binding
- Performance governor management
- Real-time scheduler support

### AURORA Distro Toolkit
- Ubuntu 24.04 remaster workflow via `aurora-distro`
- BIOS + UEFI live ISO generation
- custom branding, desktop presets, and gaming/Kali-style theme scaffolding
- host setup, USB writing, partition planning, and boot repair helpers

## Installation

### Prerequisites
- Linux kernel 5.4+
- Rust 1.75+
- (Optional) CUDA Toolkit for NVIDIA GPU support
- (Optional) ROCm for AMD GPU support

### Quick Install

```bash
# Clone repository
git clone https://github.com/aurora-runtime/aurora.git
cd aurora

# Build and install
./install/install.sh

# Or with options
./install/install.sh --prefix ~/.local/aurora --tune
```

### Manual Build

```bash
# Build release version
./build.sh release

# Or debug version
./build.sh debug

# Run
./target/release/aurora --help
```

## Usage

### Command Line

```bash
# Detect hardware
aurora detect

# Run benchmarks
aurora benchmark

# Show system status
aurora status

# Run a kernel
aurora run --kernel matmul --size 1024 --device auto
```

### Library API

```rust
use aurora_api::AuroraRuntime;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize runtime
    let runtime = AuroraRuntime::initialize()?;
    
    // Create tensors
    let a = runtime.create_tensor(TensorShape::from_2d(1024, 1024), DataType::F32)?;
    let b = runtime.create_tensor(TensorShape::from_2d(1024, 1024), DataType::F32)?;
    let mut c = runtime.create_tensor(TensorShape::from_2d(1024, 1024), DataType::F32)?;
    
    // Execute matrix multiplication
    runtime.execute_kernel(KernelType::Matmul, &[&a, &b], &mut c)?;
    
    // Shutdown
    runtime.shutdown()?;
    Ok(())
}
```

## Project Structure

```
aurora/
├── aurora-core/        # Core types and abstractions
├── aurora-profiler/    # Hardware detection and profiling
├── aurora-cpu/         # CPU compute engine with SIMD
├── aurora-gpu/         # GPU compute engine (CUDA/ROCm/Vulkan)
├── aurora-orchestrator/# Hybrid execution orchestrator
├── aurora-tensor/      # Tensor operations
├── aurora-memory/      # Memory management
├── aurora-linux/       # Linux kernel integration
├── aurora-autotune/    # Auto-tuning system
├── aurora-api/         # Public API
├── aurora-cli/         # Command-line interface
├── install/            # Installation scripts
└── docs/               # Documentation
```

## Performance Expectations

| Hardware | Expected Improvement |
|----------|---------------------|
| Weak CPU | 3-6x faster than naive CPU |
| Strong CPU | 80-95% peak utilization |
| With GPU | 10-30% improvement over poorly optimized usage |
| Hybrid | Better utilization, reduced idle time |

**Note:** AURORA will NOT beat RTX silicon limits. It will extract more from existing hardware.

## Configuration

Configuration file: `/opt/aurora/etc/aurora.conf`

```toml
[runtime]
thread_pool_size = auto
memory_pool_size = 1024

[performance]
use_hugepages = true
use_numa = true
cpu_governor = performance

[gpu]
enable_cuda = true
enable_rocm = true
enable_vulkan = true

[profiling]
enabled = true
adaptive_tuning = true
```

## Development

### Building

```bash
# Build all crates
cargo build --release

# Run tests
cargo test --all

# Run benchmarks
cargo bench
```

### Adding a New Kernel

1. Define kernel in `aurora-cpu/src/kernels/` or `aurora-gpu/src/kernels/`
2. Implement kernel trait
3. Register in kernel registry
4. Add tests and benchmarks

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option.

## Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## Acknowledgments

- Inspired by OpenBLAS, llama.cpp, and oneDNN
- Built with Rust's excellent concurrency and safety guarantees
- Linux kernel integration for maximum performance

---

**AURORA** - Extracting maximum performance from your hardware.
