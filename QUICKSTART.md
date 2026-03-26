# AURORA Quick Start Guide

## Installation (5 minutes)

### 1. Install Rust

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### 2. Build AURORA

```bash
cd aurora
./build.sh release
```

### 3. Run AURORA

```bash
./target/release/aurora detect
```

## First Steps

### Detect Your Hardware

```bash
$ ./target/release/aurora detect

╔═══════════════════════════════════════════════════════════╗
║              AURORA Hardware Profile                       ║
╚═══════════════════════════════════════════════════════════╝

┌─ CPU ─────────────────────────────────────────────────────┐
│ Model:        AMD Ryzen 9 5900X
│ Cores:        12 physical, 24 logical
│ SIMD Level:   AVX2
│ Frequency:    3700 MHz (base) / 4950 MHz (boost)
│ Cache:        L1=32KB, L2=512KB, L3=64MB
│ NUMA Nodes:   1
└───────────────────────────────────────────────────────────┘

┌─ GPU ─────────────────────────────────────────────────────┐
│ Name:         NVIDIA GeForce RTX 3080
│ Type:         CUDA
│ VRAM:         10240 MB
│ Compute:      68 units
│ Memory BW:    760.0 GB/s
└───────────────────────────────────────────────────────────┘

┌─ Memory ──────────────────────────────────────────────────┐
│ Total:        65536 MB
│ Available:    52428 MB
│ HugePages:    128 (2 MB)
└───────────────────────────────────────────────────────────┘
```

### Run Benchmarks

```bash
$ ./target/release/aurora benchmark

Running benchmarks...
  Iterations: 10

Benchmark Results:
  CPU Memory Bandwidth: 45.23 GB/s
  CPU Matmul: 125.67 GFLOPS
  CPU Vector: 89.34 GFLOPS
```

### System Status

```bash
$ ./target/release/aurora status

AURORA System Status
════════════════════
CPU Usage: 12.5%
Memory: 8192 MB / 65536 MB (12.5%)
AURORA Processes: 0
```

## Using the Library

### Add to Cargo.toml

```toml
[dependencies]
aurora-api = { path = "path/to/aurora/aurora-api" }
```

### Basic Example

```rust
use aurora_api::AuroraRuntime;
use aurora_api::{TensorShape, DataType, KernelType};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize
    let runtime = AuroraRuntime::initialize()?;
    
    // Create tensors
    let a = runtime.create_tensor(
        TensorShape::from_2d(1024, 1024),
        DataType::F32
    )?;
    
    let b = runtime.create_tensor(
        TensorShape::from_2d(1024, 1024),
        DataType::F32
    )?;
    
    let mut c = runtime.create_tensor(
        TensorShape::from_2d(1024, 1024),
        DataType::F32
    )?;
    
    // Run matmul
    runtime.execute_kernel(KernelType::Matmul, &[&a, &b], &mut c)?;
    
    // Cleanup
    runtime.shutdown()?;
    Ok(())
}
```

### Compile and Run

```bash
cargo run --release
```

## Common Tasks

### Check Version

```bash
./target/release/aurora version
```

### Install System-wide

```bash
sudo ./install/install.sh
```

### Tune System for Performance

```bash
sudo ./install/install.sh --tune
```

### Watch System Status

```bash
./target/release/aurora status --watch
```

## Troubleshooting

### "cargo: command not found"

```bash
source $HOME/.cargo/env
```

### Build fails

```bash
# Update Rust
rustup update

# Clean and rebuild
cargo clean
cargo build --release
```

### Permission denied

```bash
chmod +x build.sh install/install.sh
```

## Next Steps

1. Read [BUILD.md](BUILD.md) for detailed build instructions
2. Read [README.md](README.md) for full documentation
3. Check [PROJECT_SUMMARY.md](PROJECT_SUMMARY.md) for architecture overview
4. Explore examples in `examples/` directory

## Performance Tips

1. **Build in release mode**: Always use `--release` for performance
2. **Enable HugePages**: `sudo ./install/install.sh --tune`
3. **Set CPU governor**: Performance mode for compute workloads
4. **Use NUMA-aware allocation**: For multi-socket systems

## Getting Help

- Run `aurora --help` for command reference
- Check logs with `RUST_LOG=debug aurora detect`
- Open an issue on GitHub

---

**Happy computing with AURORA!** 🦀
