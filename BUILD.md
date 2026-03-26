# Building AURORA

## Prerequisites

- **Rust** 1.75.0 or later
- **Linux** kernel 5.4+ (x86_64)
- **Git** (for cloning)
- **Build tools**: gcc, make, pkg-config

## Installing Rust

If you don't have Rust installed:

```bash
# Install via rustup
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env

# Verify installation
rustc --version  # Should show 1.75.0 or later
cargo --version
```

## Building AURORA

### Quick Build

```bash
# Clone the repository
git clone <repository-url>
cd aurora

# Build release version
./build.sh release

# Or build debug version
./build.sh debug
```

### Manual Build

```bash
# Build all crates
cargo build --release

# Build specific crate
cargo build --release -p aurora-cli

# Run tests
cargo test --all

# Run benchmarks
cargo bench
```

## Build Outputs

After building, you'll find:

- **Binary**: `target/release/aurora` (or `target/debug/aurora`)
- **Libraries**: `target/release/lib*.so` and `target/release/lib*.a`

## Running AURORA

```bash
# Show help
./target/release/aurora --help

# Detect hardware
./target/release/aurora detect

# Run benchmarks
./target/release/aurora benchmark

# Show system status
./target/release/aurora status
```

## Installation

### System-wide Installation (requires root)

```bash
sudo ./install/install.sh
```

### User Installation

```bash
./install/install.sh --prefix ~/.local/aurora
```

### Installation Options

```bash
# Install with kernel modules
sudo ./install/install.sh --kernel

# Install with system tuning
sudo ./install/install.sh --tune

# Custom prefix
./install/install.sh --prefix /custom/path
```

## Troubleshooting

### "cargo: command not found"

Make sure Rust is installed and in your PATH:

```bash
source $HOME/.cargo/env
```

### Build failures

1. Update Rust:
   ```bash
   rustup update
   ```

2. Clean and rebuild:
   ```bash
   cargo clean
   cargo build --release
   ```

3. Check for missing dependencies:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install build-essential pkg-config

   # RHEL/CentOS/Fedora
   sudo dnf install gcc make pkgconfig
   ```

### Linker errors

If you see linker errors, you may need to install additional libraries:

```bash
# Ubuntu/Debian
sudo apt-get install libhwloc-dev

# For GPU support
sudo apt-get install nvidia-cuda-toolkit  # NVIDIA
sudo apt-get install rocm-dev             # AMD
```

## Development Build

For development with faster compile times:

```bash
# Build without optimizations
cargo build

# Run with logging
RUST_LOG=debug ./target/debug/aurora detect

# Run tests with output
cargo test --all -- --nocapture
```

## Cross-compilation

To build for a different target:

```bash
# Add target
rustup target add x86_64-unknown-linux-musl

# Build
cargo build --release --target x86_64-unknown-linux-musl
```

## Feature Flags

AURORA supports various feature flags:

```bash
# Build with CUDA support
cargo build --release --features cuda

# Build with ROCm support
cargo build --release --features rocm

# Build with Vulkan support
cargo build --release --features vulkan
```

## Verification

After building, verify the installation:

```bash
# Check version
./target/release/aurora version

# Detect hardware
./target/release/aurora detect

# Run benchmarks
./target/release/aurora benchmark
```

## Performance Build

For maximum performance:

```bash
# Set Rust flags for optimization
export RUSTFLAGS="-C target-cpu=native -C opt-level=3"

# Build
cargo build --release
```

## Docker Build

You can also build in a Docker container:

```dockerfile
FROM rust:1.75

WORKDIR /aurora
COPY . .

RUN cargo build --release

ENTRYPOINT ["./target/release/aurora"]
```

```bash
docker build -t aurora .
docker run --rm aurora detect
```

## Next Steps

After building:

1. Run `aurora detect` to see your hardware profile
2. Run `aurora benchmark` to characterize performance
3. Read the [API documentation](docs/API.md) for integration
4. Check [examples](examples/) for usage patterns

## Support

For build issues:
- Check [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
- Open an issue on GitHub
- Join our Discord community
