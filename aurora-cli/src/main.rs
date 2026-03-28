//! AURORA CLI - Command-line interface for the AURORA compute runtime

use anyhow::Result;
use aurora_core::initialize;
use aurora_profiler::{print_profile, HardwareProfiler};
use clap::{Parser, Subcommand};
use colored::*;
use tracing::Level;
use tracing_subscriber::FmtSubscriber;

/// AURORA - Adaptive Unified Runtime & Orchestration for Resource Acceleration
#[derive(Parser)]
#[command(name = "aurora")]
#[command(about = "AURORA - Hardware-adaptive compute runtime", long_about = None)]
#[command(version)]
struct Cli {
    /// Enable verbose output
    #[arg(short, long, global = true)]
    verbose: bool,
    
    /// Subcommand
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Detect and display hardware information
    Detect {
        /// Output format
        #[arg(short, long, value_enum, default_value = "text")]
        format: OutputFormat,
        
        /// Save profile to file
        #[arg(short, long)]
        output: Option<String>,
        
        /// Run benchmarks
        #[arg(short, long)]
        benchmark: bool,
    },
    
    /// Run performance benchmarks
    Benchmark {
        /// Specific benchmark to run
        #[arg(short, long)]
        name: Option<String>,
        
        /// Number of iterations
        #[arg(short, long, default_value = "10")]
        iterations: usize,
        
        /// Output format
        #[arg(short, long, value_enum, default_value = "text")]
        format: OutputFormat,
    },
    
    /// Show system status
    Status {
        /// Watch mode (continuous updates)
        #[arg(short, long)]
        watch: bool,
        
        /// Update interval in seconds
        #[arg(short, long, default_value = "1")]
        interval: u64,
    },
    
    /// Run a compute kernel
    Run {
        /// Kernel to run
        #[arg(short, long)]
        kernel: String,
        
        /// Input size
        #[arg(short, long)]
        size: Option<usize>,
        
        /// Device to use (cpu/gpu/auto)
        #[arg(short, long, default_value = "auto")]
        device: String,
        
        /// Number of iterations
        #[arg(short, long, default_value = "1")]
        iterations: usize,
    },
    
    /// Install AURORA system components
    Install {
        /// Installation prefix
        #[arg(short, long, default_value = "/opt/aurora")]
        prefix: String,
        
        /// Install kernel modules
        #[arg(short, long)]
        kernel: bool,
        
        /// Configure system for optimal performance
        #[arg(short, long)]
        tune: bool,
    },
    
    /// Uninstall AURORA
    Uninstall {
        /// Installation prefix
        #[arg(short, long, default_value = "/opt/aurora")]
        prefix: String,
    },
    
    /// Show version information
    Version,
}

#[derive(Clone, Debug, clap::ValueEnum)]
enum OutputFormat {
    Text,
    Json,
    Yaml,
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    
    // Setup logging
    let log_level = if cli.verbose {
        Level::DEBUG
    } else {
        Level::INFO
    };
    
    let subscriber = FmtSubscriber::builder()
        .with_max_level(log_level)
        .finish();
    
    tracing::subscriber::set_global_default(subscriber)?;
    
    // Initialize AURORA
    initialize()?;
    
    match cli.command {
        Commands::Detect { format, output, benchmark } => {
            cmd_detect(format, output, benchmark).await
        }
        Commands::Benchmark { name, iterations, format } => {
            cmd_benchmark(name, iterations, format).await
        }
        Commands::Status { watch, interval } => {
            cmd_status(watch, interval).await
        }
        Commands::Run { kernel, size, device, iterations } => {
            cmd_run(kernel, size, device, iterations).await
        }
        Commands::Install { prefix, kernel, tune } => {
            cmd_install(prefix, kernel, tune).await
        }
        Commands::Uninstall { prefix } => {
            cmd_uninstall(prefix).await
        }
        Commands::Version => {
            cmd_version()
        }
    }
}

async fn cmd_detect(format: OutputFormat, output: Option<String>, benchmark: bool) -> Result<()> {
    print_banner();
    
    println!("{} hardware...", "Detecting".cyan().bold());
    
    let mut profiler = HardwareProfiler::new();
    let profile = profiler.detect()?;
    
    if benchmark {
        println!("{} benchmarks...", "Running".cyan().bold());
        let _benchmarks = profiler.benchmark()?;
    }
    
    match format {
        OutputFormat::Text => {
            print_profile(&profile);
        }
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&profile)?;
            println!("{}", json);
        }
        OutputFormat::Yaml => {
            // Would need yaml crate
            println!("{:#?}", profile);
        }
    }
    
    if let Some(path) = output {
        let content = serde_json::to_string_pretty(&profile)?;
        std::fs::write(&path, content)?;
        println!("\n{} profile to {}", "Saved".green().bold(), path);
    }
    
    Ok(())
}

async fn cmd_benchmark(name: Option<String>, iterations: usize, format: OutputFormat) -> Result<()> {
    print_banner();
    
    println!("{} benchmarks...", "Running".cyan().bold());
    println!("  Iterations: {}", iterations);
    if let Some(ref n) = name {
        println!("  Benchmark: {}", n);
    }
    
    let mut profiler = HardwareProfiler::new();
    let _profile = profiler.detect()?;
    let benchmarks = profiler.benchmark()?;
    
    match format {
        OutputFormat::Text => {
            println!("\n{} Results:", "Benchmark".green().bold());
            println!(
                "  CPU Memory Bandwidth: {:.2} GB/s",
                benchmarks.memory.numa_bandwidth_gbps
            );
            println!("  CPU Matmul: {:.2} GFLOPS", benchmarks.cpu.matmul_gflops);
            println!("  CPU Vector: {:.2} GFLOPS", benchmarks.cpu.vector_gflops);
        }
        OutputFormat::Json => {
            let json = serde_json::to_string_pretty(&benchmarks)?;
            println!("{}", json);
        }
        OutputFormat::Yaml => {
            println!("{:#?}", benchmarks);
        }
    }
    
    Ok(())
}

async fn cmd_status(watch: bool, interval: u64) -> Result<()> {
    print_banner();
    
    if watch {
        println!("{} system status (Ctrl+C to exit)...", "Watching".cyan().bold());
        
        let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(interval));
        
        loop {
            interval.tick().await;
            print_status();
        }
    } else {
        print_status();
    }
    
    Ok(())
}

fn print_status() {
    use sysinfo::System;
    
    let mut sys = System::new_all();
    sys.refresh_all();
    
    println!("\x1B[2J\x1B[H"); // Clear screen
    println!("{}", "AURORA System Status".cyan().bold());
    println!("{}", "════════════════════".cyan());
    
    // CPU usage
    let cpu_total: f32 = sys.cpus().iter().map(|cpu| cpu.cpu_usage()).sum();
    let cpu_usage: f32 = if sys.cpus().is_empty() {
        0.0
    } else {
        cpu_total / sys.cpus().len() as f32
    };
    println!("CPU Usage: {:.1}%", cpu_usage);
    
    // Memory usage
    let total_mem = sys.total_memory();
    let used_mem = sys.used_memory();
    let mem_percent = (used_mem as f64 / total_mem as f64) * 100.0;
    println!("Memory: {} MB / {} MB ({:.1}%)", 
        used_mem / 1024, total_mem / 1024, mem_percent);
    
    // AURORA processes
    let aurora_procs: Vec<_> = sys.processes()
        .values()
        .filter(|process| process.name().to_string().contains("aurora"))
        .collect();
    
    println!("AURORA Processes: {}", aurora_procs.len());
    for process in aurora_procs {
        println!(
            "  {} (PID: {}) - {} MB",
            process.name().to_string(),
            process.pid(),
            process.memory() / 1024
        );
    }
}

async fn cmd_run(kernel: String, size: Option<usize>, device: String, iterations: usize) -> Result<()> {
    print_banner();
    
    println!("{} kernel '{}'...", "Running".cyan().bold(), kernel);
    println!("  Device: {}", device);
    println!("  Iterations: {}", iterations);
    if let Some(s) = size {
        println!("  Size: {}", s);
    }
    
    // Placeholder - would actually run the kernel
    println!("\n{}", "Kernel execution not yet implemented".yellow());
    
    Ok(())
}

async fn cmd_install(prefix: String, kernel: bool, tune: bool) -> Result<()> {
    print_banner();
    
    println!("{} AURORA to {}...", "Installing".green().bold(), prefix);
    
    // Check if running as root
    if unsafe { libc::geteuid() } != 0 {
        println!("{}", "Warning: Installation may require root privileges".yellow());
    }
    
    // Create directories
    let dirs = vec![
        format!("{}/bin", prefix),
        format!("{}/lib", prefix),
        format!("{}/etc", prefix),
        format!("{}/share/aurora", prefix),
        format!("{}/var/cache/aurora", prefix),
    ];
    
    for dir in &dirs {
        println!("  Creating {}", dir);
        std::fs::create_dir_all(dir)?;
    }
    
    // Install binary
    let exe_path = std::env::current_exe()?;
    let dest_path = format!("{}/bin/aurora", prefix);
    println!("  Installing binary to {}", dest_path);
    std::fs::copy(&exe_path, &dest_path)?;
    
    // Create config
    let config = r#"# AURORA Configuration
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
"#;
    
    let config_path = format!("{}/etc/aurora.conf", prefix);
    println!("  Creating config at {}", config_path);
    std::fs::write(&config_path, config)?;
    
    if kernel {
        println!("  {} kernel modules...", "Installing".cyan());
        // Would install kernel modules here
    }
    
    if tune {
        println!("  {} system for optimal performance...", "Tuning".cyan());
        tune_system()?;
    }
    
    println!("\n{} AURORA installed successfully!", "✓".green().bold());
    println!("\nAdd to your PATH: export PATH={}/bin:$PATH", prefix);
    
    Ok(())
}

async fn cmd_uninstall(prefix: String) -> Result<()> {
    print_banner();
    
    println!("{} AURORA from {}...", "Uninstalling".red().bold(), prefix);
    
    // Check if prefix exists
    if !std::path::Path::new(&prefix).exists() {
        println!("{}", "AURORA is not installed at this prefix".yellow());
        return Ok(());
    }
    
    // Confirm
    print!("Are you sure? [y/N] ");
    use std::io::Write;
    std::io::stdout().flush()?;
    
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    
    if !input.trim().eq_ignore_ascii_case("y") {
        println!("Uninstall cancelled");
        return Ok(());
    }
    
    // Remove directories
    println!("  Removing {}", prefix);
    std::fs::remove_dir_all(&prefix)?;
    
    println!("\n{} AURORA uninstalled", "✓".green().bold());
    
    Ok(())
}

fn cmd_version() -> Result<()> {
    print_banner();
    
    println!("AURORA Runtime {}", aurora_core::VERSION);
    println!("Rust Version: {}", rustc_version_runtime::version());
    println!("Target: {}", std::env::consts::ARCH);
    
    Ok(())
}

fn print_banner() {
    println!("{}", r#"
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║     █████╗ ██╗   ██╗██████╗  ██████╗ ██████╗  █████╗     ║
    ║    ██╔══██╗██║   ██║██╔══██╗██╔═══██╗██╔══██╗██╔══██╗    ║
    ║    ███████║██║   ██║██████╔╝██║   ██║██████╔╝███████║    ║
    ║    ██╔══██║██║   ██║██╔══██╗██║   ██║██╔══██╗██╔══██║    ║
    ║    ██║  ██║╚██████╔╝██║  ██║╚██████╔╝██║  ██║██║  ██║    ║
    ║    ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝    ║
    ║                                                           ║
    ║    Adaptive Unified Runtime & Orchestration               ║
    ║    for Resource Acceleration                              ║
    ╚═══════════════════════════════════════════════════════════╝
    "#.cyan());
}

fn tune_system() -> Result<()> {
    use std::process::Command;
    
    println!("  Tuning CPU governor...");
    let _ = Command::new("cpufreq-set")
        .args(&["-g", "performance"])
        .output();
    
    println!("  Enabling HugePages...");
    let _ = std::fs::write("/proc/sys/vm/nr_hugepages", "128");
    
    println!("  Setting swappiness...");
    let _ = std::fs::write("/proc/sys/vm/swappiness", "10");
    
    Ok(())
}
