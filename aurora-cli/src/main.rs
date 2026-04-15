//! AURORA CLI - Command-line interface for the AURORA compute runtime

use anyhow::Result;
use anyhow::bail;
use aurora_core::initialize;
use aurora_core::format_flops;
use aurora_gpu::{available_low_power_backends, probe_backends};
use aurora_linux::{
    configure_thp, get_cpu_affinity, load_average, memory_stats, reserve_hugepages,
    numa_topology, preferred_cpu_order, set_governor, set_idle_states, set_io_scheduler,
    set_priority, set_readahead_kb, set_swappiness,
};
use aurora_orchestrator::{ExecutionStrategy, Orchestrator};
use aurora_profiler::{print_profile, HardwareProfiler};
use aurora_profiler::profile::WorkloadCharacteristics;
use aurora_tensor::{
    add as tensor_add, attention as tensor_attention, batch_norm as tensor_batch_norm,
    conv2d as tensor_conv2d, fused_add_relu as tensor_fused_add_relu,
    gelu as tensor_gelu, im2col as tensor_im2col, layer_norm as tensor_layer_norm,
    matmul as tensor_matmul, mul as tensor_mul, quantize_i8, quantized_matmul_i8,
    reduce_max_last as tensor_reduce_max_last, reduce_sum_last as tensor_reduce_sum_last,
    rms_norm as tensor_rms_norm, relu as tensor_relu, sigmoid as tensor_sigmoid,
    softmax as tensor_softmax, transpose_last_two as tensor_transpose_last_two,
    ExecutionGraph, GraphNode, HostTensor, PersistentExecutor,
};
use clap::{Parser, Subcommand};
use colored::*;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use serde::{Deserialize, Serialize};
use std::process::Command;
use std::time::Instant;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
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

        /// CPU worker threads for runtime execution
        #[arg(short = 't', long)]
        threads: Option<usize>,
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

    /// Apply an OS-level optimization profile
    Optimize {
        /// Optimization profile
        #[arg(short, long, value_enum, default_value = "gaming")]
        profile: RuntimeProfile,

        /// Override hugepage count
        #[arg(long)]
        hugepages: Option<usize>,

        /// Override swappiness
        #[arg(long)]
        swappiness: Option<u32>,
    },
    
    /// Uninstall AURORA
    Uninstall {
        /// Installation prefix
        #[arg(short, long, default_value = "/opt/aurora")]
        prefix: String,
    },
    
    /// Show version information
    Version,

    /// Run the persistent runtime daemon loop
    Daemon {
        /// Poll interval in seconds
        #[arg(short, long, default_value = "5")]
        interval: u64,
    },

    /// Runtime-native assistant surface
    Assistant {
        /// Action to perform
        #[arg(short, long, default_value = "status")]
        action: String,

        /// Optional prompt
        #[arg(short, long)]
        prompt: Option<String>,
    },

    /// System-wide AI inference API
    Api {
        /// Action to perform: serve, status, infer
        #[arg(short, long, default_value = "status")]
        action: String,

        /// Bind address for the local HTTP API
        #[arg(long, default_value = "127.0.0.1:11435")]
        bind: String,

        /// Default local model
        #[arg(long, default_value = "llama3")]
        model: String,

        /// Optional direct prompt for `infer`
        #[arg(short, long)]
        prompt: Option<String>,
    },

    /// Inspect and drive the resident runtime fabric
    Fabric {
        /// Action to perform
        #[arg(short, long, default_value = "status")]
        action: String,

        /// Problem size for graph or stress actions
        #[arg(short, long)]
        size: Option<usize>,

        /// CPU worker threads
        #[arg(short = 't', long)]
        threads: Option<usize>,
    },
}

#[derive(Clone, Debug, clap::ValueEnum)]
enum OutputFormat {
    Text,
    Json,
    Yaml,
}

#[derive(Clone, Debug, clap::ValueEnum)]
enum RuntimeProfile {
    Balanced,
    Gaming,
    Creator,
    NoGpuTraining,
    LowPowerInference,
    MaxThroughput,
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
        Commands::Run { kernel, size, device, iterations, threads } => {
            cmd_run(kernel, size, device, iterations, threads).await
        }
        Commands::Install { prefix, kernel, tune } => {
            cmd_install(prefix, kernel, tune).await
        }
        Commands::Optimize { profile, hugepages, swappiness } => {
            cmd_optimize(profile, hugepages, swappiness).await
        }
        Commands::Uninstall { prefix } => {
            cmd_uninstall(prefix).await
        }
        Commands::Version => {
            cmd_version()
        }
        Commands::Daemon { interval } => {
            cmd_daemon(interval).await
        }
        Commands::Assistant { action, prompt } => {
            cmd_assistant(action, prompt).await
        }
        Commands::Api {
            action,
            bind,
            model,
            prompt,
        } => cmd_api(action, bind, model, prompt).await,
        Commands::Fabric { action, size, threads } => {
            cmd_fabric(action, size, threads).await
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

async fn cmd_run(
    kernel: String,
    size: Option<usize>,
    device: String,
    iterations: usize,
    threads: Option<usize>,
) -> Result<()> {
    print_banner();
    
    println!("{} kernel '{}'...", "Running".cyan().bold(), kernel);
    println!("  Device: {}", device);
    println!("  Iterations: {}", iterations);
    if let Some(s) = size {
        println!("  Size: {}", s);
    }
    if let Some(threads) = threads {
        println!("  Threads: {}", threads);
    }

    let mut profiler = HardwareProfiler::new();
    let profile = profiler.detect()?;
    let workload = describe_workload(&kernel, size.unwrap_or(default_kernel_size(&kernel)), &device);
    let orchestrator = Orchestrator::new(profile.clone());
    let plan = orchestrator.decide_strategy(&workload);

    println!(
        "  Planned strategy: {:?} (cpu={}%, gpu={}%)",
        plan.strategy,
        plan.split.cpu_percent,
        plan.split.gpu_percent
    );

    if device.eq_ignore_ascii_case("gpu") {
        let backend_status = probe_backends();
        let available = backend_status
            .iter()
            .filter(|status| status.available)
            .map(|status| status.kind.as_str())
            .collect::<Vec<_>>();

        if available.is_empty() {
            let details = backend_status
                .iter()
                .map(|status| format!("{}: {}", status.kind.as_str(), status.detail))
                .collect::<Vec<_>>()
                .join("; ");
            bail!(
                "GPU execution requested, but this host exposes no usable GPU backend. {}",
                details
            );
        }

        bail!(
            "GPU backend(s) detected ({}) but kernel execution is not wired yet; use --device cpu or --device auto on this host for now",
            available.join(", ")
        );
    }

    if matches!(plan.strategy, ExecutionStrategy::GpuOnly | ExecutionStrategy::Hybrid)
        && profile.has_gpu()
        && !device.eq_ignore_ascii_case("cpu")
    {
        println!(
            "{}",
            "GPU/hybrid was selected by the planner, but execution currently falls back to the real CPU path until the GPU backend is implemented."
                .yellow()
        );
    }

    if matches!(kernel.as_str(), "gaming-frame" | "editing-color") {
        let low_power = available_low_power_backends();
        if !low_power.is_empty() {
            println!(
                "  Low-power GPU posture: {} backend(s) detected for future iGPU acceleration",
                low_power
                    .iter()
                    .map(|backend| backend.as_str())
                    .collect::<Vec<_>>()
                    .join(", ")
            );
        }
    }

    let report = execute_cpu_kernel(
        &kernel,
        size.unwrap_or(default_kernel_size(&kernel)),
        iterations,
        threads.unwrap_or(profile.cpu.physical_cores.max(1)).max(1),
    )?;

    print_runtime_report(&report);

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
executor = persistent
graph_execution = true

[performance]
use_hugepages = true
use_numa = true
cpu_governor = performance

[gpu]
enable_cuda = true
enable_rocm = true
enable_vulkan = true

[ai]
inference_api_bind = "127.0.0.1:11435"
default_model = "llama3"
"#;
    
    let config_path = format!("{}/etc/aurora.conf", prefix);
    println!("  Creating config at {}", config_path);
    std::fs::write(&config_path, config)?;

    let systemd_dir = format!("{}/lib/systemd/system", prefix);
    std::fs::create_dir_all(&systemd_dir)?;
    let daemon_unit = format!(
        "[Unit]\nDescription=AURORA Persistent Runtime Daemon\nAfter=network-online.target\n\n[Service]\nType=simple\nExecStart={}/bin/aurora daemon --interval 5\nRestart=always\nRestartSec=2\n\n[Install]\nWantedBy=multi-user.target\n",
        prefix
    );
    let unit_path = format!("{}/aurora-runtime.service", systemd_dir);
    println!("  Creating service at {}", unit_path);
    std::fs::write(&unit_path, daemon_unit)?;

    let inference_unit = format!(
        "[Unit]\nDescription=AURORA Inference API\nAfter=network-online.target aurora-runtime.service ollama.service\nWants=network-online.target\n\n[Service]\nType=simple\nExecStart={}/bin/aurora api --action serve --bind 127.0.0.1:11435 --model llama3\nRestart=always\nRestartSec=2\n\n[Install]\nWantedBy=multi-user.target\n",
        prefix
    );
    let inference_unit_path = format!("{}/aurora-inference.service", systemd_dir);
    println!("  Creating service at {}", inference_unit_path);
    std::fs::write(&inference_unit_path, inference_unit)?;
    
    if kernel {
        println!("  {} kernel modules...", "Installing".cyan());
        // Would install kernel modules here
    }
    
    if tune {
        println!("  {} system for optimal performance...", "Tuning".cyan());
        apply_runtime_profile(RuntimeProfile::Gaming, None, None)?;
    }
    
    println!("\n{} AURORA installed successfully!", "✓".green().bold());
    println!("\nAdd to your PATH: export PATH={}/bin:$PATH", prefix);
    
    Ok(())
}

async fn cmd_optimize(
    profile: RuntimeProfile,
    hugepages: Option<usize>,
    swappiness: Option<u32>,
) -> Result<()> {
    print_banner();

    println!("{} {:?} profile...", "Applying".cyan().bold(), profile);
    apply_runtime_profile(profile, hugepages, swappiness)?;

    let affinity = get_cpu_affinity().unwrap_or_default();
    let load = load_average().unwrap_or((0.0, 0.0, 0.0));
    let memory = memory_stats().unwrap_or_default();

    println!("\n{} State:", "System".green().bold());
    println!("  Affinity CPUs: {}", affinity.len());
    println!("  Load average: {:.2} {:.2} {:.2}", load.0, load.1, load.2);
    println!(
        "  Memory available: {:.2} GiB",
        memory.available as f64 / 1024.0 / 1024.0 / 1024.0
    );

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

async fn cmd_daemon(interval: u64) -> Result<()> {
    println!("{} runtime daemon...", "Starting".cyan().bold());
    let executor = persistent_executor(std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1))?;
    let mut ticker = tokio::time::interval(tokio::time::Duration::from_secs(interval.max(1)));

    loop {
        ticker.tick().await;
        let load = load_average().unwrap_or((0.0, 0.0, 0.0));
        let mem = memory_stats().unwrap_or_default();
        let pool_stats = executor.memory_pool().stats();
        println!(
            "daemon heartbeat: threads={} load={:.2}/{:.2}/{:.2} avail_mem_gib={:.2} pooled={} reused={}",
            executor.threads(),
            load.0,
            load.1,
            load.2,
            mem.available as f64 / 1024.0 / 1024.0 / 1024.0,
            pool_stats.cached_buffers,
            pool_stats.reused_buffers,
        );
    }
}

async fn cmd_assistant(action: String, prompt: Option<String>) -> Result<()> {
    match action.as_str() {
        "status" => {
            let affinity = get_cpu_affinity().unwrap_or_default();
            let load = load_average().unwrap_or((0.0, 0.0, 0.0));
            let mem = memory_stats().unwrap_or_default();
            println!("AURORA Assistant");
            println!("  Runtime: active");
            println!("  Affinity CPUs: {}", affinity.len());
            println!("  Load: {:.2} {:.2} {:.2}", load.0, load.1, load.2);
            println!(
                "  Available memory: {:.2} GiB",
                mem.available as f64 / 1024.0 / 1024.0 / 1024.0
            );
            println!("  Recommendation: {}", suggest_runtime_mode(load, mem.available));
            Ok(())
        }
        "suggest" => {
            let load = load_average().unwrap_or((0.0, 0.0, 0.0));
            let mem = memory_stats().unwrap_or_default();
            println!("{}", suggest_runtime_mode(load, mem.available));
            Ok(())
        }
        "chat" => {
            let prompt = prompt.unwrap_or_else(|| {
                "Summarize this system and choose the best Aurora runtime mode.".to_string()
            });
            let load = load_average().unwrap_or((0.0, 0.0, 0.0));
            let mem = memory_stats().unwrap_or_default();
            let topology = numa_topology().unwrap_or_default();
            println!("AURORA Assistant Response");
            println!("  Prompt: {}", prompt);
            println!("  Suggested mode: {}", suggest_runtime_mode(load, mem.available));
            println!(
                "  Notes: keep daemon hot, use fused kernels, prefer graph-backed execution, numa_nodes={}, preferred_cores={}.",
                topology.nodes.len(),
                preferred_cpu_order().map(|cpus| cpus.len()).unwrap_or_default()
            );
            Ok(())
        }
        other => bail!("unknown assistant action '{}'; supported: status, suggest, chat", other),
    }
}

async fn cmd_api(action: String, bind: String, model: String, prompt: Option<String>) -> Result<()> {
    match action.as_str() {
        "status" => {
            let status = inference_status(&bind, &model);
            println!("{}", serde_json::to_string_pretty(&status)?);
            Ok(())
        }
        "infer" => {
            let prompt = prompt.unwrap_or_else(|| {
                "Summarize this Aurora machine and recommend the strongest workload mode.".to_string()
            });
            let response = run_inference_backend(&model, &prompt)?;
            println!("{}", response.trim());
            Ok(())
        }
        "serve" => serve_inference_api(&bind, &model).await,
        other => bail!("unknown api action '{}'; supported: serve, status, infer", other),
    }
}

async fn cmd_fabric(action: String, size: Option<usize>, threads: Option<usize>) -> Result<()> {
    let threads = threads.unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    let executor = persistent_executor(threads)?;
    match action.as_str() {
        "status" => {
            let load = load_average().unwrap_or((0.0, 0.0, 0.0));
            let mem = memory_stats().unwrap_or_default();
            let pool = executor.memory_pool().stats();
            let topology = numa_topology().unwrap_or_default();
            println!("AURORA Fabric");
            println!("  Threads: {}", executor.threads());
            println!("  Preferred CPUs: {:?}", executor.pinned_cpus());
            println!("  NUMA nodes: {}", topology.nodes.len());
            for node in topology.nodes {
                println!("    node{} => {} cpus {:?}", node.id, node.cpus.len(), node.cpus);
            }
            println!("  Load: {:.2} {:.2} {:.2}", load.0, load.1, load.2);
            println!(
                "  Available memory: {:.2} GiB",
                mem.available as f64 / 1024.0 / 1024.0 / 1024.0
            );
            println!(
                "  Pool: cached={} reused={}",
                pool.cached_buffers, pool.reused_buffers
            );
            Ok(())
        }
        "graph" => {
            let report = run_fabric_graph(size.unwrap_or(256), threads, 1)?;
            print_runtime_report(&report);
            Ok(())
        }
        "stress" => {
            let report = run_fabric_graph(size.unwrap_or(192), threads, 5)?;
            print_runtime_report(&report);
            Ok(())
        }
        other => bail!("unknown fabric action '{}'; supported: status, graph, stress", other),
    }
}

#[derive(Debug, Serialize)]
struct InferenceStatus {
    bind: String,
    default_model: String,
    backend: &'static str,
    ollama_available: bool,
    numa_nodes: usize,
    preferred_cpus: usize,
}

#[derive(Debug, Deserialize)]
struct InferenceRequest {
    prompt: String,
    model: Option<String>,
}

#[derive(Debug, Serialize)]
struct InferenceResponse {
    ok: bool,
    model: String,
    backend: &'static str,
    response: String,
}

fn inference_status(bind: &str, model: &str) -> InferenceStatus {
    let topology = numa_topology().unwrap_or_default();
    let preferred_cpus = preferred_cpu_order().map(|cpus| cpus.len()).unwrap_or_default();
    InferenceStatus {
        bind: bind.to_string(),
        default_model: model.to_string(),
        backend: "ollama",
        ollama_available: Command::new("ollama").arg("--version").output().is_ok(),
        numa_nodes: topology.nodes.len(),
        preferred_cpus,
    }
}

async fn serve_inference_api(bind: &str, model: &str) -> Result<()> {
    let listener = TcpListener::bind(bind).await?;
    println!(
        "AURORA Inference API listening on http://{bind} using model {}",
        model
    );
    loop {
        let (socket, _) = listener.accept().await?;
        let default_model = model.to_string();
        tokio::spawn(async move {
            if let Err(err) = handle_inference_client(socket, default_model).await {
                eprintln!("inference api client error: {err:#}");
            }
        });
    }
}

async fn handle_inference_client(mut socket: TcpStream, default_model: String) -> Result<()> {
    let mut buf = vec![0u8; 64 * 1024];
    let bytes_read = socket.read(&mut buf).await?;
    let request = String::from_utf8_lossy(&buf[..bytes_read]);
    let mut lines = request.lines();
    let request_line = lines.next().unwrap_or_default();
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or_default();
    let path = parts.next().unwrap_or_default();

    let response = match (method, path) {
        ("GET", "/v1/status") => {
            let body = serde_json::to_string(&inference_status("127.0.0.1:11435", &default_model))?;
            http_response(200, "application/json", &body)
        }
        ("POST", "/v1/infer") | ("POST", "/v1/chat") => {
            let body = request
                .split("\r\n\r\n")
                .nth(1)
                .or_else(|| request.split("\n\n").nth(1))
                .unwrap_or_default();
            let payload: InferenceRequest = serde_json::from_str(body)?;
            let model = payload.model.unwrap_or(default_model);
            let output = run_inference_backend(&model, &payload.prompt)?;
            let body = serde_json::to_string(&InferenceResponse {
                ok: true,
                model,
                backend: "ollama",
                response: output.trim().to_string(),
            })?;
            http_response(200, "application/json", &body)
        }
        _ => http_response(404, "application/json", "{\"ok\":false,\"error\":\"not found\"}"),
    };

    socket.write_all(response.as_bytes()).await?;
    socket.shutdown().await?;
    Ok(())
}

fn http_response(status: u16, content_type: &str, body: &str) -> String {
    let reason = match status {
        200 => "OK",
        404 => "Not Found",
        _ => "Error",
    };
    format!(
        "HTTP/1.1 {status} {reason}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    )
}

fn run_inference_backend(model: &str, prompt: &str) -> Result<String> {
    let output = Command::new("ollama")
        .args(["run", model, prompt])
        .output()
        .map_err(|err| anyhow::anyhow!("failed to start ollama backend: {err}"))?;
    if !output.status.success() {
        bail!(
            "ollama backend failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
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

#[derive(Debug, Clone)]
struct RuntimeExecutionReport {
    kernel: String,
    size: usize,
    iterations: usize,
    threads: usize,
    serial_ms: f64,
    parallel_ms: f64,
    speedup: f64,
    throughput_flops: f64,
}

fn print_runtime_report(report: &RuntimeExecutionReport) {
    println!("\n{} Results:", "Runtime".green().bold());
    println!("  Kernel: {}", report.kernel);
    println!("  Threads: {}", report.threads);
    println!("  Work size: {}", report.size);
    println!("  Iterations: {}", report.iterations);
    println!("  Serial time: {:.2} ms", report.serial_ms);
    println!("  Parallel time: {:.2} ms", report.parallel_ms);
    println!("  Speedup: {:.2}x", report.speedup);
    println!("  Throughput: {}", format_flops(report.throughput_flops));
}

fn default_kernel_size(kernel: &str) -> usize {
    match kernel {
        "matmul" => 512,
        "fused-matmul-gelu" => 512,
        "training-step-int8" => 256,
        "inference-int8" => 256,
        "gaming-frame" => 2_000_000,
        "editing-color" => 2_000_000,
        "training-step" => 256,
        "vector-add" | "vector-mul" | "relu" | "gelu" => 8_000_000,
        "sigmoid" | "fused-add-relu" | "reduce-sum" | "reduce-max" => 8_000_000,
        "softmax" | "layer-norm" | "batch-norm" | "rmsnorm" | "transpose" => 4096,
        "attention" => 256,
        "im2col" => 64,
        "conv2d" => 64,
        _ => 1_000_000,
    }
}

fn describe_workload(kernel: &str, size: usize, device: &str) -> WorkloadCharacteristics {
    let mut workload = WorkloadCharacteristics::new();

    workload = match kernel {
        "matmul" | "fused-matmul-gelu" => {
            let bytes = (size * size * std::mem::size_of::<f32>() * 3) as u64;
            workload
                .with_gpu(true)
                .with_data_size((bytes / (1024 * 1024)).max(1))
                .with_compute_intensity(size as f64)
        }
        "training-step" => {
            let bytes = (size * size * std::mem::size_of::<f32>() * 4) as u64;
            workload
                .with_gpu(true)
                .with_data_size((bytes / (1024 * 1024)).max(1))
                .with_compute_intensity((size * size) as f64)
        }
        "training-step-int8" | "inference-int8" => {
            let bytes = (size * size) as u64;
            workload
                .with_gpu(false)
                .with_data_size((bytes / (1024 * 1024)).max(1))
                .with_compute_intensity((size * size) as f64 / 2.0)
        }
        "gaming-frame" | "editing-color" => {
            let bytes = (size * 4 * std::mem::size_of::<f32>() * 2) as u64;
            workload
                .with_gpu(true)
                .with_data_size((bytes / (1024 * 1024)).max(1))
                .with_compute_intensity(6.0)
        }
        "vector-add" | "vector-mul" | "relu" | "gelu" | "sigmoid" | "fused-add-relu" => {
            let bytes = (size * std::mem::size_of::<f32>() * 3) as u64;
            workload
                .with_gpu(device.eq_ignore_ascii_case("gpu"))
                .with_data_size((bytes / (1024 * 1024)).max(1))
                .with_compute_intensity(1.5)
        }
        "softmax" | "layer-norm" | "batch-norm" | "rmsnorm" | "transpose" => {
            let bytes = (size * size * std::mem::size_of::<f32>() * 3) as u64;
            workload
                .with_gpu(device.eq_ignore_ascii_case("gpu"))
                .with_data_size((bytes / (1024 * 1024)).max(1))
                .with_compute_intensity(size as f64 / 8.0)
        }
        "reduce-sum" | "reduce-max" => workload
            .with_gpu(device.eq_ignore_ascii_case("gpu"))
            .with_data_size(((size * std::mem::size_of::<f32>()) as u64 / (1024 * 1024)).max(1))
            .with_compute_intensity(1.0),
        "attention" => {
            let seq = size;
            let bytes = (seq * seq * std::mem::size_of::<f32>() * 4) as u64;
            workload
                .with_gpu(true)
                .with_data_size((bytes / (1024 * 1024)).max(1))
                .with_compute_intensity((seq * seq) as f64)
        }
        "im2col" => {
            let spatial = size * size;
            let bytes = (spatial * 64 * std::mem::size_of::<f32>() * 3) as u64;
            workload
                .with_gpu(true)
                .with_data_size((bytes / (1024 * 1024)).max(1))
                .with_compute_intensity((size * 9 * 16) as f64)
        }
        "conv2d" => {
            let spatial = size * size;
            let bytes = (spatial * 64 * std::mem::size_of::<f32>() * 3) as u64;
            workload
                .with_gpu(true)
                .with_data_size((bytes / (1024 * 1024)).max(1))
                .with_compute_intensity((size * 9 * 64) as f64)
        }
        _ => workload.with_data_size(1).with_compute_intensity(0.1),
    };

    if device.eq_ignore_ascii_case("cpu") {
        workload.requires_gpu = false;
    }

    workload
}

fn execute_cpu_kernel(
    kernel: &str,
    size: usize,
    iterations: usize,
    threads: usize,
) -> Result<RuntimeExecutionReport> {
    match kernel {
        "vector-add" => run_vector_binary_kernel("vector-add", size, iterations, threads, |a, b| a + b),
        "vector-mul" => run_vector_binary_kernel("vector-mul", size, iterations, threads, |a, b| a * b),
        "relu" => run_relu_kernel(size, iterations, threads),
        "gelu" => run_gelu_kernel(size, iterations, threads),
        "sigmoid" => run_sigmoid_kernel(size, iterations, threads),
        "fused-add-relu" => run_fused_add_relu_kernel(size, iterations, threads),
        "softmax" => run_softmax_kernel(size, iterations, threads),
        "layer-norm" => run_layer_norm_kernel(size, iterations, threads),
        "rmsnorm" => run_rmsnorm_kernel(size, iterations, threads),
        "batch-norm" => run_batch_norm_kernel(size, iterations, threads),
        "reduce-sum" => run_reduce_sum_kernel(size, iterations, threads),
        "reduce-max" => run_reduce_max_kernel(size, iterations, threads),
        "transpose" => run_transpose_kernel(size, iterations, threads),
        "im2col" => run_im2col_kernel(size, iterations, threads),
        "attention" => run_attention_kernel(size, iterations, threads),
        "conv2d" => run_conv2d_kernel(size, iterations, threads),
        "matmul" => run_matmul_kernel(size, iterations, threads),
        "fused-matmul-gelu" => run_fused_matmul_gelu_kernel(size, iterations, threads),
        "gaming-frame" => run_gaming_frame_kernel(size, iterations, threads),
        "editing-color" => run_editing_color_kernel(size, iterations, threads),
        "training-step" => run_training_step_kernel(size, iterations, threads),
        "training-step-int8" => run_training_step_int8_kernel(size, iterations, threads),
        "inference-int8" => run_inference_int8_kernel(size, iterations, threads),
        other => bail!("unknown kernel '{}'; supported kernels: vector-add, vector-mul, relu, gelu, sigmoid, fused-add-relu, softmax, layer-norm, rmsnorm, batch-norm, reduce-sum, reduce-max, transpose, im2col, attention, conv2d, matmul, fused-matmul-gelu, gaming-frame, editing-color, training-step, training-step-int8, inference-int8", other),
    }
}

fn with_rayon_pool<T, F>(threads: usize, func: F) -> Result<T>
where
    T: Send,
    F: FnOnce() -> Result<T> + Send,
{
    let pool = ThreadPoolBuilder::new()
        .num_threads(threads.max(1))
        .build()
        .map_err(|err| anyhow::anyhow!("failed to create rayon thread pool: {err}"))?;
    pool.install(func)
}

fn persistent_executor(threads: usize) -> Result<PersistentExecutor> {
    PersistentExecutor::new(threads.max(1), true)
        .map_err(|err| anyhow::anyhow!("failed to create persistent executor: {err}"))
}

fn run_vector_binary_kernel<F>(
    kernel: &str,
    size: usize,
    iterations: usize,
    threads: usize,
    op: F,
) -> Result<RuntimeExecutionReport>
where
    F: Fn(f32, f32) -> f32 + Copy + Send + Sync + 'static,
{
    let a = vec![1.0f32; size];
    let b = vec![2.0f32; size];
    let mut serial_out = vec![0.0f32; size];
    let a_tensor = HostTensor::from_vec(&[size], a.clone())?;
    let b_tensor = HostTensor::from_vec(&[size], b.clone())?;
    let mut runtime_out = HostTensor::zeros(&[size]);

    let serial_start = Instant::now();
    for _ in 0..iterations {
        for i in 0..size {
            serial_out[i] = op(a[i], b[i]);
        }
        std::hint::black_box(&serial_out);
    }
    let serial_ms = serial_start.elapsed().as_secs_f64() * 1000.0;

    let parallel_start = Instant::now();
    with_rayon_pool(threads, || {
        for _ in 0..iterations {
            match kernel {
                "vector-add" => tensor_add(&a_tensor, &b_tensor, &mut runtime_out)?,
                "vector-mul" => tensor_mul(&a_tensor, &b_tensor, &mut runtime_out)?,
                _ => bail!("unsupported vector kernel {}", kernel),
            }
            std::hint::black_box(runtime_out.as_slice());
        }
        Ok(())
    })?;
    let parallel_ms = parallel_start.elapsed().as_secs_f64() * 1000.0;

    let flops = size as f64 * iterations as f64;
    let throughput_flops = flops / (parallel_ms / 1000.0);

    Ok(RuntimeExecutionReport {
        kernel: kernel.to_string(),
        size,
        iterations,
        threads,
        serial_ms,
        parallel_ms,
        speedup: if parallel_ms > 0.0 { serial_ms / parallel_ms } else { 0.0 },
        throughput_flops,
    })
}

fn run_matmul_kernel(size: usize, iterations: usize, threads: usize) -> Result<RuntimeExecutionReport> {
    let n = size;
    let a = vec![1.0f32; n * n];
    let b = vec![1.0f32; n * n];
    let mut serial_out = vec![0.0f32; n * n];
    let a_tensor = HostTensor::from_vec(&[n, n], a.clone())?;
    let b_tensor = HostTensor::from_vec(&[n, n], b.clone())?;
    let mut runtime_out = HostTensor::zeros(&[n, n]);

    let serial_start = Instant::now();
    for _ in 0..iterations {
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0f32;
                for k in 0..n {
                    sum += a[i * n + k] * b[k * n + j];
                }
                serial_out[i * n + j] = sum;
            }
        }
        std::hint::black_box(&serial_out);
    }
    let serial_ms = serial_start.elapsed().as_secs_f64() * 1000.0;

    let parallel_start = Instant::now();
    with_rayon_pool(threads, || {
        for _ in 0..iterations {
            tensor_matmul(&a_tensor, &b_tensor, &mut runtime_out)?;
            std::hint::black_box(runtime_out.as_slice());
        }
        Ok(())
    })?;
    let parallel_ms = parallel_start.elapsed().as_secs_f64() * 1000.0;

    let flops = 2.0 * (n as f64).powi(3) * iterations as f64;
    let throughput_flops = flops / (parallel_ms / 1000.0);

    Ok(RuntimeExecutionReport {
        kernel: "matmul".to_string(),
        size,
        iterations,
        threads,
        serial_ms,
        parallel_ms,
        speedup: if parallel_ms > 0.0 { serial_ms / parallel_ms } else { 0.0 },
        throughput_flops,
    })
}

fn run_relu_kernel(size: usize, iterations: usize, threads: usize) -> Result<RuntimeExecutionReport> {
    let input_data: Vec<f32> = (0..size)
        .map(|i| if i % 2 == 0 { i as f32 * 0.001 } else { -(i as f32) * 0.001 })
        .collect();
    let input = HostTensor::from_vec(&[size], input_data.clone())?;
    let mut runtime_out = HostTensor::zeros(&[size]);
    let mut serial_out = vec![0.0f32; size];

    let serial_start = Instant::now();
    for _ in 0..iterations {
        for i in 0..size {
            serial_out[i] = input_data[i].max(0.0);
        }
        std::hint::black_box(&serial_out);
    }
    let serial_ms = serial_start.elapsed().as_secs_f64() * 1000.0;

    let parallel_start = Instant::now();
    with_rayon_pool(threads, || {
        for _ in 0..iterations {
            tensor_relu(&input, &mut runtime_out)?;
            std::hint::black_box(runtime_out.as_slice());
        }
        Ok(())
    })?;
    let parallel_ms = parallel_start.elapsed().as_secs_f64() * 1000.0;

    let flops = size as f64 * iterations as f64;
    let throughput_flops = flops / (parallel_ms / 1000.0);

    Ok(RuntimeExecutionReport {
        kernel: "relu".to_string(),
        size,
        iterations,
        threads,
        serial_ms,
        parallel_ms,
        speedup: if parallel_ms > 0.0 { serial_ms / parallel_ms } else { 0.0 },
        throughput_flops,
    })
}

fn run_gelu_kernel(size: usize, iterations: usize, threads: usize) -> Result<RuntimeExecutionReport> {
    let input_data: Vec<f32> = (0..size)
        .map(|i| ((i % 4096) as f32 * 0.0025) - 5.0)
        .collect();
    let input = HostTensor::from_vec(&[size], input_data.clone())?;
    let mut runtime_out = HostTensor::zeros(&[size]);
    let mut serial_out = vec![0.0f32; size];

    let serial_start = Instant::now();
    for _ in 0..iterations {
        for i in 0..size {
            let x = input_data[i];
            let c = (2.0 / std::f32::consts::PI).sqrt();
            serial_out[i] = 0.5 * x * (1.0 + (c * (x + 0.044_715 * x.powi(3))).tanh());
        }
        std::hint::black_box(&serial_out);
    }
    let serial_ms = serial_start.elapsed().as_secs_f64() * 1000.0;

    let parallel_start = Instant::now();
    with_rayon_pool(threads, || {
        for _ in 0..iterations {
            tensor_gelu(&input, &mut runtime_out)?;
            std::hint::black_box(runtime_out.as_slice());
        }
        Ok(())
    })?;
    let parallel_ms = parallel_start.elapsed().as_secs_f64() * 1000.0;

    let flops = size as f64 * iterations as f64 * 8.0;
    let throughput_flops = flops / (parallel_ms / 1000.0);

    Ok(RuntimeExecutionReport {
        kernel: "gelu".to_string(),
        size,
        iterations,
        threads,
        serial_ms,
        parallel_ms,
        speedup: if parallel_ms > 0.0 { serial_ms / parallel_ms } else { 0.0 },
        throughput_flops,
    })
}

fn run_sigmoid_kernel(size: usize, iterations: usize, threads: usize) -> Result<RuntimeExecutionReport> {
    let input_data: Vec<f32> = (0..size)
        .map(|i| ((i % 4096) as f32 * 0.0025) - 5.0)
        .collect();
    let input = HostTensor::from_vec(&[size], input_data.clone())?;
    let mut runtime_out = HostTensor::zeros(&[size]);
    let mut serial_out = vec![0.0f32; size];

    let serial_start = Instant::now();
    for _ in 0..iterations {
        for i in 0..size {
            serial_out[i] = 1.0 / (1.0 + (-input_data[i]).exp());
        }
        std::hint::black_box(&serial_out);
    }
    let serial_ms = serial_start.elapsed().as_secs_f64() * 1000.0;

    let parallel_start = Instant::now();
    with_rayon_pool(threads, || {
        for _ in 0..iterations {
            tensor_sigmoid(&input, &mut runtime_out)?;
            std::hint::black_box(runtime_out.as_slice());
        }
        Ok(())
    })?;
    let parallel_ms = parallel_start.elapsed().as_secs_f64() * 1000.0;

    let flops = size as f64 * iterations as f64 * 4.0;
    let throughput_flops = flops / (parallel_ms / 1000.0);

    Ok(RuntimeExecutionReport {
        kernel: "sigmoid".to_string(),
        size,
        iterations,
        threads,
        serial_ms,
        parallel_ms,
        speedup: if parallel_ms > 0.0 { serial_ms / parallel_ms } else { 0.0 },
        throughput_flops,
    })
}

fn run_fused_add_relu_kernel(size: usize, iterations: usize, threads: usize) -> Result<RuntimeExecutionReport> {
    let a = (0..size)
        .map(|i| ((i % 2048) as f32 * 0.003) - 3.0)
        .collect::<Vec<_>>();
    let b = (0..size)
        .map(|i| 1.5 - ((i % 1024) as f32 * 0.002))
        .collect::<Vec<_>>();
    let mut serial_out = vec![0.0f32; size];
    let a_tensor = HostTensor::from_vec(&[size], a.clone())?;
    let b_tensor = HostTensor::from_vec(&[size], b.clone())?;
    let mut runtime_out = HostTensor::zeros(&[size]);

    let serial_start = Instant::now();
    for _ in 0..iterations {
        for i in 0..size {
            serial_out[i] = (a[i] + b[i]).max(0.0);
        }
        std::hint::black_box(&serial_out);
    }
    let serial_ms = serial_start.elapsed().as_secs_f64() * 1000.0;

    let parallel_start = Instant::now();
    with_rayon_pool(threads, || {
        for _ in 0..iterations {
            tensor_fused_add_relu(&a_tensor, &b_tensor, &mut runtime_out)?;
            std::hint::black_box(runtime_out.as_slice());
        }
        Ok(())
    })?;
    let parallel_ms = parallel_start.elapsed().as_secs_f64() * 1000.0;

    let flops = size as f64 * iterations as f64 * 2.0;
    let throughput_flops = flops / (parallel_ms / 1000.0);

    Ok(RuntimeExecutionReport {
        kernel: "fused-add-relu".to_string(),
        size,
        iterations,
        threads,
        serial_ms,
        parallel_ms,
        speedup: if parallel_ms > 0.0 { serial_ms / parallel_ms } else { 0.0 },
        throughput_flops,
    })
}

fn run_softmax_kernel(size: usize, iterations: usize, threads: usize) -> Result<RuntimeExecutionReport> {
    let width = size;
    let rows = 256usize;
    let total = width * rows;
    let input_data: Vec<f32> = (0..total).map(|i| (i % width) as f32 * 0.001).collect();
    let input = HostTensor::from_vec(&[rows, width], input_data.clone())?;
    let mut runtime_out = HostTensor::zeros(&[rows, width]);
    let mut serial_out = vec![0.0f32; total];

    let serial_start = Instant::now();
    for _ in 0..iterations {
        for row in 0..rows {
            let start = row * width;
            let slice = &input_data[start..start + width];
            let max_val = slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for i in 0..width {
                let value = (slice[i] - max_val).exp();
                serial_out[start + i] = value;
                sum += value;
            }
            for i in 0..width {
                serial_out[start + i] /= sum;
            }
        }
        std::hint::black_box(&serial_out);
    }
    let serial_ms = serial_start.elapsed().as_secs_f64() * 1000.0;

    let parallel_start = Instant::now();
    with_rayon_pool(threads, || {
        for _ in 0..iterations {
            tensor_softmax(&input, &mut runtime_out, -1)?;
            std::hint::black_box(runtime_out.as_slice());
        }
        Ok(())
    })?;
    let parallel_ms = parallel_start.elapsed().as_secs_f64() * 1000.0;

    let flops = total as f64 * iterations as f64 * 6.0;
    let throughput_flops = flops / (parallel_ms / 1000.0);

    Ok(RuntimeExecutionReport {
        kernel: "softmax".to_string(),
        size,
        iterations,
        threads,
        serial_ms,
        parallel_ms,
        speedup: if parallel_ms > 0.0 { serial_ms / parallel_ms } else { 0.0 },
        throughput_flops,
    })
}

fn run_layer_norm_kernel(size: usize, iterations: usize, threads: usize) -> Result<RuntimeExecutionReport> {
    let width = size;
    let rows = 256usize;
    let total = width * rows;
    let input_data: Vec<f32> = (0..total)
        .map(|i| ((i % width) as f32 * 0.002) - 3.0)
        .collect();
    let input = HostTensor::from_vec(&[rows, width], input_data.clone())?;
    let gamma = HostTensor::filled(&[width], 1.0);
    let beta = HostTensor::filled(&[width], 0.0);
    let mut runtime_out = HostTensor::zeros(&[rows, width]);
    let mut serial_out = vec![0.0f32; total];

    let serial_start = Instant::now();
    for _ in 0..iterations {
        for row in 0..rows {
            let start = row * width;
            let slice = &input_data[start..start + width];
            let mean = slice.iter().copied().sum::<f32>() / width as f32;
            let variance = slice
                .iter()
                .map(|v| {
                    let d = *v - mean;
                    d * d
                })
                .sum::<f32>()
                / width as f32;
            let denom = (variance + 1e-5).sqrt();
            for i in 0..width {
                serial_out[start + i] = (slice[i] - mean) / denom;
            }
        }
        std::hint::black_box(&serial_out);
    }
    let serial_ms = serial_start.elapsed().as_secs_f64() * 1000.0;

    let parallel_start = Instant::now();
    with_rayon_pool(threads, || {
        for _ in 0..iterations {
            tensor_layer_norm(&input, &gamma, &beta, &mut runtime_out, 1e-5)?;
            std::hint::black_box(runtime_out.as_slice());
        }
        Ok(())
    })?;
    let parallel_ms = parallel_start.elapsed().as_secs_f64() * 1000.0;

    let flops = total as f64 * iterations as f64 * 8.0;
    let throughput_flops = flops / (parallel_ms / 1000.0);

    Ok(RuntimeExecutionReport {
        kernel: "layer-norm".to_string(),
        size,
        iterations,
        threads,
        serial_ms,
        parallel_ms,
        speedup: if parallel_ms > 0.0 { serial_ms / parallel_ms } else { 0.0 },
        throughput_flops,
    })
}

fn run_batch_norm_kernel(size: usize, iterations: usize, threads: usize) -> Result<RuntimeExecutionReport> {
    let width = size;
    let rows = 256usize;
    let total = width * rows;
    let input_data: Vec<f32> = (0..total)
        .map(|i| ((i % width) as f32 * 0.0015) - 2.0)
        .collect();
    let input = HostTensor::from_vec(&[rows, width], input_data.clone())?;
    let gamma = HostTensor::filled(&[rows, width], 1.0);
    let beta = HostTensor::filled(&[rows, width], 0.0);
    let running_mean = HostTensor::filled(&[rows, width], 0.25);
    let running_var = HostTensor::filled(&[rows, width], 0.5);
    let mut runtime_out = HostTensor::zeros(&[rows, width]);
    let mut serial_out = vec![0.0f32; total];

    let serial_start = Instant::now();
    for _ in 0..iterations {
        for i in 0..total {
            serial_out[i] = (input_data[i] - 0.25) / (0.5f32 + 1e-5).sqrt();
        }
        std::hint::black_box(&serial_out);
    }
    let serial_ms = serial_start.elapsed().as_secs_f64() * 1000.0;

    let parallel_start = Instant::now();
    with_rayon_pool(threads, || {
        for _ in 0..iterations {
            tensor_batch_norm(
                &input,
                &gamma,
                &beta,
                &running_mean,
                &running_var,
                &mut runtime_out,
                1e-5,
            )?;
            std::hint::black_box(runtime_out.as_slice());
        }
        Ok(())
    })?;
    let parallel_ms = parallel_start.elapsed().as_secs_f64() * 1000.0;

    let flops = total as f64 * iterations as f64 * 6.0;
    let throughput_flops = flops / (parallel_ms / 1000.0);

    Ok(RuntimeExecutionReport {
        kernel: "batch-norm".to_string(),
        size,
        iterations,
        threads,
        serial_ms,
        parallel_ms,
        speedup: if parallel_ms > 0.0 { serial_ms / parallel_ms } else { 0.0 },
        throughput_flops,
    })
}

fn run_reduce_sum_kernel(size: usize, iterations: usize, threads: usize) -> Result<RuntimeExecutionReport> {
    let width = 1024usize;
    let rows = size.max(width) / width;
    let total = rows * width;
    let input_data: Vec<f32> = (0..total).map(|i| (i % width) as f32 * 0.125).collect();
    let input = HostTensor::from_vec(&[rows, width], input_data.clone())?;
    let mut runtime_out = HostTensor::zeros(&[rows]);
    let mut serial_out = vec![0.0f32; rows];

    let serial_start = Instant::now();
    for _ in 0..iterations {
        for row in 0..rows {
            let start = row * width;
            serial_out[row] = input_data[start..start + width].iter().copied().sum::<f32>();
        }
        std::hint::black_box(&serial_out);
    }
    let serial_ms = serial_start.elapsed().as_secs_f64() * 1000.0;

    let parallel_start = Instant::now();
    with_rayon_pool(threads, || {
        for _ in 0..iterations {
            tensor_reduce_sum_last(&input, &mut runtime_out)?;
            std::hint::black_box(runtime_out.as_slice());
        }
        Ok(())
    })?;
    let parallel_ms = parallel_start.elapsed().as_secs_f64() * 1000.0;
    let flops = total as f64 * iterations as f64;
    let throughput_flops = flops / (parallel_ms / 1000.0);

    Ok(RuntimeExecutionReport {
        kernel: "reduce-sum".to_string(),
        size,
        iterations,
        threads,
        serial_ms,
        parallel_ms,
        speedup: if parallel_ms > 0.0 { serial_ms / parallel_ms } else { 0.0 },
        throughput_flops,
    })
}

fn run_reduce_max_kernel(size: usize, iterations: usize, threads: usize) -> Result<RuntimeExecutionReport> {
    let width = 1024usize;
    let rows = size.max(width) / width;
    let total = rows * width;
    let input_data: Vec<f32> = (0..total)
        .map(|i| ((i % width) as f32 * 0.03125) - 12.0)
        .collect();
    let input = HostTensor::from_vec(&[rows, width], input_data.clone())?;
    let mut runtime_out = HostTensor::zeros(&[rows]);
    let mut serial_out = vec![0.0f32; rows];

    let serial_start = Instant::now();
    for _ in 0..iterations {
        for row in 0..rows {
            let start = row * width;
            serial_out[row] = input_data[start..start + width]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
        }
        std::hint::black_box(&serial_out);
    }
    let serial_ms = serial_start.elapsed().as_secs_f64() * 1000.0;

    let parallel_start = Instant::now();
    with_rayon_pool(threads, || {
        for _ in 0..iterations {
            tensor_reduce_max_last(&input, &mut runtime_out)?;
            std::hint::black_box(runtime_out.as_slice());
        }
        Ok(())
    })?;
    let parallel_ms = parallel_start.elapsed().as_secs_f64() * 1000.0;
    let flops = total as f64 * iterations as f64;
    let throughput_flops = flops / (parallel_ms / 1000.0);

    Ok(RuntimeExecutionReport {
        kernel: "reduce-max".to_string(),
        size,
        iterations,
        threads,
        serial_ms,
        parallel_ms,
        speedup: if parallel_ms > 0.0 { serial_ms / parallel_ms } else { 0.0 },
        throughput_flops,
    })
}

fn run_transpose_kernel(size: usize, iterations: usize, threads: usize) -> Result<RuntimeExecutionReport> {
    let rows = 256usize;
    let cols = size;
    let total = rows * cols;
    let input_data: Vec<f32> = (0..total).map(|i| i as f32).collect();
    let input = HostTensor::from_vec(&[rows, cols], input_data.clone())?;
    let mut runtime_out = HostTensor::zeros(&[cols, rows]);
    let mut serial_out = vec![0.0f32; total];

    let serial_start = Instant::now();
    for _ in 0..iterations {
        for r in 0..rows {
            for c in 0..cols {
                serial_out[c * rows + r] = input_data[r * cols + c];
            }
        }
        std::hint::black_box(&serial_out);
    }
    let serial_ms = serial_start.elapsed().as_secs_f64() * 1000.0;

    let parallel_start = Instant::now();
    with_rayon_pool(threads, || {
        for _ in 0..iterations {
            tensor_transpose_last_two(&input, &mut runtime_out)?;
            std::hint::black_box(runtime_out.as_slice());
        }
        Ok(())
    })?;
    let parallel_ms = parallel_start.elapsed().as_secs_f64() * 1000.0;
    let flops = total as f64 * iterations as f64;
    let throughput_flops = flops / (parallel_ms / 1000.0);

    Ok(RuntimeExecutionReport {
        kernel: "transpose".to_string(),
        size,
        iterations,
        threads,
        serial_ms,
        parallel_ms,
        speedup: if parallel_ms > 0.0 { serial_ms / parallel_ms } else { 0.0 },
        throughput_flops,
    })
}

fn run_rmsnorm_kernel(size: usize, iterations: usize, threads: usize) -> Result<RuntimeExecutionReport> {
    let width = size;
    let rows = 256usize;
    let total = width * rows;
    let input_data: Vec<f32> = (0..total)
        .map(|i| ((i % width) as f32 * 0.002) - 3.0)
        .collect();
    let input = HostTensor::from_vec(&[rows, width], input_data.clone())?;
    let weight = HostTensor::filled(&[width], 1.0);
    let mut runtime_out = HostTensor::zeros(&[rows, width]);
    let mut serial_out = vec![0.0f32; total];

    let serial_start = Instant::now();
    for _ in 0..iterations {
        for row in 0..rows {
            let start = row * width;
            let slice = &input_data[start..start + width];
            let rms = (slice.iter().map(|v| v * v).sum::<f32>() / width as f32 + 1e-5).sqrt();
            for i in 0..width {
                serial_out[start + i] = slice[i] / rms;
            }
        }
        std::hint::black_box(&serial_out);
    }
    let serial_ms = serial_start.elapsed().as_secs_f64() * 1000.0;

    let parallel_start = Instant::now();
    with_rayon_pool(threads, || {
        for _ in 0..iterations {
            tensor_rms_norm(&input, &weight, &mut runtime_out, 1e-5)?;
            std::hint::black_box(runtime_out.as_slice());
        }
        Ok(())
    })?;
    let parallel_ms = parallel_start.elapsed().as_secs_f64() * 1000.0;
    let flops = total as f64 * iterations as f64 * 6.0;
    let throughput_flops = flops / (parallel_ms / 1000.0);

    Ok(RuntimeExecutionReport {
        kernel: "rmsnorm".to_string(),
        size,
        iterations,
        threads,
        serial_ms,
        parallel_ms,
        speedup: if parallel_ms > 0.0 { serial_ms / parallel_ms } else { 0.0 },
        throughput_flops,
    })
}

fn run_im2col_kernel(size: usize, iterations: usize, threads: usize) -> Result<RuntimeExecutionReport> {
    let spatial = size;
    let batch = 4usize;
    let channels = 8usize;
    let kernel = (3usize, 3usize);
    let stride = (1usize, 1usize);
    let padding = (1usize, 1usize);
    let input_len = batch * channels * spatial * spatial;
    let out_h = spatial;
    let out_w = spatial;
    let rows = batch * out_h * out_w;
    let cols = channels * kernel.0 * kernel.1;
    let input_data = (0..input_len)
        .map(|i| ((i % 97) as f32 * 0.01) - 0.5)
        .collect::<Vec<_>>();
    let input = HostTensor::from_vec(&[batch, channels, spatial, spatial], input_data.clone())?;
    let mut runtime_out = HostTensor::zeros(&[rows, cols]);
    let mut serial_out = vec![0.0f32; rows * cols];

    let serial_start = Instant::now();
    for _ in 0..iterations {
        serial_im2col_nchw(&input_data, &mut serial_out, batch, channels, spatial, spatial, kernel, stride, padding);
        std::hint::black_box(&serial_out);
    }
    let serial_ms = serial_start.elapsed().as_secs_f64() * 1000.0;

    let parallel_start = Instant::now();
    with_rayon_pool(threads, || {
        for _ in 0..iterations {
            tensor_im2col(&input, &mut runtime_out, kernel, stride, padding)?;
            std::hint::black_box(runtime_out.as_slice());
        }
        Ok(())
    })?;
    let parallel_ms = parallel_start.elapsed().as_secs_f64() * 1000.0;
    let flops = (rows * cols) as f64 * iterations as f64;
    let throughput_flops = flops / (parallel_ms / 1000.0);

    Ok(RuntimeExecutionReport {
        kernel: "im2col".to_string(),
        size,
        iterations,
        threads,
        serial_ms,
        parallel_ms,
        speedup: if parallel_ms > 0.0 { serial_ms / parallel_ms } else { 0.0 },
        throughput_flops,
    })
}

fn run_attention_kernel(size: usize, iterations: usize, threads: usize) -> Result<RuntimeExecutionReport> {
    let batch = 4usize;
    let seq = size;
    let dim = 64usize;
    let value_dim = 64usize;
    let q = (0..batch * seq * dim)
        .map(|i| ((i % 127) as f32 * 0.01) - 0.5)
        .collect::<Vec<_>>();
    let k = (0..batch * seq * dim)
        .map(|i| ((i % 89) as f32 * 0.012) - 0.4)
        .collect::<Vec<_>>();
    let v = (0..batch * seq * value_dim)
        .map(|i| ((i % 73) as f32 * 0.02) - 0.7)
        .collect::<Vec<_>>();

    let query = HostTensor::from_vec(&[batch, seq, dim], q.clone())?;
    let key = HostTensor::from_vec(&[batch, seq, dim], k.clone())?;
    let value = HostTensor::from_vec(&[batch, seq, value_dim], v.clone())?;
    let mut runtime_out = HostTensor::zeros(&[batch, seq, value_dim]);
    let mut serial_out = vec![0.0f32; batch * seq * value_dim];

    let serial_start = Instant::now();
    for _ in 0..iterations {
        serial_attention(&q, &k, &v, &mut serial_out, batch, seq, dim, value_dim);
        std::hint::black_box(&serial_out);
    }
    let serial_ms = serial_start.elapsed().as_secs_f64() * 1000.0;

    let parallel_start = Instant::now();
    let executor = persistent_executor(threads)?;
    for _ in 0..iterations {
        executor.install(|_| {
            tensor_attention(&query, &key, &value, &mut runtime_out)?;
            Ok(())
        })?;
        std::hint::black_box(runtime_out.as_slice());
    }
    let parallel_ms = parallel_start.elapsed().as_secs_f64() * 1000.0;
    let flops = 2.0 * batch as f64 * seq as f64 * seq as f64 * dim as f64 * iterations as f64;
    let throughput_flops = flops / (parallel_ms / 1000.0);

    Ok(RuntimeExecutionReport {
        kernel: "attention".to_string(),
        size,
        iterations,
        threads,
        serial_ms,
        parallel_ms,
        speedup: if parallel_ms > 0.0 { serial_ms / parallel_ms } else { 0.0 },
        throughput_flops,
    })
}

fn run_conv2d_kernel(size: usize, iterations: usize, threads: usize) -> Result<RuntimeExecutionReport> {
    let spatial = size;
    let batch = 8usize;
    let in_channels = 16usize;
    let out_channels = 32usize;
    let kernel = 3usize;
    let stride = (1usize, 1usize);
    let padding = (1usize, 1usize);
    let out_h = spatial;
    let out_w = spatial;
    let input_len = batch * in_channels * spatial * spatial;
    let weight_len = out_channels * in_channels * kernel * kernel;
    let output_len = batch * out_channels * out_h * out_w;

    let input_data = (0..input_len)
        .map(|i| ((i % 97) as f32 * 0.01) - 0.5)
        .collect::<Vec<_>>();
    let weight_data = (0..weight_len)
        .map(|i| ((i % 17) as f32 * 0.02) - 0.1)
        .collect::<Vec<_>>();
    let bias_data = (0..out_channels)
        .map(|i| i as f32 * 0.001)
        .collect::<Vec<_>>();

    let input = HostTensor::from_vec(&[batch, in_channels, spatial, spatial], input_data.clone())?;
    let weight = HostTensor::from_vec(&[out_channels, in_channels, kernel, kernel], weight_data.clone())?;
    let bias = HostTensor::from_vec(&[out_channels], bias_data.clone())?;
    let mut runtime_out = HostTensor::zeros(&[batch, out_channels, out_h, out_w]);
    let mut serial_out = vec![0.0f32; output_len];

    let serial_start = Instant::now();
    for _ in 0..iterations {
        serial_conv2d_nchw(
            &input_data,
            &weight_data,
            Some(&bias_data),
            &mut serial_out,
            batch,
            in_channels,
            spatial,
            spatial,
            out_channels,
            kernel,
            kernel,
            stride,
            padding,
        );
        std::hint::black_box(&serial_out);
    }
    let serial_ms = serial_start.elapsed().as_secs_f64() * 1000.0;

    let parallel_start = Instant::now();
    with_rayon_pool(threads, || {
        for _ in 0..iterations {
            tensor_conv2d(&input, &weight, Some(&bias), &mut runtime_out, stride, padding)?;
            std::hint::black_box(runtime_out.as_slice());
        }
        Ok(())
    })?;
    let parallel_ms = parallel_start.elapsed().as_secs_f64() * 1000.0;

    let flops =
        2.0 * batch as f64 * out_channels as f64 * out_h as f64 * out_w as f64 * in_channels as f64
            * kernel as f64 * kernel as f64 * iterations as f64;
    let throughput_flops = flops / (parallel_ms / 1000.0);

    Ok(RuntimeExecutionReport {
        kernel: "conv2d".to_string(),
        size,
        iterations,
        threads,
        serial_ms,
        parallel_ms,
        speedup: if parallel_ms > 0.0 { serial_ms / parallel_ms } else { 0.0 },
        throughput_flops,
    })
}

fn run_gaming_frame_kernel(size: usize, iterations: usize, threads: usize) -> Result<RuntimeExecutionReport> {
    let pixels = size.max(1024);
    let input = (0..pixels * 4)
        .map(|i| ((i % 255) as f32) / 255.0)
        .collect::<Vec<_>>();
    let bloom = (0..pixels * 4)
        .map(|i| (((i * 7) % 255) as f32) / 512.0)
        .collect::<Vec<_>>();
    let mut serial_out = vec![0.0f32; pixels * 4];
    let mut runtime_out = vec![0.0f32; pixels * 4];

    let serial_start = Instant::now();
    for _ in 0..iterations {
        for px in 0..pixels {
            let base = px * 4;
            for ch in 0..3 {
                let v = input[base + ch] * 1.18 + bloom[base + ch] * 0.42;
                let sharpened = v + (v - 0.5) * 0.12;
                serial_out[base + ch] = sharpened.clamp(0.0, 1.0);
            }
            serial_out[base + 3] = 1.0;
        }
        std::hint::black_box(&serial_out);
    }
    let serial_ms = serial_start.elapsed().as_secs_f64() * 1000.0;

    let parallel_start = Instant::now();
    with_rayon_pool(threads, || {
        for _ in 0..iterations {
            const BLOCK_PIXELS: usize = 4096;
            runtime_out
                .par_chunks_mut(BLOCK_PIXELS * 4)
                .enumerate()
                .for_each(|(block_idx, out_chunk)| {
                    let in_chunk = &input[block_idx * BLOCK_PIXELS * 4
                        ..block_idx * BLOCK_PIXELS * 4 + out_chunk.len()];
                    let bloom_chunk = &bloom[block_idx * BLOCK_PIXELS * 4
                        ..block_idx * BLOCK_PIXELS * 4 + out_chunk.len()];
                    process_gaming_pixels_block(in_chunk, bloom_chunk, out_chunk);
                });
            std::hint::black_box(&runtime_out);
        }
        Ok(())
    })?;
    let parallel_ms = parallel_start.elapsed().as_secs_f64() * 1000.0;

    Ok(RuntimeExecutionReport {
        kernel: "gaming-frame".to_string(),
        size: pixels,
        iterations,
        threads,
        serial_ms,
        parallel_ms,
        speedup: if parallel_ms > 0.0 { serial_ms / parallel_ms } else { 0.0 },
        throughput_flops: (pixels * 16 * iterations) as f64 / (parallel_ms / 1000.0),
    })
}

fn run_editing_color_kernel(size: usize, iterations: usize, threads: usize) -> Result<RuntimeExecutionReport> {
    let pixels = size.max(1024);
    let input = (0..pixels * 4)
        .map(|i| (((i * 11) % 1024) as f32) / 1024.0)
        .collect::<Vec<_>>();
    let mut serial_out = vec![0.0f32; pixels * 4];
    let mut runtime_out = vec![0.0f32; pixels * 4];

    let serial_start = Instant::now();
    for _ in 0..iterations {
        for px in 0..pixels {
            let base = px * 4;
            let exposure = 1.09;
            let contrast = 1.16;
            for ch in 0..3 {
                let mut v = input[base + ch] * exposure;
                v = ((v - 0.5) * contrast + 0.5).clamp(0.0, 1.0);
                serial_out[base + ch] = v.powf(0.92);
            }
            serial_out[base + 3] = input[base + 3];
        }
        std::hint::black_box(&serial_out);
    }
    let serial_ms = serial_start.elapsed().as_secs_f64() * 1000.0;

    let parallel_start = Instant::now();
    with_rayon_pool(threads, || {
        for _ in 0..iterations {
            const BLOCK_PIXELS: usize = 4096;
            runtime_out
                .par_chunks_mut(BLOCK_PIXELS * 4)
                .enumerate()
                .for_each(|(block_idx, out_chunk)| {
                    let in_chunk = &input[block_idx * BLOCK_PIXELS * 4
                        ..block_idx * BLOCK_PIXELS * 4 + out_chunk.len()];
                    process_editing_pixels_block(in_chunk, out_chunk);
                });
            std::hint::black_box(&runtime_out);
        }
        Ok(())
    })?;
    let parallel_ms = parallel_start.elapsed().as_secs_f64() * 1000.0;

    Ok(RuntimeExecutionReport {
        kernel: "editing-color".to_string(),
        size: pixels,
        iterations,
        threads,
        serial_ms,
        parallel_ms,
        speedup: if parallel_ms > 0.0 { serial_ms / parallel_ms } else { 0.0 },
        throughput_flops: (pixels * 18 * iterations) as f64 / (parallel_ms / 1000.0),
    })
}

fn process_gaming_pixels_block(input: &[f32], bloom: &[f32], out: &mut [f32]) {
    for px in 0..(out.len() / 4) {
        let base = px * 4;
        let r = (input[base] * 1.18 + bloom[base] * 0.42).mul_add(1.12, -0.06);
        let g = (input[base + 1] * 1.18 + bloom[base + 1] * 0.42).mul_add(1.12, -0.06);
        let b = (input[base + 2] * 1.18 + bloom[base + 2] * 0.42).mul_add(1.12, -0.06);
        out[base] = r.clamp(0.0, 1.0);
        out[base + 1] = g.clamp(0.0, 1.0);
        out[base + 2] = b.clamp(0.0, 1.0);
        out[base + 3] = 1.0;
    }
}

fn process_editing_pixels_block(input: &[f32], out: &mut [f32]) {
    for px in 0..(out.len() / 4) {
        let base = px * 4;
        for ch in 0..3 {
            let exposed = input[base + ch] * 1.09;
            let contrasted = ((exposed - 0.5) * 1.16 + 0.5).clamp(0.0, 1.0);
            out[base + ch] = contrasted.sqrt();
        }
        out[base + 3] = input[base + 3];
    }
}

fn run_training_step_kernel(size: usize, iterations: usize, threads: usize) -> Result<RuntimeExecutionReport> {
    let n = size.max(32);
    let lhs = HostTensor::from_vec(
        &[n, n],
        (0..n * n).map(|i| ((i % 113) as f32 * 0.008) - 0.4).collect(),
    )?;
    let rhs = HostTensor::from_vec(
        &[n, n],
        (0..n * n).map(|i| ((i % 97) as f32 * 0.006) - 0.3).collect(),
    )?;
    let mut serial_mm = vec![0.0f32; n * n];
    let mut serial_gelu = vec![0.0f32; n * n];
    let mut serial_loss = vec![0.0f32; n];
    let executor = persistent_executor(threads)?;
    let gelu_c = (2.0 / std::f32::consts::PI).sqrt();

    let serial_start = Instant::now();
    for _ in 0..iterations {
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0f32;
                for k in 0..n {
                    sum += lhs.as_slice()[i * n + k] * rhs.as_slice()[k * n + j];
                }
                serial_mm[i * n + j] = sum;
                serial_gelu[i * n + j] =
                    0.5 * sum * (1.0 + (gelu_c * (sum + 0.044_715 * sum.powi(3))).tanh());
            }
        }
        for row in 0..n {
            let start = row * n;
            serial_loss[row] = serial_gelu[start..start + n].iter().copied().sum::<f32>();
        }
        std::hint::black_box(&serial_loss);
    }
    let serial_ms = serial_start.elapsed().as_secs_f64() * 1000.0;

    let parallel_start = Instant::now();
    let mut final_loss: Option<HostTensor> = None;
    for _ in 0..iterations {
        let graph = ExecutionGraph::new()
            .with_input("lhs", lhs.clone())
            .with_input("rhs", rhs.clone())
            .add_node(GraphNode::Matmul {
                lhs: "lhs".to_string(),
                rhs: "rhs".to_string(),
                output: "mm".to_string(),
            })
            .add_node(GraphNode::Gelu {
                input: "mm".to_string(),
                output: "act".to_string(),
            })
            .add_node(GraphNode::ReduceSum {
                input: "act".to_string(),
                output: "loss".to_string(),
            });
        let result = executor.execute_graph(&graph)?;
        final_loss = Some(
            result
                .tensors
                .get("loss")
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("training-step graph missing loss tensor"))?,
        );
        std::hint::black_box(final_loss.as_ref().map(HostTensor::as_slice));
    }
    let parallel_ms = parallel_start.elapsed().as_secs_f64() * 1000.0;

    Ok(RuntimeExecutionReport {
        kernel: "training-step".to_string(),
        size: n,
        iterations,
        threads,
        serial_ms,
        parallel_ms,
        speedup: if parallel_ms > 0.0 { serial_ms / parallel_ms } else { 0.0 },
        throughput_flops: (2.0 * (n as f64).powi(3) * iterations as f64) / (parallel_ms / 1000.0),
    })
}

fn run_training_step_int8_kernel(size: usize, iterations: usize, threads: usize) -> Result<RuntimeExecutionReport> {
    let n = size.max(32);
    let lhs = HostTensor::from_vec(
        &[n, n],
        (0..n * n).map(|i| ((i % 113) as f32 * 0.008) - 0.4).collect(),
    )?;
    let rhs = HostTensor::from_vec(
        &[n, n],
        (0..n * n).map(|i| ((i % 97) as f32 * 0.006) - 0.3).collect(),
    )?;
    let qlhs = quantize_i8(&lhs);
    let qrhs = quantize_i8(&rhs);
    let mut serial_out = HostTensor::zeros(&[n, n]);
    let mut runtime_out = HostTensor::zeros(&[n, n]);

    let serial_start = Instant::now();
    for _ in 0..iterations {
        tensor_matmul(&lhs, &rhs, &mut serial_out)?;
        std::hint::black_box(serial_out.as_slice());
    }
    let serial_ms = serial_start.elapsed().as_secs_f64() * 1000.0;

    let parallel_start = Instant::now();
    with_rayon_pool(threads, || {
        for _ in 0..iterations {
            quantized_matmul_i8(&qlhs, &qrhs, &mut runtime_out)?;
            std::hint::black_box(runtime_out.as_slice());
        }
        Ok(())
    })?;
    let parallel_ms = parallel_start.elapsed().as_secs_f64() * 1000.0;

    Ok(RuntimeExecutionReport {
        kernel: "training-step-int8".to_string(),
        size: n,
        iterations,
        threads,
        serial_ms,
        parallel_ms,
        speedup: if parallel_ms > 0.0 { serial_ms / parallel_ms } else { 0.0 },
        throughput_flops: (2.0 * (n as f64).powi(3) * iterations as f64) / (parallel_ms / 1000.0),
    })
}

fn run_inference_int8_kernel(size: usize, iterations: usize, threads: usize) -> Result<RuntimeExecutionReport> {
    let n = size.max(32);
    let input = HostTensor::from_vec(
        &[n, n],
        (0..n * n).map(|i| ((i % 67) as f32 * 0.01) - 0.2).collect(),
    )?;
    let weights = HostTensor::from_vec(
        &[n, n],
        (0..n * n).map(|i| ((i % 53) as f32 * 0.009) - 0.25).collect(),
    )?;
    let qinput = quantize_i8(&input);
    let qweights = quantize_i8(&weights);
    let mut serial_out = HostTensor::zeros(&[n, n]);
    let mut runtime_out = HostTensor::zeros(&[n, n]);

    let serial_start = Instant::now();
    for _ in 0..iterations {
        tensor_matmul(&input, &weights, &mut serial_out)?;
        tensor_relu(&serial_out.clone(), &mut serial_out)?;
        std::hint::black_box(serial_out.as_slice());
    }
    let serial_ms = serial_start.elapsed().as_secs_f64() * 1000.0;

    let parallel_start = Instant::now();
    with_rayon_pool(threads, || {
        for _ in 0..iterations {
            quantized_matmul_i8(&qinput, &qweights, &mut runtime_out)?;
            let relu_in = runtime_out.clone();
            tensor_relu(&relu_in, &mut runtime_out)?;
            std::hint::black_box(runtime_out.as_slice());
        }
        Ok(())
    })?;
    let parallel_ms = parallel_start.elapsed().as_secs_f64() * 1000.0;

    Ok(RuntimeExecutionReport {
        kernel: "inference-int8".to_string(),
        size: n,
        iterations,
        threads,
        serial_ms,
        parallel_ms,
        speedup: if parallel_ms > 0.0 { serial_ms / parallel_ms } else { 0.0 },
        throughput_flops: (2.0 * (n as f64).powi(3) * iterations as f64) / (parallel_ms / 1000.0),
    })
}

fn run_fused_matmul_gelu_kernel(size: usize, iterations: usize, threads: usize) -> Result<RuntimeExecutionReport> {
    let n = size;
    let a = vec![1.0f32; n * n];
    let b = vec![1.0f32; n * n];
    let mut serial_out = vec![0.0f32; n * n];
    let a_tensor = HostTensor::from_vec(&[n, n], a.clone())?;
    let b_tensor = HostTensor::from_vec(&[n, n], b.clone())?;
    let executor = persistent_executor(threads)?;
    let mut runtime_out: Option<HostTensor> = None;

    let serial_start = Instant::now();
    for _ in 0..iterations {
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0f32;
                for k in 0..n {
                    sum += a[i * n + k] * b[k * n + j];
                }
                let c = (2.0 / std::f32::consts::PI).sqrt();
                serial_out[i * n + j] = 0.5 * sum * (1.0 + (c * (sum + 0.044_715 * sum.powi(3))).tanh());
            }
        }
        std::hint::black_box(&serial_out);
    }
    let serial_ms = serial_start.elapsed().as_secs_f64() * 1000.0;

    let parallel_start = Instant::now();
    for _ in 0..iterations {
        let graph = ExecutionGraph::new()
            .with_input("lhs", a_tensor.clone())
            .with_input("rhs", b_tensor.clone())
            .add_node(GraphNode::FusedMatmulGelu {
                lhs: "lhs".to_string(),
                rhs: "rhs".to_string(),
                output: "out".to_string(),
            });
        let result = executor.execute_graph(&graph)?;
        runtime_out = Some(result
            .tensors
            .get("out")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("graph result missing fused matmul output"))?);
        std::hint::black_box(runtime_out.as_ref().map(HostTensor::as_slice));
    }
    let parallel_ms = parallel_start.elapsed().as_secs_f64() * 1000.0;
    let flops = 2.0 * (n as f64).powi(3) * iterations as f64;
    let throughput_flops = flops / (parallel_ms / 1000.0);

    Ok(RuntimeExecutionReport {
        kernel: "fused-matmul-gelu".to_string(),
        size,
        iterations,
        threads,
        serial_ms,
        parallel_ms,
        speedup: if parallel_ms > 0.0 { serial_ms / parallel_ms } else { 0.0 },
        throughput_flops,
    })
}

fn run_fabric_graph(size: usize, threads: usize, iterations: usize) -> Result<RuntimeExecutionReport> {
    let n = size;
    let lhs = HostTensor::from_vec(
        &[n, n],
        (0..n * n)
            .map(|i| ((i % 257) as f32 * 0.0075) - 0.9)
            .collect(),
    )?;
    let rhs = HostTensor::from_vec(
        &[n, n],
        (0..n * n)
            .map(|i| ((i % 193) as f32 * 0.0060) - 0.7)
            .collect(),
    )?;
    let bias = HostTensor::filled(&[n, n], 0.05);
    let serial_lhs = lhs.as_slice().to_vec();
    let serial_rhs = rhs.as_slice().to_vec();
    let mut serial_mm = vec![0.0f32; n * n];
    let mut serial_out = vec![0.0f32; n * n];

    let serial_start = Instant::now();
    for _ in 0..iterations {
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0f32;
                for k in 0..n {
                    sum += serial_lhs[i * n + k] * serial_rhs[k * n + j];
                }
                serial_mm[i * n + j] = sum + 0.05;
            }
        }
        for (dst, src) in serial_out.iter_mut().zip(serial_mm.iter().copied()) {
            *dst = src.max(0.0);
        }
        std::hint::black_box(&serial_out);
    }
    let serial_ms = serial_start.elapsed().as_secs_f64() * 1000.0;

    let executor = persistent_executor(threads)?;
    let parallel_start = Instant::now();
    let mut runtime_out: Option<HostTensor> = None;
    for _ in 0..iterations {
        let graph = ExecutionGraph::new()
            .with_input("lhs", lhs.clone())
            .with_input("rhs", rhs.clone())
            .with_input("bias", bias.clone())
            .add_node(GraphNode::Matmul {
                lhs: "lhs".to_string(),
                rhs: "rhs".to_string(),
                output: "mm".to_string(),
            })
            .add_node(GraphNode::Add {
                lhs: "mm".to_string(),
                rhs: "bias".to_string(),
                output: "biased".to_string(),
            })
            .add_node(GraphNode::Relu {
                input: "biased".to_string(),
                output: "out".to_string(),
            });
        let result = executor.execute_graph(&graph)?;
        runtime_out = Some(result
            .tensors
            .get("out")
            .cloned()
            .ok_or_else(|| anyhow::anyhow!("fabric graph missing output tensor"))?);
        std::hint::black_box(runtime_out.as_ref().map(HostTensor::as_slice));
    }
    let parallel_ms = parallel_start.elapsed().as_secs_f64() * 1000.0;
    let flops = (2.0 * (n as f64).powi(3) + (n * n * 2) as f64) * iterations as f64;

    Ok(RuntimeExecutionReport {
        kernel: "fabric-graph".to_string(),
        size,
        iterations,
        threads,
        serial_ms,
        parallel_ms,
        speedup: if parallel_ms > 0.0 { serial_ms / parallel_ms } else { 0.0 },
        throughput_flops: flops / (parallel_ms / 1000.0),
    })
}

#[allow(clippy::too_many_arguments)]
fn serial_conv2d_nchw(
    input: &[f32],
    weight: &[f32],
    bias: Option<&[f32]>,
    output: &mut [f32],
    batch: usize,
    in_channels: usize,
    in_h: usize,
    in_w: usize,
    out_channels: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride: (usize, usize),
    padding: (usize, usize),
) {
    let out_h = ((in_h + 2 * padding.0).saturating_sub(kernel_h) / stride.0) + 1;
    let out_w = ((in_w + 2 * padding.1).saturating_sub(kernel_w) / stride.1) + 1;
    for n in 0..batch {
        for oc in 0..out_channels {
            let bias_value = bias.map(|b| b[oc]).unwrap_or(0.0);
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut acc = bias_value;
                    for ic in 0..in_channels {
                        for kh in 0..kernel_h {
                            let ih = oh * stride.0 + kh;
                            if ih < padding.0 {
                                continue;
                            }
                            let ih = ih - padding.0;
                            if ih >= in_h {
                                continue;
                            }
                            for kw in 0..kernel_w {
                                let iw = ow * stride.1 + kw;
                                if iw < padding.1 {
                                    continue;
                                }
                                let iw = iw - padding.1;
                                if iw >= in_w {
                                    continue;
                                }

                                let input_idx = ((n * in_channels + ic) * in_h + ih) * in_w + iw;
                                let weight_idx =
                                    ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
                                acc += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                    let out_idx = ((n * out_channels + oc) * out_h + oh) * out_w + ow;
                    output[out_idx] = acc;
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn serial_im2col_nchw(
    input: &[f32],
    output: &mut [f32],
    batch: usize,
    channels: usize,
    in_h: usize,
    in_w: usize,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) {
    let out_h = ((in_h + 2 * padding.0).saturating_sub(kernel.0) / stride.0) + 1;
    let out_w = ((in_w + 2 * padding.1).saturating_sub(kernel.1) / stride.1) + 1;
    let row_width = channels * kernel.0 * kernel.1;
    for n in 0..batch {
        for oh in 0..out_h {
            for ow in 0..out_w {
                let row = n * out_h * out_w + oh * out_w + ow;
                let mut col = 0;
                for c in 0..channels {
                    for kh in 0..kernel.0 {
                        for kw in 0..kernel.1 {
                            let ih = oh * stride.0 + kh;
                            let iw = ow * stride.1 + kw;
                            let value = if ih < padding.0 || iw < padding.1 {
                                0.0
                            } else {
                                let ih = ih - padding.0;
                                let iw = iw - padding.1;
                                if ih >= in_h || iw >= in_w {
                                    0.0
                                } else {
                                    let idx = ((n * channels + c) * in_h + ih) * in_w + iw;
                                    input[idx]
                                }
                            };
                            output[row * row_width + col] = value;
                            col += 1;
                        }
                    }
                }
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn serial_attention(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    output: &mut [f32],
    batch: usize,
    seq: usize,
    dim: usize,
    value_dim: usize,
) {
    let scale = (dim as f32).sqrt();
    let mut scores = vec![0.0f32; seq * seq];
    for b in 0..batch {
        let q_offset = b * seq * dim;
        let k_offset = b * seq * dim;
        let v_offset = b * seq * value_dim;
        for q in 0..seq {
            for k in 0..seq {
                let mut acc = 0.0f32;
                for d in 0..dim {
                    acc += query[q_offset + q * dim + d] * key[k_offset + k * dim + d];
                }
                scores[q * seq + k] = acc / scale;
            }
            let row = &mut scores[q * seq..(q + 1) * seq];
            let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for value_score in row.iter_mut() {
                *value_score = (*value_score - max_val).exp();
                sum += *value_score;
            }
            for value_score in row.iter_mut() {
                *value_score /= sum;
            }
            for vd in 0..value_dim {
                let mut acc = 0.0f32;
                for k in 0..seq {
                    acc += row[k] * value[v_offset + k * value_dim + vd];
                }
                output[b * seq * value_dim + q * value_dim + vd] = acc;
            }
        }
    }
}

fn apply_runtime_profile(
    profile: RuntimeProfile,
    hugepages: Option<usize>,
    swappiness: Option<u32>,
) -> Result<()> {
    let profile_name = match profile {
        RuntimeProfile::Balanced => "balanced",
        RuntimeProfile::Gaming => "gaming",
        RuntimeProfile::Creator => "creator",
        RuntimeProfile::NoGpuTraining => "no-gpu-training",
        RuntimeProfile::LowPowerInference => "low-power-inference",
        RuntimeProfile::MaxThroughput => "max-throughput",
    };

    let (
        governor,
        thp_mode,
        hugepages_default,
        swappiness_default,
        readahead,
        idle_enabled,
        io_sched,
        nice,
    ) = match profile {
        RuntimeProfile::Balanced => (
            "schedutil",
            "madvise",
            128usize,
            20u32,
            1024u32,
            true,
            "mq-deadline",
            -4,
        ),
        RuntimeProfile::Gaming => (
            "performance",
            "madvise",
            256usize,
            12u32,
            2048u32,
            false,
            "mq-deadline",
            -10,
        ),
        RuntimeProfile::Creator => (
            "performance",
            "madvise",
            256usize,
            14u32,
            3072u32,
            true,
            "mq-deadline",
            -8,
        ),
        RuntimeProfile::NoGpuTraining => (
            "performance",
            "always",
            768usize,
            6u32,
            4096u32,
            false,
            "none",
            -12,
        ),
        RuntimeProfile::LowPowerInference => (
            "schedutil",
            "madvise",
            128usize,
            12u32,
            1024u32,
            true,
            "mq-deadline",
            -6,
        ),
        RuntimeProfile::MaxThroughput => (
            "performance",
            "always",
            512usize,
            8u32,
            4096u32,
            false,
            "none",
            -12,
        ),
    };

    println!("  Profile: {}", profile_name);
    apply_tuning_step("CPU governor", || Ok(set_governor(governor)?));
    apply_tuning_step("THP", || Ok(configure_thp(thp_mode)?));
    apply_tuning_step("HugePages", || Ok(reserve_hugepages(hugepages.unwrap_or(hugepages_default))?));
    apply_tuning_step("Swappiness", || Ok(set_swappiness(swappiness.unwrap_or(swappiness_default))?));
    apply_tuning_step("Read-ahead", || Ok(set_readahead_kb(readahead)?));
    apply_tuning_step("I/O scheduler", || Ok(set_io_scheduler(io_sched)?));
    apply_tuning_step("CPU idle states", || Ok(set_idle_states(idle_enabled)?));
    apply_tuning_step("Priority", || Ok(set_priority(nice)?));
    Ok(())
}

fn apply_tuning_step<F>(label: &str, action: F)
where
    F: FnOnce() -> Result<()>,
{
    match action() {
        Ok(()) => println!("  {}: applied", label),
        Err(err) => println!("  {}: skipped ({})", label, err),
    }
}

fn suggest_runtime_mode(load: (f64, f64, f64), available_memory: u64) -> &'static str {
    let available_gib = available_memory as f64 / 1024.0 / 1024.0 / 1024.0;
    if load.0 > 12.0 && available_gib > 16.0 {
        "max-throughput"
    } else if load.0 > 8.0 {
        "training"
    } else if available_gib < 4.0 {
        "balanced"
    } else if available_gib > 8.0 {
        "gaming"
    } else {
        "creator"
    }
}
