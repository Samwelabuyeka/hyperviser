use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use std::ffi::OsStr;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

#[derive(Parser, Debug)]
#[command(name = "aurora-distro")]
#[command(about = "Ubuntu remaster and installer toolkit for AURORA OS")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Create a distro build tree and default profile.
    InitTree {
        #[arg(long, default_value = "distro")]
        out: PathBuf,
        #[arg(long, default_value = "Aurora OS")]
        distro_name: String,
        #[arg(long, value_enum, default_value = "gnome")]
        desktop: DesktopPreset,
    },
    /// Validate that the host has the tools needed for remastering.
    CheckTools,
    /// Build a full Ubuntu-based ISO from the build tree.
    BuildIso {
        #[arg(long, default_value = "distro")]
        tree: PathBuf,
    },
    /// Probe the current machine to determine firmware and disk defaults.
    ScanSystem,
    /// Generate a partition plan based on host firmware and requested disk size.
    PlanPartitions {
        #[arg(long, default_value_t = 256)]
        disk_gb: u64,
    },
    /// Write a built ISO to a USB disk.
    WriteUsb {
        #[arg(long)]
        iso: PathBuf,
        #[arg(long)]
        device: PathBuf,
    },
    /// Attempt GRUB-based boot repair for BIOS and UEFI installs.
    RepairBoot {
        #[arg(long)]
        root: PathBuf,
        #[arg(long)]
        efi: Option<PathBuf>,
        #[arg(long, default_value = "/dev/sda")]
        disk: String,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum DesktopPreset {
    Minimal,
    Gnome,
    Kde,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BuildConfig {
    distro_name: String,
    iso_name: String,
    ubuntu_release: String,
    ubuntu_mirror: String,
    arch: String,
    desktop: DesktopPreset,
    package_sets: Vec<String>,
    bios_legacy: bool,
    uefi: bool,
    theme_name: String,
    accent_color: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SystemScan {
    firmware: FirmwareMode,
    architecture: String,
    disks: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum FirmwareMode {
    Bios,
    Uefi,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Partition {
    label: String,
    fs: String,
    size_mb: u64,
    mountpoint: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PartitionPlan {
    firmware: FirmwareMode,
    partitions: Vec<Partition>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::InitTree {
            out,
            distro_name,
            desktop,
        } => init_tree(&out, &distro_name, desktop),
        Commands::CheckTools => check_tools().map(|_| ()),
        Commands::BuildIso { tree } => build_iso(&tree),
        Commands::ScanSystem => {
            println!("{}", serde_json::to_string_pretty(&scan_system()?)?);
            Ok(())
        }
        Commands::PlanPartitions { disk_gb } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&plan_partitions(scan_system()?, disk_gb))?
            );
            Ok(())
        }
        Commands::WriteUsb { iso, device } => write_usb(&iso, &device),
        Commands::RepairBoot { root, efi, disk } => repair_boot(&root, efi.as_deref(), &disk),
    }
}

fn init_tree(out: &Path, distro_name: &str, desktop: DesktopPreset) -> Result<()> {
    for dir in [
        out.to_path_buf(),
        out.join("profiles"),
        out.join("overlay"),
        out.join("overlay/etc/skel"),
        out.join("overlay/usr/share/backgrounds/aurora"),
        out.join("assets"),
        out.join("assets/branding"),
        out.join("assets/theme"),
        out.join("build"),
        out.join("output"),
    ] {
        fs::create_dir_all(&dir)
            .with_context(|| format!("failed to create {}", dir.display()))?;
    }

    let config = BuildConfig {
        distro_name: distro_name.to_string(),
        iso_name: format!("{}-24.04-live.iso", distro_name.to_lowercase().replace(' ', "-")),
        ubuntu_release: "noble".to_string(),
        ubuntu_mirror: "http://archive.ubuntu.com/ubuntu".to_string(),
        arch: "amd64".to_string(),
        desktop: desktop.clone(),
        package_sets: desktop_packages(&desktop),
        bios_legacy: true,
        uefi: true,
        theme_name: "Aurora Neon Assault".to_string(),
        accent_color: "#12f7ff".to_string(),
    };

    fs::write(
        out.join("profiles/default.json"),
        serde_json::to_string_pretty(&config)?,
    )?;
    fs::write(
        out.join("assets/theme/theme.css"),
        default_theme_css(&config.distro_name, &config.accent_color),
    )?;
    fs::write(
        out.join("assets/branding/README.md"),
        "Place replacement logos, wallpapers, and boot graphics here.\n",
    )?;

    println!("Initialized distro tree at {}", out.display());
    Ok(())
}

fn check_tools() -> Result<Vec<&'static str>> {
    let tools = [
        "debootstrap",
        "rsync",
        "chroot",
        "mount",
        "umount",
        "mksquashfs",
        "grub-mkrescue",
        "xorriso",
        "grub-install",
        "update-grub",
        "lsblk",
        "dd",
    ];

    let missing: Vec<_> = tools
        .iter()
        .copied()
        .filter(|tool| !command_exists(tool))
        .collect();

    if !missing.is_empty() {
        bail!("missing required host tools: {}", missing.join(", "));
    }

    println!("All required remaster tools are installed.");
    Ok(tools.to_vec())
}

fn build_iso(tree: &Path) -> Result<()> {
    ensure_root()?;
    check_tools()?;

    let config = load_config(tree)?;
    let build_root = tree.join("build/rootfs");
    let iso_root = tree.join("build/iso");
    let live_root = iso_root.join("live");

    for dir in [&build_root, &iso_root, &live_root, &tree.join("output")] {
        fs::create_dir_all(dir)?;
    }

    run(
        "debootstrap",
        [
            "--arch",
            config.arch.as_str(),
            config.ubuntu_release.as_str(),
            build_root
                .to_str()
                .ok_or_else(|| anyhow!("non-utf8 build root path"))?,
            config.ubuntu_mirror.as_str(),
        ],
    )?;

    let mounts = [
        ("/dev", build_root.join("dev")),
        ("/proc", build_root.join("proc")),
        ("/sys", build_root.join("sys")),
    ];
    for (src, dst) in mounts.iter() {
        fs::create_dir_all(dst)?;
        run("mount", ["--bind", src, path_str(dst)?])?;
    }

    let package_script = format!(
        "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y casper grub-pc-bin grub-efi-amd64-bin linux-generic {}",
        config.package_sets.join(" ")
    );
    let chroot_cmd = format!("chroot {} /bin/bash -lc {:?}", path_str(&build_root)?, package_script);
    run_shell(&chroot_cmd)?;

    apply_overlay(tree, &build_root)?;
    install_branding(tree, &build_root, &config)?;

    fs::create_dir_all(live_root.as_path())?;
    generate_manifest(&build_root, &live_root)?;
    copy_kernel_artifacts(&build_root, &live_root)?;
    build_squashfs(&build_root, &live_root)?;
    write_grub_cfg(&iso_root, &config)?;

    let output_iso = tree.join("output").join(&config.iso_name);
    run(
        "grub-mkrescue",
        [path_str(&iso_root)?, "-o", path_str(&output_iso)?],
    )?;

    for (_, dst) in mounts.iter().rev() {
        let _ = run("umount", [path_str(dst)?]);
    }

    println!("ISO created at {}", output_iso.display());
    Ok(())
}

fn scan_system() -> Result<SystemScan> {
    let firmware = if Path::new("/sys/firmware/efi").exists() {
        FirmwareMode::Uefi
    } else {
        FirmwareMode::Bios
    };

    let disks = if command_exists("lsblk") {
        let output = Command::new("lsblk")
            .args(["-dn", "-o", "NAME"])
            .output()
            .context("failed to run lsblk")?;
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(|line| format!("/dev/{}", line.trim()))
            .filter(|line| !line.is_empty())
            .collect()
    } else {
        Vec::new()
    };

    Ok(SystemScan {
        firmware,
        architecture: std::env::consts::ARCH.to_string(),
        disks,
    })
}

fn plan_partitions(scan: SystemScan, disk_gb: u64) -> PartitionPlan {
    let mut partitions = Vec::new();
    if matches!(scan.firmware, FirmwareMode::Uefi) {
        partitions.push(Partition {
            label: "EFI".to_string(),
            fs: "fat32".to_string(),
            size_mb: 512,
            mountpoint: "/boot/efi".to_string(),
        });
    } else {
        partitions.push(Partition {
            label: "BIOS_GRUB".to_string(),
            fs: "bios_grub".to_string(),
            size_mb: 8,
            mountpoint: "bios_grub".to_string(),
        });
    }

    partitions.push(Partition {
        label: "ROOT".to_string(),
        fs: "ext4".to_string(),
        size_mb: disk_gb.saturating_mul(1024).saturating_sub(8192),
        mountpoint: "/".to_string(),
    });
    partitions.push(Partition {
        label: "SWAP".to_string(),
        fs: "swap".to_string(),
        size_mb: 8192,
        mountpoint: "swap".to_string(),
    });

    PartitionPlan {
        firmware: scan.firmware,
        partitions,
    }
}

fn write_usb(iso: &Path, device: &Path) -> Result<()> {
    ensure_root()?;
    if !iso.is_file() {
        bail!("ISO not found: {}", iso.display());
    }

    let device_str = path_str(device)?;
    if !device_str.starts_with("/dev/") {
        bail!("refusing to write to non-device path: {}", device.display());
    }

    run(
        "dd",
        [
            format!("if={}", iso.display()).as_str(),
            format!("of={}", device.display()).as_str(),
            "bs=4M",
            "status=progress",
            "oflag=sync",
        ],
    )?;
    run("sync", std::iter::empty::<&str>())?;
    println!("USB media written to {}", device.display());
    Ok(())
}

fn repair_boot(root: &Path, efi: Option<&Path>, disk: &str) -> Result<()> {
    ensure_root()?;
    let root_str = path_str(root)?;

    run("mount", ["--bind", "/dev", &format!("{root_str}/dev")])?;
    run("mount", ["--bind", "/proc", &format!("{root_str}/proc")])?;
    run("mount", ["--bind", "/sys", &format!("{root_str}/sys")])?;

    if let Some(efi_dir) = efi {
        fs::create_dir_all(root.join("boot/efi"))?;
        run("mount", [path_str(efi_dir)?, &format!("{root_str}/boot/efi")])?;
        run(
            "chroot",
            [
                root_str,
                "grub-install",
                "--target=x86_64-efi",
                "--efi-directory=/boot/efi",
                "--bootloader-id=AURORA",
                "--recheck",
            ],
        )?;
    } else {
        run(
            "chroot",
            [root_str, "grub-install", "--target=i386-pc", disk, "--recheck"],
        )?;
    }

    run("chroot", [root_str, "update-grub"])?;
    println!("Boot repair completed for {}", root.display());
    Ok(())
}

fn apply_overlay(tree: &Path, rootfs: &Path) -> Result<()> {
    let overlay = tree.join("overlay");
    if overlay.exists() {
        run("rsync", ["-a", &format!("{}/", overlay.display()), path_str(rootfs)?])?;
    }
    Ok(())
}

fn install_branding(tree: &Path, rootfs: &Path, config: &BuildConfig) -> Result<()> {
    let theme_target = rootfs.join("usr/share/aurora");
    fs::create_dir_all(&theme_target)?;
    fs::copy(
        tree.join("assets/theme/theme.css"),
        theme_target.join("theme.css"),
    )
    .ok();

    let issue = format!(
        "{}\nUbuntu 24.04 remaster with BIOS/UEFI install support.\n",
        config.distro_name
    );
    fs::write(rootfs.join("etc/issue"), issue)?;
    Ok(())
}

fn generate_manifest(rootfs: &Path, live_root: &Path) -> Result<()> {
    let output = Command::new("chroot")
        .args([path_str(rootfs)?, "dpkg-query", "-W", "--showformat=${Package} ${Version}\\n"])
        .output()
        .context("failed to generate package manifest")?;
    if !output.status.success() {
        bail!("dpkg-query failed while generating manifest");
    }
    fs::write(live_root.join("filesystem.manifest"), output.stdout)?;
    Ok(())
}

fn copy_kernel_artifacts(rootfs: &Path, live_root: &Path) -> Result<()> {
    let boot = rootfs.join("boot");
    let vmlinuz = find_first_prefixed(&boot, "vmlinuz-")?;
    let initrd = find_first_prefixed(&boot, "initrd.img-")?;
    fs::copy(vmlinuz, live_root.join("vmlinuz"))?;
    fs::copy(initrd, live_root.join("initrd"))?;
    Ok(())
}

fn build_squashfs(rootfs: &Path, live_root: &Path) -> Result<()> {
    run(
        "mksquashfs",
        [
            path_str(rootfs)?,
            path_str(&live_root.join("filesystem.squashfs"))?,
            "-e",
            "boot",
        ],
    )
}

fn write_grub_cfg(iso_root: &Path, config: &BuildConfig) -> Result<()> {
    let grub_dir = iso_root.join("boot/grub");
    fs::create_dir_all(&grub_dir)?;
    let cfg = format!(
        "set default=0\nset timeout=5\nmenuentry \"{} Live\" {{\n linux /live/vmlinuz boot=casper quiet splash ---\n initrd /live/initrd\n}}\n",
        config.distro_name
    );
    fs::write(grub_dir.join("grub.cfg"), cfg)?;
    Ok(())
}

fn load_config(tree: &Path) -> Result<BuildConfig> {
    let path = tree.join("profiles/default.json");
    let content = fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&content).context("failed to parse build config")
}

fn desktop_packages(desktop: &DesktopPreset) -> Vec<String> {
    match desktop {
        DesktopPreset::Minimal => vec![
            "ubuntu-standard".to_string(),
            "network-manager".to_string(),
            "sudo".to_string(),
        ],
        DesktopPreset::Gnome => vec![
            "ubuntu-desktop".to_string(),
            "gnome-shell-extension-manager".to_string(),
            "steam-installer".to_string(),
            "gamemode".to_string(),
        ],
        DesktopPreset::Kde => vec![
            "kubuntu-desktop".to_string(),
            "steam-installer".to_string(),
            "gamemode".to_string(),
        ],
    }
}

fn default_theme_css(name: &str, accent: &str) -> String {
    format!(
        ":root {{ --aurora-accent: {accent}; --aurora-bg: #08111d; --aurora-panel: #101c2b; }}\nbody {{ background: radial-gradient(circle at top, #14263a, var(--aurora-bg)); color: #eef7ff; font-family: 'Orbitron', sans-serif; }}\n.login-logo {{ display: none; }}\n.session-title::after {{ content: '{name}'; color: var(--aurora-accent); }}\n"
    )
}

fn ensure_root() -> Result<()> {
    #[cfg(target_family = "unix")]
    {
        if unsafe { libc::geteuid() } != 0 {
            bail!("this command must be run as root");
        }
    }
    Ok(())
}

fn command_exists(cmd: &str) -> bool {
    std::env::var_os("PATH")
        .into_iter()
        .flat_map(|paths| std::env::split_paths(&paths))
        .any(|dir| dir.join(cmd).exists() || dir.join(format!("{cmd}.exe")).exists())
}

fn run<I, S>(program: &str, args: I) -> Result<()>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let output = Command::new(program)
        .args(args)
        .output()
        .with_context(|| format!("failed to launch {program}"))?;
    if !output.status.success() {
        bail!(
            "{} failed\nstdout:\n{}\nstderr:\n{}",
            program,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
    }
    Ok(())
}

fn run_shell(command: &str) -> Result<()> {
    run("/bin/sh", ["-lc", command])
}

fn path_str(path: &Path) -> Result<&str> {
    path.to_str()
        .ok_or_else(|| anyhow!("non-utf8 path: {}", path.display()))
}

fn find_first_prefixed(dir: &Path, prefix: &str) -> Result<PathBuf> {
    let mut entries: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|entry| entry.ok())
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.starts_with(prefix))
                .unwrap_or(false)
        })
        .collect();
    entries.sort();
    entries
        .into_iter()
        .next()
        .ok_or_else(|| anyhow!("could not find {}* in {}", prefix, dir.display()))
}
