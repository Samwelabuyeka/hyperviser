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
    /// Install the Ubuntu 24 remaster host dependencies on a Linux build machine.
    PrepareHost,
    /// Validate that the host has the tools needed for remastering.
    CheckTools,
    /// Build a full Ubuntu-based ISO from the build tree.
    BuildIso {
        #[arg(long, default_value = "distro")]
        tree: PathBuf,
        #[arg(long)]
        prompt_usb: bool,
        #[arg(long)]
        usb_device: Option<PathBuf>,
        #[arg(long, value_enum, default_value = "auto")]
        system_mode: BootMode,
    },
    /// Probe the current machine to determine firmware and disk defaults.
    ScanSystem,
    /// Generate a partition plan based on host firmware and requested disk size.
    PlanPartitions {
        #[arg(long, value_enum, default_value = "auto")]
        mode: BootMode,
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

#[derive(Clone, Debug, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum BootMode {
    Auto,
    Legacy,
    Uefi,
    Both,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct BuildConfig {
    distro_name: String,
    iso_name: String,
    volume_id: String,
    branding_name: String,
    logo_path: String,
    ubuntu_release: String,
    ubuntu_mirror: String,
    arch: String,
    desktop: DesktopPreset,
    package_sets: Vec<String>,
    extra_packages: Vec<String>,
    bios_legacy: bool,
    uefi: bool,
    kali_like_theme: bool,
    theme_name: String,
    accent_color: String,
    performance_goal: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct InstallerConfig {
    wizard_title: String,
    welcome_headline: String,
    auto_scan_hardware: bool,
    offer_inbuilt_usb_writer: bool,
    support_legacy_bios: bool,
    support_uefi: bool,
    ask_user_details_first: bool,
    fallback_to_scan_when_unknown: bool,
    default_partition_mode: BootMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerformanceProfile {
    name: String,
    cpu_governor: String,
    enable_hugepages: bool,
    enable_gamemode: bool,
    enable_mangohud: bool,
    tune_sysctl: bool,
    kernel_cmdline_additions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct SystemScan {
    firmware: BootMode,
    architecture: String,
    cpu_model: String,
    ram_mb: u64,
    disks: Vec<String>,
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
    firmware: BootMode,
    partitions: Vec<Partition>,
    notes: Vec<String>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::InitTree {
            out,
            distro_name,
            desktop,
        } => init_tree(&out, &distro_name, desktop),
        Commands::PrepareHost => prepare_host(),
        Commands::CheckTools => check_tools().map(|_| ()),
        Commands::BuildIso {
            tree,
            prompt_usb,
            usb_device,
            system_mode,
        } => build_iso(&tree, prompt_usb, usb_device.as_deref(), system_mode),
        Commands::ScanSystem => {
            println!("{}", serde_json::to_string_pretty(&scan_system()?)?);
            Ok(())
        }
        Commands::PlanPartitions { mode, disk_gb } => {
            println!(
                "{}",
                serde_json::to_string_pretty(&plan_partitions(scan_system()?, mode, disk_gb))?
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
        out.join("overlay/etc/aurora-installer"),
        out.join("overlay/etc/profile.d"),
        out.join("overlay/etc/sysctl.d"),
        out.join("overlay/usr/local/bin"),
        out.join("overlay/usr/share/applications"),
        out.join("overlay/usr/share/aurora/installer"),
        out.join("overlay/usr/share/themes/Aurora-Neon/gtk-3.0"),
        out.join("overlay/usr/share/icons/Aurora-Neon"),
        out.join("overlay/usr/share/plymouth/themes/aurora-neon"),
        out.join("overlay/usr/share/grub/themes/aurora"),
        out.join("overlay/usr/share/backgrounds/aurora"),
        out.join("overlay/home/aurora/.config/autostart"),
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
        volume_id: format!("{}_2404", distro_name.to_uppercase().replace(' ', "_")),
        branding_name: distro_name.to_string(),
        logo_path: "assets/branding/logo.png".to_string(),
        ubuntu_release: "noble".to_string(),
        ubuntu_mirror: "http://archive.ubuntu.com/ubuntu".to_string(),
        arch: "amd64".to_string(),
        desktop: desktop.clone(),
        package_sets: desktop_packages(&desktop),
        extra_packages: vec![
            "gamemode".to_string(),
            "mangohud".to_string(),
            "htop".to_string(),
            "nvtop".to_string(),
            "git".to_string(),
        ],
        bios_legacy: true,
        uefi: true,
        kali_like_theme: true,
        theme_name: "Aurora Neon Assault".to_string(),
        accent_color: "#12f7ff".to_string(),
        performance_goal: "Tune for strong CPU throughput on supported hardware; 3x uplift is a target for selective workloads, not a universal guarantee.".to_string(),
    };
    let installer = InstallerConfig {
        wizard_title: format!("{} Installer", config.branding_name),
        welcome_headline: format!("Forge your high-performance {} system", config.branding_name),
        auto_scan_hardware: true,
        offer_inbuilt_usb_writer: true,
        support_legacy_bios: true,
        support_uefi: true,
        ask_user_details_first: true,
        fallback_to_scan_when_unknown: true,
        default_partition_mode: BootMode::Auto,
    };
    let performance = PerformanceProfile {
        name: "Aurora Maximum Throughput".to_string(),
        cpu_governor: "performance".to_string(),
        enable_hugepages: true,
        enable_gamemode: true,
        enable_mangohud: true,
        tune_sysctl: true,
        kernel_cmdline_additions: vec![
            "mitigations=off".to_string(),
            "transparent_hugepage=always".to_string(),
            "nowatchdog".to_string(),
        ],
    };

    fs::write(
        out.join("profiles/default.json"),
        serde_json::to_string_pretty(&config)?,
    )?;
    fs::write(
        out.join("profiles/installer.json"),
        serde_json::to_string_pretty(&installer)?,
    )?;
    fs::write(
        out.join("profiles/performance.json"),
        serde_json::to_string_pretty(&performance)?,
    )?;
    fs::write(
        out.join("assets/theme/theme.css"),
        default_theme_css(&config.distro_name, &config.accent_color),
    )?;
    fs::write(
        out.join("overlay/etc/default/grub"),
        "GRUB_TIMEOUT=5\nGRUB_GFXMODE=1920x1080\nGRUB_THEME=/usr/share/grub/themes/aurora/theme.txt\nGRUB_CMDLINE_LINUX_DEFAULT=\"quiet splash mitigations=off transparent_hugepage=always\"\n",
    )?;
    fs::write(
        out.join("overlay/etc/aurora-installer/README"),
        "The installer can scan firmware/disk details automatically and use aurora-distro plan-partitions output when users skip manual hardware entry.\n",
    )?;
    fs::write(
        out.join("overlay/etc/aurora-installer/installer.json"),
        serde_json::to_string_pretty(&installer)?,
    )?;
    fs::write(
        out.join("overlay/etc/aurora-installer/performance-profile.json"),
        serde_json::to_string_pretty(&performance)?,
    )?;
    fs::write(
        out.join("overlay/etc/profile.d/aurora-performance.sh"),
        performance_shell_script(&performance),
    )?;
    fs::write(
        out.join("overlay/etc/sysctl.d/99-aurora-gaming.conf"),
        performance_sysctl_conf(),
    )?;
    fs::write(
        out.join("overlay/usr/share/backgrounds/aurora/gaming-kali.svg"),
        default_wallpaper_svg(&config.branding_name, &config.accent_color),
    )?;
    fs::write(
        out.join("overlay/usr/share/applications/aurora-installer.desktop"),
        installer_desktop_file(),
    )?;
    fs::write(
        out.join("overlay/home/aurora/.config/autostart/aurora-installer.desktop"),
        installer_desktop_file(),
    )?;
    fs::write(
        out.join("overlay/usr/local/bin/aurora-firstboot"),
        firstboot_script(),
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/installer/index.html"),
        installer_html(&config.branding_name, &config.accent_color),
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/installer/installer.css"),
        installer_css(&config.accent_color),
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/installer/installer.js"),
        installer_js(),
    )?;
    fs::write(
        out.join("overlay/usr/share/themes/Aurora-Neon/index.theme"),
        gtk_theme_index(),
    )?;
    fs::write(
        out.join("overlay/usr/share/themes/Aurora-Neon/gtk-3.0/gtk.css"),
        gtk_theme_css(&config.accent_color),
    )?;
    fs::write(
        out.join("overlay/usr/share/icons/Aurora-Neon/index.theme"),
        icon_theme_index(),
    )?;
    fs::write(
        out.join("overlay/usr/share/grub/themes/aurora/theme.txt"),
        grub_theme_txt(&config.branding_name, &config.accent_color),
    )?;
    fs::write(
        out.join("overlay/usr/share/plymouth/themes/aurora-neon/aurora-neon.plymouth"),
        plymouth_theme_metadata(&config.theme_name),
    )?;
    fs::write(
        out.join("overlay/usr/share/plymouth/themes/aurora-neon/aurora-neon.script"),
        plymouth_theme_script(&config.branding_name, &config.accent_color),
    )?;
    fs::write(
        out.join("assets/branding/README.md"),
        "Place replacement logos, wallpapers, and boot graphics here.\n",
    )?;

    println!("Initialized distro tree at {}", out.display());
    Ok(())
}

fn prepare_host() -> Result<()> {
    ensure_root()?;
    let packages = [
        "build-essential",
        "pkg-config",
        "git",
        "curl",
        "debootstrap",
        "rsync",
        "xorriso",
        "grub-pc-bin",
        "grub-efi-amd64-bin",
        "grub-common",
        "mtools",
        "squashfs-tools",
        "casper",
        "dosfstools",
        "parted",
    ];
    run("apt-get", ["update"])?;
    let mut args = vec!["install", "-y"];
    args.extend(packages);
    run("apt-get", args)?;
    println!("Ubuntu 24 remaster host dependencies installed.");
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
        "parted",
        "mkfs.ext4",
        "mkfs.fat",
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

fn build_iso(tree: &Path, prompt_usb: bool, usb_device: Option<&Path>, system_mode: BootMode) -> Result<()> {
    ensure_root()?;
    check_tools()?;

    let config = load_config(tree)?;
    let system_scan = scan_system()?;
    let partition_plan = plan_partitions(system_scan.clone(), system_mode, 256);
    let build_root = tree.join("build/rootfs");
    let iso_root = tree.join("build/iso");
    let live_root = iso_root.join("live");

    for dir in [&build_root, &iso_root, &live_root, &tree.join("output")] {
        fs::create_dir_all(dir)?;
    }
    fs::write(
        tree.join("build/system-profile.json"),
        serde_json::to_string_pretty(&system_scan)?,
    )?;
    fs::write(
        tree.join("build/partition-plan.json"),
        serde_json::to_string_pretty(&partition_plan)?,
    )?;

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
        "apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y casper grub-pc-bin grub-efi-amd64-bin linux-generic {} {}",
        config.package_sets.join(" "),
        config.extra_packages.join(" ")
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
        [
            path_str(&iso_root)?,
            "-o",
            path_str(&output_iso)?,
            "--",
            "-volid",
            config.volume_id.as_str(),
        ],
    )?;

    for (_, dst) in mounts.iter().rev() {
        let _ = run("umount", [path_str(dst)?]);
    }

    println!("ISO created at {}", output_iso.display());
    if let Some(device) = usb_device {
        write_usb(&output_iso, device)?;
    } else if prompt_usb && prompt_yes_no("Write ISO to a USB device now?")? {
        let device = prompt_line("Enter the target device path (for example /dev/sdb): ")?;
        write_usb(&output_iso, Path::new(device.trim()))?;
    }
    Ok(())
}

fn scan_system() -> Result<SystemScan> {
    let firmware = if Path::new("/sys/firmware/efi").exists() {
        BootMode::Uefi
    } else {
        BootMode::Legacy
    };

    let cpu_model = fs::read_to_string("/proc/cpuinfo")
        .unwrap_or_default()
        .lines()
        .find(|line| line.starts_with("model name"))
        .and_then(|line| line.split(':').nth(1))
        .map(|line| line.trim().to_string())
        .unwrap_or_else(|| "Unknown CPU".to_string());
    let ram_mb = fs::read_to_string("/proc/meminfo")
        .unwrap_or_default()
        .lines()
        .find(|line| line.starts_with("MemTotal:"))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|value| value.parse::<u64>().ok())
        .map(|kb| kb / 1024)
        .unwrap_or(0);
    let disks = if command_exists("lsblk") {
        let output = Command::new("lsblk")
            .args(["-dn", "-o", "NAME,SIZE,MODEL"])
            .output()
            .context("failed to run lsblk")?;
        String::from_utf8_lossy(&output.stdout)
            .lines()
            .map(|line| line.trim().to_string())
            .filter(|line| !line.is_empty())
            .collect()
    } else {
        Vec::new()
    };

    Ok(SystemScan {
        firmware,
        architecture: std::env::consts::ARCH.to_string(),
        cpu_model,
        ram_mb,
        disks,
    })
}

fn plan_partitions(scan: SystemScan, mode: BootMode, disk_gb: u64) -> PartitionPlan {
    let firmware = match mode {
        BootMode::Auto => scan.firmware,
        other => other,
    };
    let mut partitions = Vec::new();
    let mut notes = vec!["Use GPT when targeting UEFI or dual-boot firmware support.".to_string()];
    if matches!(firmware, BootMode::Uefi | BootMode::Both) {
        partitions.push(Partition {
            label: "EFI".to_string(),
            fs: "fat32".to_string(),
            size_mb: 512,
            mountpoint: "/boot/efi".to_string(),
        });
    }
    if matches!(firmware, BootMode::Legacy | BootMode::Both) {
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
        firmware,
        partitions,
        notes: {
            notes.push("Create swap after root so installs can succeed on smaller disks.".to_string());
            notes.push(format!("Detected CPU: {} | RAM: {} MB", scan.cpu_model, scan.ram_mb));
            notes
        },
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
        "{}\nUbuntu 24.04 remaster with BIOS/UEFI install support.\n{}\n",
        config.distro_name,
        config.performance_goal
    );
    fs::write(rootfs.join("etc/issue"), issue)?;
    fs::write(
        rootfs.join("etc/motd"),
        format!(
            "{}\nTheme: {}\nGaming shell enabled.\n",
            config.branding_name, config.theme_name
        ),
    )?;
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
        config.branding_name
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
            "xfce4".to_string(),
            "lightdm".to_string(),
        ],
        DesktopPreset::Gnome => vec![
            "ubuntu-desktop".to_string(),
            "gnome-shell-extension-manager".to_string(),
            "steam-installer".to_string(),
            "gamemode".to_string(),
            "gnome-tweaks".to_string(),
        ],
        DesktopPreset::Kde => vec![
            "kubuntu-desktop".to_string(),
            "steam-installer".to_string(),
            "gamemode".to_string(),
            "plasma-workspace-wayland".to_string(),
        ],
    }
}

fn default_theme_css(name: &str, accent: &str) -> String {
    format!(
        ":root {{ --aurora-accent: {accent}; --aurora-bg: #08111d; --aurora-panel: #101c2b; }}\nbody {{ background: radial-gradient(circle at top, #14263a, var(--aurora-bg)); color: #eef7ff; font-family: 'Orbitron', sans-serif; }}\n.login-logo {{ display: none; }}\n.session-title::after {{ content: '{name}'; color: var(--aurora-accent); }}\n"
    )
}

fn installer_desktop_file() -> String {
    "[Desktop Entry]\nType=Application\nName=AURORA Installer\nExec=/usr/local/bin/aurora-firstboot\nTerminal=false\nX-GNOME-Autostart-enabled=true\nCategories=System;\n".to_string()
}

fn firstboot_script() -> String {
    "#!/usr/bin/env bash\nset -euo pipefail\nif command -v xdg-open >/dev/null 2>&1; then\n  xdg-open /usr/share/aurora/installer/index.html >/dev/null 2>&1 || true\nfi\n".to_string()
}

fn installer_html(name: &str, accent: &str) -> String {
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>{name} Installer</title><link rel=\"stylesheet\" href=\"installer.css\"></head><body><main class=\"shell\"><section class=\"hero\"><p class=\"eyebrow\">Gaming-grade Ubuntu Remaster</p><h1>{name}</h1><p class=\"lede\">A Kali-inspired neon interface for a CPU-first performance distro. If you do not know your system details, the installer can scan and build the partition plan for you.</p><div class=\"cta-row\"><button id=\"scanBtn\">Scan This System</button><button id=\"planBtn\">Create Partition Plan</button><button id=\"usbBtn\">Use Inbuilt USB Writer</button></div></section><section class=\"grid\"><article class=\"card\"><h2>Firmware Support</h2><p>Legacy BIOS, UEFI, or dual-target workflows.</p></article><article class=\"card\"><h2>Desktop Presets</h2><p>GNOME, KDE, or a lean minimal shell generated from the same Rust profile.</p></article><article class=\"card\"><h2>Performance Profile</h2><p>Huge pages, gaming overlay tools, and AURORA tuning hooks are staged into the distro tree.</p></article></section><section class=\"terminal\"><div class=\"terminal-bar\"><span></span><span></span><span></span></div><pre id=\"output\">Awaiting action...</pre></section></main><script>window.AURORA_INSTALLER={{accent:\"{accent}\",name:\"{name}\"}};</script><script src=\"installer.js\"></script></body></html>"
    )
}

fn installer_css(accent: &str) -> String {
    format!(
        ":root{{--accent:{accent};--bg:#05070d;--panel:#0f1724;--line:#17314d;--text:#edf7ff;--muted:#9db4c8;--warm:#ff6a00}}*{{box-sizing:border-box}}body{{margin:0;font-family:'Orbitron',sans-serif;background:radial-gradient(circle at top,#10233a 0%,#05070d 55%,#020304 100%);color:var(--text)}}.shell{{max-width:1100px;margin:0 auto;padding:40px 24px 72px}}.hero{{padding:48px 36px;border:1px solid var(--line);background:linear-gradient(135deg,rgba(16,31,52,.92),rgba(8,13,20,.96));border-radius:28px;box-shadow:0 20px 80px rgba(0,0,0,.45)}}.eyebrow{{letter-spacing:.22em;text-transform:uppercase;color:var(--accent);font-size:12px}}h1{{font-size:72px;line-height:.95;margin:16px 0}}.lede{{max-width:760px;color:var(--muted);font-family:'Rajdhani',sans-serif;font-size:24px}}.cta-row{{display:flex;flex-wrap:wrap;gap:16px;margin-top:28px}}button{{border:1px solid var(--line);background:linear-gradient(180deg,#102740,#0a1524);color:var(--text);padding:16px 22px;border-radius:16px;font:inherit;cursor:pointer}}button:hover{{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent),0 0 24px rgba(18,247,255,.22)}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:18px;margin-top:24px}}.card{{padding:22px;border-radius:22px;background:rgba(11,18,30,.86);border:1px solid var(--line)}}.card h2{{margin:0 0 8px;font-size:24px}}.card p{{margin:0;color:var(--muted);font-family:'Rajdhani',sans-serif;font-size:20px}}.terminal{{margin-top:24px;border-radius:22px;overflow:hidden;border:1px solid var(--line);background:#071019}}.terminal-bar{{display:flex;gap:8px;padding:12px 14px;background:#0c1725}}.terminal-bar span{{width:12px;height:12px;border-radius:50%;background:var(--warm)}}.terminal-bar span:nth-child(2){{background:#ffca28}}.terminal-bar span:nth-child(3){{background:var(--accent)}}pre{{margin:0;padding:22px;color:#d8f7ff;min-height:180px;font-family:'JetBrains Mono',monospace;white-space:pre-wrap}}@media (max-width:720px){{h1{{font-size:48px}}.lede{{font-size:20px}}}}"
    )
}

fn installer_js() -> String {
    "const output=document.getElementById('output');const write=(lines)=>output.textContent=lines.join('\\n');document.getElementById('scanBtn').addEventListener('click',()=>write(['Scanning current machine...','- detect firmware mode','- detect CPU model and RAM','- enumerate storage devices','Result: if details are missing, aurora-distro can fall back to automatic planning.']));document.getElementById('planBtn').addEventListener('click',()=>write(['Generating partition plan...','- legacy BIOS: bios_grub + root + swap','- UEFI: EFI + root + swap','- BOTH: EFI + bios_grub + root + swap','Result: installer chooses the correct layout for the detected firmware.']));document.getElementById('usbBtn').addEventListener('click',()=>write(['Inbuilt USB writer flow','1. Confirm target device','2. Verify path is a block device','3. Write ISO with dd + sync','4. Return success/failure logs to the user']));".to_string()
}

fn gtk_theme_index() -> String {
    "[Desktop Entry]\nName=Aurora Neon\nComment=Gaming-inspired dark GTK theme\n".to_string()
}

fn gtk_theme_css(accent: &str) -> String {
    format!(
        "@define-color accent {accent};\n@define-color bg #09111b;\n@define-color panel #111c2b;\n@define-color warm #ff6a00;\nwindow,dialog{{background-image:none;background-color:@bg;color:#eef7ff}}headerbar{{background:linear-gradient(to bottom,#10233a,#0a1524);border-bottom:1px solid shade(@accent,.65)}}button{{background-image:none;background:linear-gradient(to bottom,#10253b,#0c1726);border:1px solid shade(@accent,.6);border-radius:12px;color:#eef7ff}}button:hover{{box-shadow:0 0 10px alpha(@accent,.25)}}entry,spinbutton,textview{{background:#0d1520;border:1px solid #18314d;color:#eef7ff}}"
    )
}

fn icon_theme_index() -> String {
    "[Icon Theme]\nName=Aurora-Neon\nComment=Placeholder icon theme for AURORA gaming distro\nInherits=Adwaita\nDirectories=.\n".to_string()
}

fn performance_shell_script(profile: &PerformanceProfile) -> String {
    format!(
        "#!/usr/bin/env bash\nexport AURORA_PROFILE_NAME=\"{}\"\nexport AURORA_CPU_GOVERNOR=\"{}\"\nexport AURORA_ENABLE_HUGEPAGES=\"{}\"\nexport AURORA_ENABLE_GAMEMODE=\"{}\"\nexport AURORA_ENABLE_MANGOHUD=\"{}\"\n",
        profile.name,
        profile.cpu_governor,
        profile.enable_hugepages,
        profile.enable_gamemode,
        profile.enable_mangohud,
    )
}

fn performance_sysctl_conf() -> String {
    "vm.swappiness=10\nvm.dirty_ratio=5\nvm.dirty_background_ratio=2\nkernel.numa_balancing=1\nkernel.sched_autogroup_enabled=0\n".to_string()
}

fn default_wallpaper_svg(name: &str, accent: &str) -> String {
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 1920 1080\"><defs><linearGradient id=\"bg\" x1=\"0\" y1=\"0\" x2=\"1\" y2=\"1\"><stop offset=\"0%\" stop-color=\"#05070d\"/><stop offset=\"100%\" stop-color=\"#101f35\"/></linearGradient></defs><rect width=\"1920\" height=\"1080\" fill=\"url(#bg)\"/><circle cx=\"1510\" cy=\"190\" r=\"220\" fill=\"{accent}\" fill-opacity=\"0.18\"/><circle cx=\"260\" cy=\"880\" r=\"260\" fill=\"#ff5e00\" fill-opacity=\"0.14\"/><text x=\"120\" y=\"860\" fill=\"#eef7ff\" font-size=\"124\" font-family=\"Orbitron, sans-serif\">{name}</text><text x=\"128\" y=\"930\" fill=\"{accent}\" font-size=\"40\" font-family=\"Rajdhani, sans-serif\">Kali-inspired gaming shell for Ubuntu 24.04 remastering</text></svg>"
    )
}

fn grub_theme_txt(name: &str, accent: &str) -> String {
    format!(
        "title-text: \"{name}\"\ntitle-font: \"DejaVu Sans Bold 28\"\ntitle-color: \"255,255,255\"\nmessage-font: \"DejaVu Sans 16\"\nmessage-color: \"18,247,255\"\ndesktop-image: \"/boot/grub/themes/aurora/background.png\"\nprogress-bar-fg-color: \"18,247,255\"\nprogress-bar-bg-color: \"16,28,43\"\nselected-item-color: \"255,94,0\"\n"
    )
}

fn plymouth_theme_metadata(theme_name: &str) -> String {
    format!(
        "[Plymouth Theme]\nName={theme_name}\nDescription=Neon gaming boot splash for AURORA remaster\nModuleName=script\n"
    )
}

fn plymouth_theme_script(name: &str, accent: &str) -> String {
    format!(
        "Window.SetBackgroundTopColor (0.02, 0.03, 0.06);\nWindow.SetBackgroundBottomColor (0.03, 0.08, 0.15);\nlabel = Image.Text(\"{name}\", 1, 1, 1);\nlabel.SetX(Window.GetWidth() / 2 - label.GetWidth() / 2);\nlabel.SetY(Window.GetHeight() / 2 - 40);\nsub = Image.Text(\"Gaming Performance Shell\", {r}, {g}, {b});\nsub.SetX(Window.GetWidth() / 2 - sub.GetWidth() / 2);\nsub.SetY(Window.GetHeight() / 2 + 10);\n",
        r = hex_channel_to_float(accent, 1),
        g = hex_channel_to_float(accent, 3),
        b = hex_channel_to_float(accent, 5),
    )
}

fn hex_channel_to_float(hex: &str, offset: usize) -> f32 {
    let channel = hex
        .get(offset..offset + 2)
        .and_then(|value| u8::from_str_radix(value, 16).ok())
        .unwrap_or(255);
    channel as f32 / 255.0
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

fn prompt_yes_no(prompt: &str) -> Result<bool> {
    println!("{prompt} [y/N]");
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    Ok(matches!(input.trim().to_ascii_lowercase().as_str(), "y" | "yes"))
}

fn prompt_line(prompt: &str) -> Result<String> {
    print!("{prompt}");
    use std::io::Write;
    std::io::stdout().flush()?;
    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    Ok(input)
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
