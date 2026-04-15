use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand, ValueEnum};
use serde::{Deserialize, Serialize};
use std::ffi::OsStr;
use std::fs;
use std::io::{self, Write};
#[cfg(target_family = "unix")]
use std::os::unix::fs::PermissionsExt;
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
    /// Launch an Aurora application surface in terminal mode.
    App {
        #[arg(value_enum)]
        surface: AppSurface,
        #[arg(long)]
        action: Option<String>,
    },
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "kebab-case")]
enum AppSurface {
    Installer,
    ControlCenter,
    AiHub,
    DevLab,
    CreatorStudio,
    GamingCenter,
    SecurityCenter,
    PackageCenter,
    Andromeda,
    Welcome,
}

#[derive(Clone, Debug, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum DesktopPreset {
    Minimal,
    Gnome,
    Kde,
    Cosmic,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
enum BootMode {
    Auto,
    Legacy,
    Uefi,
    Both,
}

#[derive(Copy, Clone, Debug, Serialize, Deserialize, ValueEnum, PartialEq, Eq)]
#[serde(rename_all = "kebab-case")]
enum InstallMode {
    Balanced,
    Gaming,
    Creator,
    Training,
    Datacenter,
    MaxThroughput,
}

impl InstallMode {
    fn slug(self) -> &'static str {
        match self {
            InstallMode::Balanced => "balanced",
            InstallMode::Gaming => "gaming",
            InstallMode::Creator => "creator",
            InstallMode::Training => "training",
            InstallMode::Datacenter => "datacenter",
            InstallMode::MaxThroughput => "max-throughput",
        }
    }
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
    enable_memory_boost_wizard: bool,
    enable_boot_repair_tools: bool,
    available_modes: Vec<InstallMode>,
    default_mode: InstallMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerformanceProfile {
    mode: InstallMode,
    name: String,
    cpu_governor: String,
    enable_hugepages: bool,
    enable_gamemode: bool,
    enable_mangohud: bool,
    tune_sysctl: bool,
    enable_zram: bool,
    zram_fraction_percent: u8,
    use_tmpfs_for_temp: bool,
    disable_unneeded_services: bool,
    enable_preload: bool,
    io_scheduler: String,
    swap_partition_policy: String,
    readahead_kb: u32,
    vm_swappiness: u8,
    vm_dirty_ratio: u8,
    vm_dirty_background_ratio: u8,
    disable_cpu_idle: bool,
    scheduler_tune: String,
    cpu_energy_policy: String,
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
    flags: Vec<String>,
    mount_options: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct PartitionPlan {
    firmware: BootMode,
    partition_table: String,
    partitions: Vec<Partition>,
    notes: Vec<String>,
    bootloader_steps: Vec<String>,
    recovery_steps: Vec<String>,
}

#[derive(Debug, Clone)]
struct AuroraPackageSpec {
    name: String,
    version: String,
    architecture: String,
    depends: Vec<String>,
    description: String,
    payload_paths: Vec<String>,
    postinst: Option<String>,
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
        Commands::App { surface, action } => launch_app(surface, action.as_deref()),
    }
}

fn init_tree(out: &Path, distro_name: &str, desktop: DesktopPreset) -> Result<()> {
    for dir in [
        out.to_path_buf(),
        out.join("profiles"),
        out.join("overlay"),
        out.join("overlay/etc/skel"),
        out.join("overlay/etc/skel/.config"),
        out.join("overlay/etc/skel/.config/autostart"),
        out.join("overlay/etc/skel/.local"),
        out.join("overlay/etc/skel/.local/bin"),
        out.join("overlay/etc/aurora-installer"),
        out.join("overlay/etc/default"),
        out.join("overlay/etc/aurora"),
        out.join("overlay/etc/dconf/db/local.d"),
        out.join("overlay/etc/dconf/profile"),
        out.join("overlay/etc/profile.d"),
        out.join("overlay/etc/sysctl.d"),
        out.join("overlay/etc/systemd/system"),
        out.join("overlay/usr/local/bin"),
        out.join("overlay/usr/share/applications"),
        out.join("overlay/usr/share/aurora/installer"),
        out.join("overlay/usr/share/aurora/control-center"),
        out.join("overlay/usr/share/aurora/ai-hub"),
        out.join("overlay/usr/share/aurora/dev-lab"),
        out.join("overlay/usr/share/aurora/creator-studio"),
        out.join("overlay/usr/share/aurora/gaming-center"),
        out.join("overlay/usr/share/aurora/security-center"),
        out.join("overlay/usr/share/aurora/package-center"),
        out.join("overlay/usr/share/aurora/andromeda"),
        out.join("overlay/usr/share/aurora/welcome"),
        out.join("overlay/usr/share/aurora/archive"),
        out.join("overlay/usr/share/themes/Aurora-Neon/gtk-3.0"),
        out.join("overlay/usr/share/icons/Aurora-Neon"),
        out.join("overlay/usr/share/plymouth/themes/aurora-neon"),
        out.join("overlay/usr/share/grub/themes/aurora"),
        out.join("overlay/usr/share/backgrounds/aurora"),
        out.join("overlay/etc/apt/sources.list.d"),
        out.join("overlay/opt/aurora/repo/dists/stable/main/binary-amd64"),
        out.join("overlay/opt/aurora/repo/pool/main"),
        out.join("overlay/opt/aurora/repo/meta"),
        out.join("overlay/home/aurora/.config/autostart"),
        out.join("overlay/usr/local/share/aurora/scripts"),
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
        logo_path: "assets/branding/logo.svg".to_string(),
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
            "btop".to_string(),
            "zram-tools".to_string(),
            "numactl".to_string(),
            "linux-tools-generic".to_string(),
            "gnome-shell-extensions".to_string(),
            "gnome-software-plugin-flatpak".to_string(),
            "flatpak".to_string(),
            "papirus-icon-theme".to_string(),
            "fonts-cascadia-code".to_string(),
            "alacritty".to_string(),
            "vlc".to_string(),
            "pipewire-audio".to_string(),
            "pavucontrol".to_string(),
            "fastfetch".to_string(),
            "eza".to_string(),
            "earlyoom".to_string(),
            "irqbalance".to_string(),
            "thermald".to_string(),
            "ufw".to_string(),
            "fail2ban".to_string(),
            "apparmor-utils".to_string(),
            "fprintd".to_string(),
            "ollama".to_string(),
            "copyq".to_string(),
            "podman".to_string(),
            "distrobox".to_string(),
            "virt-manager".to_string(),
            "qemu-system-x86".to_string(),
            "libvirt-daemon-system".to_string(),
            "bridge-utils".to_string(),
            "dnsmasq-base".to_string(),
            "gamescope".to_string(),
            "goverlay".to_string(),
            "mesa-vulkan-drivers".to_string(),
            "mesa-utils".to_string(),
            "obs-studio".to_string(),
            "gimp".to_string(),
            "inkscape".to_string(),
            "ffmpeg".to_string(),
            "libreoffice".to_string(),
            "git-lfs".to_string(),
            "tmux".to_string(),
            "direnv".to_string(),
            "shellcheck".to_string(),
            "hyperfine".to_string(),
            "wireshark".to_string(),
            "libspa-0.2-bluetooth".to_string(),
            "valkey-server".to_string(),
            "kdump-tools".to_string(),
            "gvfs-backends".to_string(),
            "gvfs-fuse".to_string(),
            "sshfs".to_string(),
            "nautilus-extension-gnome-terminal".to_string(),
            "dconf-editor".to_string(),
            "authd".to_string(),
            "ripgrep".to_string(),
            "fd-find".to_string(),
            "bat".to_string(),
            "helix".to_string(),
            "zellij".to_string(),
            "bottom".to_string(),
            "just".to_string(),
            "neofetch".to_string(),
            "micro".to_string(),
            "ranger".to_string(),
            "tldr".to_string(),
            "duf".to_string(),
            "ncdu".to_string(),
            "glances".to_string(),
            "silversearcher-ag".to_string(),
            "fzf".to_string(),
            "jq".to_string(),
            "python3-venv".to_string(),
            "python3-pip".to_string(),
            "python3-dev".to_string(),
            "blender".to_string(),
            "krita".to_string(),
            "kdenlive".to_string(),
            "audacity".to_string(),
            "sysstat".to_string(),
            "dstat".to_string(),
            "iotop".to_string(),
            "fio".to_string(),
            "nvme-cli".to_string(),
            "prometheus-node-exporter".to_string(),
            "bpfcc-tools".to_string(),
            "tor".to_string(),
            "torsocks".to_string(),
            "macchanger".to_string(),
            "usbguard".to_string(),
        ],
        bios_legacy: true,
        uefi: true,
        kali_like_theme: true,
        theme_name: "Aurora Neon Assault".to_string(),
        accent_color: "#12f7ff".to_string(),
        performance_goal: "Tune for aggressive boot speed, stronger desktop responsiveness, faster load-time behavior, and strong CPU throughput on supported hardware; 3x uplift is a target for selective workloads, not a universal guarantee.".to_string(),
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
        enable_memory_boost_wizard: true,
        enable_boot_repair_tools: true,
        available_modes: vec![
            InstallMode::Balanced,
            InstallMode::Gaming,
            InstallMode::Creator,
            InstallMode::Training,
            InstallMode::Datacenter,
            InstallMode::MaxThroughput,
        ],
        default_mode: InstallMode::Gaming,
    };
    let balanced = performance_profile(InstallMode::Balanced);
    let gaming = performance_profile(InstallMode::Gaming);
    let creator = performance_profile(InstallMode::Creator);
    let training = performance_profile(InstallMode::Training);
    let datacenter = performance_profile(InstallMode::Datacenter);
    let performance = performance_profile(InstallMode::MaxThroughput);

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
        out.join("profiles/performance-balanced.json"),
        serde_json::to_string_pretty(&balanced)?,
    )?;
    fs::write(
        out.join("profiles/performance-gaming.json"),
        serde_json::to_string_pretty(&gaming)?,
    )?;
    fs::write(
        out.join("profiles/performance-creator.json"),
        serde_json::to_string_pretty(&creator)?,
    )?;
    fs::write(
        out.join("profiles/performance-training.json"),
        serde_json::to_string_pretty(&training)?,
    )?;
    fs::write(
        out.join("profiles/performance-datacenter.json"),
        serde_json::to_string_pretty(&datacenter)?,
    )?;
    fs::write(
        out.join("profiles/performance-max-throughput.json"),
        serde_json::to_string_pretty(&performance)?,
    )?;
    fs::write(
        out.join("assets/theme/theme.css"),
        default_theme_css(&config.distro_name, &config.accent_color),
    )?;
    fs::write(
        out.join("overlay/etc/default/grub"),
        "GRUB_TIMEOUT=3\nGRUB_GFXMODE=1920x1080\nGRUB_THEME=/usr/share/grub/themes/aurora/theme.txt\nGRUB_CMDLINE_LINUX_DEFAULT=\"quiet splash transparent_hugepage=always zswap.enabled=1 zswap.compressor=lz4 zswap.max_pool_percent=25 nowatchdog\"\n",
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
        out.join("overlay/etc/aurora-installer/autoinstall.yaml"),
        autoinstall_yaml(&config, &installer),
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
        performance_sysctl_conf(&performance),
    )?;
    fs::write(
        out.join("overlay/etc/default/aurora-performance"),
        performance_defaults(&performance),
    )?;
    fs::write(
        out.join("overlay/etc/aurora/directstream.conf"),
        directstream_defaults(),
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
        out.join("overlay/usr/share/applications/aurora-control-center.desktop"),
        aurora_control_center_desktop_file(),
    )?;
    fs::write(
        out.join("overlay/usr/share/applications/aurora-ai-hub.desktop"),
        aurora_ai_hub_desktop_file(),
    )?;
    fs::write(
        out.join("overlay/usr/share/applications/aurora-dev-lab.desktop"),
        aurora_dev_lab_desktop_file(),
    )?;
    fs::write(
        out.join("overlay/usr/share/applications/aurora-creator-studio.desktop"),
        aurora_creator_studio_desktop_file(),
    )?;
    fs::write(
        out.join("overlay/usr/share/applications/aurora-gaming-center.desktop"),
        aurora_gaming_center_desktop_file(),
    )?;
    fs::write(
        out.join("overlay/usr/share/applications/aurora-security-center.desktop"),
        aurora_security_center_desktop_file(),
    )?;
    fs::write(
        out.join("overlay/usr/share/applications/aurora-package-center.desktop"),
        aurora_package_center_desktop_file(),
    )?;
    fs::write(
        out.join("overlay/usr/share/applications/aurora-welcome.desktop"),
        aurora_welcome_desktop_file(),
    )?;
    fs::write(
        out.join("overlay/usr/share/applications/aurora-andromeda.desktop"),
        aurora_andromeda_desktop_file(),
    )?;
    fs::write(
        out.join("overlay/home/aurora/.config/autostart/aurora-installer.desktop"),
        installer_desktop_file(),
    )?;
    fs::write(
        out.join("overlay/etc/skel/.config/autostart/aurora-control-center.desktop"),
        aurora_control_center_desktop_file(),
    )?;
    fs::write(
        out.join("overlay/etc/skel/.config/autostart/aurora-ai-hub.desktop"),
        aurora_ai_hub_desktop_file(),
    )?;
    fs::write(
        out.join("overlay/etc/skel/.config/autostart/aurora-dev-lab.desktop"),
        aurora_dev_lab_desktop_file(),
    )?;
    fs::write(
        out.join("overlay/etc/skel/.config/autostart/aurora-welcome.desktop"),
        aurora_welcome_desktop_file(),
    )?;
    fs::write(
        out.join("overlay/etc/skel/.config/autostart/aurora-andromeda.desktop"),
        aurora_andromeda_desktop_file(),
    )?;
    fs::write(
        out.join("overlay/home/aurora/.config/autostart/aurora-control-center.desktop"),
        aurora_control_center_desktop_file(),
    )?;
    fs::write(
        out.join("overlay/home/aurora/.config/autostart/aurora-ai-hub.desktop"),
        aurora_ai_hub_desktop_file(),
    )?;
    fs::write(
        out.join("overlay/home/aurora/.config/autostart/aurora-dev-lab.desktop"),
        aurora_dev_lab_desktop_file(),
    )?;
    fs::write(
        out.join("overlay/home/aurora/.config/autostart/aurora-welcome.desktop"),
        aurora_welcome_desktop_file(),
    )?;
    fs::write(
        out.join("overlay/home/aurora/.config/autostart/aurora-andromeda.desktop"),
        aurora_andromeda_desktop_file(),
    )?;
    fs::write(
        out.join("overlay/usr/local/bin/aurora-firstboot"),
        firstboot_script(&installer),
    )?;
    fs::write(
        out.join("overlay/usr/local/bin/aurora-apply-desktop"),
        desktop_setup_script(&config),
    )?;
    fs::write(
        out.join("overlay/usr/local/bin/aurora-mode-switch"),
        mode_switch_script(),
    )?;
    fs::write(
        out.join("overlay/usr/local/bin/aurora-ai-setup"),
        ai_setup_script(),
    )?;
    fs::write(
        out.join("overlay/usr/local/bin/aurora-dev-setup"),
        dev_setup_script(),
    )?;
    fs::write(
        out.join("overlay/usr/local/bin/aurora-creator-setup"),
        creator_setup_script(),
    )?;
    fs::write(
        out.join("overlay/usr/local/bin/aurora-gaming-setup"),
        gaming_setup_script(),
    )?;
    fs::write(
        out.join("overlay/usr/local/bin/aurora-gpu-setup"),
        gpu_setup_script(),
    )?;
    fs::write(
        out.join("overlay/usr/local/bin/aurora-ai-helper"),
        ai_helper_script(),
    )?;
    fs::write(
        out.join("overlay/usr/local/bin/aurora-privacy-setup"),
        privacy_setup_script(),
    )?;
    fs::write(
        out.join("overlay/usr/local/bin/aurora-training-setup"),
        training_setup_script(),
    )?;
    fs::write(
        out.join("overlay/usr/local/bin/aurora-datacenter-setup"),
        datacenter_setup_script(),
    )?;
    fs::write(
        out.join("overlay/usr/local/bin/aurora-security-setup"),
        security_setup_script(),
    )?;
    fs::write(
        out.join("overlay/usr/local/bin/aurora-hardware-guard"),
        hardware_guard_script(),
    )?;
    fs::write(
        out.join("overlay/usr/local/bin/aurora-apt"),
        aurora_apt_script(),
    )?;
    fs::write(
        out.join("overlay/usr/local/bin/aurora-directstream"),
        directstream_script(),
    )?;
    fs::write(
        out.join("overlay/usr/local/bin/aurora-autosetup"),
        autosetup_script(),
    )?;
    fs::write(
        out.join("overlay/etc/systemd/system/aurora-autosetup.service"),
        autosetup_service(),
    )?;
    fs::write(
        out.join("overlay/etc/systemd/system/aurora-inference.service"),
        aurora_inference_service(),
    )?;
    fs::write(
        out.join("overlay/etc/systemd/system/aurora-zram-setup.service"),
        zram_service(),
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/installer/index.html"),
        installer_html(&config.branding_name, &config.accent_color, &installer),
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/installer/installer.css"),
        installer_css(&config.accent_color),
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/installer/installer.js"),
        installer_js(&installer),
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/installer/autoinstall.yaml"),
        autoinstall_yaml(&config, &installer),
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/control-center/index.html"),
        control_center_html(&config.branding_name, &config.accent_color),
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/control-center/control-center.css"),
        control_center_css(&config.accent_color),
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/control-center/control-center.js"),
        control_center_js(),
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/ai-hub/index.html"),
        ai_hub_html(&config.branding_name, &config.accent_color),
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/ai-hub/ai-hub.css"),
        ai_hub_css(&config.accent_color),
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/dev-lab/index.html"),
        dev_lab_html(&config.branding_name, &config.accent_color),
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/dev-lab/dev-lab.css"),
        dev_lab_css(&config.accent_color),
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/creator-studio/index.html"),
        creator_studio_html(&config.branding_name, &config.accent_color),
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/creator-studio/creator-studio.css"),
        creator_studio_css(&config.accent_color),
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/gaming-center/index.html"),
        gaming_center_html(&config.branding_name, &config.accent_color),
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/gaming-center/gaming-center.css"),
        gaming_center_css(&config.accent_color),
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/security-center/index.html"),
        security_center_html(&config.branding_name, &config.accent_color),
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/security-center/security-center.css"),
        security_center_css(&config.accent_color),
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/package-center/index.html"),
        package_center_html(&config.branding_name, &config.accent_color),
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/package-center/package-center.css"),
        package_center_css(&config.accent_color),
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/andromeda/index.html"),
        andromeda_html(&config.branding_name),
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/andromeda/andromeda.css"),
        andromeda_css(&config.accent_color),
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/archive/catalog.json"),
        aurora_archive_catalog_json(&config)?,
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/archive/meta-packages.json"),
        aurora_meta_packages_json(&config)?,
    )?;
    fs::write(
        out.join("overlay/opt/aurora/repo/dists/stable/main/binary-amd64/Packages"),
        aurora_repo_packages(&config),
    )?;
    fs::write(
        out.join("overlay/opt/aurora/repo/dists/stable/Release"),
        aurora_repo_release(&config),
    )?;
    fs::write(
        out.join("overlay/opt/aurora/repo/meta/aurora-desktop.control"),
        aurora_meta_package_control(
            "aurora-desktop",
            &config,
            &[
                "aurora-control-center",
                "aurora-app-center",
                "aurora-welcome",
                "aurora-dev-lab",
                "aurora-creator-studio",
                "aurora-gaming-center",
                "aurora-security-center",
                "aurora-ai-hub",
            ],
            "Aurora desktop surface, branding, defaults, and workflow layer.",
        ),
    )?;
    fs::write(
        out.join("overlay/opt/aurora/repo/meta/aurora-ai-stack.control"),
        aurora_meta_package_control(
            "aurora-ai-stack",
            &config,
            &["ollama", "aurora-ai-hub", "ripgrep", "fd-find", "bat", "helix"],
            "Aurora local AI stack with offline model tooling and fast Rust-native workstation utilities.",
        ),
    )?;
    fs::write(
        out.join("overlay/opt/aurora/repo/meta/aurora-gaming-stack.control"),
        aurora_meta_package_control(
            "aurora-gaming-stack",
            &config,
            &["gamemode", "mangohud", "gamescope", "goverlay", "steam-installer"],
            "Aurora gaming stack with Gamescope, GameMode, MangoHud, and curated gaming integration.",
        ),
    )?;
    fs::write(
        out.join("overlay/opt/aurora/repo/meta/aurora-creator-stack.control"),
        aurora_meta_package_control(
            "aurora-creator-stack",
            &config,
            &[
                "vlc",
                "flatpak",
                "gnome-software-plugin-flatpak",
                "obs-studio",
                "ffmpeg",
                "gimp",
                "inkscape",
                "libreoffice",
                "aurora-creator-studio",
            ],
            "Aurora creator and workstation stack with media, packaging, and terminal productivity tooling.",
        ),
    )?;
    fs::write(
        out.join("overlay/opt/aurora/repo/meta/aurora-dev-stack.control"),
        aurora_meta_package_control(
            "aurora-dev-stack",
            &config,
            &[
                "aurora-dev-lab",
                "git",
                "git-lfs",
                "podman",
                "distrobox",
                "tmux",
                "direnv",
                "shellcheck",
                "hyperfine",
            ],
            "Aurora development stack with containers, CLI tooling, shell quality checks, and workspace automation.",
        ),
    )?;
    fs::write(
        out.join("overlay/opt/aurora/repo/meta/aurora-security-stack.control"),
        aurora_meta_package_control(
            "aurora-security-stack",
            &config,
            &["ufw", "fail2ban", "apparmor-utils", "fprintd", "kdump-tools", "authd"],
            "Aurora security and hardening stack with firewall, audit, and authentication tooling.",
        ),
    )?;
    fs::write(
        out.join("overlay/opt/aurora/repo/meta/aurora-privacy-stack.control"),
        aurora_meta_package_control(
            "aurora-privacy-stack",
            &config,
            &["tor", "torsocks", "macchanger", "usbguard"],
            "Aurora privacy stack with MAC randomization posture, Tor tooling, and device access hardening.",
        ),
    )?;
    fs::write(
        out.join("overlay/opt/aurora/repo/meta/aurora-scientific-stack.control"),
        aurora_meta_package_control(
            "aurora-scientific-stack",
            &config,
            &["numactl", "linux-tools-generic", "hwloc", "valkey-server", "ripgrep", "fd-find"],
            "Aurora scientific and HPC stack with NUMA, performance counters, and high-speed data tools.",
        ),
    )?;
    fs::write(
        out.join("overlay/opt/aurora/repo/meta/aurora-training-stack.control"),
        aurora_meta_package_control(
            "aurora-training-stack",
            &config,
            &["python3-venv", "python3-pip", "python3-dev", "numactl", "hwloc", "valkey-server", "fio"],
            "Aurora training stack with Python environments, NUMA tooling, cache services, and dataset throughput helpers.",
        ),
    )?;
    fs::write(
        out.join("overlay/opt/aurora/repo/meta/aurora-datacenter-stack.control"),
        aurora_meta_package_control(
            "aurora-datacenter-stack",
            &config,
            &["sysstat", "dstat", "iotop", "prometheus-node-exporter", "bpfcc-tools", "fio", "nvme-cli"],
            "Aurora datacenter stack with observability, storage validation, and service-host tuning tools.",
        ),
    )?;
    fs::write(
        out.join("overlay/usr/local/bin/install-aurora-meta"),
        "#!/usr/bin/env bash\nset -euo pipefail\napt update\napt install -y aurora-desktop aurora-ai-stack aurora-dev-stack aurora-gaming-stack aurora-creator-stack aurora-security-stack aurora-privacy-stack aurora-scientific-stack aurora-training-stack aurora-datacenter-stack\n".to_string(),
    )?;
    fs::write(
        out.join("overlay/etc/apt/sources.list.d/aurora.sources"),
        aurora_sources_file(),
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/welcome/index.html"),
        welcome_html(&config.branding_name, &config.accent_color),
    )?;
    fs::write(
        out.join("overlay/usr/share/aurora/welcome/welcome.css"),
        welcome_css(&config.accent_color),
    )?;
    fs::write(
        out.join("overlay/etc/dconf/profile/user"),
        "user-db:user\nsystem-db:local\n".to_string(),
    )?;
    fs::write(
        out.join("overlay/etc/dconf/db/local.d/00-aurora"),
        dconf_defaults(&config),
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
    fs::write(
        out.join("assets/branding/logo.svg"),
        default_logo_svg(&config.branding_name, &config.accent_color),
    )?;

    println!("Initialized distro tree at {}", out.display());
    Ok(())
}

fn launch_app(surface: AppSurface, action: Option<&str>) -> Result<()> {
    match surface {
        AppSurface::Installer => installer_app(action),
        AppSurface::ControlCenter => control_center_app(action),
        AppSurface::AiHub => ai_hub_app(action),
        AppSurface::DevLab => dev_lab_app(action),
        AppSurface::CreatorStudio => creator_studio_app(action),
        AppSurface::GamingCenter => gaming_center_app(action),
        AppSurface::SecurityCenter => security_center_app(action),
        AppSurface::PackageCenter => package_center_app(action),
        AppSurface::Andromeda => andromeda_app(action),
        AppSurface::Welcome => welcome_app(action),
    }
}

fn installer_app(action: Option<&str>) -> Result<()> {
    let action = action.unwrap_or("summary");
    match action {
        "summary" => {
            println!("Aurora Installer");
            println!("- Modes: balanced, gaming, creator, training, datacenter, max-throughput");
            println!("- Commands:");
            println!("  aurora-distro app installer --action scan");
            println!("  aurora-distro app installer --action plan");
            println!("  aurora-distro app installer --action tools");
            Ok(())
        }
        "scan" => {
            println!("{}", serde_json::to_string_pretty(&scan_system()?)?);
            Ok(())
        }
        "plan" => {
            println!(
                "{}",
                serde_json::to_string_pretty(&plan_partitions(scan_system()?, BootMode::Auto, 256))?
            );
            Ok(())
        }
        "tools" => {
            check_tools()?;
            Ok(())
        }
        other => bail!("unknown installer action: {other}"),
    }
}

fn control_center_app(action: Option<&str>) -> Result<()> {
    match action.unwrap_or("status") {
        "status" => {
            println!("Aurora Control Center");
            println!("- Runtime: {}", command_summary("aurora", ["version"]));
            println!("- AI: {}", command_exists("ollama"));
            println!("- Gaming: {}", Path::new("/etc/aurora/directstream.conf").exists());
            println!("- Security config: {}", Path::new("/etc/default/aurora-performance").exists());
            Ok(())
        }
        "balanced" | "gaming" | "creator" | "training" | "datacenter" | "max-throughput" => {
            run("/usr/local/bin/aurora-mode-switch", [action.unwrap()])?;
            Ok(())
        }
        "desktop" => run("/usr/local/bin/aurora-apply-desktop", std::iter::empty::<&str>()),
        "ai" => ai_hub_app(Some("status")),
        "games" => gaming_center_app(Some("status")),
        "security" => security_center_app(Some("status")),
        "packages" => package_center_app(Some("status")),
        other => bail!("unknown control-center action: {other}"),
    }
}

fn ai_hub_app(action: Option<&str>) -> Result<()> {
    match action.unwrap_or("status") {
        "status" => {
            println!("Aurora AI Hub");
            println!("- Ollama installed: {}", command_exists("ollama"));
            println!("- Ollama service active: {}", service_is_active("ollama"));
            println!("- Config present: {}", Path::new("/etc/default/aurora-performance").exists());
            println!("- Suggested: ollama pull llama3");
            Ok(())
        }
        "setup" => run("/usr/local/bin/aurora-ai-setup", std::iter::empty::<&str>()),
        other => bail!("unknown ai-hub action: {other}"),
    }
}

fn dev_lab_app(action: Option<&str>) -> Result<()> {
    let action = action.unwrap_or("status");
    match action {
        "status" => {
            println!("Aurora Dev Lab");
            println!("- Containers: podman + distrobox + libvirt stack");
            println!("- Shell quality: shellcheck + hyperfine + ripgrep + fd + bat");
            println!("- Workspace bootstrap: aurora-dev-setup");
            Ok(())
        }
        "setup" => run("/usr/local/bin/aurora-dev-setup", std::iter::empty::<&str>()),
        other => bail!("unknown dev-lab action: {other}"),
    }
}

fn creator_studio_app(action: Option<&str>) -> Result<()> {
    let action = action.unwrap_or("status");
    match action {
        "status" => {
            println!("Aurora Creator Studio");
            println!("- Capture: OBS Studio + FFmpeg");
            println!("- Design: GIMP + Inkscape");
            println!("- Office: LibreOffice");
            println!("- Workspace bootstrap: aurora-creator-setup");
            Ok(())
        }
        "setup" => run(
            "/usr/local/bin/aurora-creator-setup",
            std::iter::empty::<&str>(),
        ),
        other => bail!("unknown creator-studio action: {other}"),
    }
}

fn gaming_center_app(action: Option<&str>) -> Result<()> {
    match action.unwrap_or("status") {
        "status" => {
            println!("Aurora Gaming Center");
            println!("- GameMode installed: {}", command_exists("gamemoderun"));
            println!("- Gamescope installed: {}", command_exists("gamescope"));
            println!("- MangoHud installed: {}", command_exists("mangohud"));
            println!("- DirectStream config: {}", Path::new("/etc/aurora/directstream.conf").exists());
            Ok(())
        }
        "setup" => run("/usr/local/bin/aurora-gaming-setup", std::iter::empty::<&str>()),
        "directstream" => run("/usr/local/bin/aurora-directstream", ["--prime"]),
        other => bail!("unknown gaming-center action: {other}"),
    }
}

fn security_center_app(action: Option<&str>) -> Result<()> {
    match action.unwrap_or("status") {
        "status" => {
            println!("Aurora Security Center");
            println!("- Firewall active: {}", service_or_command_state("ufw", "ufw status"));
            println!("- fail2ban active: {}", service_is_active("fail2ban"));
            println!("- apparmor active: {}", service_is_active("apparmor"));
            println!("- kdump enabled: {}", service_is_enabled("kdump-tools"));
            Ok(())
        }
        "setup" => run("/usr/local/bin/aurora-security-setup", std::iter::empty::<&str>()),
        other => bail!("unknown security-center action: {other}"),
    }
}

fn package_center_app(action: Option<&str>) -> Result<()> {
    match action.unwrap_or("status") {
        "status" => {
            println!("Aurora App Center");
            println!("- apt present: {}", command_exists("apt"));
            println!("- snap present: {}", command_exists("snap"));
            println!("- flatpak present: {}", command_exists("flatpak"));
            println!("- aurora-apt present: {}", Path::new("/usr/local/bin/aurora-apt").exists());
            println!("- aurora repo metadata: {}", Path::new("/usr/share/aurora/archive/catalog.json").exists());
            Ok(())
        }
        "classify" => {
            let package = prompt_line("Package name: ")?;
            print_package_origin(package.trim());
            Ok(())
        }
        other => bail!("unknown package-center action: {other}"),
    }
}

fn andromeda_app(action: Option<&str>) -> Result<()> {
    match action.unwrap_or("summary") {
        "summary" => {
            println!("Aurora Andromeda");
            println!("- Cinematic interstitial surface for first login.");
            println!("- Redirects into Aurora Welcome after the short boot-intro sequence.");
            Ok(())
        }
        other => bail!("unknown andromeda action: {other}"),
    }
}

fn welcome_app(action: Option<&str>) -> Result<()> {
    match action.unwrap_or("status") {
        "status" => {
            println!("Aurora Welcome");
            println!("- aurora runtime: {}", command_summary("aurora", ["version"]));
            println!("- Try:");
            println!("  aurora-distro app control-center");
            println!("  aurora-distro app ai-hub --action status");
            println!("  aurora-distro app dev-lab --action status");
            println!("  aurora-distro app creator-studio --action status");
            println!("  aurora-distro app gaming-center --action status");
            println!("  aurora-distro app security-center --action status");
            println!("  aurora-distro app package-center --action classify");
            Ok(())
        }
        other => bail!("unknown welcome action: {other}"),
    }
}

fn print_package_origin(package: &str) {
    let deb = command_status("dpkg", ["-s", package]);
    let snap = command_output("snap", ["list", package]).is_ok();
    let flatpak = command_output("flatpak", ["info", package]).is_ok();
    println!("Package: {package}");
    println!("- deb: {deb}");
    println!("- snap: {snap}");
    println!("- flatpak: {flatpak}");
}

fn command_status<I, S>(program: &str, args: I) -> bool
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    Command::new(program).args(args).status().map(|s| s.success()).unwrap_or(false)
}

fn command_output<I, S>(program: &str, args: I) -> Result<String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let output = Command::new(program)
        .args(args)
        .output()
        .with_context(|| format!("failed to launch {program}"))?;
    if !output.status.success() {
        bail!("{program} failed");
    }
    Ok(String::from_utf8_lossy(&output.stdout).trim().to_string())
}

fn command_summary<I, S>(program: &str, args: I) -> String
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    command_output(program, args).unwrap_or_else(|_| "unavailable".to_string())
}

fn service_is_active(name: &str) -> bool {
    command_status("systemctl", ["is-active", "--quiet", name])
}

fn service_is_enabled(name: &str) -> bool {
    command_status("systemctl", ["is-enabled", "--quiet", name])
}

fn service_or_command_state(service: &str, command: &str) -> bool {
    service_is_active(service) || run_shell(command).is_ok()
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
        "zram-tools",
        "linux-tools-generic",
        "numactl",
        "util-linux",
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
        "mkswap",
        "swapon",
        "systemctl",
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
    let performance = load_performance_profile(tree)?;
    let system_scan = scan_system()?;
    let partition_plan = plan_partitions(system_scan.clone(), system_mode, 256);
    let build_root = tree.join("build/rootfs");
    let iso_root = tree.join("build/iso");
    let live_root = iso_root.join("live");

    let reusable_rootfs =
        build_root.join("var/lib/dpkg/status").exists() && build_root.join("bin/bash").exists();
    if build_root.exists() && !reusable_rootfs {
        fs::remove_dir_all(&build_root)
            .with_context(|| format!("failed to reset {}", build_root.display()))?;
    }
    if iso_root.exists() {
        fs::remove_dir_all(&iso_root)
            .with_context(|| format!("failed to reset {}", iso_root.display()))?;
    }
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
    write_installer_support(tree, &partition_plan)?;

    if reusable_rootfs {
        println!("Reusing existing rootfs at {}", build_root.display());
    } else {
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
    }

    let mounts = [
        ("/dev", build_root.join("dev")),
        ("/proc", build_root.join("proc")),
        ("/sys", build_root.join("sys")),
    ];
    for (src, dst) in mounts.iter() {
        fs::create_dir_all(dst)?;
        run("mount", ["--bind", src, path_str(dst)?])?;
    }

    let (required_packages, optional_packages) =
        split_install_packages(&config.package_sets, &config.extra_packages);
    let required_install_set = [
        vec![
            "casper".to_string(),
            "grub-pc-bin".to_string(),
            "grub-efi-amd64-bin".to_string(),
            "linux-generic".to_string(),
        ],
        required_packages,
    ]
    .concat();
    let package_script = format!(
        r#"set -euo pipefail
exec > >(tee -a /var/log/aurora-package-install.log) 2>&1
export DEBIAN_FRONTEND=noninteractive
printf 'deb http://archive.ubuntu.com/ubuntu noble main universe multiverse restricted\n' > /etc/apt/sources.list
printf 'deb http://archive.ubuntu.com/ubuntu noble-updates main universe multiverse restricted\n' >> /etc/apt/sources.list
printf 'deb http://archive.ubuntu.com/ubuntu noble-security main universe multiverse restricted\n' >> /etc/apt/sources.list
cat > /etc/apt/apt.conf.d/99aurora-remaster <<'EOF'
Acquire::Retries "10";
Acquire::http::Timeout "60";
Acquire::https::Timeout "60";
Acquire::ForceIPv4 "true";
Acquire::Languages "none";
Dpkg::Use-Pty "0";
APT::Install-Recommends "true";
APT::Install-Suggests "false";
EOF
retry_apt() {{
  local attempt=1
  local max_attempts=4
  until "$@"; do
    if [ "$attempt" -ge "$max_attempts" ]; then
      echo "error: command failed after ${{max_attempts}} attempts: $*" >&2
      return 1
    fi
    echo "warning: attempt ${{attempt}} failed for: $*" >&2
    apt-get clean || true
    rm -f /var/cache/apt/archives/lock /var/lib/dpkg/lock /var/lib/dpkg/lock-frontend || true
    dpkg --configure -a || true
    sleep $((attempt * 10))
    attempt=$((attempt + 1))
  done
}}
retry_apt apt-get update
for pkg in {required}; do
  echo "Installing required package: $pkg"
  retry_apt apt-get install -y "$pkg"
done
for pkg in {optional}; do
  if apt-cache show "$pkg" >/dev/null 2>&1; then
    retry_apt apt-get install -y "$pkg" || echo "warning: optional package failed: $pkg"
  else
    echo "warning: optional package unavailable: $pkg"
  fi
done
dpkg --configure -a
apt-get clean"#,
        required = shell_words(&required_install_set),
        optional = shell_words(&optional_packages),
    );
    run("chroot", [path_str(&build_root)?, "/bin/bash", "-lc", package_script.as_str()])?;

    apply_overlay(tree, &build_root)?;
    install_branding(tree, &build_root, &config)?;
    stage_runtime(&build_root)?;
    build_aurora_repo(tree, &build_root, &config)?;
    finalize_rootfs(&build_root)?;

    // The live root must not contain mounted proc/sys/dev when we snapshot it.
    unmount_paths(&mounts);

    fs::create_dir_all(live_root.as_path())?;
    generate_manifest(&build_root, &live_root)?;
    copy_kernel_artifacts(&build_root, &live_root)?;
    build_squashfs(&build_root, &live_root)?;
    write_grub_cfg(&iso_root, &config, &performance)?;

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

fn unmount_paths(mounts: &[(&str, PathBuf)]) {
    for (_, dst) in mounts.iter().rev() {
        if let Ok(path) = path_str(dst) {
            let _ = run("umount", [path]);
        }
    }
}

fn plan_partitions(scan: SystemScan, mode: BootMode, disk_gb: u64) -> PartitionPlan {
    let firmware = match mode {
        BootMode::Auto => scan.firmware,
        other => other,
    };
    let mut partitions = Vec::new();
    let partition_table = if matches!(firmware, BootMode::Legacy) && disk_gb < 2048 {
        "msdos".to_string()
    } else {
        "gpt".to_string()
    };
    let mut notes = vec!["Use GPT when targeting UEFI or dual-boot firmware support.".to_string()];
    if matches!(firmware, BootMode::Uefi | BootMode::Both) {
        partitions.push(Partition {
            label: "EFI".to_string(),
            fs: "fat32".to_string(),
            size_mb: 512,
            mountpoint: "/boot/efi".to_string(),
            flags: vec!["esp".to_string(), "boot".to_string()],
            mount_options: "umask=0077".to_string(),
        });
    }
    if matches!(firmware, BootMode::Legacy | BootMode::Both) {
        partitions.push(Partition {
            label: "BIOS_GRUB".to_string(),
            fs: "bios_grub".to_string(),
            size_mb: 8,
            mountpoint: "bios_grub".to_string(),
            flags: vec!["bios_grub".to_string()],
            mount_options: "none".to_string(),
        });
    }

    let disk_mb = disk_gb.saturating_mul(1024);
    let swap_mb = recommended_swap_mb(scan.ram_mb, disk_gb);
    let mut reserved_mb = swap_mb;
    if matches!(firmware, BootMode::Uefi | BootMode::Both) {
        reserved_mb = reserved_mb.saturating_add(512);
    }
    if matches!(firmware, BootMode::Legacy | BootMode::Both) {
        reserved_mb = reserved_mb.saturating_add(8);
    }
    let mut available_mb = disk_mb.saturating_sub(reserved_mb);
    let root_mb = if disk_gb >= 192 {
        96 * 1024
    } else if disk_gb >= 96 {
        64 * 1024
    } else {
        available_mb.saturating_sub(16 * 1024).max(28 * 1024)
    }
    .min(available_mb);
    available_mb = available_mb.saturating_sub(root_mb);

    partitions.push(Partition {
        label: "ROOT".to_string(),
        fs: "ext4".to_string(),
        size_mb: root_mb,
        mountpoint: "/".to_string(),
        flags: Vec::new(),
        mount_options: "defaults,noatime".to_string(),
    });
    if available_mb >= 24 * 1024 {
        partitions.push(Partition {
            label: "HOME".to_string(),
            fs: "ext4".to_string(),
            size_mb: available_mb,
            mountpoint: "/home".to_string(),
            flags: Vec::new(),
            mount_options: "defaults,noatime".to_string(),
        });
    }
    partitions.push(Partition {
        label: "SWAP".to_string(),
        fs: "swap".to_string(),
        size_mb: swap_mb,
        mountpoint: "swap".to_string(),
        flags: Vec::new(),
        mount_options: "sw".to_string(),
    });

    PartitionPlan {
        firmware,
        partition_table,
        partitions,
        notes: {
            notes.push("Create swap after root so installs can succeed on smaller disks.".to_string());
            notes.push(format!(
                "Hybrid memory mode: create {} MB disk swap and pair it with zram for RAM-like overflow.",
                swap_mb
            ));
            notes.push("Enable zram on first boot so memory compression absorbs bursts before disk swap is touched.".to_string());
            notes.push("If disk size allows, create a separate /home partition so reinstalls and recovery are safer.".to_string());
            notes.push(format!("Detected CPU: {} | RAM: {} MB", scan.cpu_model, scan.ram_mb));
            notes
        },
        bootloader_steps: bootloader_steps(firmware),
        recovery_steps: recovery_steps(firmware),
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
    let mut mounted = Vec::new();
    let result = (|| -> Result<()> {
        for (src, dst) in [
            ("/dev", format!("{root_str}/dev")),
            ("/proc", format!("{root_str}/proc")),
            ("/sys", format!("{root_str}/sys")),
        ] {
            fs::create_dir_all(&dst)?;
            run("mount", ["--bind", src, &dst])?;
            mounted.push(dst);
        }

        if let Some(efi_dir) = efi {
            let efi_mount = format!("{root_str}/boot/efi");
            fs::create_dir_all(&efi_mount)?;
            run("mount", [path_str(efi_dir)?, &efi_mount])?;
            mounted.push(efi_mount.clone());
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
            run(
                "chroot",
                [root_str, "grub-install", "--target=i386-pc", disk, "--recheck"],
            )?;
            run(
                "chroot",
                [
                    root_str,
                    "/bin/sh",
                    "-lc",
                    "if [ -f /boot/efi/EFI/AURORA/grubx64.efi ]; then mkdir -p /boot/efi/EFI/BOOT && cp /boot/efi/EFI/AURORA/grubx64.efi /boot/efi/EFI/BOOT/BOOTX64.EFI; fi",
                ],
            )?;
        } else {
            run(
                "chroot",
                [root_str, "grub-install", "--target=i386-pc", disk, "--recheck"],
            )?;
        }

        run(
            "chroot",
            [
                root_str,
                "/bin/sh",
                "-lc",
                "command -v update-initramfs >/dev/null 2>&1 && update-initramfs -u || true; command -v update-grub >/dev/null 2>&1 && update-grub || grub-mkconfig -o /boot/grub/grub.cfg",
            ],
        )?;
        Ok(())
    })();

    for mount in mounted.iter().rev() {
        let _ = run("umount", [mount.as_str()]);
    }

    result?;
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
    let _ = fs::copy(
        tree.join(&config.logo_path),
        theme_target.join("logo.svg"),
    );

    let issue = format!(
        "{}\nAdaptive performance distro with BIOS/UEFI install support.\n{}\n",
        config.distro_name, config.performance_goal
    );
    fs::write(rootfs.join("etc/issue"), issue)?;
    fs::write(
        rootfs.join("etc/motd"),
        format!(
            "{}\nTheme: {}\nAdaptive shell enabled.\n",
            config.branding_name, config.theme_name
        ),
    )?;
    fs::write(
        rootfs.join("etc/os-release"),
        os_release_content(config),
    )?;
    fs::write(
        rootfs.join("etc/lsb-release"),
        lsb_release_content(config),
    )?;
    Ok(())
}

fn build_aurora_repo(tree: &Path, rootfs: &Path, config: &BuildConfig) -> Result<()> {
    let repo_root = rootfs.join("opt/aurora/repo");
    let pool_dir = repo_root.join("pool/main");
    let binary_dir = repo_root.join("dists/stable/main/binary-amd64");
    let build_dir = tree.join("build/aurora-repo-build");
    fs::create_dir_all(&pool_dir)?;
    fs::create_dir_all(&binary_dir)?;
    if build_dir.exists() {
        fs::remove_dir_all(&build_dir)?;
    }
    fs::create_dir_all(&build_dir)?;

    let specs = aurora_package_specs(config);
    let mut package_paths = Vec::new();
    for spec in specs {
        let package_root = build_dir.join(&spec.name);
        let debian_dir = package_root.join("DEBIAN");
        fs::create_dir_all(&debian_dir)?;
        fs::write(
            debian_dir.join("control"),
            aurora_binary_package_control(&spec, config),
        )?;
        if let Some(postinst) = &spec.postinst {
            let postinst_path = debian_dir.join("postinst");
            fs::write(&postinst_path, postinst)?;
            #[cfg(target_family = "unix")]
            {
                let mut perms = fs::metadata(&postinst_path)?.permissions();
                perms.set_mode(0o755);
                fs::set_permissions(&postinst_path, perms)?;
            }
        }

        for rel_path in &spec.payload_paths {
            copy_path_into_package(rootfs, &package_root, rel_path)?;
        }

        let doc_dir = package_root.join(format!("usr/share/doc/{}", spec.name));
        fs::create_dir_all(&doc_dir)?;
        fs::write(
            doc_dir.join("README.Aurora"),
            format!(
                "{}\nPackage: {}\nBuilt as part of the Aurora local archive layer.\n",
                spec.description, spec.name
            ),
        )?;

        let deb_path = pool_dir.join(format!("{}_{}_{}.deb", spec.name, spec.version, spec.architecture));
        run(
            "dpkg-deb",
            ["--build", path_str(&package_root)?, path_str(&deb_path)?],
        )?;
        package_paths.push(deb_path);
    }

    fs::write(binary_dir.join("Packages"), aurora_repo_packages_file(&package_paths)?)?;
    fs::write(repo_root.join("dists/stable/Release"), aurora_repo_release(config))?;
    Ok(())
}

fn stage_runtime(rootfs: &Path) -> Result<()> {
    run("cargo", ["build", "--release", "-p", "aurora-cli"])?;

    let built_binary = Path::new("target/release/aurora");
    if !built_binary.is_file() {
        bail!("expected built runtime at {}", built_binary.display());
    }

    let runtime_dir = rootfs.join("usr/local/bin");
    fs::create_dir_all(&runtime_dir)?;
    let runtime_path = runtime_dir.join("aurora");
    fs::copy(built_binary, &runtime_path)
        .with_context(|| format!("failed to stage runtime to {}", runtime_path.display()))?;

    #[cfg(target_family = "unix")]
    {
        let mut perms = fs::metadata(&runtime_path)?.permissions();
        perms.set_mode(0o755);
        fs::set_permissions(&runtime_path, perms)?;
    }

    let systemd_dir = rootfs.join("etc/systemd/system");
    fs::create_dir_all(&systemd_dir)?;
    fs::write(
        systemd_dir.join("aurora-runtime.service"),
        aurora_runtime_service(),
    )?;
    fs::write(
        systemd_dir.join("aurora-inference.service"),
        aurora_inference_service(),
    )?;

    Ok(())
}

fn copy_path_into_package(rootfs: &Path, package_root: &Path, rel_path: &str) -> Result<()> {
    let trimmed = rel_path.trim_start_matches('/');
    let src = rootfs.join(trimmed);
    if !src.exists() {
        return Ok(());
    }
    let dst = package_root.join(trimmed);
    if src.is_dir() {
        fs::create_dir_all(&dst)?;
        for entry in fs::read_dir(&src)? {
            let entry = entry?;
            let child_rel = if trimmed.is_empty() {
                entry.file_name().to_string_lossy().to_string()
            } else {
                format!("{trimmed}/{}", entry.file_name().to_string_lossy())
            };
            copy_path_into_package(rootfs, package_root, &child_rel)?;
        }
    } else {
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&src, &dst).with_context(|| format!("failed to copy {} into package", src.display()))?;
        #[cfg(target_family = "unix")]
        {
            let perms = fs::metadata(&src)?.permissions();
            fs::set_permissions(&dst, perms)?;
        }
    }
    Ok(())
}

fn write_installer_support(tree: &Path, partition_plan: &PartitionPlan) -> Result<()> {
    let build_dir = tree.join("build");
    fs::create_dir_all(&build_dir)?;
    fs::write(
        build_dir.join("partition-apply.sh"),
        partition_apply_script(partition_plan),
    )?;
    fs::write(build_dir.join("repair-boot.sh"), repair_boot_script())?;
    Ok(())
}

fn finalize_rootfs(rootfs: &Path) -> Result<()> {
    let command = "command -v dconf >/dev/null 2>&1 && dconf update || true; \
                   command -v apt-get >/dev/null 2>&1 && apt-get update >/dev/null 2>&1 || true; \
                   command -v apt-get >/dev/null 2>&1 && DEBIAN_FRONTEND=noninteractive apt-get install -y aurora-branding aurora-runtime-tools aurora-installer aurora-control-center aurora-app-center aurora-ai-hub aurora-dev-lab aurora-creator-studio aurora-gaming-center aurora-security-center aurora-welcome aurora-desktop aurora-dev-stack aurora-creator-stack aurora-gaming-stack aurora-security-stack aurora-scientific-stack aurora-training-stack aurora-datacenter-stack aurora-privacy-stack >/dev/null 2>&1 || true; \
                   command -v flatpak >/dev/null 2>&1 && flatpak remote-add --if-not-exists flathub https://dl.flathub.org/repo/flathub.flatpakrepo || true; \
                   command -v systemctl >/dev/null 2>&1 && systemctl enable aurora-autosetup.service aurora-zram-setup.service aurora-runtime.service aurora-inference.service >/dev/null 2>&1 || true; \
                   command -v systemctl >/dev/null 2>&1 && systemctl enable ollama >/dev/null 2>&1 || true; \
                   command -v systemctl >/dev/null 2>&1 && systemctl enable valkey-server >/dev/null 2>&1 || true; \
                   command -v systemctl >/dev/null 2>&1 && systemctl enable kdump-tools >/dev/null 2>&1 || true; \
                   command -v plymouth-set-default-theme >/dev/null 2>&1 && plymouth-set-default-theme aurora-neon >/dev/null 2>&1 || true; \
                   [ -d /usr/share/plymouth/themes/aurora-neon ] && ln -sf /usr/share/plymouth/themes/aurora-neon/aurora-neon.plymouth /etc/alternatives/default.plymouth || true; \
                   command -v update-initramfs >/dev/null 2>&1 && update-initramfs -u >/dev/null 2>&1 || true; \
                   command -v update-alternatives >/dev/null 2>&1 && update-alternatives --set x-session-manager /usr/bin/gnome-shell >/dev/null 2>&1 || true";
    run("chroot", [path_str(rootfs)?, "/bin/sh", "-lc", command])
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

fn write_grub_cfg(iso_root: &Path, config: &BuildConfig, performance: &PerformanceProfile) -> Result<()> {
    let grub_dir = iso_root.join("boot/grub");
    fs::create_dir_all(&grub_dir)?;
    let cmdline = if performance.kernel_cmdline_additions.is_empty() {
        "boot=casper".to_string()
    } else {
        format!("boot=casper {}", performance.kernel_cmdline_additions.join(" "))
    };
    let cfg = format!(
        "set default=0\nset timeout=3\nmenuentry \"{} Live\" {{\n linux /live/vmlinuz {} ---\n initrd /live/initrd\n}}\n",
        config.branding_name,
        cmdline
    );
    fs::write(grub_dir.join("grub.cfg"), cfg)?;
    Ok(())
}

fn performance_profile(mode: InstallMode) -> PerformanceProfile {
    match mode {
        InstallMode::Balanced => PerformanceProfile {
            mode,
            name: "Aurora Balanced".to_string(),
            cpu_governor: "schedutil".to_string(),
            enable_hugepages: true,
            enable_gamemode: false,
            enable_mangohud: false,
            tune_sysctl: true,
            enable_zram: true,
            zram_fraction_percent: 35,
            use_tmpfs_for_temp: false,
            disable_unneeded_services: false,
            enable_preload: true,
            io_scheduler: "mq-deadline".to_string(),
            swap_partition_policy: "moderate-zram-disk".to_string(),
            readahead_kb: 1024,
            vm_swappiness: 25,
            vm_dirty_ratio: 10,
            vm_dirty_background_ratio: 5,
            disable_cpu_idle: false,
            scheduler_tune: "balanced".to_string(),
            cpu_energy_policy: "balance_performance".to_string(),
            kernel_cmdline_additions: vec!["quiet".to_string(), "splash".to_string()],
        },
        InstallMode::Gaming => PerformanceProfile {
            mode,
            name: "Aurora Gaming".to_string(),
            cpu_governor: "performance".to_string(),
            enable_hugepages: true,
            enable_gamemode: true,
            enable_mangohud: true,
            tune_sysctl: true,
            enable_zram: true,
            zram_fraction_percent: 45,
            use_tmpfs_for_temp: true,
            disable_unneeded_services: true,
            enable_preload: true,
            io_scheduler: "none".to_string(),
            swap_partition_policy: "latency-first-zram-disk".to_string(),
            readahead_kb: 2048,
            vm_swappiness: 18,
            vm_dirty_ratio: 6,
            vm_dirty_background_ratio: 3,
            disable_cpu_idle: false,
            scheduler_tune: "desktop-low-latency".to_string(),
            cpu_energy_policy: "performance".to_string(),
            kernel_cmdline_additions: vec![
                "quiet".to_string(),
                "splash".to_string(),
                "transparent_hugepage=always".to_string(),
                "noatime".to_string(),
            ],
        },
        InstallMode::Creator => PerformanceProfile {
            mode,
            name: "Aurora Creator Studio".to_string(),
            cpu_governor: "performance".to_string(),
            enable_hugepages: true,
            enable_gamemode: false,
            enable_mangohud: false,
            tune_sysctl: true,
            enable_zram: true,
            zram_fraction_percent: 50,
            use_tmpfs_for_temp: true,
            disable_unneeded_services: false,
            enable_preload: true,
            io_scheduler: "mq-deadline".to_string(),
            swap_partition_policy: "cache-heavy-zram-disk".to_string(),
            readahead_kb: 3072,
            vm_swappiness: 14,
            vm_dirty_ratio: 12,
            vm_dirty_background_ratio: 4,
            disable_cpu_idle: false,
            scheduler_tune: "creator-cache-biased".to_string(),
            cpu_energy_policy: "performance".to_string(),
            kernel_cmdline_additions: vec![
                "quiet".to_string(),
                "splash".to_string(),
                "transparent_hugepage=madvise".to_string(),
            ],
        },
        InstallMode::Training => PerformanceProfile {
            mode,
            name: "Aurora Training".to_string(),
            cpu_governor: "performance".to_string(),
            enable_hugepages: true,
            enable_gamemode: false,
            enable_mangohud: false,
            tune_sysctl: true,
            enable_zram: true,
            zram_fraction_percent: 65,
            use_tmpfs_for_temp: true,
            disable_unneeded_services: true,
            enable_preload: false,
            io_scheduler: "none".to_string(),
            swap_partition_policy: "dataset-staging-zram-disk".to_string(),
            readahead_kb: 4096,
            vm_swappiness: 10,
            vm_dirty_ratio: 18,
            vm_dirty_background_ratio: 6,
            disable_cpu_idle: true,
            scheduler_tune: "training-throughput".to_string(),
            cpu_energy_policy: "performance".to_string(),
            kernel_cmdline_additions: vec![
                "quiet".to_string(),
                "transparent_hugepage=always".to_string(),
                "numa_balancing=disable".to_string(),
            ],
        },
        InstallMode::Datacenter => PerformanceProfile {
            mode,
            name: "Aurora Datacenter".to_string(),
            cpu_governor: "performance".to_string(),
            enable_hugepages: true,
            enable_gamemode: false,
            enable_mangohud: false,
            tune_sysctl: true,
            enable_zram: true,
            zram_fraction_percent: 30,
            use_tmpfs_for_temp: false,
            disable_unneeded_services: true,
            enable_preload: false,
            io_scheduler: "mq-deadline".to_string(),
            swap_partition_policy: "predictable-disk-backed".to_string(),
            readahead_kb: 1024,
            vm_swappiness: 8,
            vm_dirty_ratio: 8,
            vm_dirty_background_ratio: 3,
            disable_cpu_idle: false,
            scheduler_tune: "service-isolated".to_string(),
            cpu_energy_policy: "balance_performance".to_string(),
            kernel_cmdline_additions: vec![
                "quiet".to_string(),
                "systemd.unified_cgroup_hierarchy=1".to_string(),
            ],
        },
        InstallMode::MaxThroughput => PerformanceProfile {
            mode,
            name: "Aurora Maximum Throughput".to_string(),
            cpu_governor: "performance".to_string(),
            enable_hugepages: true,
            enable_gamemode: true,
            enable_mangohud: true,
            tune_sysctl: true,
            enable_zram: true,
            zram_fraction_percent: 60,
            use_tmpfs_for_temp: true,
            disable_unneeded_services: true,
            enable_preload: true,
            io_scheduler: "none".to_string(),
            swap_partition_policy: "hybrid-zram-disk".to_string(),
            readahead_kb: 4096,
            vm_swappiness: 15,
            vm_dirty_ratio: 5,
            vm_dirty_background_ratio: 2,
            disable_cpu_idle: true,
            scheduler_tune: "throughput-aggressive".to_string(),
            cpu_energy_policy: "performance".to_string(),
            kernel_cmdline_additions: vec![
                "mitigations=off".to_string(),
                "transparent_hugepage=always".to_string(),
                "nowatchdog".to_string(),
                "quiet".to_string(),
                "splash".to_string(),
                "noatime".to_string(),
            ],
        },
    }
}

fn split_install_packages(
    package_sets: &[String],
    extra_packages: &[String],
) -> (Vec<String>, Vec<String>) {
    let mut required = Vec::new();
    let mut optional = Vec::new();

    for package in package_sets.iter().chain(extra_packages.iter()) {
        if is_optional_package(package) {
            optional.push(package.clone());
        } else {
            required.push(package.clone());
        }
    }

    (required, optional)
}

fn is_optional_package(package: &str) -> bool {
    matches!(
        package,
        "gamemode"
            | "mangohud"
            | "nvtop"
            | "btop"
            | "zram-tools"
            | "numactl"
            | "linux-tools-generic"
            | "gnome-shell-extension-manager"
            | "steam-installer"
            | "gnome-tweaks"
            | "gnome-shell-extensions"
            | "gnome-software-plugin-flatpak"
            | "flatpak"
            | "papirus-icon-theme"
            | "fonts-cascadia-code"
            | "alacritty"
            | "vlc"
            | "pipewire-audio"
            | "pavucontrol"
            | "fastfetch"
            | "eza"
            | "earlyoom"
            | "irqbalance"
            | "thermald"
            | "ufw"
            | "fail2ban"
            | "apparmor-utils"
            | "fprintd"
            | "ollama"
            | "copyq"
            | "podman"
            | "distrobox"
            | "virt-manager"
            | "qemu-system-x86"
            | "libvirt-daemon-system"
            | "bridge-utils"
            | "dnsmasq-base"
            | "gamescope"
            | "goverlay"
            | "mesa-vulkan-drivers"
            | "mesa-utils"
            | "vulkan-tools"
            | "lutris"
            | "wine64"
            | "winetricks"
            | "pciutils"
            | "inxi"
            | "clinfo"
            | "obs-studio"
            | "gimp"
            | "inkscape"
            | "ffmpeg"
            | "libreoffice"
            | "kdenlive"
            | "blender"
            | "krita"
            | "audacity"
            | "git-lfs"
            | "tmux"
            | "direnv"
            | "shellcheck"
            | "hyperfine"
            | "wireshark"
            | "nmap"
            | "tcpdump"
            | "strace"
            | "bpfcc-tools"
            | "gh"
            | "libspa-0.2-bluetooth"
            | "valkey-server"
            | "kdump-tools"
            | "gvfs-backends"
            | "gvfs-fuse"
            | "sshfs"
            | "nautilus-extension-gnome-terminal"
            | "dconf-editor"
            | "authd"
            | "ripgrep"
            | "fd-find"
            | "bat"
            | "helix"
            | "zellij"
            | "bottom"
            | "just"
            | "neofetch"
            | "micro"
            | "ranger"
            | "tldr"
            | "duf"
            | "ncdu"
            | "glances"
            | "silversearcher-ag"
            | "fzf"
            | "jq"
            | "python3-venv"
            | "python3-pip"
            | "python3-dev"
            | "sysstat"
            | "dstat"
            | "iotop"
            | "fio"
            | "nvme-cli"
            | "prometheus-node-exporter"
            | "tor"
            | "torsocks"
            | "macchanger"
            | "usbguard"
    )
}

fn shell_words(items: &[String]) -> String {
    items.iter()
        .map(|item| format!("'{}'", item.replace('\'', "'\\''")))
        .collect::<Vec<_>>()
        .join(" ")
}

fn aurora_package_specs(config: &BuildConfig) -> Vec<AuroraPackageSpec> {
    let arch = config.arch.clone();
    let version = "1.0".to_string();
    vec![
        AuroraPackageSpec {
            name: "aurora-branding".to_string(),
            version: version.clone(),
            architecture: "all".to_string(),
            depends: vec![],
            description: format!("{} branding, icons, themes, wallpaper, and boot assets.", config.branding_name),
            payload_paths: vec![
                "usr/share/aurora/logo.svg".to_string(),
                "usr/share/backgrounds/aurora".to_string(),
                "usr/share/themes/Aurora-Neon".to_string(),
                "usr/share/icons/Aurora-Neon".to_string(),
                "usr/share/plymouth/themes/aurora-neon".to_string(),
                "usr/share/grub/themes/aurora".to_string(),
            ],
            postinst: None,
        },
        AuroraPackageSpec {
            name: "aurora-runtime-tools".to_string(),
            version: version.clone(),
            architecture: arch.clone(),
            depends: vec!["bash".to_string()],
            description: "Aurora runtime binary, tuning scripts, and adaptive system control tooling.".to_string(),
            payload_paths: vec![
                "usr/local/bin/aurora".to_string(),
                "usr/local/bin/aurora-mode-switch".to_string(),
                "usr/local/bin/aurora-apply-desktop".to_string(),
                "usr/local/bin/aurora-autosetup".to_string(),
                "usr/local/bin/aurora-ai-setup".to_string(),
                "usr/local/bin/aurora-dev-setup".to_string(),
                "usr/local/bin/aurora-creator-setup".to_string(),
                "usr/local/bin/aurora-gaming-setup".to_string(),
                "usr/local/bin/aurora-gpu-setup".to_string(),
                "usr/local/bin/aurora-ai-helper".to_string(),
                "usr/local/bin/aurora-privacy-setup".to_string(),
                "usr/local/bin/aurora-training-setup".to_string(),
                "usr/local/bin/aurora-datacenter-setup".to_string(),
                "usr/local/bin/aurora-security-setup".to_string(),
                "usr/local/bin/aurora-hardware-guard".to_string(),
                "usr/local/bin/aurora-directstream".to_string(),
                "etc/default/aurora-performance".to_string(),
                "etc/aurora/directstream.conf".to_string(),
                "etc/systemd/system/aurora-autosetup.service".to_string(),
                "etc/systemd/system/aurora-inference.service".to_string(),
                "etc/systemd/system/aurora-runtime.service".to_string(),
                "etc/systemd/system/aurora-zram-setup.service".to_string(),
            ],
            postinst: Some(
                "#!/usr/bin/env bash\nset -e\ncommand -v systemctl >/dev/null 2>&1 && systemctl daemon-reload >/dev/null 2>&1 || true\n".to_string(),
            ),
        },
        AuroraPackageSpec {
            name: "aurora-installer".to_string(),
            version: version.clone(),
            architecture: "all".to_string(),
            depends: vec!["aurora-branding".to_string()],
            description: "Aurora installer assets, generated plans, and installation launchers.".to_string(),
            payload_paths: vec![
                "usr/share/aurora/installer".to_string(),
                "usr/share/applications/aurora-installer.desktop".to_string(),
                "etc/aurora-installer".to_string(),
            ],
            postinst: None,
        },
        AuroraPackageSpec {
            name: "aurora-control-center".to_string(),
            version: version.clone(),
            architecture: "all".to_string(),
            depends: vec!["aurora-runtime-tools".to_string()],
            description: "Aurora Control Center package with system mode switching and runtime status entrypoints.".to_string(),
            payload_paths: vec![
                "usr/share/aurora/control-center".to_string(),
                "usr/share/applications/aurora-control-center.desktop".to_string(),
            ],
            postinst: None,
        },
        AuroraPackageSpec {
            name: "aurora-app-center".to_string(),
            version: version.clone(),
            architecture: "all".to_string(),
            depends: vec!["apt".to_string()],
            description: "Aurora App Center with Aurora archive metadata, labels, and curated package presentation.".to_string(),
            payload_paths: vec![
                "usr/share/aurora/package-center".to_string(),
                "usr/share/aurora/archive".to_string(),
                "usr/share/applications/aurora-package-center.desktop".to_string(),
                "etc/apt/sources.list.d/aurora.sources".to_string(),
                "opt/aurora/repo/meta".to_string(),
            ],
            postinst: None,
        },
        AuroraPackageSpec {
            name: "aurora-ai-hub".to_string(),
            version: version.clone(),
            architecture: "all".to_string(),
            depends: vec!["aurora-runtime-tools".to_string()],
            description: "Aurora AI Hub with local-model workflow, Ollama setup, and offline AI integration scaffolding.".to_string(),
            payload_paths: vec![
                "usr/share/aurora/ai-hub".to_string(),
                "usr/share/applications/aurora-ai-hub.desktop".to_string(),
                "usr/local/bin/aurora-ai-setup".to_string(),
            ],
            postinst: None,
        },
        AuroraPackageSpec {
            name: "aurora-dev-lab".to_string(),
            version: version.clone(),
            architecture: "all".to_string(),
            depends: vec!["aurora-runtime-tools".to_string()],
            description: "Aurora Dev Lab with containers, benchmarking, shell tooling, and development workspace setup.".to_string(),
            payload_paths: vec![
                "usr/share/aurora/dev-lab".to_string(),
                "usr/share/applications/aurora-dev-lab.desktop".to_string(),
                "usr/local/bin/aurora-dev-setup".to_string(),
            ],
            postinst: None,
        },
        AuroraPackageSpec {
            name: "aurora-creator-studio".to_string(),
            version: version.clone(),
            architecture: "all".to_string(),
            depends: vec!["aurora-runtime-tools".to_string()],
            description: "Aurora Creator Studio with capture, media, graphics, and office workflow entrypoints.".to_string(),
            payload_paths: vec![
                "usr/share/aurora/creator-studio".to_string(),
                "usr/share/applications/aurora-creator-studio.desktop".to_string(),
                "usr/local/bin/aurora-creator-setup".to_string(),
            ],
            postinst: None,
        },
        AuroraPackageSpec {
            name: "aurora-gaming-center".to_string(),
            version: version.clone(),
            architecture: "all".to_string(),
            depends: vec!["aurora-runtime-tools".to_string()],
            description: "Aurora Gaming Center with gaming setup, DirectStream, and performance-oriented desktop integration.".to_string(),
            payload_paths: vec![
                "usr/share/aurora/gaming-center".to_string(),
                "usr/share/applications/aurora-gaming-center.desktop".to_string(),
                "usr/local/bin/aurora-gaming-setup".to_string(),
                "usr/local/bin/aurora-directstream".to_string(),
                "etc/aurora/directstream.conf".to_string(),
            ],
            postinst: None,
        },
        AuroraPackageSpec {
            name: "aurora-security-center".to_string(),
            version: version.clone(),
            architecture: "all".to_string(),
            depends: vec!["aurora-runtime-tools".to_string()],
            description: "Aurora Security Center with hardening defaults, firewall workflow, and crash-debug posture.".to_string(),
            payload_paths: vec![
                "usr/share/aurora/security-center".to_string(),
                "usr/share/applications/aurora-security-center.desktop".to_string(),
                "usr/local/bin/aurora-security-setup".to_string(),
            ],
            postinst: None,
        },
        AuroraPackageSpec {
            name: "aurora-welcome".to_string(),
            version: version.clone(),
            architecture: "all".to_string(),
            depends: vec!["aurora-branding".to_string(), "aurora-control-center".to_string()],
            description: "Aurora welcome flow, firstboot launcher, and curated getting-started surface.".to_string(),
            payload_paths: vec![
                "usr/share/aurora/welcome".to_string(),
                "usr/share/aurora/andromeda".to_string(),
                "usr/share/applications/aurora-welcome.desktop".to_string(),
                "usr/share/applications/aurora-andromeda.desktop".to_string(),
                "usr/local/bin/aurora-firstboot".to_string(),
            ],
            postinst: None,
        },
        AuroraPackageSpec {
            name: "aurora-desktop".to_string(),
            version: version.clone(),
            architecture: "all".to_string(),
            depends: vec![
                "aurora-branding".to_string(),
                "aurora-control-center".to_string(),
                "aurora-app-center".to_string(),
                "aurora-installer".to_string(),
                "aurora-welcome".to_string(),
                "aurora-dev-lab".to_string(),
                "aurora-creator-studio".to_string(),
            ],
            description: format!("{} desktop identity, workflows, and integrated Aurora user experience layer.", config.branding_name),
            payload_paths: vec![],
            postinst: None,
        },
        AuroraPackageSpec {
            name: "aurora-ai-stack".to_string(),
            version: version.clone(),
            architecture: "all".to_string(),
            depends: vec!["aurora-ai-hub".to_string(), "ollama".to_string()],
            description: "Aurora AI stack meta-package for local models and AI workflows.".to_string(),
            payload_paths: vec![],
            postinst: None,
        },
        AuroraPackageSpec {
            name: "aurora-gaming-stack".to_string(),
            version: version.clone(),
            architecture: "all".to_string(),
            depends: vec![
                "aurora-gaming-center".to_string(),
                "gamemode".to_string(),
                "mangohud".to_string(),
                "gamescope".to_string(),
            ],
            description: "Aurora gaming stack meta-package for tuned gaming workflows and tooling.".to_string(),
            payload_paths: vec![],
            postinst: None,
        },
        AuroraPackageSpec {
            name: "aurora-creator-stack".to_string(),
            version: version.clone(),
            architecture: "all".to_string(),
            depends: vec![
                "aurora-creator-studio".to_string(),
                "aurora-app-center".to_string(),
                "flatpak".to_string(),
                "vlc".to_string(),
                "obs-studio".to_string(),
                "ffmpeg".to_string(),
                "gimp".to_string(),
                "inkscape".to_string(),
                "libreoffice".to_string(),
            ],
            description: "Aurora creator stack meta-package for workstation and media tooling.".to_string(),
            payload_paths: vec![],
            postinst: None,
        },
        AuroraPackageSpec {
            name: "aurora-dev-stack".to_string(),
            version: version.clone(),
            architecture: "all".to_string(),
            depends: vec![
                "aurora-dev-lab".to_string(),
                "git".to_string(),
                "git-lfs".to_string(),
                "podman".to_string(),
                "distrobox".to_string(),
                "tmux".to_string(),
                "direnv".to_string(),
                "shellcheck".to_string(),
                "hyperfine".to_string(),
            ],
            description: "Aurora development stack meta-package for containers, shell workflows, and benchmarking.".to_string(),
            payload_paths: vec![],
            postinst: None,
        },
        AuroraPackageSpec {
            name: "aurora-security-stack".to_string(),
            version: version.clone(),
            architecture: "all".to_string(),
            depends: vec![
                "aurora-security-center".to_string(),
                "ufw".to_string(),
                "fail2ban".to_string(),
                "apparmor-utils".to_string(),
                "fprintd".to_string(),
                "kdump-tools".to_string(),
                "authd".to_string(),
            ],
            description: "Aurora security stack meta-package for hardening and recovery tooling.".to_string(),
            payload_paths: vec![],
            postinst: None,
        },
        AuroraPackageSpec {
            name: "aurora-scientific-stack".to_string(),
            version,
            architecture: "all".to_string(),
            depends: vec![
                "numactl".to_string(),
                "linux-tools-generic".to_string(),
                "hwloc".to_string(),
                "valkey-server".to_string(),
                "ripgrep".to_string(),
                "fd-find".to_string(),
            ],
            description: "Aurora scientific stack meta-package for profiling, NUMA placement, and high-speed data work.".to_string(),
            payload_paths: vec![],
            postinst: None,
        },
        AuroraPackageSpec {
            name: "aurora-training-stack".to_string(),
            version: "1.0".to_string(),
            architecture: "all".to_string(),
            depends: vec![
                "aurora-ai-hub".to_string(),
                "python3-venv".to_string(),
                "python3-pip".to_string(),
                "numactl".to_string(),
                "hwloc".to_string(),
                "valkey-server".to_string(),
            ],
            description: "Aurora training stack meta-package for local AI and throughput-oriented data workflows.".to_string(),
            payload_paths: vec![],
            postinst: None,
        },
        AuroraPackageSpec {
            name: "aurora-datacenter-stack".to_string(),
            version: "1.0".to_string(),
            architecture: "all".to_string(),
            depends: vec![
                "prometheus-node-exporter".to_string(),
                "sysstat".to_string(),
                "iotop".to_string(),
                "fio".to_string(),
                "nvme-cli".to_string(),
            ],
            description: "Aurora datacenter stack meta-package for service observability, storage validation, and host operations.".to_string(),
            payload_paths: vec![],
            postinst: None,
        },
        AuroraPackageSpec {
            name: "aurora-privacy-stack".to_string(),
            version: "1.0".to_string(),
            architecture: "all".to_string(),
            depends: vec![
                "tor".to_string(),
                "torsocks".to_string(),
                "macchanger".to_string(),
                "usbguard".to_string(),
            ],
            description: "Aurora privacy stack meta-package for local hardening, MAC hygiene, and host privacy defaults.".to_string(),
            payload_paths: vec![],
            postinst: None,
        },
    ]
}

fn aurora_binary_package_control(spec: &AuroraPackageSpec, config: &BuildConfig) -> String {
    let depends = if spec.depends.is_empty() {
        String::new()
    } else {
        format!("Depends: {}\n", spec.depends.join(", "))
    };
    format!(
        "Package: {name}\nVersion: {version}\nSection: aurora\nPriority: optional\nArchitecture: {arch}\nMaintainer: {maintainer}\n{depends}Description: {description}\n",
        name = spec.name,
        version = spec.version,
        arch = spec.architecture,
        maintainer = config.branding_name,
        depends = depends,
        description = spec.description,
    )
}

fn aurora_repo_packages_file(package_paths: &[PathBuf]) -> Result<String> {
    let mut entries = Vec::new();
    for package_path in package_paths {
        let package_name = command_summary("dpkg-deb", ["-f", path_str(package_path)?, "Package"]);
        let version = command_summary("dpkg-deb", ["-f", path_str(package_path)?, "Version"]);
        let arch = command_summary("dpkg-deb", ["-f", path_str(package_path)?, "Architecture"]);
        let depends = command_summary("dpkg-deb", ["-f", path_str(package_path)?, "Depends"]);
        let description = command_summary("dpkg-deb", ["-f", path_str(package_path)?, "Description"]);
        let bytes = fs::read(package_path)?;
        let size = bytes.len();
        let md5 = md5_hex(&bytes);
        let sha256 = sha256_hex(&bytes);
        let filename = format!(
            "pool/main/{}",
            package_path
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or_else(|| anyhow!("invalid package file name"))?
        );
        let depends_line = if depends == "unavailable" || depends.is_empty() {
            String::new()
        } else {
            format!("Depends: {depends}\n")
        };
        entries.push(format!(
            "Package: {package}\nVersion: {version}\nArchitecture: {arch}\n{depends}Filename: {filename}\nSize: {size}\nMD5sum: {md5}\nSHA256: {sha256}\nDescription: {description}\n",
            package = package_name,
            depends = depends_line,
        ));
    }
    Ok(entries.join("\n"))
}

fn md5_hex(bytes: &[u8]) -> String {
    let digest = md5_simple(bytes);
    digest.iter().map(|b| format!("{b:02x}")).collect()
}

fn sha256_hex(bytes: &[u8]) -> String {
    command_summary("sha256sum", std::iter::empty::<&str>());
    let output = Command::new("sha256sum")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .spawn()
        .and_then(|mut child| {
            if let Some(stdin) = child.stdin.as_mut() {
                stdin.write_all(bytes)?;
            }
            child.wait_with_output()
        });
    match output {
        Ok(output) if output.status.success() => String::from_utf8_lossy(&output.stdout)
            .split_whitespace()
            .next()
            .unwrap_or_default()
            .to_string(),
        _ => String::new(),
    }
}

fn md5_simple(input: &[u8]) -> [u8; 16] {
    let mut message = input.to_vec();
    let bit_len = (message.len() as u64) * 8;
    message.push(0x80);
    while (message.len() % 64) != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_le_bytes());

    let mut a0: u32 = 0x67452301;
    let mut b0: u32 = 0xefcdab89;
    let mut c0: u32 = 0x98badcfe;
    let mut d0: u32 = 0x10325476;

    let s: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14,
        20, 5, 9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11,
        16, 23, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    let k: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
        0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
        0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
        0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
        0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
        0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
        0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
        0xeb86d391,
    ];

    for chunk in message.chunks_exact(64) {
        let mut m = [0u32; 16];
        for (i, word) in m.iter_mut().enumerate() {
            let start = i * 4;
            *word = u32::from_le_bytes([
                chunk[start],
                chunk[start + 1],
                chunk[start + 2],
                chunk[start + 3],
            ]);
        }

        let mut a = a0;
        let mut b = b0;
        let mut c = c0;
        let mut d = d0;

        for i in 0..64 {
            let (f, g) = if i < 16 {
                ((b & c) | ((!b) & d), i)
            } else if i < 32 {
                ((d & b) | ((!d) & c), (5 * i + 1) % 16)
            } else if i < 48 {
                (b ^ c ^ d, (3 * i + 5) % 16)
            } else {
                (c ^ (b | (!d)), (7 * i) % 16)
            };
            let temp = d;
            d = c;
            c = b;
            b = b.wrapping_add(
                a.wrapping_add(f)
                    .wrapping_add(k[i])
                    .wrapping_add(m[g])
                    .rotate_left(s[i]),
            );
            a = temp;
        }

        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }

    let mut digest = [0u8; 16];
    digest[0..4].copy_from_slice(&a0.to_le_bytes());
    digest[4..8].copy_from_slice(&b0.to_le_bytes());
    digest[8..12].copy_from_slice(&c0.to_le_bytes());
    digest[12..16].copy_from_slice(&d0.to_le_bytes());
    digest
}

fn aurora_sources_file() -> String {
    "Types: deb\nURIs: file:///opt/aurora/repo\nSuites: stable\nComponents: main\nTrusted: yes\n".to_string()
}

fn aurora_repo_release(config: &BuildConfig) -> String {
    format!(
        "Origin: {name}\nLabel: {name} Archive\nSuite: stable\nCodename: aurora-stable\nArchitectures: amd64\nComponents: main\nDescription: Curated Aurora package archive layered on top of a Debian-compatible base system\n",
        name = config.branding_name
    )
}

fn aurora_repo_packages(config: &BuildConfig) -> String {
    [
        aurora_repo_package_entry(
            "aurora-desktop",
            "1.0",
            "aurora-control-center, aurora-app-center, aurora-welcome",
            &format!("Core {} desktop identity and UX layer.", config.branding_name),
        ),
        aurora_repo_package_entry(
            "aurora-ai-stack",
            "1.0",
            "ollama, aurora-ai-hub",
            "Aurora local AI stack with curated offline model integration.",
        ),
        aurora_repo_package_entry(
            "aurora-dev-stack",
            "1.0",
            "aurora-dev-lab, git, git-lfs, podman, distrobox, tmux, direnv",
            "Aurora development stack with containers, shell automation, and benchmarking.",
        ),
        aurora_repo_package_entry(
            "aurora-gaming-stack",
            "1.0",
            "gamemode, mangohud, gamescope",
            "Aurora gaming stack with tuned performance and session tooling.",
        ),
        aurora_repo_package_entry(
            "aurora-creator-stack",
            "1.0",
            "aurora-creator-studio, obs-studio, ffmpeg, gimp, inkscape, libreoffice",
            "Aurora creator stack with capture, media, design, and workstation tooling.",
        ),
        aurora_repo_package_entry(
            "aurora-security-stack",
            "1.0",
            "aurora-security-center, ufw, fail2ban, apparmor-utils, kdump-tools",
            "Aurora security stack with hardening, access control, and recovery tooling.",
        ),
        aurora_repo_package_entry(
            "aurora-scientific-stack",
            "1.0",
            "numactl, linux-tools-generic, hwloc, valkey-server",
            "Aurora scientific stack with profiling, NUMA, and high-throughput support tools.",
        ),
        aurora_repo_package_entry(
            "aurora-training-stack",
            "1.0",
            "aurora-ai-hub, python3-venv, python3-pip, numactl, hwloc, valkey-server",
            "Aurora training stack with local AI, memory locality, and sustained throughput helpers.",
        ),
        aurora_repo_package_entry(
            "aurora-datacenter-stack",
            "1.0",
            "prometheus-node-exporter, sysstat, iotop, fio, nvme-cli",
            "Aurora datacenter stack with observability and storage-validation tools.",
        ),
        aurora_repo_package_entry(
            "aurora-privacy-stack",
            "1.0",
            "tor, torsocks, macchanger, usbguard",
            "Aurora privacy stack with local network and device-hardening posture.",
        ),
    ]
    .join("\n")
}

fn aurora_repo_package_entry(name: &str, version: &str, depends: &str, description: &str) -> String {
    format!(
        "Package: {name}\nVersion: {version}\nArchitecture: amd64\nMaintainer: Aurora OS\nDepends: {depends}\nDescription: {description}\n"
    )
}

fn aurora_meta_package_control(
    name: &str,
    config: &BuildConfig,
    depends: &[&str],
    description: &str,
) -> String {
    format!(
        "Package: {name}\nVersion: 1.0\nSection: metapackages\nPriority: optional\nArchitecture: all\nMaintainer: {maintainer}\nDepends: {depends}\nDescription: {description}\n",
        maintainer = config.branding_name,
        depends = depends.join(", "),
    )
}

fn aurora_archive_catalog_json(config: &BuildConfig) -> Result<String> {
    serde_json::to_string_pretty(&serde_json::json!({
        "archive": {
            "name": format!("{} Archive", config.branding_name),
            "suite": "stable",
            "component": "main",
            "channel": "aurora-curated",
            "base": "noble-compatible"
        },
        "labels": [
            {"id": "aurora", "name": "Aurora Curated"},
            {"id": "gaming", "name": "Gaming Ready"},
            {"id": "ai", "name": "AI Native"},
            {"id": "creator", "name": "Creator Stack"},
            {"id": "secure", "name": "Security Hardened"},
            {"id": "training", "name": "Training Stack"},
            {"id": "datacenter", "name": "Datacenter Ops"},
            {"id": "privacy", "name": "Privacy Guard"}
        ],
        "featured": [
            {"package": "aurora-desktop", "title": "Aurora Desktop", "origin": "Aurora Archive", "label": "Aurora Curated"},
            {"package": "aurora-ai-stack", "title": "Aurora AI Stack", "origin": "Aurora Archive", "label": "AI Native"},
            {"package": "aurora-gaming-stack", "title": "Aurora Gaming Stack", "origin": "Aurora Archive", "label": "Gaming Ready"},
            {"package": "aurora-creator-stack", "title": "Aurora Creator Stack", "origin": "Aurora Archive", "label": "Creator Stack"},
            {"package": "aurora-training-stack", "title": "Aurora Training Stack", "origin": "Aurora Archive", "label": "Training Stack"},
            {"package": "aurora-datacenter-stack", "title": "Aurora Datacenter Stack", "origin": "Aurora Archive", "label": "Datacenter Ops"},
            {"package": "aurora-privacy-stack", "title": "Aurora Privacy Stack", "origin": "Aurora Archive", "label": "Privacy Guard"}
        ]
    }))
    .context("failed to render aurora archive catalog")
}

fn aurora_meta_packages_json(config: &BuildConfig) -> Result<String> {
    serde_json::to_string_pretty(&serde_json::json!({
        "meta_packages": [
            {
                "name": "aurora-desktop",
                "title": format!("{} Desktop", config.branding_name),
                "depends": ["aurora-control-center", "aurora-app-center", "aurora-welcome", "aurora-dev-lab", "aurora-creator-studio", "aurora-gaming-center", "aurora-ai-hub", "aurora-security-center"]
            },
            {
                "name": "aurora-ai-stack",
                "title": "Aurora AI Stack",
                "depends": ["ollama", "aurora-ai-hub", "ripgrep", "fd-find", "bat", "helix"]
            },
            {
                "name": "aurora-dev-stack",
                "title": "Aurora Dev Stack",
                "depends": ["aurora-dev-lab", "git", "git-lfs", "podman", "distrobox", "tmux", "direnv", "shellcheck", "hyperfine"]
            },
            {
                "name": "aurora-gaming-stack",
                "title": "Aurora Gaming Stack",
                "depends": ["gamemode", "mangohud", "gamescope", "goverlay", "steam-installer"]
            },
            {
                "name": "aurora-creator-stack",
                "title": "Aurora Creator Stack",
                "depends": ["aurora-creator-studio", "obs-studio", "ffmpeg", "gimp", "inkscape", "libreoffice"]
            },
            {
                "name": "aurora-security-stack",
                "title": "Aurora Security Stack",
                "depends": ["aurora-security-center", "ufw", "fail2ban", "apparmor-utils", "fprintd", "kdump-tools", "authd"]
            },
            {
                "name": "aurora-scientific-stack",
                "title": "Aurora Scientific Stack",
                "depends": ["numactl", "linux-tools-generic", "hwloc", "valkey-server", "ripgrep", "fd-find"]
            },
            {
                "name": "aurora-training-stack",
                "title": "Aurora Training Stack",
                "depends": ["aurora-ai-hub", "python3-venv", "python3-pip", "numactl", "hwloc", "valkey-server"]
            },
            {
                "name": "aurora-datacenter-stack",
                "title": "Aurora Datacenter Stack",
                "depends": ["prometheus-node-exporter", "sysstat", "iotop", "fio", "nvme-cli"]
            },
            {
                "name": "aurora-privacy-stack",
                "title": "Aurora Privacy Stack",
                "depends": ["tor", "torsocks", "macchanger", "usbguard"]
            }
        ]
    }))
    .context("failed to render aurora meta package catalog")
}

fn load_config(tree: &Path) -> Result<BuildConfig> {
    let path = tree.join("profiles/default.json");
    let content = fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&content).context("failed to parse build config")
}

fn load_performance_profile(tree: &Path) -> Result<PerformanceProfile> {
    let path = tree.join("profiles/performance.json");
    let content = fs::read_to_string(&path)
        .with_context(|| format!("failed to read {}", path.display()))?;
    serde_json::from_str(&content).context("failed to parse performance profile")
}

fn desktop_packages(desktop: &DesktopPreset) -> Vec<String> {
    match desktop {
        DesktopPreset::Minimal => vec![
            "ubuntu-standard".to_string(),
            "network-manager".to_string(),
            "sudo".to_string(),
            "xfce4".to_string(),
            "lightdm".to_string(),
            "gamemode".to_string(),
            "mangohud".to_string(),
        ],
        DesktopPreset::Gnome => vec![
            "ubuntu-desktop".to_string(),
            "gnome-shell-extension-manager".to_string(),
            "gnome-shell-extensions".to_string(),
            "steam-installer".to_string(),
            "gamemode".to_string(),
            "gnome-tweaks".to_string(),
            "flatpak".to_string(),
            "gnome-software-plugin-flatpak".to_string(),
            "obs-studio".to_string(),
            "gimp".to_string(),
            "inkscape".to_string(),
            "ffmpeg".to_string(),
            "libreoffice".to_string(),
            "podman".to_string(),
            "distrobox".to_string(),
            "virt-manager".to_string(),
        ],
        DesktopPreset::Kde => vec![
            "kubuntu-desktop".to_string(),
            "steam-installer".to_string(),
            "gamemode".to_string(),
            "plasma-workspace-wayland".to_string(),
        ],
        DesktopPreset::Cosmic => vec![
            "ubuntu-desktop".to_string(),
            "flatpak".to_string(),
            "gnome-software-plugin-flatpak".to_string(),
            "gamemode".to_string(),
        ],
    }
}

fn default_theme_css(name: &str, accent: &str) -> String {
    format!(
        ":root {{ --aurora-accent: {accent}; --aurora-bg: #08111d; --aurora-panel: #101c2b; }}\nbody {{ background: radial-gradient(circle at top, #14263a, var(--aurora-bg)); color: #eef7ff; font-family: 'Orbitron', sans-serif; }}\n.login-logo {{ display: none; }}\n.session-title::after {{ content: '{name}'; color: var(--aurora-accent); }}\n"
    )
}

fn installer_desktop_file() -> String {
    desktop_entry("AURORA Installer", "installer", "System")
}

fn aurora_control_center_desktop_file() -> String {
    desktop_entry("Aurora Control Center", "control-center", "Settings;System")
}

fn aurora_ai_hub_desktop_file() -> String {
    desktop_entry("Aurora AI Hub", "ai-hub", "Utility;Development")
}

fn aurora_dev_lab_desktop_file() -> String {
    desktop_entry("Aurora Dev Lab", "dev-lab", "Development;System")
}

fn aurora_creator_studio_desktop_file() -> String {
    desktop_entry("Aurora Creator Studio", "creator-studio", "AudioVideo;Graphics;Office")
}

fn aurora_gaming_center_desktop_file() -> String {
    desktop_entry("Aurora Gaming Center", "gaming-center", "Game;System")
}

fn aurora_security_center_desktop_file() -> String {
    desktop_entry("Aurora Security Center", "security-center", "Security;Settings;System")
}

fn aurora_package_center_desktop_file() -> String {
    desktop_entry("Aurora App Center", "package-center", "System;PackageManager")
}

fn aurora_welcome_desktop_file() -> String {
    desktop_entry("Aurora Welcome", "welcome", "System")
}

fn aurora_andromeda_desktop_file() -> String {
    "[Desktop Entry]\nType=Application\nName=Aurora Andromeda\nExec=/bin/sh -lc 'xdg-open /usr/share/aurora/andromeda/index.html || gio open /usr/share/aurora/andromeda/index.html || firefox /usr/share/aurora/andromeda/index.html'\nTerminal=false\nX-GNOME-Autostart-enabled=true\nCategories=System;\n".to_string()
}

fn desktop_entry(name: &str, surface: &str, categories: &str) -> String {
    format!(
        "[Desktop Entry]\nType=Application\nName={name}\nExec=/bin/sh -lc 'x-terminal-emulator -e aurora-distro app {surface} || gnome-terminal -- aurora-distro app {surface} || alacritty -e aurora-distro app {surface} || xterm -e aurora-distro app {surface}'\nTerminal=false\nX-GNOME-Autostart-enabled=true\nCategories={categories};\n"
    )
}

fn firstboot_script(installer: &InstallerConfig) -> String {
    format!(
        "#!/usr/bin/env bash\nset -euo pipefail\nMODE_FILE=/etc/aurora-installer/selected-mode\nif [ ! -f \"$MODE_FILE\" ]; then\n  printf '%s\\n' '{}' > \"$MODE_FILE\"\nfi\nif systemctl list-unit-files | grep -q '^aurora-autosetup.service'; then\n  systemctl enable aurora-autosetup.service >/dev/null 2>&1 || true\n  systemctl start aurora-autosetup.service >/dev/null 2>&1 || true\nfi\ncommand -v aurora-apply-desktop >/dev/null 2>&1 && aurora-apply-desktop || true\ncommand -v aurora-gpu-setup >/dev/null 2>&1 && aurora-gpu-setup --firstboot || true\ncommand -v aurora-ai-setup >/dev/null 2>&1 && aurora-ai-setup --firstboot || true\ncommand -v aurora-ai-helper >/dev/null 2>&1 && aurora-ai-helper status >/dev/null 2>&1 || true\ncommand -v aurora-dev-setup >/dev/null 2>&1 && aurora-dev-setup || true\ncommand -v aurora-creator-setup >/dev/null 2>&1 && aurora-creator-setup || true\ncommand -v aurora-gaming-setup >/dev/null 2>&1 && aurora-gaming-setup --firstboot || true\ncommand -v aurora-training-setup >/dev/null 2>&1 && aurora-training-setup || true\ncommand -v aurora-datacenter-setup >/dev/null 2>&1 && aurora-datacenter-setup || true\ncommand -v aurora-security-setup >/dev/null 2>&1 && aurora-security-setup --firstboot || true\ncommand -v aurora-privacy-setup >/dev/null 2>&1 && aurora-privacy-setup --firstboot || true\ncommand -v aurora-hardware-guard >/dev/null 2>&1 && aurora-hardware-guard --seal || true\nprintf '%s\\n' \"$(cat \"$MODE_FILE\")\" >/tmp/aurora-installer-mode\nif command -v xdg-open >/dev/null 2>&1; then\n  xdg-open /usr/share/aurora/andromeda/index.html >/dev/null 2>&1 || true\nfi\n",
        install_mode_name(&installer.default_mode)
    )
}

fn installer_html(name: &str, accent: &str, installer: &InstallerConfig) -> String {
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>{name} Installer</title><link rel=\"stylesheet\" href=\"installer.css\"></head><body><main class=\"shell\"><section class=\"hero\"><div class=\"brand-row\"><img class=\"brand-mark\" src=\"/usr/share/aurora/logo.svg\" alt=\"{name} logo\"><div><p class=\"eyebrow\">Adaptive Neon Workstation</p><h1>{name}</h1></div></div><p class=\"lede\">Shape the system before install: choose your runtime mode, deploy a tuned partition plan, and boot into a shell curated around speed, visuals, and stronger day-one tooling.</p><div class=\"cta-row\"><button id=\"scanBtn\">Scan This System</button><button id=\"planBtn\">Create Partition Plan</button><button id=\"usbBtn\">Use Inbuilt USB Writer</button><button id=\"repairBtn\">Show Boot Repair</button></div></section><section class=\"grid\"><article class=\"card mode-card\"><h2>Balanced</h2><p>Moderate zram, calmer I/O tuning, and lower background disruption for mixed desktop workloads.</p><button data-mode=\"balanced\">Choose Balanced</button></article><article class=\"card mode-card\"><h2>Gaming</h2><p>Frame-time stability, low latency, DirectStream asset priming, and launcher-ready gaming defaults.</p><button data-mode=\"gaming\">Choose Gaming</button></article><article class=\"card mode-card\"><h2>Creator</h2><p>Disk, cache, and memory posture for editing, rendering, OBS capture, and FFmpeg or Blender-class sessions.</p><button data-mode=\"creator\">Choose Creator</button></article><article class=\"card mode-card\"><h2>Training</h2><p>Hugepages, dataset staging, memory locality, and long-running throughput for local AI training.</p><button data-mode=\"training\">Choose Training</button></article><article class=\"card mode-card\"><h2>Datacenter</h2><p>Predictable scheduling, service isolation, I/O policy, and observability for service hosts.</p><button data-mode=\"datacenter\">Choose Datacenter</button></article><article class=\"card mode-card\"><h2>Max Throughput</h2><p>Pushes CPU throughput, hugepage usage, zram, and aggressive boot/runtime knobs for compute-heavy work.</p><button data-mode=\"max-throughput\">Choose Max Throughput</button></article><article class=\"card\"><h2>Out-of-Box Stack</h2><p>Flatpak-ready app flow, tuned shell defaults, bold theming, runtime tooling, repair scripts, GPU staging, and a performance control surface.</p></article></section><section class=\"terminal\"><div class=\"terminal-bar\"><span></span><span></span><span></span></div><pre id=\"output\">Awaiting action...</pre></section></main><script>window.AURORA_INSTALLER={{accent:\"{accent}\",name:\"{name}\",defaultMode:\"{}\",availableModes:{},memoryBoost:{},bootRepair:{}}};</script><script src=\"installer.js\"></script></body></html>",
        install_mode_name(&installer.default_mode),
        serde_json::to_string(&installer.available_modes).unwrap_or_else(|_| "[]".to_string()),
        installer.enable_memory_boost_wizard,
        installer.enable_boot_repair_tools
    )
}

fn installer_css(accent: &str) -> String {
    format!(
        ":root{{--accent:{accent};--bg:#05070d;--panel:#0f1724;--line:#17314d;--text:#edf7ff;--muted:#9db4c8;--warm:#ff6a00}}*{{box-sizing:border-box}}body{{margin:0;font-family:'Orbitron',sans-serif;background:radial-gradient(circle at top,#10233a 0%,#05070d 55%,#020304 100%);color:var(--text)}}.shell{{max-width:1100px;margin:0 auto;padding:40px 24px 72px}}.hero{{padding:48px 36px;border:1px solid var(--line);background:linear-gradient(135deg,rgba(16,31,52,.92),rgba(8,13,20,.96));border-radius:28px;box-shadow:0 20px 80px rgba(0,0,0,.45)}}.brand-row{{display:flex;align-items:center;gap:20px}}.brand-mark{{width:94px;height:94px;filter:drop-shadow(0 0 24px rgba(18,247,255,.28))}}.eyebrow{{letter-spacing:.22em;text-transform:uppercase;color:var(--accent);font-size:12px}}h1{{font-size:72px;line-height:.95;margin:16px 0}}.lede{{max-width:760px;color:var(--muted);font-family:'Rajdhani',sans-serif;font-size:24px}}.cta-row{{display:flex;flex-wrap:wrap;gap:16px;margin-top:28px}}button{{border:1px solid var(--line);background:linear-gradient(180deg,#102740,#0a1524);color:var(--text);padding:16px 22px;border-radius:16px;font:inherit;cursor:pointer}}button:hover{{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent),0 0 24px rgba(18,247,255,.22)}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:18px;margin-top:24px}}.card{{padding:22px;border-radius:22px;background:rgba(11,18,30,.86);border:1px solid var(--line)}}.card h2{{margin:0 0 8px;font-size:24px}}.card p{{margin:0;color:var(--muted);font-family:'Rajdhani',sans-serif;font-size:20px}}.terminal{{margin-top:24px;border-radius:22px;overflow:hidden;border:1px solid var(--line);background:#071019}}.terminal-bar{{display:flex;gap:8px;padding:12px 14px;background:#0c1725}}.terminal-bar span{{width:12px;height:12px;border-radius:50%;background:var(--warm)}}.terminal-bar span:nth-child(2){{background:#ffca28}}.terminal-bar span:nth-child(3){{background:var(--accent)}}pre{{margin:0;padding:22px;color:#d8f7ff;min-height:180px;font-family:'JetBrains Mono',monospace;white-space:pre-wrap}}@media (max-width:720px){{.brand-row{{align-items:flex-start;flex-direction:column}}.brand-mark{{width:72px;height:72px}}h1{{font-size:48px}}.lede{{font-size:20px}}}}"
    )
}

fn installer_js(_installer: &InstallerConfig) -> String {
    "const cfg=window.AURORA_INSTALLER;const output=document.getElementById('output');const write=(lines)=>output.textContent=lines.join('\\n');const modes={balanced:['Balanced mode selected','- schedutil governor','- moderate zram and preload','- quieter background-service policy','- good default for mixed desktop work'],gaming:['Gaming mode selected','- performance governor','- frame-time focused storage/read-ahead posture','- enables GameMode, MangoHud, Gamescope, and DirectStream hints','- best fit for desktop responsiveness and games'],creator:['Creator mode selected','- performance governor with cache-biased writeback','- strong fit for OBS, FFmpeg, Blender, and editing workflows','- favors responsive export and media ingest'],training:['Training mode selected','- hugepages and throughput-first memory posture','- dataset/cache staging for long-running jobs','- mixed CPU/GPU orchestration target when hardware exists'],datacenter:['Datacenter mode selected','- predictable I/O scheduler and lower swappiness','- service isolation and observability posture','- favors consistent service latency over desktop feel'],['max-throughput']:['Max Throughput selected','- performance governor + aggressive hugepage posture','- highest zram allocation and stronger cache tuning','- trims more services and boosts readahead','- best fit for compute-heavy sustained workloads']};document.getElementById('scanBtn').addEventListener('click',()=>write(['Scanning current machine...','- detect firmware mode','- detect CPU model and RAM','- enumerate storage devices','- estimate swap + zram balance','Result: if details are missing, aurora-distro can fall back to automatic planning.']));document.getElementById('planBtn').addEventListener('click',()=>write(['Generating partition plan...','- choose GPT for UEFI or hybrid installs','- add BIOS_GRUB when legacy GRUB embedding is needed','- size swap from RAM and disk pressure','- keep partition-apply.sh ready for manual replay','Result: installer can switch between workload profiles after partitioning.']));document.getElementById('usbBtn').addEventListener('click',()=>write(['Inbuilt USB writer flow','1. Confirm target device','2. Verify path is a block device','3. Write ISO with dd + sync','4. Return success/failure logs to the user']));document.getElementById('repairBtn').addEventListener('click',()=>write(['Boot repair toolkit','- bind-mount /dev /proc /sys','- reinstall GRUB for EFI and BIOS targets when available','- refresh initramfs and grub.cfg','- keep BOOTX64.EFI fallback in place for UEFI firmware']));document.querySelectorAll('[data-mode]').forEach(btn=>btn.addEventListener('click',()=>write(modes[btn.dataset.mode]||['Unknown mode'])));write(['Default installer mode: '+cfg.defaultMode,'Available modes: '+cfg.availableModes.join(', '),'Choose a mode, then scan hardware or generate a partition plan.']);".to_string()
}

fn control_center_html(name: &str, accent: &str) -> String {
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>{name} Control Center</title><link rel=\"stylesheet\" href=\"control-center.css\"></head><body><main class=\"frame\"><section class=\"hero\"><p class=\"eyebrow\">Aurora Runtime Control</p><h1>{name} Control Center</h1><p class=\"lede\">Switch system modes, inspect runtime placement, and keep the distro personality consistent after install.</p></section><section class=\"grid\"><button data-mode=\"balanced\">Balanced</button><button data-mode=\"gaming\">Gaming</button><button data-mode=\"creator\">Creator</button><button data-mode=\"training\">Training</button><button data-mode=\"datacenter\">Datacenter</button><button data-mode=\"max-throughput\">Max Throughput</button><button id=\"desktopBtn\">Apply Desktop Layer</button><button id=\"runtimeBtn\">Runtime Status</button><button id=\"tourBtn\">Open Welcome Tour</button><button id=\"aiBtn\">Open AI Hub</button><button id=\"gamingBtn\">Open Gaming Center</button><button id=\"securityBtn\">Open Security Center</button><button id=\"packageBtn\">Open Package Center</button></section><pre id=\"terminal\">Aurora control surface ready.</pre></main><script>window.AURORA_CC={{name:\"{name}\",accent:\"{accent}\"}};</script><script src=\"control-center.js\"></script></body></html>"
    )
}

fn control_center_css(accent: &str) -> String {
    format!(
        ":root{{--accent:{accent};--bg:#04060b;--panel:#0f1724;--line:#17314d;--text:#edf7ff;--muted:#9db4c8}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at top,#10233a,#04060b 60%);color:var(--text);font-family:'Orbitron',sans-serif}}.frame{{max-width:1080px;margin:0 auto;padding:40px 24px 72px}}.hero{{padding:32px;border:1px solid var(--line);border-radius:24px;background:rgba(10,17,28,.92)}}.eyebrow{{text-transform:uppercase;letter-spacing:.2em;color:var(--accent);font-size:12px}}h1{{margin:12px 0 10px;font-size:56px}}.lede{{font-family:'Rajdhani',sans-serif;color:var(--muted);font-size:22px;max-width:760px}}.grid{{margin-top:22px;display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}}button{{padding:18px 20px;border-radius:18px;border:1px solid var(--line);background:linear-gradient(180deg,#102740,#0a1524);color:var(--text);font:inherit;cursor:pointer}}button:hover{{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent),0 0 24px rgba(18,247,255,.2)}}pre{{margin-top:22px;min-height:180px;padding:20px;border-radius:20px;border:1px solid var(--line);background:#071019;color:#d8f7ff;font-family:'JetBrains Mono',monospace;white-space:pre-wrap}}"
    )
}

fn control_center_js() -> String {
    "const term=document.getElementById('terminal');const write=(lines)=>term.textContent=lines.join('\\n');document.querySelectorAll('[data-mode]').forEach(btn=>btn.addEventListener('click',()=>write(['Requested mode: '+btn.dataset.mode,'CLI: sudo aurora-mode-switch '+btn.dataset.mode,'Effect: updates /etc/default/aurora-performance and triggers aurora-autosetup.'])));document.getElementById('desktopBtn').addEventListener('click',()=>write(['Desktop refresh','CLI: aurora-apply-desktop','Effect: reapplies Aurora shell defaults, favorites, wallpaper, fonts, and dock posture.']));document.getElementById('runtimeBtn').addEventListener('click',()=>write(['Runtime status','Binary: /usr/local/bin/aurora','Quick checks: aurora version | aurora detect | aurora status','AI helper: aurora-ai-helper status','GPU posture: cat /etc/aurora/gpu-profile.conf']));document.getElementById('tourBtn').addEventListener('click',()=>window.location='/usr/share/aurora/welcome/index.html');document.getElementById('aiBtn').addEventListener('click',()=>window.location='/usr/share/aurora/ai-hub/index.html');document.getElementById('gamingBtn').addEventListener('click',()=>window.location='/usr/share/aurora/gaming-center/index.html');document.getElementById('securityBtn').addEventListener('click',()=>window.location='/usr/share/aurora/security-center/index.html');document.getElementById('packageBtn').addEventListener('click',()=>window.location='/usr/share/aurora/package-center/index.html');".to_string()
}

fn ai_hub_html(name: &str, accent: &str) -> String {
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>{name} AI Hub</title><link rel=\"stylesheet\" href=\"ai-hub.css\"></head><body><main class=\"wrap\"><section class=\"hero\"><p class=\"eyebrow\">AI-Native Desktop</p><h1>{name} AI Hub</h1><p class=\"lede\">Private local models, desktop search integration, a local AI helper, and GPU-aware AI tooling prepared for offline use through Ollama and Aurora wrappers.</p></section><section class=\"cards\"><article><h2>Local Models</h2><p>Ollama is treated as the default local model service. Suggested first pull: <code>ollama pull llama3</code>.</p></article><article><h2>AI Helper</h2><p><code>aurora-ai-helper</code> gives the desktop a local chat and status surface without pushing prompts to remote services.</p></article><article><h2>GPU Readiness</h2><p>Driver/tooling posture is prepared for NVIDIA, AMD, and Intel acceleration through Mesa, CUDA, ROCm, and OpenCL-aware setup flows where available.</p></article><article><h2>Rust Toolchain Surface</h2><p>Rust-native helpers like <code>ripgrep</code>, <code>fd</code>, <code>bat</code>, <code>helix</code>, <code>zellij</code>, <code>bottom</code>, and <code>just</code> are part of Aurora's developer posture when available.</p></article></section></main></body></html>"
    )
}

fn ai_hub_css(accent: &str) -> String {
    format!(
        ":root{{--accent:{accent};--bg:#05070d;--panel:#0d1827;--line:#15314d;--text:#eef7ff;--muted:#9eb5cb}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at top,#10233a,#05070d 60%);color:var(--text);font-family:'Orbitron',sans-serif}}.wrap{{max-width:1100px;margin:0 auto;padding:48px 24px 72px}}.hero{{padding:34px;border:1px solid var(--line);border-radius:26px;background:rgba(11,18,29,.92)}}.eyebrow{{color:var(--accent);text-transform:uppercase;letter-spacing:.2em;font-size:12px}}h1{{margin:14px 0;font-size:60px}}.lede{{max-width:780px;font-family:'Rajdhani',sans-serif;font-size:24px;color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:18px;margin-top:24px}}article{{padding:22px;border-radius:22px;background:rgba(10,18,29,.88);border:1px solid var(--line)}}article h2{{margin:0 0 8px;font-size:24px}}article p{{margin:0;font-family:'Rajdhani',sans-serif;font-size:20px;color:var(--muted)}}code{{color:var(--accent)}}"
    )
}

fn gaming_center_html(name: &str, accent: &str) -> String {
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>{name} Gaming Center</title><link rel=\"stylesheet\" href=\"gaming-center.css\"></head><body><main class=\"wrap\"><section class=\"hero\"><p class=\"eyebrow\">Unified Gaming Stack</p><h1>{name} Gaming Center</h1><p class=\"lede\">Aurora bundles a single surface for Proton, Wine, Mesa, Gamescope, anti-cheat posture, MangoHud, low-latency runtime defaults, and DirectStorage-style asset streaming.</p></section><section class=\"cards\"><article><h2>Wayland First</h2><p>Wayland remains the default target. NVIDIA stability and zero-tear posture are part of the distro direction.</p></article><article><h2>Kernel/Game Tweaks</h2><p>Gaming profiles include higher mapping limits, low-latency desktop defaults, and aggressive memory posture.</p></article><article><h2>Driver Stack</h2><p>Mesa, Vulkan, MangoHud, GameMode, Gamescope, and related tooling are preselected where available.</p></article><article><h2>DirectStream</h2><p><code>aurora-directstream</code> applies a Linux-native fast asset loading posture with larger readahead, I/O priority, and game cache hints.</p></article></section></main></body></html>"
    )
}

fn gaming_center_css(accent: &str) -> String {
    format!(
        ":root{{--accent:{accent};--bg:#05070d;--panel:#111a2a;--line:#17314d;--text:#eef7ff;--muted:#9eb5cb;--warm:#ff6a00}}*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(180deg,#111c30,#05070d);color:var(--text);font-family:'Orbitron',sans-serif}}.wrap{{max-width:1100px;margin:0 auto;padding:48px 24px 72px}}.hero{{padding:34px;border:1px solid var(--line);border-radius:26px;background:rgba(11,18,29,.94)}}.eyebrow{{color:var(--warm);text-transform:uppercase;letter-spacing:.2em;font-size:12px}}h1{{margin:14px 0;font-size:60px}}.lede{{max-width:780px;font-family:'Rajdhani',sans-serif;font-size:24px;color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:18px;margin-top:24px}}article{{padding:22px;border-radius:22px;background:rgba(10,18,29,.88);border:1px solid var(--line)}}article h2{{margin:0 0 8px;font-size:24px}}article p{{margin:0;font-family:'Rajdhani',sans-serif;font-size:20px;color:var(--muted)}}code{{color:var(--accent)}}"
    )
}

fn welcome_html(name: &str, accent: &str) -> String {
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Welcome to {name}</title><link rel=\"stylesheet\" href=\"welcome.css\"></head><body><main class=\"wrap\"><section class=\"hero\"><p class=\"eyebrow\">Welcome To {name}</p><h1>Operator-grade desktop</h1><p class=\"lede\">Aurora boots like a sci-fi workstation, not a generic Ubuntu respin. The runtime, AI hub, gaming center, creator tooling, privacy posture, repair surface, and performance profiles are meant to make the machine feel purpose-built from the first session.</p><div class=\"hero-grid\"><div><span>Gaming</span><strong>DirectStream, Proton posture, Gamescope, MangoHud, Wayland-first</strong></div><div><span>Creator</span><strong>OBS, FFmpeg, Blender, Krita, Kdenlive, low-friction export flow</strong></div><div><span>Operator</span><strong>Containers, tracing, observability, local AI, privacy, clone detection</strong></div></div></section><section class=\"cards\"><article><h2>Runtime Layer</h2><p><code>aurora</code> is staged into the live system and exposed as a first-class command surface.</p></article><article><h2>AI-Native</h2><p>Local model access and <code>aurora-ai-helper</code> are treated as first-party workflow pieces, not browser-only extras.</p></article><article><h2>Gaming Center</h2><p>Wayland, Proton, Mesa, Gamescope, DirectStream, and low-latency posture are pulled into one visible layer.</p></article><article><h2>Rust-Native Tools</h2><p>Fast search, terminal, editor, benchmarking, and workspace tools are staged where the archive provides them.</p></article></section></main></body></html>"
    )
}

fn security_center_html(name: &str, accent: &str) -> String {
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>{name} Security Center</title><link rel=\"stylesheet\" href=\"security-center.css\"></head><body><main class=\"wrap\"><section class=\"hero\"><p class=\"eyebrow\">Security Center</p><h1>{name} Security Center</h1><p class=\"lede\">Aurora centralizes host hardening, snap permission posture, kdump visibility, fingerprint support, firewall state, and future enterprise-auth hooks.</p></section><section class=\"cards\"><article><h2>Granular Controls</h2><p>Security posture includes firewall, fail2ban, AppArmor, fingerprint support, and future snap permission prompting direction.</p></article><article><h2>Kdump</h2><p><code>kdump-tools</code> is included so kernel crash capture can be enabled for debugging and reliability workflows.</p></article><article><h2>Repositories</h2><p>Aurora assumes stronger repository trust posture and discourages weak third-party signature paths.</p></article><article><h2>Enterprise Prep</h2><p><code>authd</code> is treated as a future-facing enterprise authentication path rather than legacy-only PAM glue.</p></article></section></main></body></html>"
    )
}

fn security_center_css(accent: &str) -> String {
    format!(
        ":root{{--accent:{accent};--bg:#061018;--panel:#101a28;--line:#16314d;--text:#eef7ff;--muted:#9fb5ca;--warn:#ff9d2f}}*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(180deg,#091320,#061018);color:var(--text);font-family:'Orbitron',sans-serif}}.wrap{{max-width:1100px;margin:0 auto;padding:48px 24px 72px}}.hero{{padding:34px;border:1px solid var(--line);border-radius:26px;background:rgba(11,18,29,.94)}}.eyebrow{{color:var(--warn);text-transform:uppercase;letter-spacing:.2em;font-size:12px}}h1{{margin:14px 0;font-size:58px}}.lede{{max-width:780px;font-family:'Rajdhani',sans-serif;font-size:24px;color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:18px;margin-top:24px}}article{{padding:22px;border-radius:22px;background:rgba(10,18,29,.88);border:1px solid var(--line)}}article h2{{margin:0 0 8px;font-size:24px}}article p{{margin:0;font-family:'Rajdhani',sans-serif;font-size:20px;color:var(--muted)}}code{{color:var(--accent)}}"
    )
}

fn package_center_html(name: &str, accent: &str) -> String {
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>{name} App Center</title><link rel=\"stylesheet\" href=\"package-center.css\"></head><body><main class=\"wrap\"><section class=\"hero\"><p class=\"eyebrow\">Aurora Archive</p><h1>{name} App Center</h1><p class=\"lede\">Aurora makes package origin explicit and puts Aurora-owned stacks, gamer tooling, creator tooling, AI helpers, and operator utilities ahead of raw upstream naming.</p></section><section class=\"cards\"><article><h2>Aurora Archive</h2><p>A local Aurora repository is staged into the image so curated Aurora packages and meta-packages have a distinct channel and identity.</p></article><article><h2>Curated Stacks</h2><p><code>Gaming</code>, <code>Creator</code>, <code>Training</code>, <code>Datacenter</code>, and <code>Privacy</code> stacks make the image feel like a platform instead of a bare package list.</p></article><article><h2>Native DEB, Snap, Flatpak</h2><p>The App Center still distinguishes archive-backed DEBs, Snaps, and Flatpaks so users can reason about provenance and trust.</p></article><article><h2>Selective Replacement</h2><p>Aurora replaces only the layers that improve UX, runtime behavior, and workflow density while preserving a stable upstream base elsewhere.</p></article></section></main></body></html>"
    )
}

fn package_center_css(accent: &str) -> String {
    format!(
        ":root{{--accent:{accent};--bg:#05070d;--panel:#111a2a;--line:#17314d;--text:#eef7ff;--muted:#9db4c8}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at top,#10233a,#05070d 60%);color:var(--text);font-family:'Orbitron',sans-serif}}.wrap{{max-width:1100px;margin:0 auto;padding:48px 24px 72px}}.hero{{padding:34px;border:1px solid var(--line);border-radius:26px;background:rgba(11,18,29,.94)}}.eyebrow{{color:var(--accent);text-transform:uppercase;letter-spacing:.2em;font-size:12px}}h1{{margin:14px 0;font-size:58px}}.lede{{max-width:780px;font-family:'Rajdhani',sans-serif;font-size:24px;color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:18px;margin-top:24px}}article{{padding:22px;border-radius:22px;background:rgba(10,18,29,.88);border:1px solid var(--line)}}article h2{{margin:0 0 8px;font-size:24px}}article p{{margin:0;font-family:'Rajdhani',sans-serif;font-size:20px;color:var(--muted)}}code{{color:var(--accent)}}"
    )
}

fn welcome_css(accent: &str) -> String {
    format!(
        ":root{{--accent:{accent};--bg:#05070d;--panel:#0e1827;--line:#16304d;--text:#eef7ff;--muted:#9bb2c8}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at top,#152947 0%,#08101b 40%,#03050a 100%);color:var(--text);font-family:'Orbitron',sans-serif}}.wrap{{max-width:1180px;margin:0 auto;padding:48px 24px 72px}}.hero{{padding:40px;border:1px solid var(--line);border-radius:30px;background:linear-gradient(135deg,rgba(11,17,28,.96),rgba(3,6,10,.92));box-shadow:0 30px 120px rgba(0,0,0,.45)}}.eyebrow{{letter-spacing:.22em;text-transform:uppercase;color:var(--accent);font-size:12px}}h1{{margin:14px 0;font-size:64px;line-height:.95}}.lede{{max-width:860px;font-family:'Rajdhani',sans-serif;font-size:24px;color:var(--muted)}}.hero-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px;margin-top:28px}}.hero-grid div{{padding:18px;border-radius:18px;border:1px solid var(--line);background:rgba(10,18,29,.76)}}.hero-grid span{{display:block;margin-bottom:8px;color:var(--accent);text-transform:uppercase;letter-spacing:.18em;font-size:12px}}.hero-grid strong{{font-family:'Rajdhani',sans-serif;font-size:20px;line-height:1.2}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:18px;margin-top:24px}}article{{padding:22px;border-radius:22px;background:rgba(10,18,29,.88);border:1px solid var(--line)}}article h2{{margin:0 0 8px;font-size:24px}}article p{{margin:0;font-family:'Rajdhani',sans-serif;font-size:20px;color:var(--muted)}}"
    )
}

fn andromeda_html(name: &str) -> String {
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Andromeda</title><link rel=\"stylesheet\" href=\"andromeda.css\"></head><body><main class=\"frame\"><div class=\"veil\"></div><section class=\"hero\"><p class=\"eyebrow\">{name}</p><h1>ANDROMEDA</h1><p class=\"lede\">Cinematic runtime surface loading. Gaming, creation, local AI, privacy, and operator tooling are staging now.</p><div class=\"scanline\"></div></section></main><script>setTimeout(()=>window.location='/usr/share/aurora/welcome/index.html',4200);</script></body></html>"
    )
}

fn andromeda_css(accent: &str) -> String {
    format!(
        ":root{{--accent:{accent};--text:#ffffff}}*{{box-sizing:border-box}}body{{margin:0;background:#000;color:var(--text);font-family:'Orbitron',sans-serif;overflow:hidden}}.frame{{min-height:100vh;display:grid;place-items:center;position:relative;background:radial-gradient(circle at center,rgba(14,18,24,.25),#000 62%)}}.veil{{position:absolute;inset:0;background:linear-gradient(180deg,rgba(255,255,255,.03),rgba(0,0,0,.18) 22%,rgba(0,0,0,.55));mix-blend-mode:screen}}.hero{{position:relative;text-align:center;padding:48px 32px}}.eyebrow{{margin:0 0 18px;letter-spacing:.38em;text-transform:uppercase;font-size:14px;color:rgba(255,255,255,.72)}}h1{{margin:0;font-size:min(16vw,180px);letter-spacing:.28em;text-indent:.28em;font-weight:800;text-shadow:0 0 32px rgba(255,255,255,.15),0 0 90px rgba(255,255,255,.08)}}.lede{{max-width:720px;margin:28px auto 0;color:rgba(255,255,255,.78);font-family:'Rajdhani',sans-serif;font-size:28px}}.scanline{{width:min(720px,78vw);height:2px;margin:34px auto 0;background:linear-gradient(90deg,transparent,rgba(255,255,255,.9),transparent);box-shadow:0 0 28px rgba(255,255,255,.35),0 0 120px rgba(255,255,255,.18);animation:scan 3.2s ease-in-out infinite}}@keyframes scan{{0%,100%{{transform:scaleX(.62);opacity:.45}}50%{{transform:scaleX(1);opacity:1}}}}"
    )
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

fn dconf_defaults(config: &BuildConfig) -> String {
    format!(
        "[org/gnome/desktop/interface]\ngtk-theme='Aurora-Neon'\nicon-theme='Papirus'\ncolor-scheme='prefer-dark'\nclock-format='24h'\nfont-name='Cantarell 11'\ndocument-font-name='Cantarell 11'\nmonospace-font-name='Cascadia Code 11'\nenable-hot-corners=false\n\n[org/gnome/desktop/background]\npicture-uri='file:///usr/share/backgrounds/aurora/gaming-kali.svg'\npicture-uri-dark='file:///usr/share/backgrounds/aurora/gaming-kali.svg'\nprimary-color='#05070d'\nsecondary-color='{}'\n\n[org/gnome/desktop/wm/preferences]\nbutton-layout='appmenu:minimize,maximize,close'\nfocus-mode='click'\n\n[org/gnome/mutter]\nedge-tiling=true\noverlay-key='Super_L'\nworkspaces-only-on-primary=false\n\n[org/gnome/shell]\nfavorite-apps=['org.gnome.Nautilus.desktop','org.gnome.Terminal.desktop','alacritty.desktop','firefox_firefox.desktop','org.gnome.Software.desktop','aurora-control-center.desktop','aurora-dev-lab.desktop','aurora-creator-studio.desktop','aurora-ai-hub.desktop','aurora-gaming-center.desktop','aurora-installer.desktop']\ndisable-user-extensions=false\nenabled-extensions=['apps-menu@gnome-shell-extensions.gcampax.github.com','places-menu@gnome-shell-extensions.gcampax.github.com','user-theme@gnome-shell-extensions.gcampax.github.com']\nwelcome-dialog-last-shown-version='999999'\n\n[org/gnome/shell/keybindings]\ntoggle-application-view=['Super_L']\n\n[org/gnome/shell/extensions/dash-to-dock]\ndock-position='BOTTOM'\nextend-height=false\nshow-trash=true\nshow-mounts=true\ntransparency-mode='FIXED'\nbackground-opacity=0.25\nclick-action='minimize-or-overview'\nshow-apps-at-top=true\n\n[org/gnome/login-screen]\nlogo='/usr/share/aurora/logo.svg'\n",
        config.accent_color
    )
}

fn os_release_content(config: &BuildConfig) -> String {
    format!(
        "PRETTY_NAME=\"{}\"\nNAME=\"{}\"\nVERSION_ID=\"24.04\"\nVERSION=\"24.04 Neon Assault\"\nVERSION_CODENAME=noble\nID=aurora-neon\nID_LIKE=ubuntu debian\nHOME_URL=\"https://aurora.local\"\nSUPPORT_URL=\"https://aurora.local/support\"\nBUG_REPORT_URL=\"https://aurora.local/issues\"\nPRIVACY_POLICY_URL=\"https://aurora.local/privacy\"\nUBUNTU_CODENAME=noble\nLOGO=distributor-logo\n",
        config.branding_name, config.branding_name
    )
}

fn lsb_release_content(config: &BuildConfig) -> String {
    format!(
        "DISTRIB_ID={}\nDISTRIB_RELEASE=24.04\nDISTRIB_CODENAME=noble\nDISTRIB_DESCRIPTION=\"{}\"\n",
        config.branding_name.replace(' ', "_"),
        config.branding_name
    )
}

fn performance_shell_script(profile: &PerformanceProfile) -> String {
    format!(
        "#!/usr/bin/env bash\nexport AURORA_INSTALL_MODE=\"{}\"\nexport AURORA_PROFILE_NAME=\"{}\"\nexport AURORA_CPU_GOVERNOR=\"{}\"\nexport AURORA_ENABLE_HUGEPAGES=\"{}\"\nexport AURORA_ENABLE_GAMEMODE=\"{}\"\nexport AURORA_ENABLE_MANGOHUD=\"{}\"\nexport AURORA_ENABLE_ZRAM=\"{}\"\nexport AURORA_ZRAM_FRACTION_PERCENT=\"{}\"\nexport AURORA_USE_TMPFS_FOR_TEMP=\"{}\"\nexport AURORA_DISABLE_UNNEEDED_SERVICES=\"{}\"\nexport AURORA_ENABLE_PRELOAD=\"{}\"\nexport AURORA_IO_SCHEDULER=\"{}\"\nexport AURORA_SWAP_POLICY=\"{}\"\nexport AURORA_READAHEAD_KB=\"{}\"\nexport AURORA_VM_SWAPPINESS=\"{}\"\nexport AURORA_VM_DIRTY_RATIO=\"{}\"\nexport AURORA_VM_DIRTY_BACKGROUND_RATIO=\"{}\"\nexport AURORA_DISABLE_CPU_IDLE=\"{}\"\nexport AURORA_SCHEDULER_TUNE=\"{}\"\nexport AURORA_CPU_ENERGY_POLICY=\"{}\"\n",
        install_mode_name(&profile.mode),
        profile.name,
        profile.cpu_governor,
        profile.enable_hugepages,
        profile.enable_gamemode,
        profile.enable_mangohud,
        profile.enable_zram,
        profile.zram_fraction_percent,
        profile.use_tmpfs_for_temp,
        profile.disable_unneeded_services,
        profile.enable_preload,
        profile.io_scheduler,
        profile.swap_partition_policy,
        profile.readahead_kb,
        profile.vm_swappiness,
        profile.vm_dirty_ratio,
        profile.vm_dirty_background_ratio,
        profile.disable_cpu_idle,
        profile.scheduler_tune,
        profile.cpu_energy_policy,
    )
}

fn performance_sysctl_conf(profile: &PerformanceProfile) -> String {
    format!(
        "vm.swappiness={}\nvm.dirty_ratio={}\nvm.dirty_background_ratio={}\nvm.vfs_cache_pressure=50\nvm.page-cluster=0\nkernel.numa_balancing=1\nkernel.sched_autogroup_enabled=0\nvm.max_map_count=1048576\nvm.watermark_boost_factor=0\nvm.watermark_scale_factor=125\nvm.compaction_proactiveness=20\nvm.min_free_kbytes=131072\nkernel.kptr_restrict=2\nkernel.dmesg_restrict=1\nkernel.unprivileged_bpf_disabled=1\nkernel.yama.ptrace_scope=1\nfs.protected_fifos=2\nfs.protected_regular=2\nnet.ipv4.tcp_fastopen=3\nnet.core.default_qdisc=fq_pie\n",
        profile.vm_swappiness,
        profile.vm_dirty_ratio,
        profile.vm_dirty_background_ratio,
    )
}

fn performance_defaults(profile: &PerformanceProfile) -> String {
    format!(
        "AURORA_INSTALL_MODE={}\nAURORA_PROFILE_NAME={}\nAURORA_CPU_GOVERNOR={}\nAURORA_ENABLE_ZRAM={}\nAURORA_ZRAM_FRACTION_PERCENT={}\nAURORA_ENABLE_PRELOAD={}\nAURORA_IO_SCHEDULER={}\nAURORA_READAHEAD_KB={}\nAURORA_DISABLE_UNNEEDED_SERVICES={}\nAURORA_SWAP_POLICY={}\nAURORA_VM_SWAPPINESS={}\nAURORA_VM_DIRTY_RATIO={}\nAURORA_VM_DIRTY_BACKGROUND_RATIO={}\nAURORA_DISABLE_CPU_IDLE={}\nAURORA_SCHEDULER_TUNE={}\nAURORA_CPU_ENERGY_POLICY={}\nAURORA_ENABLE_FIREWALL=true\nAURORA_ENABLE_FAIL2BAN=true\nAURORA_ENABLE_THERMALD=true\nAURORA_ENABLE_IRQBALANCE=true\nAURORA_ENABLE_EARLYOOM=true\nAURORA_ENABLE_DIRECTSTREAM=true\nAURORA_ENABLE_DNS_OVER_TLS=true\nAURORA_ENABLE_MAC_RANDOMIZATION=true\nAURORA_ENABLE_PRIVACY_TOOLS=true\n",
        install_mode_name(&profile.mode),
        profile.name,
        profile.cpu_governor,
        profile.enable_zram,
        profile.zram_fraction_percent,
        profile.enable_preload,
        profile.io_scheduler,
        profile.readahead_kb,
        profile.disable_unneeded_services,
        profile.swap_partition_policy,
        profile.vm_swappiness,
        profile.vm_dirty_ratio,
        profile.vm_dirty_background_ratio,
        profile.disable_cpu_idle,
        profile.scheduler_tune,
        profile.cpu_energy_policy
    )
}

fn autosetup_script() -> String {
    "#!/usr/bin/env bash\nset -euo pipefail\nCONFIG=/etc/default/aurora-performance\nif [ -f \"$CONFIG\" ]; then\n  # shellcheck disable=SC1091\n  . \"$CONFIG\"\nfi\nfor cpu in /sys/devices/system/cpu/cpu[0-9]*; do\n  if [ -w \"$cpu/cpufreq/scaling_governor\" ]; then\n    printf '%s' \"${AURORA_CPU_GOVERNOR:-performance}\" > \"$cpu/cpufreq/scaling_governor\" || true\n  fi\n  if [ -w \"$cpu/cpufreq/energy_performance_preference\" ]; then\n    printf '%s' \"${AURORA_CPU_ENERGY_POLICY:-performance}\" > \"$cpu/cpufreq/energy_performance_preference\" || true\n  fi\n done\nif [ \"${AURORA_USE_TMPFS_FOR_TEMP:-true}\" = \"true\" ]; then\n  grep -q '^tmpfs /tmp tmpfs' /etc/fstab || printf 'tmpfs /tmp tmpfs defaults,noatime,mode=1777 0 0\\n' >> /etc/fstab\nfi\nif [ \"${AURORA_DISABLE_UNNEEDED_SERVICES:-true}\" = \"true\" ] && command -v systemctl >/dev/null 2>&1; then\n  for svc in bluetooth cups apport whoopsie snapd ModemManager; do\n    systemctl disable \"$svc\" >/dev/null 2>&1 || true\n    systemctl stop \"$svc\" >/dev/null 2>&1 || true\n  done\nfi\nif [ -n \"${AURORA_READAHEAD_KB:-}\" ]; then\n  for block in /sys/block/*/queue/read_ahead_kb; do\n    [ -w \"$block\" ] && printf '%s' \"$AURORA_READAHEAD_KB\" > \"$block\" || true\n  done\nfi\nif [ -n \"${AURORA_IO_SCHEDULER:-}\" ]; then\n  for sched in /sys/block/*/queue/scheduler; do\n    [ -w \"$sched\" ] && grep -qw \"$AURORA_IO_SCHEDULER\" \"$sched\" && printf '%s' \"$AURORA_IO_SCHEDULER\" > \"$sched\" || true\n  done\nfi\nif [ \"${AURORA_DISABLE_CPU_IDLE:-false}\" = \"true\" ] && [ -w /sys/module/intel_idle/parameters/max_cstate ]; then\n  printf '1' > /sys/module/intel_idle/parameters/max_cstate || true\nfi\nif [ \"${AURORA_ENABLE_MAC_RANDOMIZATION:-true}\" = \"true\" ]; then\n  mkdir -p /etc/NetworkManager/conf.d\n  cat > /etc/NetworkManager/conf.d/90-aurora-privacy.conf <<'EOF'\n[device]\nwifi.scan-rand-mac-address=yes\n[connection]\nethernet.cloned-mac-address=stable\nwifi.cloned-mac-address=stable-ssid\nEOF\nfi\nif [ \"${AURORA_ENABLE_DNS_OVER_TLS:-true}\" = \"true\" ]; then\n  mkdir -p /etc/systemd/resolved.conf.d\n  cat > /etc/systemd/resolved.conf.d/90-aurora-privacy.conf <<'EOF'\n[Resolve]\nDNSOverTLS=opportunistic\nMulticastDNS=no\nLLMNR=no\nEOF\nfi\nif [ \"${AURORA_ENABLE_ZRAM:-true}\" = \"true\" ] && command -v systemctl >/dev/null 2>&1; then\n  systemctl enable aurora-zram-setup.service >/dev/null 2>&1 || true\n  systemctl start aurora-zram-setup.service >/dev/null 2>&1 || true\nfi\nif [ \"${AURORA_ENABLE_IRQBALANCE:-true}\" = \"true\" ] && command -v systemctl >/dev/null 2>&1; then\n  systemctl enable irqbalance >/dev/null 2>&1 || true\n  systemctl start irqbalance >/dev/null 2>&1 || true\nfi\nif [ \"${AURORA_ENABLE_THERMALD:-true}\" = \"true\" ] && command -v systemctl >/dev/null 2>&1; then\n  systemctl enable thermald >/dev/null 2>&1 || true\n  systemctl start thermald >/dev/null 2>&1 || true\nfi\nif [ \"${AURORA_ENABLE_EARLYOOM:-true}\" = \"true\" ] && command -v systemctl >/dev/null 2>&1; then\n  systemctl enable earlyoom >/dev/null 2>&1 || true\n  systemctl start earlyoom >/dev/null 2>&1 || true\nfi\nif [ \"${AURORA_ENABLE_FAIL2BAN:-true}\" = \"true\" ] && command -v systemctl >/dev/null 2>&1; then\n  systemctl enable fail2ban >/dev/null 2>&1 || true\n  systemctl start fail2ban >/dev/null 2>&1 || true\nfi\nif [ \"${AURORA_ENABLE_FIREWALL:-true}\" = \"true\" ] && command -v ufw >/dev/null 2>&1; then\n  ufw default deny incoming >/dev/null 2>&1 || true\n  ufw default allow outgoing >/dev/null 2>&1 || true\n  ufw --force enable >/dev/null 2>&1 || true\nfi\ncommand -v systemctl >/dev/null 2>&1 && systemctl enable fstrim.timer >/dev/null 2>&1 || true\ncommand -v systemctl >/dev/null 2>&1 && systemctl restart systemd-resolved NetworkManager >/dev/null 2>&1 || true\nsysctl --system >/dev/null 2>&1 || true\n".to_string()
}

fn desktop_setup_script(config: &BuildConfig) -> String {
    format!(
        "#!/usr/bin/env bash\nset -euo pipefail\nexport XDG_CURRENT_DESKTOP=\"${{XDG_CURRENT_DESKTOP:-GNOME}}\"\nWALL='/usr/share/backgrounds/aurora/gaming-kali.svg'\nrun_gsettings() {{\n  if command -v gsettings >/dev/null 2>&1; then\n    gsettings set \"$1\" \"$2\" \"$3\" >/dev/null 2>&1 || true\n  fi\n}}\nrun_gsettings org.gnome.desktop.interface gtk-theme 'Aurora-Neon'\nrun_gsettings org.gnome.desktop.interface icon-theme 'Papirus'\nrun_gsettings org.gnome.desktop.interface color-scheme 'prefer-dark'\nrun_gsettings org.gnome.desktop.interface monospace-font-name 'Cascadia Code 11'\nrun_gsettings org.gnome.desktop.interface enable-hot-corners false\nrun_gsettings org.gnome.desktop.background picture-uri \"file://${{WALL}}\"\nrun_gsettings org.gnome.desktop.background picture-uri-dark \"file://${{WALL}}\"\nrun_gsettings org.gnome.desktop.wm.preferences button-layout 'appmenu:minimize,maximize,close'\nrun_gsettings org.gnome.mutter overlay-key 'Super_L'\nrun_gsettings org.gnome.shell favorite-apps \"['org.gnome.Nautilus.desktop','org.gnome.Terminal.desktop','alacritty.desktop','firefox_firefox.desktop','org.gnome.Software.desktop','aurora-control-center.desktop','aurora-dev-lab.desktop','aurora-creator-studio.desktop','aurora-ai-hub.desktop','aurora-installer.desktop']\"\nrun_gsettings org.gnome.shell enabled-extensions \"['apps-menu@gnome-shell-extensions.gcampax.github.com','places-menu@gnome-shell-extensions.gcampax.github.com','user-theme@gnome-shell-extensions.gcampax.github.com']\"\nrun_gsettings org.gnome.shell.extensions.dash-to-dock dock-position 'BOTTOM'\nrun_gsettings org.gnome.shell.extensions.dash-to-dock extend-height false\nrun_gsettings org.gnome.shell.extensions.dash-to-dock show-trash true\nrun_gsettings org.gnome.shell.extensions.dash-to-dock show-mounts true\nrun_gsettings org.gnome.shell.extensions.dash-to-dock show-apps-at-top true\nmkdir -p \"$HOME/.config\"\ncat > \"$HOME/.config/aurora-desktop-summary\" <<'EOF'\n{name}\nMode-ready GNOME shell\nTheme: Aurora-Neon\nWallpaper: $WALL\nRuntime: /usr/local/bin/aurora\nSecurity: ufw fail2ban apparmor earlyoom thermald irqbalance\nStudios: Dev Lab + Creator Studio + AI Hub\nEOF\n",
        name = config.branding_name
    )
}

fn mode_switch_script() -> String {
    "#!/usr/bin/env bash\nset -euo pipefail\nMODE=\"${1:-gaming}\"\nCONFIG=/etc/default/aurora-performance\nset_key() {\n  local key=\"$1\"\n  local value=\"$2\"\n  grep -q \"^${key}=\" \"$CONFIG\" && sed -i \"s|^${key}=.*|${key}=${value}|\" \"$CONFIG\" || printf '%s=%s\\n' \"$key\" \"$value\" >> \"$CONFIG\"\n}\ncase \"$MODE\" in\n  balanced)\n    set_key AURORA_CPU_GOVERNOR schedutil\n    set_key AURORA_READAHEAD_KB 1024\n    set_key AURORA_VM_SWAPPINESS 25\n    set_key AURORA_IO_SCHEDULER mq-deadline\n    ;;\n  gaming)\n    set_key AURORA_CPU_GOVERNOR performance\n    set_key AURORA_READAHEAD_KB 2048\n    set_key AURORA_VM_SWAPPINESS 18\n    set_key AURORA_IO_SCHEDULER none\n    ;;\n  creator)\n    set_key AURORA_CPU_GOVERNOR performance\n    set_key AURORA_READAHEAD_KB 3072\n    set_key AURORA_VM_SWAPPINESS 14\n    set_key AURORA_IO_SCHEDULER mq-deadline\n    ;;\n  training)\n    set_key AURORA_CPU_GOVERNOR performance\n    set_key AURORA_READAHEAD_KB 4096\n    set_key AURORA_VM_SWAPPINESS 10\n    set_key AURORA_IO_SCHEDULER none\n    set_key AURORA_DISABLE_CPU_IDLE true\n    ;;\n  datacenter)\n    set_key AURORA_CPU_GOVERNOR performance\n    set_key AURORA_READAHEAD_KB 1024\n    set_key AURORA_VM_SWAPPINESS 8\n    set_key AURORA_IO_SCHEDULER mq-deadline\n    set_key AURORA_DISABLE_UNNEEDED_SERVICES true\n    ;;\n  max-throughput)\n    set_key AURORA_CPU_GOVERNOR performance\n    set_key AURORA_READAHEAD_KB 4096\n    set_key AURORA_VM_SWAPPINESS 15\n    set_key AURORA_DISABLE_CPU_IDLE true\n    set_key AURORA_IO_SCHEDULER none\n    ;;\n  *)\n    echo \"unknown mode: $MODE\" >&2\n    exit 1\n    ;;\n esac\nset_key AURORA_INSTALL_MODE \"$MODE\"\nset_key AURORA_PROFILE_NAME \"$MODE\"\nif command -v systemctl >/dev/null 2>&1; then\n  systemctl restart aurora-autosetup.service >/dev/null 2>&1 || true\nfi\necho \"Aurora mode switched to $MODE\"\n".to_string()
}

fn ai_setup_script() -> String {
    "#!/usr/bin/env bash\nset -euo pipefail\nFIRSTBOOT=\"${1:-}\"\nif command -v systemctl >/dev/null 2>&1; then\n  systemctl enable ollama >/dev/null 2>&1 || true\n  [ \"$FIRSTBOOT\" = \"--firstboot\" ] && systemctl start ollama >/dev/null 2>&1 || true\nfi\nmkdir -p \"$HOME/.config/aurora\"\ncat > \"$HOME/.config/aurora/ai-hub.conf\" <<'EOF'\nprovider=ollama\ndefault_model=llama3\nsearch_integration=aurora-ai-helper\nfile_manager_actions=planned\naccelerators=auto-detect\nEOF\nif command -v ollama >/dev/null 2>&1; then\n  echo 'Aurora AI Hub configured for Ollama. Suggested next step: ollama pull llama3'\nelse\n  echo 'Ollama not available in this image build.'\nfi\n".to_string()
}

fn dev_setup_script() -> String {
    "#!/usr/bin/env bash\nset -euo pipefail\nmkdir -p \"$HOME/.config/aurora\"\ncat > \"$HOME/.config/aurora/dev-lab.conf\" <<'EOF'\ncontainer_runtime=podman\nworkspace_shell=tmux\nbenchmark_tool=hyperfine\nrepo_acceleration=git-lfs\nEOF\nif command -v systemctl >/dev/null 2>&1; then\n  systemctl enable libvirtd >/dev/null 2>&1 || true\n  systemctl start libvirtd >/dev/null 2>&1 || true\nfi\ncommand -v git-lfs >/dev/null 2>&1 && git lfs install >/dev/null 2>&1 || true\necho 'Aurora Dev Lab workspace applied.'\n".to_string()
}

fn creator_setup_script() -> String {
    "#!/usr/bin/env bash\nset -euo pipefail\nmkdir -p \"$HOME/Videos\" \"$HOME/Pictures\" \"$HOME/Projects\" \"$HOME/.cache/aurora/preview\"\nmkdir -p \"$HOME/.config/aurora\"\ncat > \"$HOME/.config/aurora/creator-studio.conf\" <<'EOF'\nrecording=obs-studio\nvideo_pipeline=ffmpeg\nraster_editor=gimp\nvector_editor=inkscape\nvideo_editor=kdenlive\n3d_suite=blender\noffice_suite=libreoffice\npreview_cache=$HOME/.cache/aurora/preview\nEOF\necho 'Aurora Creator Studio workspace applied.'\n".to_string()
}

fn gaming_setup_script() -> String {
    "#!/usr/bin/env bash\nset -euo pipefail\nsysctl_conf=/etc/sysctl.d/99-aurora-gaming.conf\nif [ -f \"$sysctl_conf\" ]; then\n  grep -q '^vm.max_map_count=' \"$sysctl_conf\" || printf 'vm.max_map_count=2147483642\\n' >> \"$sysctl_conf\"\n  grep -q '^fs.file-max=' \"$sysctl_conf\" || printf 'fs.file-max=2097152\\n' >> \"$sysctl_conf\"\nfi\nmkdir -p \"$HOME/.config/aurora\"\ncat > \"$HOME/.config/aurora/gaming-center.conf\" <<'EOF'\nwayland_default=true\nnvidia_wayland=preferred\ngamescope=enabled_when_available\nmangohud=enabled_when_available\nanti_cheat_posture=steam_proton_focus\ndirectstream=enabled\nEOF\ncommand -v aurora-directstream >/dev/null 2>&1 && aurora-directstream --prime || true\nsysctl --system >/dev/null 2>&1 || true\necho 'Aurora Gaming Center profile applied.'\n".to_string()
}

fn training_setup_script() -> String {
    "#!/usr/bin/env bash\nset -euo pipefail\nmkdir -p \"$HOME/.cache/aurora/datasets\" \"$HOME/.cache/aurora/checkpoints\" \"$HOME/.config/aurora\"\ncat > \"$HOME/.config/aurora/training.conf\" <<'EOF'\ndataset_cache=$HOME/.cache/aurora/datasets\ncheckpoint_dir=$HOME/.cache/aurora/checkpoints\npython_env=venv\nnuma_policy=preferred\ngpu_orchestration=mixed-when-available\nEOF\ncommand -v python3 >/dev/null 2>&1 && python3 -m venv \"$HOME/.cache/aurora/venv\" >/dev/null 2>&1 || true\ncommand -v valkey-server >/dev/null 2>&1 && command -v systemctl >/dev/null 2>&1 && systemctl enable valkey-server >/dev/null 2>&1 || true\necho 'Aurora Training profile applied.'\n".to_string()
}

fn datacenter_setup_script() -> String {
    "#!/usr/bin/env bash\nset -euo pipefail\nmkdir -p \"$HOME/.config/aurora\"\ncat > \"$HOME/.config/aurora/datacenter.conf\" <<'EOF'\nservice_isolation=enabled\nobservability=prometheus-node-exporter\nstorage_validation=fio\nio_watch=iotop\nsched_profile=service-isolated\nEOF\nif command -v systemctl >/dev/null 2>&1; then\n  systemctl enable prometheus-node-exporter >/dev/null 2>&1 || true\n  systemctl start prometheus-node-exporter >/dev/null 2>&1 || true\nfi\necho 'Aurora Datacenter profile applied.'\n".to_string()
}

fn gpu_setup_script() -> String {
    "#!/usr/bin/env bash\nset -euo pipefail\nFIRSTBOOT=\"${1:-}\"\nmkdir -p /etc/aurora \"$HOME/.config/aurora\"\nGPU_VENDOR=cpu-only\nif command -v nvidia-smi >/dev/null 2>&1; then\n  GPU_VENDOR=nvidia\nelif lspci 2>/dev/null | grep -Eqi 'vga|3d|display' && lspci 2>/dev/null | grep -qi 'AMD/ATI'; then\n  GPU_VENDOR=amd\nelif lspci 2>/dev/null | grep -Eqi 'vga|3d|display' && lspci 2>/dev/null | grep -qi 'Intel'; then\n  GPU_VENDOR=intel\nfi\ncat > /etc/aurora/gpu-profile.conf <<EOF\nvendor=${GPU_VENDOR}\nmesa_vulkan=enabled_when_available\nopencl=enabled_when_available\ncuda=enabled_when_available\nrocm=enabled_when_available\nEOF\ncat > \"$HOME/.config/aurora/gpu-profile.conf\" <<EOF\nvendor=${GPU_VENDOR}\nmode=auto\nEOF\nif [ \"$GPU_VENDOR\" = nvidia ] && command -v systemctl >/dev/null 2>&1; then\n  systemctl enable nvidia-persistenced >/dev/null 2>&1 || true\n  [ \"$FIRSTBOOT\" = \"--firstboot\" ] && systemctl start nvidia-persistenced >/dev/null 2>&1 || true\nfi\necho \"Aurora GPU posture prepared for ${GPU_VENDOR}\"\n".to_string()
}

fn ai_helper_script() -> String {
    "#!/usr/bin/env bash\nset -euo pipefail\nACTION=\"${1:-chat}\"\nshift || true\nMODEL=\"${AURORA_AI_MODEL:-llama3}\"\nPROMPT=\"${*:-}\"\ncase \"$ACTION\" in\n  status)\n    if command -v ollama >/dev/null 2>&1; then\n      echo \"Aurora AI Helper ready with Ollama model ${MODEL}\"\n    else\n      echo \"Aurora AI Helper installed but Ollama is unavailable\"\n    fi\n    ;;\n  chat)\n    if ! command -v ollama >/dev/null 2>&1; then\n      echo 'ollama not installed or not in PATH' >&2\n      exit 1\n    fi\n    if [ -z \"$PROMPT\" ]; then\n      PROMPT='Summarize this Aurora system and suggest the best workload mode.'\n    fi\n    ollama run \"$MODEL\" \"$PROMPT\"\n    ;;\n  *)\n    echo \"unknown action: $ACTION\" >&2\n    exit 1\n    ;;\n esac\n".to_string()
}

fn hardware_guard_script() -> String {
    "#!/usr/bin/env bash\nset -euo pipefail\nACTION=\"${1:-status}\"\nSTATE_DIR=/etc/aurora\nSTATE_FILE=${STATE_DIR}/hardware.identity\nmkdir -p \"$STATE_DIR\"\nfingerprint() {\n  local machine_id product_uuid board serial\n  machine_id=$(cat /etc/machine-id 2>/dev/null || true)\n  product_uuid=$(cat /sys/class/dmi/id/product_uuid 2>/dev/null || true)\n  board=$(cat /sys/class/dmi/id/board_name 2>/dev/null || true)\n  serial=$(cat /sys/class/dmi/id/product_serial 2>/dev/null || true)\n  printf '%s|%s|%s|%s' \"$machine_id\" \"$product_uuid\" \"$board\" \"$serial\" | sha256sum | awk '{print $1}'\n}\ncurrent=$(fingerprint)\ncase \"$ACTION\" in\n  --seal|seal)\n    if [ ! -f \"$STATE_FILE\" ]; then\n      printf '%s\\n' \"$current\" > \"$STATE_FILE\"\n      echo 'Aurora hardware identity sealed.'\n    else\n      echo 'Aurora hardware identity already sealed.'\n    fi\n    ;;\n  --check|check|status)\n    if [ -f \"$STATE_FILE\" ]; then\n      sealed=$(cat \"$STATE_FILE\")\n      if [ \"$sealed\" != \"$current\" ]; then\n        echo 'warning: Aurora hardware identity mismatch detected; image may have been moved or cloned.' >&2\n        exit 2\n      fi\n      echo 'Aurora hardware identity matches this machine.'\n    else\n      echo 'Aurora hardware identity not sealed yet.'\n    fi\n    ;;\n  *)\n    echo \"unknown action: $ACTION\" >&2\n    exit 1\n    ;;\n esac\n".to_string()
}

fn directstream_defaults() -> String {
    "AURORA_DIRECTSTREAM_READAHEAD_KB=4096\nAURORA_DIRECTSTREAM_CACHE_DIR=/var/cache/aurora/directstream\nAURORA_DIRECTSTREAM_IO_NICE=2\nAURORA_DIRECTSTREAM_PRIME_ON_BOOT=true\n".to_string()
}

fn directstream_script() -> String {
    "#!/usr/bin/env bash\nset -euo pipefail\nCONFIG=/etc/aurora/directstream.conf\n[ -f \"$CONFIG\" ] && . \"$CONFIG\"\nCACHE_DIR=\"${AURORA_DIRECTSTREAM_CACHE_DIR:-/var/cache/aurora/directstream}\"\nREADAHEAD=\"${AURORA_DIRECTSTREAM_READAHEAD_KB:-4096}\"\nIONICE_CLASS=\"${AURORA_DIRECTSTREAM_IO_NICE:-2}\"\nmkdir -p \"$CACHE_DIR\"\nfor block in /sys/block/*/queue/read_ahead_kb; do\n  [ -w \"$block\" ] && printf '%s' \"$READAHEAD\" > \"$block\" || true\ndone\ncommand -v ionice >/dev/null 2>&1 && ionice -c 2 -n \"$IONICE_CLASS\" true >/dev/null 2>&1 || true\nif [ \"${1:-}\" = \"--prime\" ]; then\n  find /usr /opt -maxdepth 3 \\( -name '*.so' -o -name '*.vkd3d' -o -name '*.dxvk' -o -name '*.pak' \\) -type f 2>/dev/null | head -n 400 | xargs -r cat >/dev/null 2>&1 || true\nfi\necho \"Aurora DirectStream applied with readahead ${READAHEAD} KB and cache dir ${CACHE_DIR}\"\n".to_string()
}

fn security_setup_script() -> String {
    "#!/usr/bin/env bash\nset -euo pipefail\nif command -v systemctl >/dev/null 2>&1; then\n  systemctl enable fail2ban >/dev/null 2>&1 || true\n  systemctl enable apparmor >/dev/null 2>&1 || true\n  systemctl enable kdump-tools >/dev/null 2>&1 || true\n  [ \"${1:-}\" = \"--firstboot\" ] && systemctl start fail2ban >/dev/null 2>&1 || true\nfi\ncommand -v ufw >/dev/null 2>&1 && ufw --force enable >/dev/null 2>&1 || true\nmkdir -p \"$HOME/.config/aurora\"\ncat > \"$HOME/.config/aurora/security-center.conf\" <<'EOF'\nfirewall=enabled\nfail2ban=enabled\napparmor=enabled\nkdump=enabled_if_available\nauthd=prepared_if_available\nEOF\necho 'Aurora Security Center profile applied.'\n".to_string()
}

fn privacy_setup_script() -> String {
    "#!/usr/bin/env bash\nset -euo pipefail\nFIRSTBOOT=\"${1:-}\"\nmkdir -p /etc/NetworkManager/conf.d /etc/systemd/resolved.conf.d \"$HOME/.config/aurora\"\ncat > /etc/NetworkManager/conf.d/90-aurora-privacy.conf <<'EOF'\n[device]\nwifi.scan-rand-mac-address=yes\n[connection]\nethernet.cloned-mac-address=stable\nwifi.cloned-mac-address=stable-ssid\nEOF\ncat > /etc/systemd/resolved.conf.d/90-aurora-privacy.conf <<'EOF'\n[Resolve]\nDNSOverTLS=opportunistic\nMulticastDNS=no\nLLMNR=no\nEOF\ncat > \"$HOME/.config/aurora/privacy.conf\" <<'EOF'\ndns_over_tls=opportunistic\nmac_randomization=enabled\ntor_tools=enabled_when_available\nusbguard=enabled_when_available\nEOF\nif command -v systemctl >/dev/null 2>&1; then\n  systemctl enable usbguard >/dev/null 2>&1 || true\n  systemctl enable tor >/dev/null 2>&1 || true\n  [ \"$FIRSTBOOT\" = \"--firstboot\" ] && systemctl start usbguard >/dev/null 2>&1 || true\n  [ \"$FIRSTBOOT\" = \"--firstboot\" ] && systemctl restart systemd-resolved NetworkManager >/dev/null 2>&1 || true\nfi\necho 'Aurora Privacy profile applied.'\n".to_string()
}

fn dev_lab_html(name: &str, accent: &str) -> String {
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>{name} Dev Lab</title><link rel=\"stylesheet\" href=\"dev-lab.css\"></head><body><main class=\"wrap\"><section class=\"hero\"><p class=\"eyebrow\">Aurora Engineering Surface</p><h1>{name} Dev Lab</h1><p class=\"lede\">Container-native workflows, terminal acceleration, repo hygiene tools, and virtualization support are staged directly into the distro instead of being an afterthought.</p></section><section class=\"grid\"><article><h2>Containers</h2><p>Podman, Distrobox, libvirt, and QEMU give the image a workstation-grade build and test posture.</p></article><article><h2>Shell Flow</h2><p>tmux, direnv, shellcheck, hyperfine, ripgrep, fd, and bat are included for fast local iteration.</p></article><article><h2>Repo Work</h2><p>Git LFS, benchmarking, and quality checks are part of the default Aurora stack.</p></article></section></main></body></html>"
    )
}

fn dev_lab_css(accent: &str) -> String {
    format!(
        ":root{{--accent:{accent};--bg:#06101a;--panel:#0f1d2b;--line:#193550;--text:#eef7ff;--muted:#9bb3c9}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at top,#14304d,#06101a 58%);color:var(--text);font-family:'Orbitron',sans-serif}}.wrap{{max-width:1080px;margin:0 auto;padding:40px 24px 72px}}.hero{{padding:34px;border-radius:26px;border:1px solid var(--line);background:linear-gradient(135deg,rgba(14,30,46,.94),rgba(7,12,20,.96))}}.eyebrow{{text-transform:uppercase;letter-spacing:.22em;color:var(--accent);font-size:12px}}h1{{margin:12px 0 14px;font-size:58px}}.lede{{max-width:760px;color:var(--muted);font-size:22px;font-family:'Rajdhani',sans-serif}}.grid{{margin-top:22px;display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:18px}}article{{padding:22px;border-radius:22px;border:1px solid var(--line);background:rgba(10,18,29,.88)}}h2{{margin:0 0 8px}}p{{margin:0;color:var(--muted);font-family:'Rajdhani',sans-serif;font-size:20px}}"
    )
}

fn creator_studio_html(name: &str, accent: &str) -> String {
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>{name} Creator Studio</title><link rel=\"stylesheet\" href=\"creator-studio.css\"></head><body><main class=\"wrap\"><section class=\"hero\"><p class=\"eyebrow\">Aurora Media Surface</p><h1>{name} Creator Studio</h1><p class=\"lede\">Recording, editing, graphics, office work, and production-ready defaults are layered into the image so the live ISO feels like a real workstation.</p></section><section class=\"grid\"><article><h2>Capture</h2><p>OBS Studio and FFmpeg are staged for streaming, screen recording, and quick export workflows.</p></article><article><h2>Design</h2><p>GIMP and Inkscape cover raster and vector work out of the box.</p></article><article><h2>Delivery</h2><p>LibreOffice, VLC, and Flatpak-ready distribution keep the system useful for both production and review.</p></article></section></main></body></html>"
    )
}

fn creator_studio_css(accent: &str) -> String {
    format!(
        ":root{{--accent:{accent};--bg:#11060c;--panel:#24101a;--line:#4a1832;--text:#fff1f5;--muted:#d2aebb}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at top,#4a1832,#11060c 62%);color:var(--text);font-family:'Orbitron',sans-serif}}.wrap{{max-width:1080px;margin:0 auto;padding:40px 24px 72px}}.hero{{padding:34px;border-radius:26px;border:1px solid var(--line);background:linear-gradient(135deg,rgba(37,12,24,.94),rgba(14,7,11,.97))}}.eyebrow{{text-transform:uppercase;letter-spacing:.22em;color:var(--accent);font-size:12px}}h1{{margin:12px 0 14px;font-size:58px}}.lede{{max-width:760px;color:var(--muted);font-size:22px;font-family:'Rajdhani',sans-serif}}.grid{{margin-top:22px;display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:18px}}article{{padding:22px;border-radius:22px;border:1px solid var(--line);background:rgba(31,10,18,.88)}}h2{{margin:0 0 8px}}p{{margin:0;color:var(--muted);font-family:'Rajdhani',sans-serif;font-size:20px}}"
    )
}

fn aurora_apt_script() -> String {
    "#!/usr/bin/env bash\nset -euo pipefail\nexport APT_CONFIG=/dev/null\nif [ \"$#\" -eq 0 ]; then\n  echo 'usage: aurora-apt <apt-args>' >&2\n  exit 1\nfi\napt -o APT::Color=1 -o Dpkg::Progress-Fancy=1 \"$@\"\n".to_string()
}

fn autosetup_service() -> String {
    "[Unit]\nDescription=AURORA automatic performance setup\nAfter=multi-user.target\n\n[Service]\nType=oneshot\nExecStart=/usr/local/bin/aurora-autosetup\nRemainAfterExit=yes\n\n[Install]\nWantedBy=multi-user.target\n".to_string()
}

fn aurora_inference_service() -> String {
    "[Unit]\nDescription=AURORA local inference API\nAfter=network-online.target aurora-runtime.service ollama.service\nWants=network-online.target\n\n[Service]\nType=simple\nExecStart=/usr/local/bin/aurora api --action serve --bind 127.0.0.1:11435 --model llama3\nRestart=always\nRestartSec=2\nNice=-5\n\n[Install]\nWantedBy=multi-user.target\n".to_string()
}

fn aurora_runtime_service() -> String {
    "[Unit]\nDescription=AURORA persistent runtime executor\nAfter=network-online.target multi-user.target\nWants=network-online.target\n\n[Service]\nType=simple\nExecStart=/usr/local/bin/aurora daemon --interval 5\nRestart=always\nRestartSec=2\nNice=-10\nLimitNOFILE=1048576\nTasksMax=infinity\n\n[Install]\nWantedBy=multi-user.target\n".to_string()
}

fn zram_service() -> String {
    "[Unit]\nDescription=AURORA zram memory boost\nAfter=local-fs.target\n\n[Service]\nType=oneshot\nExecStart=/bin/bash -lc 'modprobe zram || true; echo lz4 > /sys/block/zram0/comp_algorithm 2>/dev/null || true; mem_total_kb=$(awk \"/MemTotal/ {print \\$2}\" /proc/meminfo); size_bytes=$((mem_total_kb * 1024 * 60 / 100)); echo ${size_bytes:-0} > /sys/block/zram0/disksize 2>/dev/null || true; mkswap /dev/zram0 >/dev/null 2>&1 || true; swapon -p 100 /dev/zram0 >/dev/null 2>&1 || true'\nRemainAfterExit=yes\n\n[Install]\nWantedBy=multi-user.target\n".to_string()
}

fn default_wallpaper_svg(name: &str, accent: &str) -> String {
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 1920 1080\"><defs><linearGradient id=\"bg\" x1=\"0\" y1=\"0\" x2=\"1\" y2=\"1\"><stop offset=\"0%\" stop-color=\"#030814\"/><stop offset=\"45%\" stop-color=\"#0b1630\"/><stop offset=\"100%\" stop-color=\"#02050d\"/></linearGradient><radialGradient id=\"aura\" cx=\"50%\" cy=\"45%\" r=\"46%\"><stop offset=\"0%\" stop-color=\"{accent}\" stop-opacity=\"0.8\"><animate attributeName=\"stop-opacity\" values=\"0.35;0.85;0.35\" dur=\"4.2s\" repeatCount=\"indefinite\"/></stop><stop offset=\"55%\" stop-color=\"#63d7ff\" stop-opacity=\"0.26\"/><stop offset=\"100%\" stop-color=\"#030814\" stop-opacity=\"0\"/></radialGradient><linearGradient id=\"blade\" x1=\"0\" y1=\"0\" x2=\"1\" y2=\"1\"><stop offset=\"0%\" stop-color=\"#eff9ff\"/><stop offset=\"100%\" stop-color=\"{accent}\"/></linearGradient></defs><rect width=\"1920\" height=\"1080\" fill=\"url(#bg)\"/><g opacity=\"0.9\"><circle cx=\"975\" cy=\"410\" r=\"330\" fill=\"url(#aura)\"/><path d=\"M912 268c80-48 174-36 241 22 55 48 95 123 101 202 5 75-26 160-78 215-61 64-150 104-250 100-77-4-150-34-206-88-58-56-88-128-91-209-4-102 41-188 117-242 50-36 107-53 166-51Z\" fill=\"#0a1220\" fill-opacity=\"0.9\"/><path d=\"M820 335c63-53 113-86 149-98 49-16 94-17 140-2 33 11 53 27 74 57-43-9-74-2-98 14 27 18 47 43 61 73-44-15-84-11-116 15 25 9 43 25 58 46-41-8-79-1-117 23-33 20-62 48-95 92 10-80 25-137 45-170-26 8-48 24-67 44 9-41 31-77 66-106-36 0-69 6-100 12Z\" fill=\"#d8f4ff\" fill-opacity=\"0.95\"/><path d=\"M1084 460c-36 10-67 30-95 58 44 8 88 25 132 54-66-11-125 2-173 38 66 1 123 15 172 39-73 8-133 30-184 66-42 30-82 70-121 122 28-100 38-180 29-240-28 14-48 35-67 57 7-52 36-97 80-137-37-4-68 1-105 13 55-45 112-75 172-92 60-17 112-10 160 22Z\" fill=\"#bcecff\" fill-opacity=\"0.78\"/><rect x=\"758\" y=\"474\" width=\"440\" height=\"12\" rx=\"6\" fill=\"{accent}\" opacity=\"0.8\"><animate attributeName=\"x\" values=\"758;736;758\" dur=\"3.6s\" repeatCount=\"indefinite\"/></rect><rect x=\"734\" y=\"516\" width=\"488\" height=\"8\" rx=\"4\" fill=\"#eff9ff\" opacity=\"0.7\"><animate attributeName=\"width\" values=\"420;488;420\" dur=\"2.8s\" repeatCount=\"indefinite\"/></rect><path d=\"M1188 332c84 34 146 86 186 155 26 44 39 83 44 133-35-40-71-66-111-78 18 41 24 79 18 118-23-29-48-49-80-60 0 38-9 69-24 100-18-51-48-92-92-121 19-91 35-173 59-247Z\" fill=\"#63d7ff\" fill-opacity=\"0.16\"/></g><g opacity=\"0.65\"><circle cx=\"319\" cy=\"211\" r=\"4\" fill=\"#e8fbff\"><animate attributeName=\"cy\" values=\"211;185;211\" dur=\"3.4s\" repeatCount=\"indefinite\"/></circle><circle cx=\"1547\" cy=\"182\" r=\"5\" fill=\"{accent}\"><animate attributeName=\"cy\" values=\"182;156;182\" dur=\"2.9s\" repeatCount=\"indefinite\"/></circle><circle cx=\"1630\" cy=\"742\" r=\"3\" fill=\"#e8fbff\"><animate attributeName=\"cy\" values=\"742;705;742\" dur=\"4.1s\" repeatCount=\"indefinite\"/></circle><circle cx=\"280\" cy=\"803\" r=\"4\" fill=\"{accent}\"><animate attributeName=\"cy\" values=\"803;771;803\" dur=\"3.1s\" repeatCount=\"indefinite\"/></circle></g><text x=\"110\" y=\"865\" fill=\"#eef7ff\" font-size=\"122\" font-family=\"Orbitron, sans-serif\">{name}</text><text x=\"118\" y=\"932\" fill=\"{accent}\" font-size=\"38\" font-family=\"Rajdhani, sans-serif\">Original anime-inspired acceleration shell</text></svg>"
    )
}

fn autoinstall_yaml(config: &BuildConfig, installer: &InstallerConfig) -> String {
    format!(
        "#cloud-config\nautoinstall:\n  version: 1\n  locale: en_US.UTF-8\n  keyboard:\n    layout: us\n  identity:\n    hostname: aurora\n    username: aurora\n    password: \"$6$rounds=4096$aurora$replace-me-with-a-real-hash\"\n  storage:\n    layout:\n      name: direct\n  ssh:\n    install-server: true\n  packages:\n    - aurora-runtime-tools\n    - aurora-ai-hub\n    - ollama\n    - gamescope\n    - mangohud\n    - obs-studio\n  late-commands:\n    - curtin in-target --target=/target systemctl enable aurora-runtime.service\n    - curtin in-target --target=/target systemctl enable aurora-inference.service\n    - curtin in-target --target=/target systemctl enable aurora-autosetup.service\n    - curtin in-target --target=/target /usr/local/bin/aurora-mode-switch {}\n  user-data:\n    write_files:\n      - path: /etc/aurora-installer/branding.txt\n        content: |\n          {}\n      - path: /etc/aurora-installer/autoinstall-mode.txt\n        content: |\n          {}\n",
        installer.default_mode.slug(),
        config.branding_name,
        installer.default_mode.slug()
    )
}

fn grub_theme_txt(name: &str, _accent: &str) -> String {
    format!(
        "title-text: \"{name}\"\ntitle-font: \"DejaVu Sans Bold 28\"\ntitle-color: \"255,255,255\"\nmessage-font: \"DejaVu Sans 16\"\nmessage-color: \"18,247,255\"\nprogress-bar-fg-color: \"18,247,255\"\nprogress-bar-bg-color: \"16,28,43\"\nselected-item-color: \"255,94,0\"\n"
    )
}

fn default_logo_svg(name: &str, accent: &str) -> String {
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 512 512\"><defs><linearGradient id=\"g\" x1=\"0\" y1=\"0\" x2=\"1\" y2=\"1\"><stop offset=\"0%\" stop-color=\"{accent}\"/><stop offset=\"100%\" stop-color=\"#ff6a00\"/></linearGradient><radialGradient id=\"pulse\" cx=\"50%\" cy=\"45%\" r=\"60%\"><stop offset=\"0%\" stop-color=\"{accent}\" stop-opacity=\"0.55\"><animate attributeName=\"stop-opacity\" values=\"0.2;0.55;0.2\" dur=\"2.8s\" repeatCount=\"indefinite\"/></stop><stop offset=\"100%\" stop-color=\"#09111b\" stop-opacity=\"0\"/></radialGradient></defs><rect width=\"512\" height=\"512\" rx=\"120\" fill=\"#09111b\"/><circle cx=\"256\" cy=\"228\" r=\"180\" fill=\"url(#pulse)\"/><g><animateTransform attributeName=\"transform\" type=\"scale\" values=\"1;1.035;1\" dur=\"2.8s\" repeatCount=\"indefinite\" additive=\"sum\"/><path d=\"M256 72 116 146v109c0 96 61 153 140 185 79-32 140-89 140-185V146L256 72Zm0 53 92 49v76c0 64-37 109-92 136-55-27-92-72-92-136v-76l92-49Zm-41 72h82l45 118h-50l-13-36h-46l-13 36h-50l45-118Zm41 36-12 34h24l-12-34Z\" fill=\"url(#g)\"/></g><rect x=\"108\" y=\"388\" width=\"296\" height=\"6\" rx=\"3\" fill=\"{accent}\" opacity=\"0.7\"><animate attributeName=\"width\" values=\"180;296;180\" dur=\"2.4s\" repeatCount=\"indefinite\"/></rect><text x=\"256\" y=\"438\" text-anchor=\"middle\" fill=\"#eef7ff\" font-size=\"34\" font-family=\"Orbitron, sans-serif\">{name}</text></svg>"
    )
}

fn plymouth_theme_metadata(theme_name: &str) -> String {
    format!(
        "[Plymouth Theme]\nName={theme_name}\nDescription=Neon gaming boot splash for AURORA remaster\nModuleName=script\n"
    )
}

fn plymouth_theme_script(name: &str, accent: &str) -> String {
    format!(
        "Window.SetBackgroundTopColor (0.0, 0.0, 0.0);\nWindow.SetBackgroundBottomColor (0.02, 0.02, 0.03);\nlogo = Image.Text(\"{name}\", 1, 1, 1);\nlogo.SetX(Window.GetWidth() / 2 - logo.GetWidth() / 2);\nlogo.SetY(Window.GetHeight() / 2 - 120);\nphase = Image.Text(\"ANDROMEDA\", 1, 1, 1);\nphase.SetX(Window.GetWidth() / 2 - phase.GetWidth() / 2);\nphase.SetY(Window.GetHeight() / 2 - 34);\ntag = Image.Text(\"cinematic boot sequence\", {r}, {g}, {b});\ntag.SetX(Window.GetWidth() / 2 - tag.GetWidth() / 2);\ntag.SetY(Window.GetHeight() / 2 + 12);\npulse = Image.Text(\"[=======     ]\", {r}, {g}, {b});\npulse.SetX(Window.GetWidth() / 2 - pulse.GetWidth() / 2);\npulse.SetY(Window.GetHeight() / 2 + 56);\nstatus = Image.Text(\"Loading Aurora userspace\", 0.90, 0.94, 1.0);\nstatus.SetX(Window.GetWidth() / 2 - status.GetWidth() / 2);\nstatus.SetY(Window.GetHeight() / 2 + 96);\n",
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

fn recommended_swap_mb(ram_mb: u64, disk_gb: u64) -> u64 {
    let base = if ram_mb <= 8 * 1024 {
        ram_mb.max(4096)
    } else if ram_mb <= 32 * 1024 {
        (ram_mb / 2).max(8192)
    } else {
        (ram_mb / 3).max(12288)
    };
    let disk_cap = disk_gb.saturating_mul(1024) / 5;
    base.min(disk_cap).max(4096)
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
    let paths: Vec<_> = std::env::var_os("PATH")
        .into_iter()
        .flat_map(|value| std::env::split_paths(&value).collect::<Vec<_>>())
        .collect();
    paths
        .into_iter()
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

fn install_mode_name(mode: &InstallMode) -> &'static str {
    match mode {
        InstallMode::Balanced => "balanced",
        InstallMode::Gaming => "gaming",
        InstallMode::Creator => "creator",
        InstallMode::Training => "training",
        InstallMode::Datacenter => "datacenter",
        InstallMode::MaxThroughput => "max-throughput",
    }
}

fn bootloader_steps(firmware: BootMode) -> Vec<String> {
    let mut steps = Vec::new();
    if matches!(firmware, BootMode::Uefi | BootMode::Both) {
        steps.push("Mount the EFI system partition at /boot/efi before running grub-install.".to_string());
        steps.push("Install x86_64-efi GRUB and keep a BOOTX64.EFI fallback in EFI/BOOT.".to_string());
    }
    if matches!(firmware, BootMode::Legacy | BootMode::Both) {
        steps.push("Embed i386-pc GRUB into the target disk so BIOS fallback remains available.".to_string());
    }
    steps.push("Regenerate initramfs and grub.cfg after repair or partition replay.".to_string());
    steps
}

fn recovery_steps(firmware: BootMode) -> Vec<String> {
    let mut steps = vec![
        "Mount the installed root filesystem under /mnt/target-root.".to_string(),
        "Bind /dev, /proc, and /sys into the target root before chroot work.".to_string(),
        "Run filesystem checks before reinstalling bootloaders when corruption is suspected.".to_string(),
    ];
    if matches!(firmware, BootMode::Uefi | BootMode::Both) {
        steps.push("Mount the EFI partition and reinstall UEFI GRUB before refreshing grub.cfg.".to_string());
    }
    if matches!(firmware, BootMode::Legacy | BootMode::Both) {
        steps.push("Reinstall BIOS GRUB to the target disk MBR/embedding area for legacy firmware.".to_string());
    }
    steps
}

fn partition_apply_script(plan: &PartitionPlan) -> String {
    let mut out = String::from(
        "#!/usr/bin/env bash\nset -euo pipefail\nDISK=\"${1:-}\"\nif [ -z \"$DISK\" ]; then\n  echo \"usage: $0 /dev/sdX\" >&2\n  exit 1\nfi\npartprobe \"$DISK\" || true\nwipefs -a \"$DISK\"\nparted -s \"$DISK\" mklabel ",
    );
    out.push_str(&plan.partition_table);
    out.push('\n');

    let mut start_mb = 1u64;
    for (idx, part) in plan.partitions.iter().enumerate() {
        let end_mb = if part.fs == "bios_grub" {
            start_mb + part.size_mb
        } else {
            start_mb + part.size_mb.saturating_sub(1)
        };
        let part_num = idx + 1;
        let part_type = match part.fs.as_str() {
            "fat32" => "fat32",
            "swap" => "linux-swap",
            "bios_grub" => "ext4",
            _ => "ext4",
        };
        out.push_str(&format!(
            "parted -s \"$DISK\" unit MiB mkpart {} {} {}\n",
            part.label, start_mb, end_mb
        ));
        for flag in &part.flags {
            out.push_str(&format!("parted -s \"$DISK\" set {} {} on\n", part_num, flag));
        }
        match part.fs.as_str() {
            "fat32" => out.push_str(&format!("mkfs.fat -F32 \"${{DISK}}{}\"\n", part_num)),
            "swap" => out.push_str(&format!("mkswap \"${{DISK}}{}\"\n", part_num)),
            "bios_grub" => {}
            _ => out.push_str(&format!("mkfs.{} \"${{DISK}}{}\"\n", part_type, part_num)),
        }
        start_mb = end_mb + 1;
    }

    out.push_str("partprobe \"$DISK\" || true\n");
    out
}

fn repair_boot_script() -> String {
    "#!/usr/bin/env bash\nset -euo pipefail\nROOT=\"${1:-/mnt/target-root}\"\nEFI=\"${2:-}\"\nDISK=\"${3:-/dev/sda}\"\nmount --bind /dev \"$ROOT/dev\"\nmount --bind /proc \"$ROOT/proc\"\nmount --bind /sys \"$ROOT/sys\"\nif [ -n \"$EFI\" ]; then\n  mkdir -p \"$ROOT/boot/efi\"\n  mount \"$EFI\" \"$ROOT/boot/efi\"\n  chroot \"$ROOT\" grub-install --target=x86_64-efi --efi-directory=/boot/efi --bootloader-id=AURORA --recheck\n  chroot \"$ROOT\" grub-install --target=i386-pc \"$DISK\" --recheck\nelse\n  chroot \"$ROOT\" grub-install --target=i386-pc \"$DISK\" --recheck\nfi\nchroot \"$ROOT\" /bin/sh -lc 'command -v update-initramfs >/dev/null 2>&1 && update-initramfs -u || true; command -v update-grub >/dev/null 2>&1 && update-grub || grub-mkconfig -o /boot/grub/grub.cfg'\n".to_string()
}
