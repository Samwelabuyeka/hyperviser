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
    GamingCenter,
    SecurityCenter,
    PackageCenter,
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
    MaxThroughput,
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
        out.join("overlay/usr/share/aurora/gaming-center"),
        out.join("overlay/usr/share/aurora/security-center"),
        out.join("overlay/usr/share/aurora/package-center"),
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
            "gamescope".to_string(),
            "goverlay".to_string(),
            "mesa-vulkan-drivers".to_string(),
            "mesa-utils".to_string(),
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
            InstallMode::MaxThroughput,
        ],
        default_mode: InstallMode::Gaming,
    };
    let balanced = performance_profile(InstallMode::Balanced);
    let gaming = performance_profile(InstallMode::Gaming);
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
        out.join("overlay/etc/skel/.config/autostart/aurora-welcome.desktop"),
        aurora_welcome_desktop_file(),
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
        out.join("overlay/home/aurora/.config/autostart/aurora-welcome.desktop"),
        aurora_welcome_desktop_file(),
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
        out.join("overlay/usr/local/bin/aurora-gaming-setup"),
        gaming_setup_script(),
    )?;
    fs::write(
        out.join("overlay/usr/local/bin/aurora-security-setup"),
        security_setup_script(),
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
            &["vlc", "flatpak", "gnome-software-plugin-flatpak", "zellij", "bottom", "just"],
            "Aurora creator and workstation stack with media, packaging, and terminal productivity tooling.",
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
        out.join("overlay/opt/aurora/repo/meta/aurora-scientific-stack.control"),
        aurora_meta_package_control(
            "aurora-scientific-stack",
            &config,
            &["numactl", "linux-tools-generic", "hwloc", "valkey-server", "ripgrep", "fd-find"],
            "Aurora scientific and HPC stack with NUMA, performance counters, and high-speed data tools.",
        ),
    )?;
    fs::write(
        out.join("overlay/usr/local/bin/install-aurora-meta"),
        "#!/usr/bin/env bash\nset -euo pipefail\napt update\napt install -y aurora-desktop aurora-ai-stack aurora-gaming-stack aurora-creator-stack aurora-security-stack aurora-scientific-stack\n".to_string(),
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
        AppSurface::GamingCenter => gaming_center_app(action),
        AppSurface::SecurityCenter => security_center_app(action),
        AppSurface::PackageCenter => package_center_app(action),
        AppSurface::Welcome => welcome_app(action),
    }
}

fn installer_app(action: Option<&str>) -> Result<()> {
    let action = action.unwrap_or("summary");
    match action {
        "summary" => {
            println!("Aurora Installer");
            println!("- Modes: balanced, gaming, max-throughput");
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
        "balanced" | "gaming" | "max-throughput" => {
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

fn welcome_app(action: Option<&str>) -> Result<()> {
    match action.unwrap_or("status") {
        "status" => {
            println!("Aurora Welcome");
            println!("- aurora runtime: {}", command_summary("aurora", ["version"]));
            println!("- Try:");
            println!("  aurora-distro app control-center");
            println!("  aurora-distro app ai-hub --action status");
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
        "set -euo pipefail\n\
         exec > >(tee -a /var/log/aurora-package-install.log) 2>&1\n\
         export DEBIAN_FRONTEND=noninteractive\n\
         printf 'deb http://archive.ubuntu.com/ubuntu noble main universe multiverse restricted\\n' > /etc/apt/sources.list\n\
         printf 'deb http://archive.ubuntu.com/ubuntu noble-updates main universe multiverse restricted\\n' >> /etc/apt/sources.list\n\
         printf 'deb http://archive.ubuntu.com/ubuntu noble-security main universe multiverse restricted\\n' >> /etc/apt/sources.list\n\
         cat > /etc/apt/apt.conf.d/99aurora-remaster <<'EOF'\n\
 Acquire::Retries \"10\";\n\
 Acquire::http::Timeout \"60\";\n\
 Acquire::https::Timeout \"60\";\n\
 Acquire::ForceIPv4 \"true\";\n\
 Acquire::Languages \"none\";\n\
 Dpkg::Use-Pty \"0\";\n\
 APT::Install-Recommends \"true\";\n\
 APT::Install-Suggests \"false\";\n\
 EOF\n\
         retry_apt() {{\n\
           local attempt=1\n\
           local max_attempts=4\n\
           until \"$@\"; do\n\
             if [ \"$attempt\" -ge \"$max_attempts\" ]; then\n\
               echo \"error: command failed after ${{max_attempts}} attempts: $*\" >&2\n\
               return 1\n\
             fi\n\
             echo \"warning: attempt ${{attempt}} failed for: $*\" >&2\n\
             apt-get clean || true\n\
             rm -f /var/cache/apt/archives/lock /var/lib/dpkg/lock /var/lib/dpkg/lock-frontend || true\n\
             dpkg --configure -a || true\n\
             sleep $((attempt * 10))\n\
             attempt=$((attempt + 1))\n\
           done\n\
         }}\n\
         retry_apt apt-get update\n\
         for pkg in {}; do\n\
           echo \"Installing required package: $pkg\"\n\
           retry_apt apt-get install -y \"$pkg\"\n\
         done\n\
         for pkg in {}; do \
           if apt-cache show \"$pkg\" >/dev/null 2>&1; then \
             retry_apt apt-get install -y \"$pkg\" || echo \"warning: optional package failed: $pkg\"; \
           else \
             echo \"warning: optional package unavailable: $pkg\"; \
           fi; \
         done\n\
         dpkg --configure -a\n\
         apt-get clean",
        shell_words(&required_install_set),
        shell_words(&optional_packages),
    );
    let chroot_cmd = format!("chroot {} /bin/bash -lc {:?}", path_str(&build_root)?, package_script);
    run_shell(&chroot_cmd)?;

    apply_overlay(tree, &build_root)?;
    install_branding(tree, &build_root, &config)?;
    stage_runtime(&build_root)?;
    build_aurora_repo(tree, &build_root, &config)?;
    finalize_rootfs(&build_root)?;

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
                   command -v apt-get >/dev/null 2>&1 && DEBIAN_FRONTEND=noninteractive apt-get install -y aurora-branding aurora-runtime-tools aurora-installer aurora-control-center aurora-app-center aurora-ai-hub aurora-gaming-center aurora-security-center aurora-welcome aurora-desktop >/dev/null 2>&1 || true; \
                   command -v flatpak >/dev/null 2>&1 && flatpak remote-add --if-not-exists flathub https://dl.flathub.org/repo/flathub.flatpakrepo || true; \
                   command -v systemctl >/dev/null 2>&1 && systemctl enable aurora-autosetup.service aurora-zram-setup.service >/dev/null 2>&1 || true; \
                   command -v systemctl >/dev/null 2>&1 && systemctl enable ollama >/dev/null 2>&1 || true; \
                   command -v systemctl >/dev/null 2>&1 && systemctl enable valkey-server >/dev/null 2>&1 || true; \
                   command -v systemctl >/dev/null 2>&1 && systemctl enable kdump-tools >/dev/null 2>&1 || true; \
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
            | "gamescope"
            | "goverlay"
            | "mesa-vulkan-drivers"
            | "mesa-utils"
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
                "usr/local/bin/aurora-gaming-setup".to_string(),
                "usr/local/bin/aurora-security-setup".to_string(),
                "usr/local/bin/aurora-directstream".to_string(),
                "etc/default/aurora-performance".to_string(),
                "etc/aurora/directstream.conf".to_string(),
                "etc/systemd/system/aurora-autosetup.service".to_string(),
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
                "usr/share/applications/aurora-welcome.desktop".to_string(),
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
            version,
            architecture: "all".to_string(),
            depends: vec![
                "aurora-app-center".to_string(),
                "flatpak".to_string(),
                "vlc".to_string(),
                "zellij".to_string(),
                "helix".to_string(),
            ],
            description: "Aurora creator stack meta-package for workstation and media tooling.".to_string(),
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
        "Origin: {name}\nLabel: {name} Archive\nSuite: stable\nCodename: aurora-stable\nArchitectures: amd64\nComponents: main\nDescription: Curated Aurora package archive layered on top of Ubuntu\n",
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
            "aurora-gaming-stack",
            "1.0",
            "gamemode, mangohud, gamescope",
            "Aurora gaming stack with tuned performance and session tooling.",
        ),
        aurora_repo_package_entry(
            "aurora-creator-stack",
            "1.0",
            "flatpak, vlc, zellij, helix",
            "Aurora creator and workstation stack with media and Rust-native utilities.",
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
            "base": "ubuntu-noble"
        },
        "labels": [
            {"id": "aurora", "name": "Aurora Curated"},
            {"id": "gaming", "name": "Gaming Ready"},
            {"id": "ai", "name": "AI Native"},
            {"id": "creator", "name": "Creator Stack"},
            {"id": "secure", "name": "Security Hardened"}
        ],
        "featured": [
            {"package": "aurora-desktop", "title": "Aurora Desktop", "origin": "Aurora Archive", "label": "Aurora Curated"},
            {"package": "aurora-ai-stack", "title": "Aurora AI Stack", "origin": "Aurora Archive", "label": "AI Native"},
            {"package": "aurora-gaming-stack", "title": "Aurora Gaming Stack", "origin": "Aurora Archive", "label": "Gaming Ready"},
            {"package": "aurora-creator-stack", "title": "Aurora Creator Stack", "origin": "Aurora Archive", "label": "Creator Stack"}
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
                "depends": ["aurora-control-center", "aurora-app-center", "aurora-welcome", "aurora-gaming-center", "aurora-ai-hub", "aurora-security-center"]
            },
            {
                "name": "aurora-ai-stack",
                "title": "Aurora AI Stack",
                "depends": ["ollama", "aurora-ai-hub", "ripgrep", "fd-find", "bat", "helix"]
            },
            {
                "name": "aurora-gaming-stack",
                "title": "Aurora Gaming Stack",
                "depends": ["gamemode", "mangohud", "gamescope", "goverlay", "steam-installer"]
            },
            {
                "name": "aurora-creator-stack",
                "title": "Aurora Creator Stack",
                "depends": ["flatpak", "gnome-software-plugin-flatpak", "vlc", "zellij", "bottom", "just"]
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

fn desktop_entry(name: &str, surface: &str, categories: &str) -> String {
    format!(
        "[Desktop Entry]\nType=Application\nName={name}\nExec=/bin/sh -lc 'x-terminal-emulator -e aurora-distro app {surface} || gnome-terminal -- aurora-distro app {surface} || alacritty -e aurora-distro app {surface} || xterm -e aurora-distro app {surface}'\nTerminal=false\nX-GNOME-Autostart-enabled=true\nCategories={categories};\n"
    )
}

fn firstboot_script(installer: &InstallerConfig) -> String {
    format!(
        "#!/usr/bin/env bash\nset -euo pipefail\nMODE_FILE=/etc/aurora-installer/selected-mode\nif [ ! -f \"$MODE_FILE\" ]; then\n  printf '%s\\n' '{}' > \"$MODE_FILE\"\nfi\nif systemctl list-unit-files | grep -q '^aurora-autosetup.service'; then\n  systemctl enable aurora-autosetup.service >/dev/null 2>&1 || true\n  systemctl start aurora-autosetup.service >/dev/null 2>&1 || true\nfi\ncommand -v aurora-apply-desktop >/dev/null 2>&1 && aurora-apply-desktop || true\ncommand -v aurora-ai-setup >/dev/null 2>&1 && aurora-ai-setup --firstboot || true\ncommand -v aurora-gaming-setup >/dev/null 2>&1 && aurora-gaming-setup --firstboot || true\ncommand -v aurora-security-setup >/dev/null 2>&1 && aurora-security-setup --firstboot || true\nprintf '%s\\n' \"$(cat \"$MODE_FILE\")\" >/tmp/aurora-installer-mode\nif command -v xdg-open >/dev/null 2>&1; then\n  xdg-open /usr/share/aurora/installer/index.html >/dev/null 2>&1 || true\nfi\n",
        install_mode_name(&installer.default_mode)
    )
}

fn installer_html(name: &str, accent: &str, installer: &InstallerConfig) -> String {
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width, initial-scale=1\"><title>{name} Installer</title><link rel=\"stylesheet\" href=\"installer.css\"></head><body><main class=\"shell\"><section class=\"hero\"><div class=\"brand-row\"><img class=\"brand-mark\" src=\"/usr/share/aurora/logo.svg\" alt=\"{name} logo\"><div><p class=\"eyebrow\">Adaptive Neon Workstation</p><h1>{name}</h1></div></div><p class=\"lede\">Shape the system before install: choose your runtime mode, deploy a tuned partition plan, and boot into a shell curated around speed, visuals, and stronger day-one tooling.</p><div class=\"cta-row\"><button id=\"scanBtn\">Scan This System</button><button id=\"planBtn\">Create Partition Plan</button><button id=\"usbBtn\">Use Inbuilt USB Writer</button><button id=\"repairBtn\">Show Boot Repair</button></div></section><section class=\"grid\"><article class=\"card mode-card\"><h2>Balanced</h2><p>Moderate zram, calmer I/O tuning, and lower background disruption for mixed desktop workloads.</p><button data-mode=\"balanced\">Choose Balanced</button></article><article class=\"card mode-card\"><h2>Gaming</h2><p>Lower-latency desktop tuning with GameMode, MangoHud, GNOME shell extras, and faster storage defaults.</p><button data-mode=\"gaming\">Choose Gaming</button></article><article class=\"card mode-card\"><h2>Max Throughput</h2><p>Pushes CPU throughput, hugepage usage, zram, and aggressive boot/runtime knobs for compute-heavy work.</p><button data-mode=\"max-throughput\">Choose Max Throughput</button></article><article class=\"card\"><h2>Out-of-Box Stack</h2><p>Flatpak-ready app flow, tuned shell defaults, bold theming, runtime tooling, repair scripts, and a performance control surface.</p></article></section><section class=\"terminal\"><div class=\"terminal-bar\"><span></span><span></span><span></span></div><pre id=\"output\">Awaiting action...</pre></section></main><script>window.AURORA_INSTALLER={{accent:\"{accent}\",name:\"{name}\",defaultMode:\"{}\",availableModes:{},memoryBoost:{},bootRepair:{}}};</script><script src=\"installer.js\"></script></body></html>",
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
    "const cfg=window.AURORA_INSTALLER;const output=document.getElementById('output');const write=(lines)=>output.textContent=lines.join('\\n');const modes={balanced:['Balanced mode selected','- schedutil governor','- moderate zram and preload','- quieter background-service policy','- good default for general workstations'],gaming:['Gaming mode selected','- performance governor','- enable GameMode and MangoHud','- lower-latency readahead and I/O profile','- best fit for desktop responsiveness and games'],['max-throughput']:['Max Throughput selected','- performance governor + aggressive hugepage posture','- highest zram allocation and stronger cache tuning','- trims more services and boosts readahead','- best fit for compute-heavy sustained workloads']};document.getElementById('scanBtn').addEventListener('click',()=>write(['Scanning current machine...','- detect firmware mode','- detect CPU model and RAM','- enumerate storage devices','- estimate swap + zram balance','Result: if details are missing, aurora-distro can fall back to automatic planning.']));document.getElementById('planBtn').addEventListener('click',()=>write(['Generating partition plan...','- choose GPT for UEFI or hybrid installs','- add BIOS_GRUB when legacy GRUB embedding is needed','- size swap from RAM and disk pressure','- keep partition-apply.sh ready for manual replay','Result: installer can switch between balanced, gaming, and throughput presets after partitioning.']));document.getElementById('usbBtn').addEventListener('click',()=>write(['Inbuilt USB writer flow','1. Confirm target device','2. Verify path is a block device','3. Write ISO with dd + sync','4. Return success/failure logs to the user']));document.getElementById('repairBtn').addEventListener('click',()=>write(['Boot repair toolkit','- bind-mount /dev /proc /sys','- reinstall GRUB for EFI and BIOS targets when available','- refresh initramfs and grub.cfg','- keep BOOTX64.EFI fallback in place for UEFI firmware']));document.querySelectorAll('[data-mode]').forEach(btn=>btn.addEventListener('click',()=>write(modes[btn.dataset.mode]||['Unknown mode'])));write(['Default installer mode: '+cfg.defaultMode,'Available modes: '+cfg.availableModes.join(', '),'Choose a mode, then scan hardware or generate a partition plan.']);".to_string()
}

fn control_center_html(name: &str, accent: &str) -> String {
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>{name} Control Center</title><link rel=\"stylesheet\" href=\"control-center.css\"></head><body><main class=\"frame\"><section class=\"hero\"><p class=\"eyebrow\">Aurora Runtime Control</p><h1>{name} Control Center</h1><p class=\"lede\">Switch system modes, inspect runtime placement, and keep the distro personality consistent after install.</p></section><section class=\"grid\"><button data-mode=\"balanced\">Balanced</button><button data-mode=\"gaming\">Gaming</button><button data-mode=\"max-throughput\">Max Throughput</button><button id=\"desktopBtn\">Apply Desktop Layer</button><button id=\"runtimeBtn\">Runtime Status</button><button id=\"tourBtn\">Open Welcome Tour</button><button id=\"aiBtn\">Open AI Hub</button><button id=\"gamingBtn\">Open Gaming Center</button><button id=\"securityBtn\">Open Security Center</button><button id=\"packageBtn\">Open Package Center</button></section><pre id=\"terminal\">Aurora control surface ready.</pre></main><script>window.AURORA_CC={{name:\"{name}\",accent:\"{accent}\"}};</script><script src=\"control-center.js\"></script></body></html>"
    )
}

fn control_center_css(accent: &str) -> String {
    format!(
        ":root{{--accent:{accent};--bg:#04060b;--panel:#0f1724;--line:#17314d;--text:#edf7ff;--muted:#9db4c8}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at top,#10233a,#04060b 60%);color:var(--text);font-family:'Orbitron',sans-serif}}.frame{{max-width:1080px;margin:0 auto;padding:40px 24px 72px}}.hero{{padding:32px;border:1px solid var(--line);border-radius:24px;background:rgba(10,17,28,.92)}}.eyebrow{{text-transform:uppercase;letter-spacing:.2em;color:var(--accent);font-size:12px}}h1{{margin:12px 0 10px;font-size:56px}}.lede{{font-family:'Rajdhani',sans-serif;color:var(--muted);font-size:22px;max-width:760px}}.grid{{margin-top:22px;display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:16px}}button{{padding:18px 20px;border-radius:18px;border:1px solid var(--line);background:linear-gradient(180deg,#102740,#0a1524);color:var(--text);font:inherit;cursor:pointer}}button:hover{{border-color:var(--accent);box-shadow:0 0 0 1px var(--accent),0 0 24px rgba(18,247,255,.2)}}pre{{margin-top:22px;min-height:180px;padding:20px;border-radius:20px;border:1px solid var(--line);background:#071019;color:#d8f7ff;font-family:'JetBrains Mono',monospace;white-space:pre-wrap}}"
    )
}

fn control_center_js() -> String {
    "const term=document.getElementById('terminal');const write=(lines)=>term.textContent=lines.join('\\n');document.querySelectorAll('[data-mode]').forEach(btn=>btn.addEventListener('click',()=>write(['Requested mode: '+btn.dataset.mode,'CLI: sudo aurora-mode-switch '+btn.dataset.mode,'Effect: updates /etc/default/aurora-performance and triggers aurora-autosetup.'])));document.getElementById('desktopBtn').addEventListener('click',()=>write(['Desktop refresh','CLI: aurora-apply-desktop','Effect: reapplies Aurora shell defaults, favorites, wallpaper, fonts, and dock posture.']));document.getElementById('runtimeBtn').addEventListener('click',()=>write(['Runtime status','Binary: /usr/local/bin/aurora','Quick checks: aurora version | aurora detect | aurora status']));document.getElementById('tourBtn').addEventListener('click',()=>window.location='/usr/share/aurora/welcome/index.html');document.getElementById('aiBtn').addEventListener('click',()=>window.location='/usr/share/aurora/ai-hub/index.html');document.getElementById('gamingBtn').addEventListener('click',()=>window.location='/usr/share/aurora/gaming-center/index.html');document.getElementById('securityBtn').addEventListener('click',()=>window.location='/usr/share/aurora/security-center/index.html');document.getElementById('packageBtn').addEventListener('click',()=>window.location='/usr/share/aurora/package-center/index.html');".to_string()
}

fn ai_hub_html(name: &str, accent: &str) -> String {
    format!(
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>{name} AI Hub</title><link rel=\"stylesheet\" href=\"ai-hub.css\"></head><body><main class=\"wrap\"><section class=\"hero\"><p class=\"eyebrow\">AI-Native Desktop</p><h1>{name} AI Hub</h1><p class=\"lede\">Private local models, desktop search integration, and GPU-aware AI tooling prepared for offline use through Ollama and Aurora wrappers.</p></section><section class=\"cards\"><article><h2>Local Models</h2><p>Ollama is treated as the default local model service. Suggested first pull: <code>ollama pull llama3</code>.</p></article><article><h2>Desktop Hooks</h2><p>Use Aurora AI Hub as the launcher for local assistance tied to files, search, and terminal workflows.</p></article><article><h2>GPU Readiness</h2><p>Driver/tooling posture is prepared for NVIDIA, AMD, and Intel acceleration through Mesa/NVIDIA-aware gaming and compute stacks.</p></article><article><h2>Rust Toolchain Surface</h2><p>Rust-native helpers like <code>ripgrep</code>, <code>fd</code>, <code>bat</code>, <code>helix</code>, <code>zellij</code>, <code>bottom</code>, and <code>just</code> are part of Aurora's developer posture when available.</p></article></section></main></body></html>"
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
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>Welcome to {name}</title><link rel=\"stylesheet\" href=\"welcome.css\"></head><body><main class=\"wrap\"><section class=\"hero\"><p class=\"eyebrow\">Welcome To {name}</p><h1>Not just a rebrand</h1><p class=\"lede\">Aurora Neon boots with a tuned runtime, adaptive installer, local AI hub, gaming center, control center, security center, package-awareness center, richer shell defaults, repair tooling, performance profiles, and a Rust-native workstation stack intended to make the system feel opinionated from minute one.</p></section><section class=\"cards\"><article><h2>Runtime Layer</h2><p><code>aurora</code> is staged into the live system and exposed as a first-class command.</p></article><article><h2>AI-Native</h2><p>Ollama-first local model access is part of the distro feature surface rather than a browser-only story.</p></article><article><h2>Gaming Center</h2><p>Unified Wayland/gaming direction for Proton, Mesa, Gamescope, DirectStream, and low-latency posture.</p></article><article><h2>Rust-Native Tools</h2><p>Aurora tries to ship fast Rust-based search, editing, terminal, and workspace tools where the base archive makes them available.</p></article></section></main></body></html>"
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
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>{name} App Center</title><link rel=\"stylesheet\" href=\"package-center.css\"></head><body><main class=\"wrap\"><section class=\"hero\"><p class=\"eyebrow\">Aurora Archive</p><h1>{name} App Center</h1><p class=\"lede\">Aurora makes package origin explicit and puts Aurora-owned packages, stacks, and curated labels ahead of raw upstream naming.</p></section><section class=\"cards\"><article><h2>Aurora Archive</h2><p>A local Aurora repository is staged into the image so curated Aurora packages and meta-packages have a distinct channel and identity.</p></article><article><h2>Aurora Labels</h2><p>User-facing tooling can present Aurora-owned labels like <code>AI Stack</code>, <code>Gaming Stack</code>, and <code>Creator Stack</code> instead of raw dependency names.</p></article><article><h2>Native DEB, Snap, Flatpak</h2><p>The App Center still distinguishes archive-backed DEBs, Snaps, and Flatpaks so users can reason about provenance.</p></article><article><h2>Selective Replacement</h2><p>Aurora can replace or augment only the packages that add real value, while still inheriting Ubuntu's mature base elsewhere.</p></article></section></main></body></html>"
    )
}

fn package_center_css(accent: &str) -> String {
    format!(
        ":root{{--accent:{accent};--bg:#05070d;--panel:#111a2a;--line:#17314d;--text:#eef7ff;--muted:#9db4c8}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at top,#10233a,#05070d 60%);color:var(--text);font-family:'Orbitron',sans-serif}}.wrap{{max-width:1100px;margin:0 auto;padding:48px 24px 72px}}.hero{{padding:34px;border:1px solid var(--line);border-radius:26px;background:rgba(11,18,29,.94)}}.eyebrow{{color:var(--accent);text-transform:uppercase;letter-spacing:.2em;font-size:12px}}h1{{margin:14px 0;font-size:58px}}.lede{{max-width:780px;font-family:'Rajdhani',sans-serif;font-size:24px;color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:18px;margin-top:24px}}article{{padding:22px;border-radius:22px;background:rgba(10,18,29,.88);border:1px solid var(--line)}}article h2{{margin:0 0 8px;font-size:24px}}article p{{margin:0;font-family:'Rajdhani',sans-serif;font-size:20px;color:var(--muted)}}code{{color:var(--accent)}}"
    )
}

fn welcome_css(accent: &str) -> String {
    format!(
        ":root{{--accent:{accent};--bg:#05070d;--panel:#0e1827;--line:#16304d;--text:#eef7ff;--muted:#9bb2c8}}*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(180deg,#091120,#05070d);color:var(--text);font-family:'Orbitron',sans-serif}}.wrap{{max-width:1100px;margin:0 auto;padding:48px 24px 72px}}.hero{{padding:36px;border:1px solid var(--line);border-radius:28px;background:rgba(11,17,28,.92)}}.eyebrow{{letter-spacing:.22em;text-transform:uppercase;color:var(--accent);font-size:12px}}h1{{margin:14px 0;font-size:64px;line-height:.95}}.lede{{max-width:780px;font-family:'Rajdhani',sans-serif;font-size:24px;color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:18px;margin-top:24px}}article{{padding:22px;border-radius:22px;background:rgba(10,18,29,.88);border:1px solid var(--line)}}article h2{{margin:0 0 8px;font-size:24px}}article p{{margin:0;font-family:'Rajdhani',sans-serif;font-size:20px;color:var(--muted)}}"
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
        "[org/gnome/desktop/interface]\ngtk-theme='Aurora-Neon'\nicon-theme='Papirus'\ncolor-scheme='prefer-dark'\nclock-format='24h'\nfont-name='Cantarell 11'\ndocument-font-name='Cantarell 11'\nmonospace-font-name='Cascadia Code 11'\nenable-hot-corners=false\n\n[org/gnome/desktop/background]\npicture-uri='file:///usr/share/backgrounds/aurora/gaming-kali.svg'\npicture-uri-dark='file:///usr/share/backgrounds/aurora/gaming-kali.svg'\nprimary-color='#05070d'\nsecondary-color='{}'\n\n[org/gnome/desktop/wm/preferences]\nbutton-layout='appmenu:minimize,maximize,close'\nfocus-mode='click'\n\n[org/gnome/mutter]\nedge-tiling=true\noverlay-key='Super_L'\nworkspaces-only-on-primary=false\n\n[org/gnome/shell]\nfavorite-apps=['org.gnome.Nautilus.desktop','org.gnome.Terminal.desktop','alacritty.desktop','firefox_firefox.desktop','org.gnome.Software.desktop','aurora-control-center.desktop','aurora-ai-hub.desktop','aurora-gaming-center.desktop','aurora-installer.desktop']\ndisable-user-extensions=false\nenabled-extensions=['apps-menu@gnome-shell-extensions.gcampax.github.com','places-menu@gnome-shell-extensions.gcampax.github.com','user-theme@gnome-shell-extensions.gcampax.github.com']\nwelcome-dialog-last-shown-version='999999'\n\n[org/gnome/shell/keybindings]\ntoggle-application-view=['Super_L']\n\n[org/gnome/shell/extensions/dash-to-dock]\ndock-position='BOTTOM'\nextend-height=false\nshow-trash=true\nshow-mounts=true\ntransparency-mode='FIXED'\nbackground-opacity=0.25\nclick-action='minimize-or-overview'\nshow-apps-at-top=true\n\n[org/gnome/login-screen]\nlogo='/usr/share/aurora/logo.svg'\n",
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
        "AURORA_INSTALL_MODE={}\nAURORA_PROFILE_NAME={}\nAURORA_CPU_GOVERNOR={}\nAURORA_ENABLE_ZRAM={}\nAURORA_ZRAM_FRACTION_PERCENT={}\nAURORA_ENABLE_PRELOAD={}\nAURORA_IO_SCHEDULER={}\nAURORA_READAHEAD_KB={}\nAURORA_DISABLE_UNNEEDED_SERVICES={}\nAURORA_SWAP_POLICY={}\nAURORA_VM_SWAPPINESS={}\nAURORA_VM_DIRTY_RATIO={}\nAURORA_VM_DIRTY_BACKGROUND_RATIO={}\nAURORA_DISABLE_CPU_IDLE={}\nAURORA_SCHEDULER_TUNE={}\nAURORA_CPU_ENERGY_POLICY={}\nAURORA_ENABLE_FIREWALL=true\nAURORA_ENABLE_FAIL2BAN=true\nAURORA_ENABLE_THERMALD=true\nAURORA_ENABLE_IRQBALANCE=true\nAURORA_ENABLE_EARLYOOM=true\nAURORA_ENABLE_DIRECTSTREAM=true\n",
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
    "#!/usr/bin/env bash\nset -euo pipefail\nCONFIG=/etc/default/aurora-performance\nif [ -f \"$CONFIG\" ]; then\n  # shellcheck disable=SC1091\n  . \"$CONFIG\"\nfi\nfor cpu in /sys/devices/system/cpu/cpu[0-9]*; do\n  if [ -w \"$cpu/cpufreq/scaling_governor\" ]; then\n    printf '%s' \"${AURORA_CPU_GOVERNOR:-performance}\" > \"$cpu/cpufreq/scaling_governor\" || true\n  fi\n  if [ -w \"$cpu/cpufreq/energy_performance_preference\" ]; then\n    printf '%s' \"${AURORA_CPU_ENERGY_POLICY:-performance}\" > \"$cpu/cpufreq/energy_performance_preference\" || true\n  fi\n done\nif [ \"${AURORA_USE_TMPFS_FOR_TEMP:-true}\" = \"true\" ]; then\n  grep -q '^tmpfs /tmp tmpfs' /etc/fstab || printf 'tmpfs /tmp tmpfs defaults,noatime,mode=1777 0 0\\n' >> /etc/fstab\nfi\nif [ \"${AURORA_DISABLE_UNNEEDED_SERVICES:-true}\" = \"true\" ] && command -v systemctl >/dev/null 2>&1; then\n  for svc in bluetooth cups apport whoopsie snapd ModemManager; do\n    systemctl disable \"$svc\" >/dev/null 2>&1 || true\n    systemctl stop \"$svc\" >/dev/null 2>&1 || true\n  done\nfi\nif [ -n \"${AURORA_READAHEAD_KB:-}\" ]; then\n  for block in /sys/block/*/queue/read_ahead_kb; do\n    [ -w \"$block\" ] && printf '%s' \"$AURORA_READAHEAD_KB\" > \"$block\" || true\n  done\nfi\nif [ -n \"${AURORA_IO_SCHEDULER:-}\" ]; then\n  for sched in /sys/block/*/queue/scheduler; do\n    [ -w \"$sched\" ] && grep -qw \"$AURORA_IO_SCHEDULER\" \"$sched\" && printf '%s' \"$AURORA_IO_SCHEDULER\" > \"$sched\" || true\n  done\nfi\nif [ \"${AURORA_DISABLE_CPU_IDLE:-false}\" = \"true\" ] && [ -w /sys/module/intel_idle/parameters/max_cstate ]; then\n  printf '1' > /sys/module/intel_idle/parameters/max_cstate || true\nfi\nif [ \"${AURORA_ENABLE_ZRAM:-true}\" = \"true\" ] && command -v systemctl >/dev/null 2>&1; then\n  systemctl enable aurora-zram-setup.service >/dev/null 2>&1 || true\n  systemctl start aurora-zram-setup.service >/dev/null 2>&1 || true\nfi\nif [ \"${AURORA_ENABLE_IRQBALANCE:-true}\" = \"true\" ] && command -v systemctl >/dev/null 2>&1; then\n  systemctl enable irqbalance >/dev/null 2>&1 || true\n  systemctl start irqbalance >/dev/null 2>&1 || true\nfi\nif [ \"${AURORA_ENABLE_THERMALD:-true}\" = \"true\" ] && command -v systemctl >/dev/null 2>&1; then\n  systemctl enable thermald >/dev/null 2>&1 || true\n  systemctl start thermald >/dev/null 2>&1 || true\nfi\nif [ \"${AURORA_ENABLE_EARLYOOM:-true}\" = \"true\" ] && command -v systemctl >/dev/null 2>&1; then\n  systemctl enable earlyoom >/dev/null 2>&1 || true\n  systemctl start earlyoom >/dev/null 2>&1 || true\nfi\nif [ \"${AURORA_ENABLE_FAIL2BAN:-true}\" = \"true\" ] && command -v systemctl >/dev/null 2>&1; then\n  systemctl enable fail2ban >/dev/null 2>&1 || true\n  systemctl start fail2ban >/dev/null 2>&1 || true\nfi\nif [ \"${AURORA_ENABLE_FIREWALL:-true}\" = \"true\" ] && command -v ufw >/dev/null 2>&1; then\n  ufw default deny incoming >/dev/null 2>&1 || true\n  ufw default allow outgoing >/dev/null 2>&1 || true\n  ufw --force enable >/dev/null 2>&1 || true\nfi\ncommand -v systemctl >/dev/null 2>&1 && systemctl enable fstrim.timer >/dev/null 2>&1 || true\nsysctl --system >/dev/null 2>&1 || true\n".to_string()
}

fn desktop_setup_script(config: &BuildConfig) -> String {
    format!(
        "#!/usr/bin/env bash\nset -euo pipefail\nexport XDG_CURRENT_DESKTOP=\"${{XDG_CURRENT_DESKTOP:-GNOME}}\"\nWALL='/usr/share/backgrounds/aurora/gaming-kali.svg'\nrun_gsettings() {{\n  if command -v gsettings >/dev/null 2>&1; then\n    gsettings set \"$1\" \"$2\" \"$3\" >/dev/null 2>&1 || true\n  fi\n}}\nrun_gsettings org.gnome.desktop.interface gtk-theme 'Aurora-Neon'\nrun_gsettings org.gnome.desktop.interface icon-theme 'Papirus'\nrun_gsettings org.gnome.desktop.interface color-scheme 'prefer-dark'\nrun_gsettings org.gnome.desktop.interface monospace-font-name 'Cascadia Code 11'\nrun_gsettings org.gnome.desktop.interface enable-hot-corners false\nrun_gsettings org.gnome.desktop.background picture-uri \"file://${{WALL}}\"\nrun_gsettings org.gnome.desktop.background picture-uri-dark \"file://${{WALL}}\"\nrun_gsettings org.gnome.desktop.wm.preferences button-layout 'appmenu:minimize,maximize,close'\nrun_gsettings org.gnome.mutter overlay-key 'Super_L'\nrun_gsettings org.gnome.shell favorite-apps \"['org.gnome.Nautilus.desktop','org.gnome.Terminal.desktop','alacritty.desktop','firefox_firefox.desktop','org.gnome.Software.desktop','aurora-control-center.desktop','aurora-installer.desktop']\"\nrun_gsettings org.gnome.shell enabled-extensions \"['apps-menu@gnome-shell-extensions.gcampax.github.com','places-menu@gnome-shell-extensions.gcampax.github.com','user-theme@gnome-shell-extensions.gcampax.github.com']\"\nrun_gsettings org.gnome.shell.extensions.dash-to-dock dock-position 'BOTTOM'\nrun_gsettings org.gnome.shell.extensions.dash-to-dock extend-height false\nrun_gsettings org.gnome.shell.extensions.dash-to-dock show-trash true\nrun_gsettings org.gnome.shell.extensions.dash-to-dock show-mounts true\nrun_gsettings org.gnome.shell.extensions.dash-to-dock show-apps-at-top true\nmkdir -p \"$HOME/.config\"\ncat > \"$HOME/.config/aurora-desktop-summary\" <<'EOF'\n{name}\nMode-ready GNOME shell\nTheme: Aurora-Neon\nWallpaper: $WALL\nRuntime: /usr/local/bin/aurora\nSecurity: ufw fail2ban apparmor earlyoom thermald irqbalance\nEOF\n",
        name = config.branding_name
    )
}

fn mode_switch_script() -> String {
    "#!/usr/bin/env bash\nset -euo pipefail\nMODE=\"${1:-gaming}\"\nCONFIG=/etc/default/aurora-performance\ncase \"$MODE\" in\n  balanced)\n    sed -i 's/^AURORA_CPU_GOVERNOR=.*/AURORA_CPU_GOVERNOR=schedutil/' \"$CONFIG\" || true\n    sed -i 's/^AURORA_READAHEAD_KB=.*/AURORA_READAHEAD_KB=1024/' \"$CONFIG\" || true\n    sed -i 's/^AURORA_VM_SWAPPINESS=.*/AURORA_VM_SWAPPINESS=25/' \"$CONFIG\" || true\n    ;;\n  gaming)\n    sed -i 's/^AURORA_CPU_GOVERNOR=.*/AURORA_CPU_GOVERNOR=performance/' \"$CONFIG\" || true\n    sed -i 's/^AURORA_READAHEAD_KB=.*/AURORA_READAHEAD_KB=2048/' \"$CONFIG\" || true\n    sed -i 's/^AURORA_VM_SWAPPINESS=.*/AURORA_VM_SWAPPINESS=18/' \"$CONFIG\" || true\n    ;;\n  max-throughput)\n    sed -i 's/^AURORA_CPU_GOVERNOR=.*/AURORA_CPU_GOVERNOR=performance/' \"$CONFIG\" || true\n    sed -i 's/^AURORA_READAHEAD_KB=.*/AURORA_READAHEAD_KB=4096/' \"$CONFIG\" || true\n    sed -i 's/^AURORA_VM_SWAPPINESS=.*/AURORA_VM_SWAPPINESS=15/' \"$CONFIG\" || true\n    sed -i 's/^AURORA_DISABLE_CPU_IDLE=.*/AURORA_DISABLE_CPU_IDLE=true/' \"$CONFIG\" || true\n    ;;\n  *)\n    echo \"unknown mode: $MODE\" >&2\n    exit 1\n    ;;\n esac\nif command -v systemctl >/dev/null 2>&1; then\n  systemctl restart aurora-autosetup.service >/dev/null 2>&1 || true\nfi\necho \"Aurora mode switched to $MODE\"\n".to_string()
}

fn ai_setup_script() -> String {
    "#!/usr/bin/env bash\nset -euo pipefail\nFIRSTBOOT=\"${1:-}\"\nif command -v systemctl >/dev/null 2>&1; then\n  systemctl enable ollama >/dev/null 2>&1 || true\n  [ \"$FIRSTBOOT\" = \"--firstboot\" ] && systemctl start ollama >/dev/null 2>&1 || true\nfi\nmkdir -p \"$HOME/.config/aurora\"\ncat > \"$HOME/.config/aurora/ai-hub.conf\" <<'EOF'\nprovider=ollama\ndefault_model=llama3\nsearch_integration=planned\nfile_manager_actions=planned\nEOF\nif command -v ollama >/dev/null 2>&1; then\n  echo 'Aurora AI Hub configured for Ollama. Suggested next step: ollama pull llama3'\nelse\n  echo 'Ollama not available in this image build.'\nfi\n".to_string()
}

fn gaming_setup_script() -> String {
    "#!/usr/bin/env bash\nset -euo pipefail\nsysctl_conf=/etc/sysctl.d/99-aurora-gaming.conf\nif [ -f \"$sysctl_conf\" ]; then\n  grep -q '^vm.max_map_count=' \"$sysctl_conf\" || printf 'vm.max_map_count=2147483642\\n' >> \"$sysctl_conf\"\n  grep -q '^fs.file-max=' \"$sysctl_conf\" || printf 'fs.file-max=2097152\\n' >> \"$sysctl_conf\"\nfi\nmkdir -p \"$HOME/.config/aurora\"\ncat > \"$HOME/.config/aurora/gaming-center.conf\" <<'EOF'\nwayland_default=true\nnvidia_wayland=preferred\ngamescope=enabled_when_available\nmangohud=enabled_when_available\nanti_cheat_posture=steam_proton_focus\ndirectstream=enabled\nEOF\ncommand -v aurora-directstream >/dev/null 2>&1 && aurora-directstream --prime || true\nsysctl --system >/dev/null 2>&1 || true\necho 'Aurora Gaming Center profile applied.'\n".to_string()
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

fn aurora_apt_script() -> String {
    "#!/usr/bin/env bash\nset -euo pipefail\nexport APT_CONFIG=/dev/null\nif [ \"$#\" -eq 0 ]; then\n  echo 'usage: aurora-apt <apt-args>' >&2\n  exit 1\nfi\napt -o APT::Color=1 -o Dpkg::Progress-Fancy=1 \"$@\"\n".to_string()
}

fn autosetup_service() -> String {
    "[Unit]\nDescription=AURORA automatic performance setup\nAfter=multi-user.target\n\n[Service]\nType=oneshot\nExecStart=/usr/local/bin/aurora-autosetup\nRemainAfterExit=yes\n\n[Install]\nWantedBy=multi-user.target\n".to_string()
}

fn zram_service() -> String {
    "[Unit]\nDescription=AURORA zram memory boost\nAfter=local-fs.target\n\n[Service]\nType=oneshot\nExecStart=/bin/bash -lc 'modprobe zram || true; echo lz4 > /sys/block/zram0/comp_algorithm 2>/dev/null || true; mem_total_kb=$(awk \"/MemTotal/ {print \\$2}\" /proc/meminfo); size_bytes=$((mem_total_kb * 1024 * 60 / 100)); echo ${size_bytes:-0} > /sys/block/zram0/disksize 2>/dev/null || true; mkswap /dev/zram0 >/dev/null 2>&1 || true; swapon -p 100 /dev/zram0 >/dev/null 2>&1 || true'\nRemainAfterExit=yes\n\n[Install]\nWantedBy=multi-user.target\n".to_string()
}

fn default_wallpaper_svg(name: &str, accent: &str) -> String {
    format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 1920 1080\"><defs><linearGradient id=\"bg\" x1=\"0\" y1=\"0\" x2=\"1\" y2=\"1\"><stop offset=\"0%\" stop-color=\"#05070d\"/><stop offset=\"100%\" stop-color=\"#101f35\"/></linearGradient></defs><rect width=\"1920\" height=\"1080\" fill=\"url(#bg)\"/><circle cx=\"1510\" cy=\"190\" r=\"220\" fill=\"{accent}\" fill-opacity=\"0.18\"/><circle cx=\"260\" cy=\"880\" r=\"260\" fill=\"#ff5e00\" fill-opacity=\"0.14\"/><text x=\"120\" y=\"860\" fill=\"#eef7ff\" font-size=\"124\" font-family=\"Orbitron, sans-serif\">{name}</text><text x=\"128\" y=\"930\" fill=\"{accent}\" font-size=\"40\" font-family=\"Rajdhani, sans-serif\">Kali-inspired gaming shell for Ubuntu 24.04 remastering</text></svg>"
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
