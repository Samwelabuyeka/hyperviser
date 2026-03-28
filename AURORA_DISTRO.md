# AURORA Distro Toolkit

`aurora-distro` is the Ubuntu 24.04 remaster toolkit for building an AURORA-branded live ISO with:

- BIOS + UEFI support
- custom distro name and branding assets
- desktop presets for `minimal`, `gnome`, and `kde`
- Kali-inspired dark visuals with a gaming-oriented neon theme
- optional inbuilt USB writing after ISO creation
- system scanning and partition planning when hardware details are unknown
- GRUB-based boot repair for BIOS and UEFI installs
- autosetup performance services for first boot
- hybrid compressed-memory tuning with zram plus disk swap

## Performance Goal

The target is to push CPU-heavy workloads much harder than a stock install by combining:

- performance governor defaults
- huge page friendly configuration
- lower-latency boot/runtime defaults
- zram-backed compressed memory to delay disk swap pressure
- service trimming and preload-style caching
- stronger storage readahead and I/O defaults
- the AURORA runtime and tuning stack

This is a performance-oriented distro goal, not a universal guarantee that every workload will be 3x faster.

## Ubuntu 24 Host Setup

On the Linux remaster host:

```bash
cargo run -p aurora-distro -- prepare-host
```

That installs the main Ubuntu 24 remaster dependencies required by the toolkit.

## Docker Remaster Path

If you prefer a more self-contained Ubuntu 24 builder, use:

```bash
./scripts/remaster-in-docker.sh
```

That script builds `aurora-distro/Dockerfile.remaster`, mounts the repo into the container, initializes the distro tree, scans the system, plans partitions, validates tools, and starts the ISO build.

Important:

- Docker still needs privileged access for mount/chroot-heavy remaster steps.
- This is meant for Linux hosts with Docker available.
- In the current Windows Codex session, Docker is not installed and WSL access is blocked, so the repo can be prepared for this workflow here but not executed end-to-end here.

## Initialize A Build Tree

```bash
cargo run -p aurora-distro -- init-tree --out distro --distro-name "Aurora Neon" --desktop gnome
```

This generates:

- `distro/profiles/default.json`
- `distro/profiles/installer.json`
- `distro/profiles/performance.json`
- branding and theme asset folders
- overlay files for GRUB and Plymouth
- a default gaming/Kali-style wallpaper and shell theme scaffold
- installer UX files under `overlay/usr/share/aurora/installer/`
- first-boot launcher and autostart entries for the installer shell
- performance profile shell/sysctl files under `overlay/etc/`
- autosetup services under `overlay/etc/systemd/system/`
- `overlay/usr/local/bin/aurora-autosetup` for boot-time tuning

## Inspect Hardware

```bash
cargo run -p aurora-distro -- scan-system
```

If the user does not know firmware or disk details, use:

```bash
cargo run -p aurora-distro -- plan-partitions --mode auto --disk-gb 512
```

The tool will fall back to system scanning and create a BIOS/UEFI-aware partition plan.

The generated plan now aims for:

- EFI support where needed
- BIOS GRUB support where needed
- root plus optional home partitioning on larger disks
- a dedicated swap partition sized from available RAM and disk
- zram enabled on first boot so compressed memory behaves like fast overflow before disk swap is used heavily

## Build ISO

```bash
sudo cargo run -p aurora-distro -- build-iso --tree distro --prompt-usb
```

You can also provide the USB target directly:

```bash
sudo cargo run -p aurora-distro -- build-iso --tree distro --usb-device /dev/sdb
```

The build command now also writes:

- `distro/build/system-profile.json`
- `distro/build/partition-plan.json`

so the remaster output keeps the detected hardware scan and generated partition strategy together.

The generated runtime profile now also stages:

- `zram-tools`
- `preload`
- `numactl`
- `linux-tools-generic`
- AURORA autosetup and zram systemd services

## Boot Repair

UEFI example:

```bash
sudo cargo run -p aurora-distro -- repair-boot --root /mnt/target-root --efi /dev/sda1 --disk /dev/sda
```

Legacy BIOS example:

```bash
sudo cargo run -p aurora-distro -- repair-boot --root /mnt/target-root --disk /dev/sda
```
