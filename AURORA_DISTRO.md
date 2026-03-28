# AURORA Distro Toolkit

`aurora-distro` is the Ubuntu 24.04 remaster toolkit for building an AURORA-branded live ISO with:

- BIOS + UEFI support
- custom distro name and branding assets
- desktop presets for `minimal`, `gnome`, and `kde`
- Kali-inspired dark visuals with a gaming-oriented neon theme
- optional inbuilt USB writing after ISO creation
- system scanning and partition planning when hardware details are unknown
- GRUB-based boot repair for BIOS and UEFI installs

## Performance Goal

The target is to push CPU-heavy workloads much harder than a stock install by combining:

- performance governor defaults
- huge page friendly configuration
- lower-latency boot/runtime defaults
- the AURORA runtime and tuning stack

This is a performance-oriented distro goal, not a universal guarantee that every workload will be 3x faster.

## Ubuntu 24 Host Setup

On the Linux remaster host:

```bash
cargo run -p aurora-distro -- prepare-host
```

That installs the main Ubuntu 24 remaster dependencies required by the toolkit.

## Initialize A Build Tree

```bash
cargo run -p aurora-distro -- init-tree --out distro --distro-name "Aurora Neon" --desktop gnome
```

This generates:

- `distro/profiles/default.json`
- branding and theme asset folders
- overlay files for GRUB and Plymouth
- a default gaming/Kali-style wallpaper and shell theme scaffold

## Inspect Hardware

```bash
cargo run -p aurora-distro -- scan-system
```

If the user does not know firmware or disk details, use:

```bash
cargo run -p aurora-distro -- plan-partitions --mode auto --disk-gb 512
```

The tool will fall back to system scanning and create a BIOS/UEFI-aware partition plan.

## Build ISO

```bash
sudo cargo run -p aurora-distro -- build-iso --tree distro --prompt-usb
```

You can also provide the USB target directly:

```bash
sudo cargo run -p aurora-distro -- build-iso --tree distro --usb-device /dev/sdb
```

## Boot Repair

UEFI example:

```bash
sudo cargo run -p aurora-distro -- repair-boot --root /mnt/target-root --efi /dev/sda1 --disk /dev/sda
```

Legacy BIOS example:

```bash
sudo cargo run -p aurora-distro -- repair-boot --root /mnt/target-root --disk /dev/sda
```
