#!/bin/bash
#
# AURORA System Tuning Script
# Optimizes Linux system for high-performance computing
#

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
SYSCTL_CONF="/etc/sysctl.d/99-aurora.conf"
SYSTEMD_CONF="/etc/systemd/system.conf.d/aurora.conf"
LIMITS_CONF="/etc/security/limits.d/99-aurora.conf"
CPU_GOVERNOR="performance"

# Check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        echo -e "${RED}Error: This script must be run as root${NC}"
        exit 1
    fi
}

# Print banner
print_banner() {
    echo -e "${BLUE}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║              AURORA System Tuning                         ║
║         Optimizing Linux for HPC Workloads               ║
╚═══════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

# Tune sysctl parameters
tune_sysctl() {
    echo -e "${BLUE}Tuning kernel parameters...${NC}"
    
    cat > "$SYSCTL_CONF" << 'EOF'
# AURORA HPC Optimizations

# Memory settings
vm.swappiness = 10
vm.dirty_ratio = 5
vm.dirty_background_ratio = 2
vm.vfs_cache_pressure = 50
vm.zone_reclaim_mode = 0

# HugePages
vm.nr_hugepages = 512
vm.nr_overcommit_hugepages = 256

# Transparent HugePages
kernel.shmmax = 68719476736
kernel.shmall = 4294967296

# Network (for distributed computing)
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.netdev_max_backlog = 300000
net.ipv4.tcp_congestion_control = bbr

# Scheduler
kernel.sched_min_granularity_ns = 10000000
kernel.sched_wakeup_granularity_ns = 15000000
kernel.sched_migration_cost_ns = 5000000
kernel.sched_nr_migrate = 256

# IRQ balancing
kernel.irqbalance = 0
EOF
    
    sysctl --system
    echo -e "  ${GREEN}✓${NC} Kernel parameters tuned"
}

# Tune CPU governor
tune_cpu() {
    echo -e "${BLUE}Tuning CPU settings...${NC}"
    
    # Set CPU governor
    for cpu in /sys/devices/system/cpu/cpu[0-9]*; do
        if [ -f "$cpu/cpufreq/scaling_governor" ]; then
            echo "$CPU_GOVERNOR" > "$cpu/cpufreq/scaling_governor" 2>/dev/null || true
        fi
        
        # Disable C-states for compute cores
        if [ -f "$cpu/cpuidle/state3/disable" ]; then
            echo 1 > "$cpu/cpuidle/state3/disable" 2>/dev/null || true
        fi
        if [ -f "$cpu/cpuidle/state4/disable" ]; then
            echo 1 > "$cpu/cpuidle/state4/disable" 2>/dev/null || true
        fi
    done
    
    echo -e "  ${GREEN}✓${NC} CPU governor set to $CPU_GOVERNOR"
    
    # Disable CPU frequency scaling
    if command -v cpupower &> /dev/null; then
        cpupower frequency-set -g performance 2>/dev/null || true
    fi
}

# Tune memory
tune_memory() {
    echo -e "${BLUE}Tuning memory settings...${NC}"
    
    # Enable Transparent HugePages
    if [ -f /sys/kernel/mm/transparent_hugepage/enabled ]; then
        echo always > /sys/kernel/mm/transparent_hugepage/enabled 2>/dev/null || true
    fi
    
    # Disable THP defragmentation
    if [ -f /sys/kernel/mm/transparent_hugepage/defrag ]; then
        echo never > /sys/kernel/mm/transparent_hugepage/defrag 2>/dev/null || true
    fi
    
    # Set compaction proactiveness
    if [ -f /proc/sys/vm/compaction_proactiveness ]; then
        echo 0 > /proc/sys/vm/compaction_proactiveness 2>/dev/null || true
    fi
    
    echo -e "  ${GREEN}✓${NC} Memory settings tuned"
}

# Tune NUMA
tune_numa() {
    echo -e "${BLUE}Tuning NUMA settings...${NC}"
    
    # Check if NUMA is available
    if command -v numactl &> /dev/null; then
        # Enable NUMA balancing
        if [ -f /proc/sys/kernel/numa_balancing ]; then
            echo 1 > /proc/sys/kernel/numa_balancing 2>/dev/null || true
        fi
        
        echo -e "  ${GREEN}✓${NC} NUMA settings tuned"
    else
        echo -e "  ${YELLOW}!${NC} NUMA tools not installed"
    fi
}

# Tune I/O scheduler
tune_io() {
    echo -e "${BLUE}Tuning I/O scheduler...${NC}"
    
    # Set I/O scheduler to none for NVMe/SSD
    for disk in /sys/block/nvme*n1/queue/scheduler /sys/block/sd*/queue/scheduler; do
        if [ -f "$disk" ]; then
            echo none > "$disk" 2>/dev/null || true
        fi
    done
    
    # Increase I/O queue depth
    for queue in /sys/block/*/queue/nr_requests; do
        if [ -f "$queue" ]; then
            echo 4096 > "$queue" 2>/dev/null || true
        fi
    done
    
    echo -e "  ${GREEN}✓${NC} I/O scheduler tuned"
}

# Configure systemd
tune_systemd() {
    echo -e "${BLUE}Configuring systemd...${NC}"
    
    mkdir -p "$(dirname $SYSTEMD_CONF)"
    
    cat > "$SYSTEMD_CONF" << 'EOF'
[Manager]
# Increase file descriptor limits
DefaultLimitNOFILE=65536:65536
DefaultLimitMEMLOCK=infinity

# CPU affinity for system services
# DefaultCPUAffinity=0
EOF
    
    systemctl daemon-reload
    echo -e "  ${GREEN}✓${NC} Systemd configured"
}

# Configure limits
tune_limits() {
    echo -e "${BLUE}Configuring resource limits...${NC}"
    
    cat > "$LIMITS_CONF" << 'EOF'
# AURORA HPC Resource Limits

# File descriptors
* soft nofile 65536
* hard nofile 65536

# Memory locking
* soft memlock unlimited
* hard memlock unlimited

# Real-time priority
* soft rtprio 99
* hard rtprio 99

# Nice values
* soft nice -20
* hard nice -20
EOF
    
    echo -e "  ${GREEN}✓${NC} Resource limits configured"
}

# Disable unnecessary services
disable_services() {
    echo -e "${BLUE}Disabling unnecessary services...${NC}"
    
    local services=(
        "bluetooth"
        "cups"
        "avahi-daemon"
        "ModemManager"
        "snapd"
        "apport"
        "whoopsie"
    )
    
    for service in "${services[@]}"; do
        if systemctl is-enabled "$service" 2>/dev/null | grep -q enabled; then
            systemctl disable "$service" 2>/dev/null || true
            systemctl stop "$service" 2>/dev/null || true
            echo -e "  ${GREEN}✓${NC} Disabled $service"
        fi
    done
}

# Create CPU isolation script
create_isolation_script() {
    echo -e "${BLUE}Creating CPU isolation script...${NC}"
    
    cat > /usr/local/bin/aurora-isolate-cpus << 'EOF'
#!/bin/bash
# Isolate CPUs for compute workloads

if [ "$EUID" -ne 0 ]; then
    echo "Error: Must run as root"
    exit 1
fi

# Get number of CPUs
NUM_CPUS=$(nproc)

# Isolate all but first 2 CPUs for system
ISOLATED=""
for ((i=2; i<NUM_CPUS; i++)); do
    ISOLATED="$ISOLATED,$i"
done
ISOLATED=${ISOLATED#,}

# Add to kernel cmdline
if ! grep -q "isolcpus" /etc/default/grub; then
    sed -i "s/GRUB_CMDLINE_LINUX_DEFAULT=\"/GRUB_CMDLINE_LINUX_DEFAULT=\"isolcpus=$ISOLATED /" /etc/default/grub
    update-grub
    echo "CPU isolation configured. Reboot required."
else
    echo "CPU isolation already configured."
fi
EOF
    
    chmod +x /usr/local/bin/aurora-isolate-cpus
    echo -e "  ${GREEN}✓${NC} CPU isolation script created"
}

# Apply IRQ affinity
apply_irq_affinity() {
    echo -e "${BLUE}Applying IRQ affinity...${NC}"
    
    # Move all IRQs to CPU 0
    for irq in /proc/irq/*/smp_affinity; do
        if [ -f "$irq" ]; then
            echo 1 > "$irq" 2>/dev/null || true
        fi
    done
    
    echo -e "  ${GREEN}✓${NC} IRQ affinity applied"
}

# Verify tuning
verify_tuning() {
    echo -e "${BLUE}Verifying system tuning...${NC}"
    
    local errors=0
    
    # Check CPU governor
    local gov=$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo "unknown")
    if [ "$gov" != "$CPU_GOVERNOR" ]; then
        echo -e "  ${YELLOW}!${NC} CPU governor is $gov (expected $CPU_GOVERNOR)"
        ((errors++))
    else
        echo -e "  ${GREEN}✓${NC} CPU governor: $gov"
    fi
    
    # Check swappiness
    local swap=$(cat /proc/sys/vm/swappiness)
    if [ "$swap" -gt 10 ]; then
        echo -e "  ${YELLOW}!${NC} Swappiness is $swap (expected <= 10)"
        ((errors++))
    else
        echo -e "  ${GREEN}✓${NC} Swappiness: $swap"
    fi
    
    # Check HugePages
    local hp=$(cat /proc/sys/vm/nr_hugepages)
    if [ "$hp" -lt 128 ]; then
        echo -e "  ${YELLOW}!${NC} HugePages: $hp (expected >= 128)"
        ((errors++))
    else
        echo -e "  ${GREEN}✓${NC} HugePages: $hp"
    fi
    
    # Check THP
    if [ -f /sys/kernel/mm/transparent_hugepage/enabled ]; then
        local thp=$(cat /sys/kernel/mm/transparent_hugepage/enabled)
        if echo "$thp" | grep -q "\[always\]"; then
            echo -e "  ${GREEN}✓${NC} Transparent HugePages: always"
        else
            echo -e "  ${YELLOW}!${NC} Transparent HugePages: $thp"
        fi
    fi
    
    return $errors
}

# Print summary
print_summary() {
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}  System tuning complete!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    echo "Tuning applied:"
    echo "  - Kernel parameters: $SYSCTL_CONF"
    echo "  - Systemd config: $SYSTEMD_CONF"
    echo "  - Resource limits: $LIMITS_CONF"
    echo ""
    echo "Some changes require a reboot to take full effect."
    echo ""
    echo "To isolate CPUs for compute:"
    echo "  sudo /usr/local/bin/aurora-isolate-cpus"
    echo "  sudo reboot"
    echo ""
}

# Main
main() {
    print_banner
    check_root
    
    tune_sysctl
    tune_cpu
    tune_memory
    tune_numa
    tune_io
    tune_systemd
    tune_limits
    disable_services
    create_isolation_script
    apply_irq_affinity
    
    verify_tuning
    
    print_summary
}

# Handle arguments
case "${1:-}" in
    --verify)
        verify_tuning
        exit $?
        ;;
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --verify    Verify current tuning"
        echo "  --help      Show this help"
        echo ""
        exit 0
        ;;
    *)
        main
        ;;
esac
