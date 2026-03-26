/*
 * AURORA Compute Scheduler Kernel Module
 * 
 * Provides kernel-level compute scheduling and performance monitoring
 * for the AURORA high-performance compute runtime.
 */

#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/init.h>
#include <linux/kthread.h>
#include <linux/sched.h>
#include <linux/cpumask.h>
#include <linux/slab.h>
#include <linux/fs.h>
#include <linux/cdev.h>
#include <linux/device.h>
#include <linux/uaccess.h>
#include <linux/perf_event.h>
#include <linux/hw_breakpoint.h>
#include <linux/mm.h>
#include <linux/mman.h>
#include <linux/hugetlb.h>
#include <linux/numa.h>

#define AURORA_MODULE_NAME "aurora_compute"
#define AURORA_VERSION "0.1.0"
#define AURORA_DEVICE_NAME "aurora"

/* IOCTL commands */
#define AURORA_MAGIC 'A'
#define AURORA_IOCTL_SET_AFFINITY    _IOW(AURORA_MAGIC, 0, struct aurora_affinity_req)
#define AURORA_IOCTL_GET_AFFINITY    _IOR(AURORA_MAGIC, 1, struct aurora_affinity_req)
#define AURORA_IOCTL_LOCK_MEMORY     _IOW(AURORA_MAGIC, 2, struct aurora_memory_req)
#define AURORA_IOCTL_UNLOCK_MEMORY   _IOW(AURORA_MAGIC, 3, struct aurora_memory_req)
#define AURORA_IOCTL_RESERVE_HUGEPAGE _IOW(AURORA_MAGIC, 4, int)
#define AURORA_IOCTL_SET_NUMA_POLICY _IOW(AURORA_MAGIC, 5, struct aurora_numa_req)
#define AURORA_IOCTL_GET_PERF_COUNTERS _IOR(AURORA_MAGIC, 6, struct aurora_perf_counters)
#define AURORA_IOCTL_SET_CPU_GOV     _IOW(AURORA_MAGIC, 7, char[16])

/* Request structures */
struct aurora_affinity_req {
    pid_t pid;
    cpumask_t mask;
};

struct aurora_memory_req {
    unsigned long addr;
    size_t size;
};

struct aurora_numa_req {
    int node;
    unsigned long addr;
    size_t size;
};

struct aurora_perf_counters {
    u64 cycles;
    u64 instructions;
    u64 cache_misses;
    u64 cache_references;
    u64 branch_misses;
};

/* Module parameters */
static int aurora_debug = 0;
module_param(aurora_debug, int, 0644);
MODULE_PARM_DESC(aurora_debug, "Enable debug messages");

static int default_hugepages = 128;
module_param(default_hugepages, int, 0644);
MODULE_PARM_DESC(default_hugepages, "Default HugePages to reserve");

/* Device structure */
static struct class *aurora_class;
static struct cdev aurora_cdev;
static dev_t aurora_dev;

/* Performance monitoring */
static struct perf_event *aurora_pe_cycles;
static struct perf_event *aurora_pe_instructions;
static struct perf_event *aurora_pe_cache_misses;

/* Debug macro */
#define aurora_dbg(fmt, ...) \
    do { if (aurora_debug) pr_info("[AURORA] " fmt, ##__VA_ARGS__); } while(0)

/*
 * Set CPU affinity for a process
 */
static int aurora_set_affinity(struct aurora_affinity_req *req)
{
    struct task_struct *task;
    int ret;
    
    aurora_dbg("Setting affinity for PID %d\n", req->pid);
    
    rcu_read_lock();
    task = find_task_by_vpid(req->pid);
    if (!task) {
        rcu_read_unlock();
        return -ESRCH;
    }
    get_task_struct(task);
    rcu_read_unlock();
    
    ret = sched_setaffinity(req->pid, &req->mask);
    
    put_task_struct(task);
    return ret;
}

/*
 * Get CPU affinity for a process
 */
static int aurora_get_affinity(struct aurora_affinity_req *req)
{
    struct task_struct *task;
    int ret;
    
    rcu_read_lock();
    task = find_task_by_vpid(req->pid);
    if (!task) {
        rcu_read_unlock();
        return -ESRCH;
    }
    get_task_struct(task);
    rcu_read_unlock();
    
    ret = sched_getaffinity(req->pid, &req->mask);
    
    put_task_struct(task);
    return ret;
}

/*
 * Lock memory pages to prevent swapping
 */
static int aurora_lock_memory(struct aurora_memory_req *req)
{
    struct mm_struct *mm = current->mm;
    unsigned long start, end;
    int ret;
    
    aurora_dbg("Locking memory at %lx, size %zu\n", req->addr, req->size);
    
    start = req->addr & PAGE_MASK;
    end = PAGE_ALIGN(req->addr + req->size);
    
    mmap_read_lock(mm);
    ret = mlock_vma_pages_range(mm->mmap, start, end);
    mmap_read_unlock(mm);
    
    return ret;
}

/*
 * Unlock memory pages
 */
static int aurora_unlock_memory(struct aurora_memory_req *req)
{
    struct mm_struct *mm = current->mm;
    unsigned long start, end;
    
    aurora_dbg("Unlocking memory at %lx, size %zu\n", req->addr, req->size);
    
    start = req->addr & PAGE_MASK;
    end = PAGE_ALIGN(req->addr + req->size);
    
    mmap_read_lock(mm);
    munlock_vma_pages_range(mm->mmap, start, end);
    mmap_read_unlock(mm);
    
    return 0;
}

/*
 * Reserve HugePages
 */
static int aurora_reserve_hugepages(int count)
{
    int ret;
    
    aurora_dbg("Reserving %d HugePages\n", count);
    
    ret = hugetlb_reserve_pages(
        NULL, 0, count, NULL, 
        HUGETLB_SHMFS_INODE, HPAGE_SIZE
    );
    
    return ret;
}

/*
 * Set NUMA memory policy
 */
static int aurora_set_numa_policy(struct aurora_numa_req *req)
{
    struct mempolicy *pol;
    nodemask_t nodes;
    
    aurora_dbg("Setting NUMA policy for node %d\n", req->node);
    
    nodes_clear(nodes);
    node_set(req->node, nodes);
    
    pol = mpol_new(MPOL_BIND, &nodes);
    if (IS_ERR(pol))
        return PTR_ERR(pol);
    
    mpol_put(pol);
    return 0;
}

/*
 * Read performance counters
 */
static int aurora_get_perf_counters(struct aurora_perf_counters *counters)
{
    memset(counters, 0, sizeof(*counters));
    
    if (aurora_pe_cycles)
        counters->cycles = perf_event_read_value(aurora_pe_cycles, NULL, NULL);
    
    if (aurora_pe_instructions)
        counters->instructions = perf_event_read_value(aurora_pe_instructions, NULL, NULL);
    
    if (aurora_pe_cache_misses)
        counters->cache_misses = perf_event_read_value(aurora_pe_cache_misses, NULL, NULL);
    
    return 0;
}

/*
 * Set CPU governor
 */
static int aurora_set_cpu_governor(const char *governor)
{
    /* This would require cpufreq integration */
    pr_info("[AURORA] Setting CPU governor to %s\n", governor);
    return 0;
}

/*
 * IOCTL handler
 */
static long aurora_ioctl(struct file *file, unsigned int cmd, unsigned long arg)
{
    int ret = 0;
    
    switch (cmd) {
    case AURORA_IOCTL_SET_AFFINITY: {
        struct aurora_affinity_req req;
        if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
            return -EFAULT;
        ret = aurora_set_affinity(&req);
        break;
    }
    
    case AURORA_IOCTL_GET_AFFINITY: {
        struct aurora_affinity_req req;
        if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
            return -EFAULT;
        ret = aurora_get_affinity(&req);
        if (!ret && copy_to_user((void __user *)arg, &req, sizeof(req)))
            ret = -EFAULT;
        break;
    }
    
    case AURORA_IOCTL_LOCK_MEMORY: {
        struct aurora_memory_req req;
        if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
            return -EFAULT;
        ret = aurora_lock_memory(&req);
        break;
    }
    
    case AURORA_IOCTL_UNLOCK_MEMORY: {
        struct aurora_memory_req req;
        if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
            return -EFAULT;
        ret = aurora_unlock_memory(&req);
        break;
    }
    
    case AURORA_IOCTL_RESERVE_HUGEPAGE: {
        int count;
        if (copy_from_user(&count, (void __user *)arg, sizeof(count)))
            return -EFAULT;
        ret = aurora_reserve_hugepages(count);
        break;
    }
    
    case AURORA_IOCTL_SET_NUMA_POLICY: {
        struct aurora_numa_req req;
        if (copy_from_user(&req, (void __user *)arg, sizeof(req)))
            return -EFAULT;
        ret = aurora_set_numa_policy(&req);
        break;
    }
    
    case AURORA_IOCTL_GET_PERF_COUNTERS: {
        struct aurora_perf_counters counters;
        ret = aurora_get_perf_counters(&counters);
        if (!ret && copy_to_user((void __user *)arg, &counters, sizeof(counters)))
            ret = -EFAULT;
        break;
    }
    
    case AURORA_IOCTL_SET_CPU_GOV: {
        char governor[16];
        if (copy_from_user(governor, (void __user *)arg, sizeof(governor)))
            return -EFAULT;
        ret = aurora_set_cpu_governor(governor);
        break;
    }
    
    default:
        ret = -EINVAL;
    }
    
    return ret;
}

/*
 * File operations
 */
static const struct file_operations aurora_fops = {
    .owner = THIS_MODULE,
    .unlocked_ioctl = aurora_ioctl,
    .compat_ioctl = aurora_ioctl,
};

/*
 * Module initialization
 */
static int __init aurora_init(void)
{
    int ret;
    struct perf_event_attr pe_attr;
    
    pr_info("[AURORA] Loading AURORA Compute Scheduler v%s\n", AURORA_VERSION);
    
    /* Allocate device number */
    ret = alloc_chrdev_region(&aurora_dev, 0, 1, AURORA_DEVICE_NAME);
    if (ret < 0) {
        pr_err("[AURORA] Failed to allocate device number\n");
        return ret;
    }
    
    /* Create device class */
    aurora_class = class_create(THIS_MODULE, AURORA_DEVICE_NAME);
    if (IS_ERR(aurora_class)) {
        pr_err("[AURORA] Failed to create device class\n");
        ret = PTR_ERR(aurora_class);
        goto err_unregister;
    }
    
    /* Create device */
    device_create(aurora_class, NULL, aurora_dev, NULL, AURORA_DEVICE_NAME);
    
    /* Initialize character device */
    cdev_init(&aurora_cdev, &aurora_fops);
    aurora_cdev.owner = THIS_MODULE;
    
    ret = cdev_add(&aurora_cdev, aurora_dev, 1);
    if (ret) {
        pr_err("[AURORA] Failed to add character device\n");
        goto err_class;
    }
    
    /* Initialize performance counters */
    memset(&pe_attr, 0, sizeof(pe_attr));
    pe_attr.type = PERF_TYPE_HARDWARE;
    pe_attr.size = sizeof(pe_attr);
    pe_attr.disabled = 1;
    pe_attr.exclude_kernel = 1;
    pe_attr.exclude_hv = 1;
    
    pe_attr.config = PERF_COUNT_HW_CPU_CYCLES;
    aurora_pe_cycles = perf_event_create_kernel_counter(&pe_attr, -1, NULL, NULL, NULL);
    
    pe_attr.config = PERF_COUNT_HW_INSTRUCTIONS;
    aurora_pe_instructions = perf_event_create_kernel_counter(&pe_attr, -1, NULL, NULL, NULL);
    
    pe_attr.config = PERF_COUNT_HW_CACHE_MISSES;
    aurora_pe_cache_misses = perf_event_create_kernel_counter(&pe_attr, -1, NULL, NULL, NULL);
    
    /* Reserve default HugePages */
    aurora_reserve_hugepages(default_hugepages);
    
    pr_info("[AURORA] AURORA Compute Scheduler loaded successfully\n");
    return 0;

err_class:
    device_destroy(aurora_class, aurora_dev);
    class_destroy(aurora_class);
err_unregister:
    unregister_chrdev_region(aurora_dev, 1);
    return ret;
}

/*
 * Module cleanup
 */
static void __exit aurora_exit(void)
{
    pr_info("[AURORA] Unloading AURORA Compute Scheduler\n");
    
    /* Cleanup performance counters */
    if (aurora_pe_cycles)
        perf_event_release_kernel(aurora_pe_cycles);
    if (aurora_pe_instructions)
        perf_event_release_kernel(aurora_pe_instructions);
    if (aurora_pe_cache_misses)
        perf_event_release_kernel(aurora_pe_cache_misses);
    
    /* Cleanup device */
    cdev_del(&aurora_cdev);
    device_destroy(aurora_class, aurora_dev);
    class_destroy(aurora_class);
    unregister_chrdev_region(aurora_dev, 1);
    
    pr_info("[AURORA] AURORA Compute Scheduler unloaded\n");
}

module_init(aurora_init);
module_exit(aurora_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("AURORA Team");
MODULE_DESCRIPTION("AURORA Compute Scheduler Kernel Module");
MODULE_VERSION(AURORA_VERSION);
