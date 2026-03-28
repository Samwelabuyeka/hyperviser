//! Memory management and allocation

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use parking_lot::Mutex;
use crate::device::DeviceId;
use crate::error::{AuroraError, Result};

/// Memory type for allocations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MemoryType {
    /// Host memory (pageable)
    Host,
    /// Pinned/host-registered memory
    Pinned,
    /// Device memory (GPU)
    Device,
    /// Unified/shared memory
    Unified,
    /// NUMA-aware host memory
    Numa { node: i32 },
}

impl MemoryType {
    /// Check if this is host-accessible memory
    pub fn is_host_accessible(&self) -> bool {
        matches!(self, MemoryType::Host | MemoryType::Pinned | MemoryType::Unified | MemoryType::Numa { .. })
    }
    
    /// Check if this is device memory
    pub fn is_device(&self) -> bool {
        matches!(self, MemoryType::Device)
    }
    
    /// Check if this is pinned memory
    pub fn is_pinned(&self) -> bool {
        matches!(self, MemoryType::Pinned)
    }
}

/// Memory allocation handle
#[derive(Debug, Clone)]
pub struct Allocation {
    /// Allocation ID
    pub id: u64,
    /// Device where memory is allocated
    pub device: DeviceId,
    /// Memory type
    pub memory_type: MemoryType,
    /// Size in bytes
    pub size: usize,
    /// Alignment
    pub alignment: usize,
    /// Host pointer (if host-accessible)
    pub host_ptr: Option<usize>,
    /// Device pointer (if device memory)
    pub device_ptr: Option<u64>,
}

impl Allocation {
    /// Create a new allocation descriptor
    pub fn new(
        id: u64,
        device: DeviceId,
        memory_type: MemoryType,
        size: usize,
        alignment: usize,
    ) -> Self {
        Self {
            id,
            device,
            memory_type,
            size,
            alignment,
            host_ptr: None,
            device_ptr: None,
        }
    }
    
    /// Check if allocation is valid
    pub fn is_valid(&self) -> bool {
        self.host_ptr.is_some() || self.device_ptr.is_some()
    }
    
    /// Get pointer as u64 for kernel arguments
    pub fn as_u64(&self) -> u64 {
        if let Some(ptr) = self.device_ptr {
            ptr
        } else if let Some(ptr) = self.host_ptr {
            ptr as u64
        } else {
            0
        }
    }
}

/// Memory pool for efficient allocation
#[derive(Debug)]
pub struct MemoryPool {
    /// Pool ID
    id: u64,
    /// Device for this pool
    device: DeviceId,
    /// Memory type
    memory_type: MemoryType,
    /// Total capacity
    capacity: usize,
    /// Used memory
    used: Arc<Mutex<usize>>,
    /// Free blocks (size -> list of allocations)
    free_blocks: Arc<Mutex<std::collections::BTreeMap<usize, Vec<Allocation>>>>,
    /// Active allocations
    active: Arc<Mutex<std::collections::HashMap<u64, Allocation>>>,
    /// Next allocation ID
    next_id: Arc<Mutex<u64>>,
}

impl MemoryPool {
    /// Create a new memory pool
    pub fn new(id: u64, device: DeviceId, memory_type: MemoryType, capacity: usize) -> Self {
        Self {
            id,
            device,
            memory_type,
            capacity,
            used: Arc::new(Mutex::new(0)),
            free_blocks: Arc::new(Mutex::new(std::collections::BTreeMap::new())),
            active: Arc::new(Mutex::new(std::collections::HashMap::new())),
            next_id: Arc::new(Mutex::new(1)),
        }
    }
    
    /// Get pool ID
    pub fn id(&self) -> u64 {
        self.id
    }
    
    /// Get device
    pub fn device(&self) -> DeviceId {
        self.device
    }
    
    /// Get memory type
    pub fn memory_type(&self) -> MemoryType {
        self.memory_type
    }
    
    /// Get total capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }
    
    /// Get used memory
    pub fn used(&self) -> usize {
        *self.used.lock()
    }
    
    /// Get available memory
    pub fn available(&self) -> usize {
        self.capacity - self.used()
    }
    
    /// Get usage percentage
    pub fn usage_percent(&self) -> f64 {
        if self.capacity == 0 {
            return 0.0;
        }
        (self.used() as f64 / self.capacity as f64) * 100.0
    }
    
    /// Allocate memory from the pool
    pub fn allocate(&self, size: usize, alignment: usize) -> Result<Allocation> {
        if size == 0 {
            return Err(AuroraError::invalid_arg("Cannot allocate 0 bytes"));
        }
        
        // Round up to alignment
        let aligned_size = ((size + alignment - 1) / alignment) * alignment;
        
        // Check if we have enough capacity
        if aligned_size > self.available() {
            return Err(AuroraError::oom(aligned_size, self.available()));
        }
        
        // Try to find a free block
        let mut free_blocks = self.free_blocks.lock();
        let mut active = self.active.lock();
        let mut used = self.used.lock();
        let mut next_id = self.next_id.lock();
        
        // Look for a suitable free block
        let candidates: Vec<_> = free_blocks
            .range(aligned_size..)
            .take(5)
            .map(|(k, v)| (*k, v.clone()))
            .collect();
        
        for (block_size, mut blocks) in candidates {
            if let Some(mut alloc) = blocks.pop() {
                // Update the allocation
                alloc.size = size;
                alloc.alignment = alignment;
                
                // Remove the block from free list if empty
                if blocks.is_empty() {
                    free_blocks.remove(&block_size);
                } else {
                    free_blocks.insert(block_size, blocks);
                }
                
                // Add to active allocations
                let id = *next_id;
                *next_id += 1;
                alloc.id = id;
                active.insert(id, alloc.clone());
                *used += aligned_size;
                
                return Ok(alloc);
            }
        }
        
        // No suitable free block, create new allocation
        let id = *next_id;
        *next_id += 1;
        
        let alloc = Allocation::new(id, self.device, self.memory_type, size, alignment);
        active.insert(id, alloc.clone());
        *used += aligned_size;
        
        Ok(alloc)
    }
    
    /// Free an allocation back to the pool
    pub fn free(&self, allocation: Allocation) -> Result<()> {
        let mut active = self.active.lock();
        let mut free_blocks = self.free_blocks.lock();
        let mut used = self.used.lock();
        
        // Remove from active
        if active.remove(&allocation.id).is_none() {
            return Err(AuroraError::invalid_arg(
                format!("Allocation {} not found in pool", allocation.id)
            ));
        }
        
        // Add to free blocks
        let aligned_size = ((allocation.size + allocation.alignment - 1) / allocation.alignment) * allocation.alignment;
        *used -= aligned_size;
        
        free_blocks
            .entry(aligned_size)
            .or_insert_with(Vec::new)
            .push(allocation);
        
        Ok(())
    }
    
    /// Defragment the memory pool
    pub fn defragment(&self) -> Result<usize> {
        let mut free_blocks = self.free_blocks.lock();
        let before = free_blocks.len();
        
        // Simple defragmentation: merge adjacent blocks of same size
        // In a real implementation, this would be more sophisticated
        
        let after = free_blocks.len();
        Ok(before - after)
    }
    
    /// Clear all allocations
    pub fn clear(&self) -> Result<()> {
        let mut active = self.active.lock();
        let mut free_blocks = self.free_blocks.lock();
        let mut used = self.used.lock();
        
        active.clear();
        free_blocks.clear();
        *used = 0;
        
        Ok(())
    }
}

/// Memory manager for all devices
#[derive(Debug, Default)]
pub struct MemoryManager {
    /// Pools by device
    pools: std::collections::HashMap<DeviceId, Vec<MemoryPool>>,
}

impl MemoryManager {
    /// Create a new memory manager
    pub fn new() -> Self {
        Self {
            pools: std::collections::HashMap::new(),
        }
    }
    
    /// Register a memory pool for a device
    pub fn register_pool(&mut self, pool: MemoryPool) {
        self.pools
            .entry(pool.device())
            .or_insert_with(Vec::new)
            .push(pool);
    }
    
    /// Get pools for a device
    pub fn get_pools(&self, device: DeviceId) -> Option<&Vec<MemoryPool>> {
        self.pools.get(&device)
    }
    
    /// Allocate memory on a specific device
    pub fn allocate(
        &self,
        device: DeviceId,
        memory_type: MemoryType,
        size: usize,
        alignment: usize,
    ) -> Result<Allocation> {
        if let Some(pools) = self.pools.get(&device) {
            for pool in pools {
                if pool.memory_type() == memory_type {
                    return pool.allocate(size, alignment);
                }
            }
        }
        
        Err(AuroraError::device_error(
            device.to_string(),
            format!("No memory pool found for type {:?}", memory_type)
        ))
    }
    
    /// Get total memory usage across all pools
    pub fn total_usage(&self) -> usize {
        self.pools
            .values()
            .flatten()
            .map(|p| p.used())
            .sum()
    }
    
    /// Get total capacity across all pools
    pub fn total_capacity(&self) -> usize {
        self.pools
            .values()
            .flatten()
            .map(|p| p.capacity())
            .sum()
    }
}

/// Memory statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryStats {
    /// Total allocated bytes
    pub allocated_bytes: usize,
    /// Total reserved bytes
    pub reserved_bytes: usize,
    /// Peak allocated bytes
    pub peak_allocated_bytes: usize,
    /// Number of allocations
    pub num_allocations: usize,
    /// Number of cached blocks
    pub num_cached_blocks: usize,
}
