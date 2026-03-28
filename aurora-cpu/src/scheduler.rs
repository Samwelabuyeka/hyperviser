//! Lightweight CPU scheduler helpers.

use crate::thread_pool::ThreadPool;
use aurora_core::error::Result;
use std::ops::Range;

/// Scheduler facade over the CPU thread pool.
#[derive(Debug)]
pub struct Scheduler {
    pool: ThreadPool,
}

impl Scheduler {
    /// Create a scheduler with a fixed worker count.
    pub fn new(workers: usize) -> Result<Self> {
        Ok(Self {
            pool: ThreadPool::new(workers)?,
        })
    }

    /// Run a parallel loop over the provided range.
    pub fn parallel_for<F>(&self, range: Range<usize>, func: F) -> Result<()>
    where
        F: Fn(usize) + Send + Sync + 'static,
    {
        self.pool.parallel_for(range, func)
    }

    /// Expose the backing pool for advanced callers.
    pub fn pool(&self) -> &ThreadPool {
        &self.pool
    }
}
