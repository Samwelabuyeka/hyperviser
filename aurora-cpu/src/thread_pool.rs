//! Work-stealing thread pool for CPU execution

use aurora_core::error::{AuroraError, Result};
use crossbeam::deque::{Stealer, Worker as DequeWorker};
use parking_lot::{Condvar, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

/// Task identifier
pub type TaskId = u64;

/// Task function type
pub type TaskFn = Box<dyn FnOnce() + Send + 'static>;

/// A task to be executed
pub struct Task {
    /// Task ID
    pub id: TaskId,
    /// Task function
    pub func: TaskFn,
    /// Priority (lower = higher priority)
    pub priority: u32,
}

impl Task {
    /// Create a new task
    pub fn new<F>(id: TaskId, func: F) -> Self
    where
        F: FnOnce() + Send + 'static,
    {
        Self {
            id,
            func: Box::new(func),
            priority: 0,
        }
    }
    
    /// Set priority
    pub fn with_priority(mut self, priority: u32) -> Self {
        self.priority = priority;
        self
    }
    
    /// Execute the task
    pub fn run(self) {
        (self.func)()
    }
}

/// Thread pool for parallel execution
pub struct ThreadPool {
    /// Number of worker threads
    num_workers: usize,
    /// Worker threads
    workers: Vec<Worker>,
    /// Global task queue
    global_queue: Arc<Mutex<Vec<Task>>>,
    /// Condition variable for task notification
    condvar: Arc<Condvar>,
    /// Shutdown flag
    shutdown: Arc<AtomicBool>,
    /// Task counter
    task_counter: Arc<AtomicU64>,
}

impl ThreadPool {
    /// Create a new thread pool
    pub fn new(num_workers: usize) -> Result<Self> {
        if num_workers == 0 {
            return Err(AuroraError::invalid_arg("Thread pool must have at least 1 worker"));
        }
        
        let global_queue = Arc::new(Mutex::new(Vec::new()));
        let condvar = Arc::new(Condvar::new());
        let shutdown = Arc::new(AtomicBool::new(false));
        let task_counter = Arc::new(AtomicU64::new(0));
        
        let mut workers = Vec::with_capacity(num_workers);
        
        for id in 0..num_workers {
            let worker = Worker::new(
                id,
                global_queue.clone(),
                condvar.clone(),
                shutdown.clone(),
                task_counter.clone(),
            );
            workers.push(worker);
        }
        
        Ok(Self {
            num_workers,
            workers,
            global_queue,
            condvar,
            shutdown,
            task_counter,
        })
    }
    
    /// Get number of workers
    pub fn num_workers(&self) -> usize {
        self.num_workers
    }
    
    /// Submit a task to the pool
    pub fn submit<F>(&self, func: F) -> TaskId
    where
        F: FnOnce() + Send + 'static,
    {
        let id = self.task_counter.fetch_add(1, Ordering::SeqCst);
        let task = Task::new(id, func);
        
        {
            let mut queue = self.global_queue.lock();
            queue.push(task);
        }
        
        self.condvar.notify_one();
        id
    }
    
    /// Submit a task with priority
    pub fn submit_priority<F>(&self, func: F, priority: u32) -> TaskId
    where
        F: FnOnce() + Send + 'static,
    {
        let id = self.task_counter.fetch_add(1, Ordering::SeqCst);
        let task = Task::new(id, func).with_priority(priority);
        
        {
            let mut queue = self.global_queue.lock();
            // Insert based on priority (simple insertion sort)
            let pos = queue.iter().position(|t| t.priority > priority).unwrap_or(queue.len());
            queue.insert(pos, task);
        }
        
        self.condvar.notify_one();
        id
    }
    
    /// Execute a function and wait for result
    pub fn execute<F, T>(&self, func: F) -> Result<T>
    where
        F: FnOnce() -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let (tx, rx) = std::sync::mpsc::channel();
        
        self.submit(move || {
            let result = func();
            let _ = tx.send(result);
        });
        
        rx.recv().map_err(|_| AuroraError::invalid_arg("Task execution failed"))
    }
    
    /// Execute a function in parallel across all workers
    pub fn parallel_for<F>(&self, range: std::ops::Range<usize>, func: F) -> Result<()>
    where
        F: Fn(usize) + Send + Sync + 'static,
    {
        let func = Arc::new(func);
        let chunk_size = (range.end - range.start + self.num_workers - 1) / self.num_workers;
        
        let mut handles = Vec::new();
        
        for worker_id in 0..self.num_workers {
            let func = func.clone();
            let start = range.start + worker_id * chunk_size;
            let end = (start + chunk_size).min(range.end);
            
            if start >= end {
                continue;
            }
            
            let handle = thread::spawn(move || {
                for i in start..end {
                    func(i);
                }
            });
            
            handles.push(handle);
        }
        
        for handle in handles {
            handle.join().map_err(|_| AuroraError::invalid_arg("Worker thread panicked"))?;
        }
        
        Ok(())
    }
    
    /// Shutdown the thread pool
    pub fn shutdown(&mut self) -> Result<()> {
        self.shutdown.store(true, Ordering::SeqCst);
        self.condvar.notify_all();
        
        for worker in self.workers.drain(..) {
            worker.join()?;
        }
        
        Ok(())
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        let _ = self.shutdown();
    }
}

impl std::fmt::Debug for ThreadPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThreadPool")
            .field("num_workers", &self.num_workers)
            .finish()
    }
}

/// Worker thread
pub struct Worker {
    /// Worker ID
    id: usize,
    /// Thread handle
    handle: Option<JoinHandle<()>>,
    /// Local task queue (work-stealing)
    local_queue: DequeWorker<Task>,
    /// Stealers from other workers
    stealers: Vec<Stealer<Task>>,
}

impl Worker {
    /// Create and start a new worker
    fn new(
        id: usize,
        global_queue: Arc<Mutex<Vec<Task>>>,
        condvar: Arc<Condvar>,
        shutdown: Arc<AtomicBool>,
        task_counter: Arc<AtomicU64>,
    ) -> Self {
        let local_queue = DequeWorker::new_fifo();
        
        let handle = thread::spawn(move || {
            Self::run_loop(
                id,
                global_queue,
                condvar,
                shutdown,
                task_counter,
            );
        });
        
        Self {
            id,
            handle: Some(handle),
            local_queue,
            stealers: Vec::new(),
        }
    }
    
    /// Main worker loop
    fn run_loop(
        id: usize,
        global_queue: Arc<Mutex<Vec<Task>>>,
        condvar: Arc<Condvar>,
        shutdown: Arc<AtomicBool>,
        _task_counter: Arc<AtomicU64>,
    ) {
        loop {
            // Try to get a task
            let task = {
                let mut queue = global_queue.lock();
                
                // Wait for task or shutdown
                while queue.is_empty() && !shutdown.load(Ordering::SeqCst) {
                    condvar.wait(&mut queue);
                }
                
                if shutdown.load(Ordering::SeqCst) && queue.is_empty() {
                    return;
                }
                
                queue.pop()
            };
            
            if let Some(task) = task {
                task.run();
            }
        }
    }
    
    /// Join the worker thread
    fn join(mut self) -> Result<()> {
        if let Some(handle) = self.handle.take() {
            handle.join().map_err(|_| AuroraError::invalid_arg("Worker thread panicked"))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    #[test]
    fn test_thread_pool_creation() {
        let pool = ThreadPool::new(4).unwrap();
        assert_eq!(pool.num_workers(), 4);
    }

    #[test]
    fn test_task_submission() {
        let pool = ThreadPool::new(2).unwrap();
        let counter = Arc::new(AtomicUsize::new(0));
        
        let c = counter.clone();
        pool.submit(move || {
            c.fetch_add(1, Ordering::SeqCst);
        });
        
        // Give time for execution
        std::thread::sleep(std::time::Duration::from_millis(100));
        
        assert!(counter.load(Ordering::SeqCst) >= 1);
    }

    #[test]
    fn test_parallel_for() {
        let pool = ThreadPool::new(4).unwrap();
        let results = Arc::new(Mutex::new(vec![0; 100]));
        
        let r = results.clone();
        pool.parallel_for(0..100, move |i| {
            r.lock()[i] = i * i;
        }).unwrap();
        
        let final_results = results.lock();
        assert_eq!(final_results[10], 100);
        assert_eq!(final_results[20], 400);
    }
}
