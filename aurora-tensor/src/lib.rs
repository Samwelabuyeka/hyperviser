//! AURORA Tensor Operations
//!
//! Host-side tensor execution, graph execution, pooled memory, and a
//! persistent work-stealing executor for the current runtime.

#![warn(missing_docs)]

use aurora_core::device::DeviceId;
use aurora_core::error::{AuroraError, Result};
use aurora_core::tensor::{DataType, Tensor, TensorShape};
use aurora_linux::{preferred_cpu_order, set_cpu_affinity};
use matrixmultiply::sgemm;
use parking_lot::Mutex;
use rayon::prelude::*;
use rayon::ThreadPool;
use rayon::ThreadPoolBuilder;
use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use wide::f32x8;

/// Host-resident tensor storage used by the current runtime execution path.
#[derive(Debug, Clone)]
pub struct HostTensor {
    desc: Tensor,
    data: Vec<f32>,
}

/// Symmetrically quantized int8 tensor for CPU-first low-power execution.
#[derive(Debug, Clone)]
pub struct QuantizedTensorI8 {
    shape: Vec<usize>,
    scale: f32,
    data: Vec<i8>,
}

impl QuantizedTensorI8 {
    /// Tensor shape.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Quantization scale.
    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Raw int8 storage.
    pub fn as_slice(&self) -> &[i8] {
        &self.data
    }
}

impl HostTensor {
    /// Create a zero-initialized host tensor.
    pub fn zeros(shape: &[usize]) -> Self {
        Self::filled(shape, 0.0)
    }

    /// Create a host tensor filled with a constant value.
    pub fn filled(shape: &[usize], value: f32) -> Self {
        let shape = TensorShape::new(shape);
        let numel = shape.numel();
        Self {
            desc: Tensor::new(0, shape, DataType::F32, DeviceId::CPU),
            data: vec![value; numel],
        }
    }

    /// Create a host tensor from existing data.
    pub fn from_vec(shape: &[usize], data: Vec<f32>) -> Result<Self> {
        let shape = TensorShape::new(shape);
        if shape.numel() != data.len() {
            return Err(AuroraError::shape_mismatch(
                vec![shape.numel()],
                vec![data.len()],
            ));
        }
        Ok(Self {
            desc: Tensor::new(0, shape, DataType::F32, DeviceId::CPU),
            data,
        })
    }

    /// Create a host tensor from an owned buffer.
    pub fn from_buffer(shape: &[usize], data: Vec<f32>) -> Result<Self> {
        Self::from_vec(shape, data)
    }

    /// Get the tensor descriptor.
    pub fn descriptor(&self) -> &Tensor {
        &self.desc
    }

    /// Get the tensor shape.
    pub fn shape(&self) -> &TensorShape {
        &self.desc.shape
    }

    /// Get the tensor data.
    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }

    /// Get mutable tensor data.
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        &mut self.data
    }

    /// Number of elements.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the tensor has no elements.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Consume the tensor and return the underlying storage.
    pub fn into_vec(self) -> Vec<f32> {
        self.data
    }
}

/// Quantize a host tensor into symmetric int8 storage.
pub fn quantize_i8(input: &HostTensor) -> QuantizedTensorI8 {
    let max_abs = input
        .as_slice()
        .iter()
        .copied()
        .map(f32::abs)
        .fold(0.0f32, f32::max)
        .max(1e-8);
    let scale = max_abs / 127.0;
    let inv_scale = 1.0 / scale;
    let data = input
        .as_slice()
        .iter()
        .copied()
        .map(|v| (v * inv_scale).round().clamp(-127.0, 127.0) as i8)
        .collect::<Vec<_>>();
    QuantizedTensorI8 {
        shape: input.shape().dims().to_vec(),
        scale,
        data,
    }
}

/// Dequantize an int8 tensor back to host `f32`.
pub fn dequantize_i8(input: &QuantizedTensorI8) -> Result<HostTensor> {
    HostTensor::from_vec(
        &input.shape,
        input
            .data
            .iter()
            .copied()
            .map(|v| v as f32 * input.scale)
            .collect(),
    )
}

/// Blocked int8 matrix multiplication with dequantized output.
pub fn quantized_matmul_i8(
    a: &QuantizedTensorI8,
    b: &QuantizedTensorI8,
    c: &mut HostTensor,
) -> Result<()> {
    if a.shape.len() != 2 || b.shape.len() != 2 || c.shape().dims().len() != 2 {
        return Err(AuroraError::invalid_arg(
            "quantized_matmul_i8 expects 2D tensors",
        ));
    }
    let m = a.shape[0];
    let k_a = a.shape[1];
    let k_b = b.shape[0];
    let n = b.shape[1];
    let out = c.shape().dims();
    if k_a != k_b {
        return Err(AuroraError::shape_mismatch(vec![m, k_a], vec![k_b, n]));
    }
    if out != [m, n] {
        return Err(AuroraError::shape_mismatch(vec![m, n], out.to_vec()));
    }

    let scale = a.scale * b.scale;
    let mut b_t = vec![0i8; b.data.len()];
    for row in 0..k_a {
        for col in 0..n {
            b_t[col * k_a + row] = b.data[row * n + col];
        }
    }

    const ROW_BLOCK: usize = 16;
    const COL_BLOCK: usize = 32;
    const K_BLOCK: usize = 64;

    c.as_mut_slice()
        .par_chunks_mut(ROW_BLOCK * n)
        .enumerate()
        .for_each(|(row_block_idx, out_block)| {
            let row_start = row_block_idx * ROW_BLOCK;
            let rows = (m - row_start).min(ROW_BLOCK);
            for col_start in (0..n).step_by(COL_BLOCK) {
                let cols = (n - col_start).min(COL_BLOCK);
                let mut acc = vec![0i32; rows * cols];
                for k_start in (0..k_a).step_by(K_BLOCK) {
                    let depth = (k_a - k_start).min(K_BLOCK);
                    for r in 0..rows {
                        let a_row = &a.data[(row_start + r) * k_a + k_start..(row_start + r) * k_a + k_start + depth];
                        for c_idx in 0..cols {
                            let b_col = &b_t[(col_start + c_idx) * k_a + k_start..(col_start + c_idx) * k_a + k_start + depth];
                            let mut local = 0i32;
                            for kk in 0..depth {
                                local += (a_row[kk] as i32) * (b_col[kk] as i32);
                            }
                            acc[r * cols + c_idx] += local;
                        }
                    }
                }
                for r in 0..rows {
                    let out_row = &mut out_block[r * n..(r + 1) * n];
                    for c_idx in 0..cols {
                        out_row[col_start + c_idx] = acc[r * cols + c_idx] as f32 * scale;
                    }
                }
            }
        });
    Ok(())
}

/// Memory pool statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct MemoryPoolStats {
    /// Total bytes allocated by the pool.
    pub allocated_bytes: usize,
    /// Total buffers currently cached.
    pub cached_buffers: usize,
    /// Number of buffer reuses.
    pub reused_buffers: usize,
}

/// Simple host memory pool for reusable `f32` buffers.
#[derive(Debug, Default)]
pub struct HostMemoryPool {
    free: Mutex<HashMap<usize, Vec<Vec<f32>>>>,
    allocated_bytes: AtomicUsize,
    reused_buffers: AtomicUsize,
}

impl HostMemoryPool {
    /// Create an empty pool.
    pub fn new() -> Self {
        Self::default()
    }

    /// Acquire a reusable buffer sized for `len` elements.
    pub fn take(&self, len: usize) -> Vec<f32> {
        let mut free: parking_lot::lock_api::MutexGuard<'_, parking_lot::RawMutex, HashMap<usize, Vec<Vec<f32>>>> =
            self.free.lock();
        if let Some(bucket) = free.get_mut(&len) {
            if let Some(mut buffer) = bucket.pop() {
                buffer.resize(len, 0.0);
                self.reused_buffers.fetch_add(1, Ordering::Relaxed);
                return buffer;
            }
        }
        self.allocated_bytes
            .fetch_add(len * std::mem::size_of::<f32>(), Ordering::Relaxed);
        vec![0.0; len]
    }

    /// Return a buffer to the pool.
    pub fn release(&self, mut buffer: Vec<f32>) {
        let len = buffer.len();
        buffer.fill(0.0);
        self.free.lock().entry(len).or_default().push(buffer);
    }

    /// Allocate a pooled tensor filled with zeros.
    pub fn tensor_zeros(&self, shape: &[usize]) -> Result<HostTensor> {
        let len = TensorShape::new(shape).numel();
        HostTensor::from_buffer(shape, self.take(len))
    }

    /// Snapshot current pool stats.
    pub fn stats(&self) -> MemoryPoolStats {
        let free: parking_lot::lock_api::MutexGuard<'_, parking_lot::RawMutex, HashMap<usize, Vec<Vec<f32>>>> =
            self.free.lock();
        let cached_buffers = free
            .values()
            .map(|buffers: &Vec<Vec<f32>>| buffers.len())
            .sum();
        MemoryPoolStats {
            allocated_bytes: self.allocated_bytes.load(Ordering::Relaxed),
            cached_buffers,
            reused_buffers: self.reused_buffers.load(Ordering::Relaxed),
        }
    }
}

/// Persistent executor backed by a pinned Rayon work-stealing pool.
pub struct PersistentExecutor {
    pool: ThreadPool,
    memory_pool: Arc<HostMemoryPool>,
    threads: usize,
    pinned_cpus: Vec<usize>,
}

impl PersistentExecutor {
    /// Create a new persistent executor.
    pub fn new(threads: usize, pin_threads: bool) -> Result<Self> {
        let threads = threads.max(1);
        let cpu_order = preferred_cpu_order().unwrap_or_else(|_| {
            let cpu_count = std::thread::available_parallelism()
                .map(|count| count.get())
                .unwrap_or(1)
                .max(1);
            (0..cpu_count).collect::<Vec<_>>()
        });
        let pinned_cpus = if pin_threads {
            (0..threads)
                .map(|index| cpu_order[index % cpu_order.len().max(1)])
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let start_cpus = pinned_cpus.clone();
        let pool = ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|idx| format!("aurora-worker-{idx}"))
            .start_handler(move |idx| {
                if let Some(&cpu) = start_cpus.get(idx) {
                    let _ = set_cpu_affinity(&[cpu]);
                }
            })
            .build()
            .map_err(|err| AuroraError::Other(format!("failed to build persistent executor: {err}")))?;
        Ok(Self {
            pool,
            memory_pool: Arc::new(HostMemoryPool::new()),
            threads,
            pinned_cpus,
        })
    }

    /// Number of worker threads.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// CPU pinning layout.
    pub fn pinned_cpus(&self) -> &[usize] {
        &self.pinned_cpus
    }

    /// Access the shared memory pool.
    pub fn memory_pool(&self) -> &Arc<HostMemoryPool> {
        &self.memory_pool
    }

    /// Run work inside the persistent work-stealing pool.
    pub fn install<T, F>(&self, func: F) -> Result<T>
    where
        T: Send,
        F: FnOnce(&HostMemoryPool) -> Result<T> + Send,
    {
        let pool = self.memory_pool.clone();
        self.pool.install(|| func(&pool))
    }

    /// Execute a graph in the persistent executor.
    pub fn execute_graph(&self, graph: &ExecutionGraph) -> Result<GraphExecution> {
        self.install(|pool| execute_graph_with_pool(graph, pool))
    }
}

/// Graph node kinds for reusable runtime execution.
#[derive(Debug, Clone)]
pub enum GraphNode {
    /// ReLU node.
    Relu { input: String, output: String },
    /// Sigmoid node.
    Sigmoid { input: String, output: String },
    /// GELU node.
    Gelu { input: String, output: String },
    /// Softmax node.
    Softmax { input: String, output: String, axis: i32 },
    /// LayerNorm node.
    LayerNorm {
        input: String,
        gamma: String,
        beta: String,
        output: String,
        epsilon: f32,
    },
    /// RMSNorm node.
    RmsNorm {
        input: String,
        weight: String,
        output: String,
        epsilon: f32,
    },
    /// Add node.
    Add { lhs: String, rhs: String, output: String },
    /// Multiply node.
    Mul { lhs: String, rhs: String, output: String },
    /// Fused add + ReLU node.
    FusedAddRelu { lhs: String, rhs: String, output: String },
    /// Matmul node.
    Matmul { lhs: String, rhs: String, output: String },
    /// Fused matmul + GELU node.
    FusedMatmulGelu { lhs: String, rhs: String, output: String },
    /// Reduce-sum node.
    ReduceSum { input: String, output: String },
    /// Reduce-max node.
    ReduceMax { input: String, output: String },
    /// Transpose node.
    Transpose { input: String, output: String },
    /// Im2Col node.
    Im2Col {
        input: String,
        output: String,
        kernel: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
    },
    /// Attention node.
    Attention {
        query: String,
        key: String,
        value: String,
        output: String,
    },
}

/// Declarative execution graph.
#[derive(Debug, Clone, Default)]
pub struct ExecutionGraph {
    inputs: HashMap<String, HostTensor>,
    nodes: Vec<GraphNode>,
}

impl ExecutionGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a named input tensor.
    pub fn with_input(mut self, name: impl Into<String>, tensor: HostTensor) -> Self {
        self.inputs.insert(name.into(), tensor);
        self
    }

    /// Add a graph node.
    pub fn add_node(mut self, node: GraphNode) -> Self {
        self.nodes.push(node);
        self
    }
}

/// Graph execution result.
#[derive(Debug, Clone)]
pub struct GraphExecution {
    /// Materialized tensors by name.
    pub tensors: HashMap<String, HostTensor>,
    /// Pool stats after execution.
    pub pool_stats: MemoryPoolStats,
}

fn execute_graph_with_pool(graph: &ExecutionGraph, pool: &HostMemoryPool) -> Result<GraphExecution> {
    let mut tensors = graph.inputs.clone();

    for node in &graph.nodes {
        match node {
            GraphNode::Relu { input, output } => {
                let source = tensor_ref(&tensors, input)?;
                let mut result = pool.tensor_zeros(source.shape().dims())?;
                relu(source, &mut result)?;
                tensors.insert(output.clone(), result);
            }
            GraphNode::Sigmoid { input, output } => {
                let source = tensor_ref(&tensors, input)?;
                let mut result = pool.tensor_zeros(source.shape().dims())?;
                sigmoid(source, &mut result)?;
                tensors.insert(output.clone(), result);
            }
            GraphNode::Gelu { input, output } => {
                let source = tensor_ref(&tensors, input)?;
                let mut result = pool.tensor_zeros(source.shape().dims())?;
                gelu(source, &mut result)?;
                tensors.insert(output.clone(), result);
            }
            GraphNode::Softmax { input, output, axis } => {
                let source = tensor_ref(&tensors, input)?;
                let mut result = pool.tensor_zeros(source.shape().dims())?;
                softmax(source, &mut result, *axis)?;
                tensors.insert(output.clone(), result);
            }
            GraphNode::LayerNorm {
                input,
                gamma,
                beta,
                output,
                epsilon,
            } => {
                let source = tensor_ref(&tensors, input)?;
                let gamma = tensor_ref(&tensors, gamma)?;
                let beta = tensor_ref(&tensors, beta)?;
                let mut result = pool.tensor_zeros(source.shape().dims())?;
                layer_norm(source, gamma, beta, &mut result, *epsilon)?;
                tensors.insert(output.clone(), result);
            }
            GraphNode::RmsNorm {
                input,
                weight,
                output,
                epsilon,
            } => {
                let source = tensor_ref(&tensors, input)?;
                let weight = tensor_ref(&tensors, weight)?;
                let mut result = pool.tensor_zeros(source.shape().dims())?;
                rms_norm(source, weight, &mut result, *epsilon)?;
                tensors.insert(output.clone(), result);
            }
            GraphNode::Add { lhs, rhs, output } => {
                let lhs = tensor_ref(&tensors, lhs)?;
                let rhs = tensor_ref(&tensors, rhs)?;
                let mut result = pool.tensor_zeros(lhs.shape().dims())?;
                add(lhs, rhs, &mut result)?;
                tensors.insert(output.clone(), result);
            }
            GraphNode::Mul { lhs, rhs, output } => {
                let lhs = tensor_ref(&tensors, lhs)?;
                let rhs = tensor_ref(&tensors, rhs)?;
                let mut result = pool.tensor_zeros(lhs.shape().dims())?;
                mul(lhs, rhs, &mut result)?;
                tensors.insert(output.clone(), result);
            }
            GraphNode::FusedAddRelu { lhs, rhs, output } => {
                let lhs = tensor_ref(&tensors, lhs)?;
                let rhs = tensor_ref(&tensors, rhs)?;
                let mut result = pool.tensor_zeros(lhs.shape().dims())?;
                fused_add_relu(lhs, rhs, &mut result)?;
                tensors.insert(output.clone(), result);
            }
            GraphNode::Matmul { lhs, rhs, output } => {
                let lhs = tensor_ref(&tensors, lhs)?;
                let rhs = tensor_ref(&tensors, rhs)?;
                let (m, _) = matrix_dims(lhs)?;
                let (_, n) = matrix_dims(rhs)?;
                let mut result = pool.tensor_zeros(&[m, n])?;
                matmul(lhs, rhs, &mut result)?;
                tensors.insert(output.clone(), result);
            }
            GraphNode::FusedMatmulGelu { lhs, rhs, output } => {
                let lhs = tensor_ref(&tensors, lhs)?;
                let rhs = tensor_ref(&tensors, rhs)?;
                let (m, _) = matrix_dims(lhs)?;
                let (_, n) = matrix_dims(rhs)?;
                let mut result = pool.tensor_zeros(&[m, n])?;
                fused_matmul_gelu(lhs, rhs, &mut result)?;
                tensors.insert(output.clone(), result);
            }
            GraphNode::ReduceSum { input, output } => {
                let source = tensor_ref(&tensors, input)?;
                let mut result = pool.tensor_zeros(&reduced_last_shape(source.shape().dims()))?;
                reduce_sum_last(source, &mut result)?;
                tensors.insert(output.clone(), result);
            }
            GraphNode::ReduceMax { input, output } => {
                let source = tensor_ref(&tensors, input)?;
                let mut result = pool.tensor_zeros(&reduced_last_shape(source.shape().dims()))?;
                reduce_max_last(source, &mut result)?;
                tensors.insert(output.clone(), result);
            }
            GraphNode::Transpose { input, output } => {
                let source = tensor_ref(&tensors, input)?;
                let mut result = pool.tensor_zeros(source.shape().transpose().dims())?;
                transpose_last_two(source, &mut result)?;
                tensors.insert(output.clone(), result);
            }
            GraphNode::Im2Col {
                input,
                output,
                kernel,
                stride,
                padding,
            } => {
                let source = tensor_ref(&tensors, input)?;
                let shape = im2col_output_shape(source.shape().dims(), *kernel, *stride, *padding)?;
                let mut result = pool.tensor_zeros(&shape)?;
                im2col(source, &mut result, *kernel, *stride, *padding)?;
                tensors.insert(output.clone(), result);
            }
            GraphNode::Attention {
                query,
                key,
                value,
                output,
            } => {
                let query = tensor_ref(&tensors, query)?;
                let key = tensor_ref(&tensors, key)?;
                let value = tensor_ref(&tensors, value)?;
                let shape = attention_output_shape(query, key, value)?;
                let mut result = pool.tensor_zeros(&shape)?;
                attention(query, key, value, &mut result)?;
                tensors.insert(output.clone(), result);
            }
        }
    }

    Ok(GraphExecution {
        tensors,
        pool_stats: pool.stats(),
    })
}

fn tensor_ref<'a>(tensors: &'a HashMap<String, HostTensor>, name: &str) -> Result<&'a HostTensor> {
    tensors
        .get(name)
        .ok_or_else(|| AuroraError::not_found(format!("graph tensor '{name}' is missing")))
}

/// Matrix multiplication on host tensors.
pub fn matmul(a: &HostTensor, b: &HostTensor, c: &mut HostTensor) -> Result<()> {
    let (m, k_a) = matrix_dims(a)?;
    let (k_b, n) = matrix_dims(b)?;
    let (m_c, n_c) = matrix_dims(c)?;

    if k_a != k_b {
        return Err(AuroraError::shape_mismatch(vec![m, k_a], vec![k_b, n]));
    }
    if m != m_c || n != n_c {
        return Err(AuroraError::shape_mismatch(vec![m, n], vec![m_c, n_c]));
    }

    unsafe {
        sgemm(
            m,
            k_a,
            n,
            1.0,
            a.as_slice().as_ptr(),
            k_a as isize,
            1,
            b.as_slice().as_ptr(),
            n as isize,
            1,
            0.0,
            c.as_mut_slice().as_mut_ptr(),
            n as isize,
            1,
        );
    }

    Ok(())
}

/// Fused matrix multiplication followed by GELU.
pub fn fused_matmul_gelu(a: &HostTensor, b: &HostTensor, c: &mut HostTensor) -> Result<()> {
    matmul(a, b, c)?;
    gelu_in_place(c.as_mut_slice());
    Ok(())
}

/// 2D convolution for NCHW host tensors.
pub fn conv2d(
    input: &HostTensor,
    weight: &HostTensor,
    bias: Option<&HostTensor>,
    output: &mut HostTensor,
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<()> {
    let in_dims = input.shape().dims();
    let weight_dims = weight.shape().dims();
    let out_dims = output.shape().dims();

    if in_dims.len() != 4 || weight_dims.len() != 4 || out_dims.len() != 4 {
        return Err(AuroraError::invalid_arg(
            "conv2d expects NCHW input/output and OIHW weights",
        ));
    }

    let batch = in_dims[0];
    let in_channels = in_dims[1];
    let in_h = in_dims[2];
    let in_w = in_dims[3];
    let out_channels = weight_dims[0];
    let weight_in_channels = weight_dims[1];
    let kernel_h = weight_dims[2];
    let kernel_w = weight_dims[3];

    if in_channels != weight_in_channels {
        return Err(AuroraError::shape_mismatch(
            vec![in_channels],
            vec![weight_in_channels],
        ));
    }

    if let Some(bias) = bias {
        if bias.shape().dims() != [out_channels] {
            return Err(AuroraError::shape_mismatch(
                vec![out_channels],
                bias.shape().dims().to_vec(),
            ));
        }
    }

    let expected_out_h = ((in_h + 2 * padding.0).saturating_sub(kernel_h) / stride.0) + 1;
    let expected_out_w = ((in_w + 2 * padding.1).saturating_sub(kernel_w) / stride.1) + 1;
    let expected_out = [batch, out_channels, expected_out_h, expected_out_w];
    if out_dims != expected_out {
        return Err(AuroraError::shape_mismatch(expected_out.to_vec(), out_dims.to_vec()));
    }

    output
        .as_mut_slice()
        .par_chunks_mut(out_channels * expected_out_h * expected_out_w)
        .enumerate()
        .for_each(|(n, out_batch)| {
            for oc in 0..out_channels {
                let bias_value = bias.map(|b| b.as_slice()[oc]).unwrap_or(0.0);
                for oh in 0..expected_out_h {
                    for ow in 0..expected_out_w {
                        let mut acc = bias_value;
                        for ic in 0..in_channels {
                            for kh in 0..kernel_h {
                                let ih = oh * stride.0 + kh;
                                if ih < padding.0 {
                                    continue;
                                }
                                let ih = ih - padding.0;
                                if ih >= in_h {
                                    continue;
                                }
                                for kw in 0..kernel_w {
                                    let iw = ow * stride.1 + kw;
                                    if iw < padding.1 {
                                        continue;
                                    }
                                    let iw = iw - padding.1;
                                    if iw >= in_w {
                                        continue;
                                    }

                                    let input_idx = ((n * in_channels + ic) * in_h + ih) * in_w + iw;
                                    let weight_idx =
                                        ((oc * in_channels + ic) * kernel_h + kh) * kernel_w + kw;
                                    acc += input.as_slice()[input_idx] * weight.as_slice()[weight_idx];
                                }
                            }
                        }
                        let out_idx = (oc * expected_out_h + oh) * expected_out_w + ow;
                        out_batch[out_idx] = acc;
                    }
                }
            }
        });

    Ok(())
}

/// Convert an NCHW tensor to an im2col matrix.
pub fn im2col(
    input: &HostTensor,
    output: &mut HostTensor,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<()> {
    let in_dims = input.shape().dims();
    if in_dims.len() != 4 {
        return Err(AuroraError::invalid_arg("im2col expects a 4D NCHW tensor"));
    }
    let out_shape = im2col_output_shape(in_dims, kernel, stride, padding)?;
    if output.shape().dims() != out_shape {
        return Err(AuroraError::shape_mismatch(out_shape, output.shape().dims().to_vec()));
    }

    let batch = in_dims[0];
    let channels = in_dims[1];
    let in_h = in_dims[2];
    let in_w = in_dims[3];
    let out_h = ((in_h + 2 * padding.0).saturating_sub(kernel.0) / stride.0) + 1;
    let out_w = ((in_w + 2 * padding.1).saturating_sub(kernel.1) / stride.1) + 1;
    let row_width = channels * kernel.0 * kernel.1;

    output
        .as_mut_slice()
        .par_chunks_mut(row_width)
        .enumerate()
        .for_each(|(row, out_row)| {
            let n = row / (out_h * out_w);
            let spatial = row % (out_h * out_w);
            let oh = spatial / out_w;
            let ow = spatial % out_w;
            let mut col = 0;
            for c in 0..channels {
                for kh in 0..kernel.0 {
                    let ih = oh * stride.0 + kh;
                    for kw in 0..kernel.1 {
                        let iw = ow * stride.1 + kw;
                        let value = if ih < padding.0 || iw < padding.1 {
                            0.0
                        } else {
                            let ih = ih - padding.0;
                            let iw = iw - padding.1;
                            if ih >= in_h || iw >= in_w {
                                0.0
                            } else {
                                let idx = ((n * channels + c) * in_h + ih) * in_w + iw;
                                input.as_slice()[idx]
                            }
                        };
                        out_row[col] = value;
                        col += 1;
                    }
                }
            }
        });

    Ok(())
}

/// Batch normalization.
pub fn batch_norm(
    input: &HostTensor,
    gamma: &HostTensor,
    beta: &HostTensor,
    running_mean: &HostTensor,
    running_var: &HostTensor,
    output: &mut HostTensor,
    epsilon: f32,
) -> Result<()> {
    ensure_same_shape(input, output)?;
    ensure_same_shape(input, gamma)?;
    ensure_same_shape(input, beta)?;
    ensure_same_shape(input, running_mean)?;
    ensure_same_shape(input, running_var)?;

    output
        .as_mut_slice()
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, out)| {
            let denom = (running_var.as_slice()[i] + epsilon).sqrt();
            *out = ((input.as_slice()[i] - running_mean.as_slice()[i]) / denom)
                * gamma.as_slice()[i]
                + beta.as_slice()[i];
        });
    Ok(())
}

/// Layer normalization over the last axis.
pub fn layer_norm(
    input: &HostTensor,
    gamma: &HostTensor,
    beta: &HostTensor,
    output: &mut HostTensor,
    epsilon: f32,
) -> Result<()> {
    ensure_same_shape(input, output)?;
    let dims = input.shape().dims();
    let width = *dims
        .last()
        .ok_or_else(|| AuroraError::invalid_arg("layer_norm requires at least one dimension"))?;

    if gamma.len() != width || beta.len() != width {
        return Err(AuroraError::shape_mismatch(
            vec![width],
            vec![gamma.len().max(beta.len())],
        ));
    }

    output
        .as_mut_slice()
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(row_idx, out_row)| {
            let start = row_idx * width;
            let input_row = &input.as_slice()[start..start + width];
            let mean = input_row.iter().copied().sum::<f32>() / width as f32;
            let variance = input_row
                .iter()
                .map(|v| {
                    let d = *v - mean;
                    d * d
                })
                .sum::<f32>()
                / width as f32;
            let denom = (variance + epsilon).sqrt();
            for i in 0..width {
                out_row[i] = ((input_row[i] - mean) / denom) * gamma.as_slice()[i] + beta.as_slice()[i];
            }
        });
    Ok(())
}

/// RMS normalization over the last axis.
pub fn rms_norm(
    input: &HostTensor,
    weight: &HostTensor,
    output: &mut HostTensor,
    epsilon: f32,
) -> Result<()> {
    ensure_same_shape(input, output)?;
    let dims = input.shape().dims();
    let width = *dims
        .last()
        .ok_or_else(|| AuroraError::invalid_arg("rms_norm requires at least one dimension"))?;
    if weight.len() != width {
        return Err(AuroraError::shape_mismatch(vec![width], vec![weight.len()]));
    }

    output
        .as_mut_slice()
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(row_idx, out_row)| {
            let start = row_idx * width;
            let input_row = &input.as_slice()[start..start + width];
            let rms = (input_row.iter().map(|v| v * v).sum::<f32>() / width as f32 + epsilon).sqrt();
            for i in 0..width {
                out_row[i] = (input_row[i] / rms) * weight.as_slice()[i];
            }
        });
    Ok(())
}

/// ReLU activation.
pub fn relu(input: &HostTensor, output: &mut HostTensor) -> Result<()> {
    ensure_same_shape(input, output)?;
    output
        .as_mut_slice()
        .par_iter_mut()
        .zip(input.as_slice().par_iter())
        .for_each(|(out, &value)| *out = value.max(0.0));
    Ok(())
}

/// GELU activation.
pub fn gelu(input: &HostTensor, output: &mut HostTensor) -> Result<()> {
    ensure_same_shape(input, output)?;
    output
        .as_mut_slice()
        .par_chunks_mut(1024)
        .zip(input.as_slice().par_chunks(1024))
        .for_each(|(out_chunk, in_chunk)| gelu_chunk(in_chunk, out_chunk));
    Ok(())
}

/// Sigmoid activation.
pub fn sigmoid(input: &HostTensor, output: &mut HostTensor) -> Result<()> {
    ensure_same_shape(input, output)?;
    output
        .as_mut_slice()
        .par_iter_mut()
        .zip(input.as_slice().par_iter())
        .for_each(|(out, &value)| *out = 1.0 / (1.0 + (-value).exp()));
    Ok(())
}

/// Softmax over the last axis.
pub fn softmax(input: &HostTensor, output: &mut HostTensor, axis: i32) -> Result<()> {
    ensure_same_shape(input, output)?;
    let dims = input.shape().dims();
    let width = *dims
        .last()
        .ok_or_else(|| AuroraError::invalid_arg("softmax requires at least one dimension"))?;

    if axis != -1 && axis != (dims.len() as i32 - 1) {
        return Err(AuroraError::unsupported(
            "host softmax currently supports only the last axis",
        ));
    }

    output
        .as_mut_slice()
        .par_chunks_mut(width)
        .enumerate()
        .for_each(|(row_idx, out_row)| {
            let start = row_idx * width;
            let input_row = &input.as_slice()[start..start + width];
            let max_val = input_row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for i in 0..width {
                let value = (input_row[i] - max_val).exp();
                out_row[i] = value;
                sum += value;
            }
            if sum != 0.0 {
                for out in out_row.iter_mut() {
                    *out /= sum;
                }
            }
        });
    Ok(())
}

/// Reduce sum over the last axis.
pub fn reduce_sum_last(input: &HostTensor, output: &mut HostTensor) -> Result<()> {
    let dims = input.shape().dims();
    let width = *dims
        .last()
        .ok_or_else(|| AuroraError::invalid_arg("reduce_sum requires at least one dimension"))?;
    let expected = reduced_last_shape(dims);
    if output.shape().dims() != expected {
        return Err(AuroraError::shape_mismatch(expected, output.shape().dims().to_vec()));
    }

    output
        .as_mut_slice()
        .par_iter_mut()
        .enumerate()
        .for_each(|(row, out)| {
            let start = row * width;
            *out = simd_sum(&input.as_slice()[start..start + width]);
        });
    Ok(())
}

/// Reduce max over the last axis.
pub fn reduce_max_last(input: &HostTensor, output: &mut HostTensor) -> Result<()> {
    let dims = input.shape().dims();
    let width = *dims
        .last()
        .ok_or_else(|| AuroraError::invalid_arg("reduce_max requires at least one dimension"))?;
    let expected = reduced_last_shape(dims);
    if output.shape().dims() != expected {
        return Err(AuroraError::shape_mismatch(expected, output.shape().dims().to_vec()));
    }

    output
        .as_mut_slice()
        .par_iter_mut()
        .enumerate()
        .for_each(|(row, out)| {
            let start = row * width;
            *out = simd_max(&input.as_slice()[start..start + width]);
        });
    Ok(())
}

/// Swap the last two dimensions of a tensor.
pub fn transpose_last_two(input: &HostTensor, output: &mut HostTensor) -> Result<()> {
    let in_dims = input.shape().dims();
    let expected = input.shape().transpose();
    if output.shape().dims() != expected.dims() {
        return Err(AuroraError::shape_mismatch(
            expected.dims().to_vec(),
            output.shape().dims().to_vec(),
        ));
    }

    if in_dims.len() < 2 {
        output.as_mut_slice().copy_from_slice(input.as_slice());
        return Ok(());
    }

    let outer = in_dims[..in_dims.len() - 2].iter().product::<usize>().max(1);
    let rows = in_dims[in_dims.len() - 2];
    let cols = in_dims[in_dims.len() - 1];
    let out_slice = output.as_mut_slice();
    let in_slice = input.as_slice();

    out_slice
        .par_chunks_mut(rows * cols)
        .enumerate()
        .for_each(|(batch, out_batch)| {
            let in_batch = &in_slice[batch * rows * cols..(batch + 1) * rows * cols];
            for r in 0..rows {
                for c in 0..cols {
                    out_batch[c * rows + r] = in_batch[r * cols + c];
                }
            }
        });
    debug_assert_eq!(outer * rows * cols, input.len());
    Ok(())
}

/// Scaled dot-product attention.
pub fn attention(
    query: &HostTensor,
    key: &HostTensor,
    value: &HostTensor,
    output: &mut HostTensor,
) -> Result<()> {
    let out_shape = attention_output_shape(query, key, value)?;
    if output.shape().dims() != out_shape {
        return Err(AuroraError::shape_mismatch(out_shape, output.shape().dims().to_vec()));
    }

    let q_dims = query.shape().dims();
    let k_dims = key.shape().dims();
    let v_dims = value.shape().dims();
    let batch = q_dims[0];
    let q_seq = q_dims[1];
    let dim = q_dims[2];
    let k_seq = k_dims[1];
    let value_dim = v_dims[2];
    let scale = (dim as f32).sqrt();

    output
        .as_mut_slice()
        .par_chunks_mut(q_seq * value_dim)
        .enumerate()
        .for_each(|(batch_idx, out_batch)| {
            let q_offset = batch_idx * q_seq * dim;
            let k_offset = batch_idx * k_seq * dim;
            let v_offset = batch_idx * k_seq * value_dim;
            let q_batch = &query.as_slice()[q_offset..q_offset + q_seq * dim];
            let k_batch = &key.as_slice()[k_offset..k_offset + k_seq * dim];
            let v_batch = &value.as_slice()[v_offset..v_offset + k_seq * value_dim];
            let mut scores = vec![0.0f32; q_seq * k_seq];

            for q in 0..q_seq {
                for k in 0..k_seq {
                    let q_row = &q_batch[q * dim..(q + 1) * dim];
                    let k_row = &k_batch[k * dim..(k + 1) * dim];
                    let acc = simd_dot(q_row, k_row);
                    scores[q * k_seq + k] = acc / scale;
                }

                let row = &mut scores[q * k_seq..(q + 1) * k_seq];
                let max_val = simd_max(row);
                let mut sum = 0.0f32;
                for value in row.iter_mut() {
                    *value = (*value - max_val).exp();
                    sum += *value;
                }
                if sum != 0.0 {
                    for value in row.iter_mut() {
                        *value /= sum;
                    }
                }

                for vd in 0..value_dim {
                    let mut acc = 0.0f32;
                    for k in 0..k_seq {
                        acc += row[k] * v_batch[k * value_dim + vd];
                    }
                    out_batch[q * value_dim + vd] = acc;
                }
            }
        });

    Ok(())
}

/// Element-wise addition.
pub fn add(a: &HostTensor, b: &HostTensor, c: &mut HostTensor) -> Result<()> {
    ensure_same_shape(a, b)?;
    ensure_same_shape(a, c)?;
    c.as_mut_slice()
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, out)| *out = a.as_slice()[i] + b.as_slice()[i]);
    Ok(())
}

/// Element-wise multiplication.
pub fn mul(a: &HostTensor, b: &HostTensor, c: &mut HostTensor) -> Result<()> {
    ensure_same_shape(a, b)?;
    ensure_same_shape(a, c)?;
    c.as_mut_slice()
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, out)| *out = a.as_slice()[i] * b.as_slice()[i]);
    Ok(())
}

/// Element-wise add followed by ReLU.
pub fn fused_add_relu(a: &HostTensor, b: &HostTensor, c: &mut HostTensor) -> Result<()> {
    ensure_same_shape(a, b)?;
    ensure_same_shape(a, c)?;
    c.as_mut_slice()
        .par_iter_mut()
        .enumerate()
        .for_each(|(i, out)| *out = (a.as_slice()[i] + b.as_slice()[i]).max(0.0));
    Ok(())
}

fn gelu_in_place(values: &mut [f32]) {
    values
        .par_chunks_mut(1024)
        .for_each(|chunk| {
            let input = chunk.to_vec();
            gelu_chunk(&input, chunk);
        });
}

fn gelu_scalar(value: f32) -> f32 {
    let c = (2.0 / std::f32::consts::PI).sqrt();
    0.5 * value * (1.0 + (c * (value + 0.044_715 * value.powi(3))).tanh())
}

fn gelu_chunk(input: &[f32], output: &mut [f32]) {
    debug_assert_eq!(input.len(), output.len());
    let mut i = 0;
    while i + 8 <= input.len() {
        let lanes = f32x8::from([
            input[i],
            input[i + 1],
            input[i + 2],
            input[i + 3],
            input[i + 4],
            input[i + 5],
            input[i + 6],
            input[i + 7],
        ]);
        let lanes_arr: [f32; 8] = lanes.into();
        let mut out = [0.0f32; 8];
        for lane in 0..8 {
            out[lane] = gelu_scalar(lanes_arr[lane]);
        }
        output[i..i + 8].copy_from_slice(&out);
        i += 8;
    }
    while i < input.len() {
        output[i] = gelu_scalar(input[i]);
        i += 1;
    }
}

fn simd_sum(values: &[f32]) -> f32 {
    let mut i = 0;
    let mut acc = f32x8::ZERO;
    while i + 8 <= values.len() {
        acc += f32x8::from([
            values[i],
            values[i + 1],
            values[i + 2],
            values[i + 3],
            values[i + 4],
            values[i + 5],
            values[i + 6],
            values[i + 7],
        ]);
        i += 8;
    }
    let lanes: [f32; 8] = acc.into();
    let mut total = lanes.iter().copied().sum::<f32>();
    while i < values.len() {
        total += values[i];
        i += 1;
    }
    total
}

fn simd_max(values: &[f32]) -> f32 {
    let mut i = 0;
    let mut acc = f32x8::splat(f32::NEG_INFINITY);
    while i + 8 <= values.len() {
        let lanes = f32x8::from([
            values[i],
            values[i + 1],
            values[i + 2],
            values[i + 3],
            values[i + 4],
            values[i + 5],
            values[i + 6],
            values[i + 7],
        ]);
        acc = acc.max(lanes);
        i += 8;
    }
    let lanes: [f32; 8] = acc.into();
    let mut max_val = lanes.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    while i < values.len() {
        max_val = max_val.max(values[i]);
        i += 1;
    }
    max_val
}

fn simd_dot(lhs: &[f32], rhs: &[f32]) -> f32 {
    debug_assert_eq!(lhs.len(), rhs.len());
    let mut i = 0;
    let mut acc = f32x8::ZERO;
    while i + 8 <= lhs.len() {
        let left = f32x8::from([
            lhs[i],
            lhs[i + 1],
            lhs[i + 2],
            lhs[i + 3],
            lhs[i + 4],
            lhs[i + 5],
            lhs[i + 6],
            lhs[i + 7],
        ]);
        let right = f32x8::from([
            rhs[i],
            rhs[i + 1],
            rhs[i + 2],
            rhs[i + 3],
            rhs[i + 4],
            rhs[i + 5],
            rhs[i + 6],
            rhs[i + 7],
        ]);
        acc += left * right;
        i += 8;
    }
    let lanes: [f32; 8] = acc.into();
    let mut total = lanes.iter().copied().sum::<f32>();
    while i < lhs.len() {
        total += lhs[i] * rhs[i];
        i += 1;
    }
    total
}

fn reduced_last_shape(dims: &[usize]) -> Vec<usize> {
    if dims.len() <= 1 {
        Vec::new()
    } else {
        dims[..dims.len() - 1].to_vec()
    }
}

fn im2col_output_shape(
    dims: &[usize],
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
) -> Result<Vec<usize>> {
    if dims.len() != 4 {
        return Err(AuroraError::invalid_arg("im2col expects a 4D NCHW tensor"));
    }
    let batch = dims[0];
    let channels = dims[1];
    let in_h = dims[2];
    let in_w = dims[3];
    let out_h = ((in_h + 2 * padding.0).saturating_sub(kernel.0) / stride.0) + 1;
    let out_w = ((in_w + 2 * padding.1).saturating_sub(kernel.1) / stride.1) + 1;
    Ok(vec![batch * out_h * out_w, channels * kernel.0 * kernel.1])
}

fn attention_output_shape(
    query: &HostTensor,
    key: &HostTensor,
    value: &HostTensor,
) -> Result<Vec<usize>> {
    let q_dims = query.shape().dims();
    let k_dims = key.shape().dims();
    let v_dims = value.shape().dims();
    if q_dims.len() != 3 || k_dims.len() != 3 || v_dims.len() != 3 {
        return Err(AuroraError::invalid_arg(
            "attention expects [batch, seq, dim] query/key/value tensors",
        ));
    }
    if q_dims[0] != k_dims[0] || q_dims[0] != v_dims[0] {
        return Err(AuroraError::shape_mismatch(q_dims.to_vec(), k_dims.to_vec()));
    }
    if k_dims[1] != v_dims[1] || q_dims[2] != k_dims[2] {
        return Err(AuroraError::shape_mismatch(k_dims.to_vec(), v_dims.to_vec()));
    }
    Ok(vec![q_dims[0], q_dims[1], v_dims[2]])
}

fn ensure_same_shape(a: &HostTensor, b: &HostTensor) -> Result<()> {
    if a.shape().dims() != b.shape().dims() {
        return Err(AuroraError::shape_mismatch(
            a.shape().dims().to_vec(),
            b.shape().dims().to_vec(),
        ));
    }
    Ok(())
}

fn matrix_dims(tensor: &HostTensor) -> Result<(usize, usize)> {
    if !tensor.shape().is_matrix() {
        return Err(AuroraError::invalid_arg(format!(
            "expected matrix tensor, got shape {}",
            tensor.shape()
        )));
    }
    Ok((
        tensor.shape().rows().unwrap_or(0),
        tensor.shape().cols().unwrap_or(0),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_works() {
        let a = HostTensor::from_vec(&[4], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = HostTensor::from_vec(&[4], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
        let mut c = HostTensor::zeros(&[4]);
        add(&a, &b, &mut c).unwrap();
        assert_eq!(c.as_slice(), &[6.0, 8.0, 10.0, 12.0]);
    }

    #[test]
    fn gelu_works() {
        let input = HostTensor::from_vec(&[2], vec![0.0, 1.0]).unwrap();
        let mut output = HostTensor::zeros(&[2]);
        gelu(&input, &mut output).unwrap();
        assert!(output.as_slice()[0].abs() < 1e-6);
        assert!(output.as_slice()[1] > 0.8 && output.as_slice()[1] < 0.85);
    }

    #[test]
    fn reduce_sum_works() {
        let input = HostTensor::from_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mut output = HostTensor::zeros(&[2]);
        reduce_sum_last(&input, &mut output).unwrap();
        assert_eq!(output.as_slice(), &[6.0, 15.0]);
    }

    #[test]
    fn transpose_works() {
        let input = HostTensor::from_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let mut output = HostTensor::zeros(&[3, 2]);
        transpose_last_two(&input, &mut output).unwrap();
        assert_eq!(output.as_slice(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn fused_matmul_gelu_works() {
        let a = HostTensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let b = HostTensor::from_vec(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let mut c = HostTensor::zeros(&[2, 2]);
        fused_matmul_gelu(&a, &b, &mut c).unwrap();
        assert!(c.as_slice()[0] > 0.8);
        assert!(c.as_slice()[3] > 3.9);
    }

    #[test]
    fn im2col_works() {
        let input = HostTensor::from_vec(
            &[1, 1, 3, 3],
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .unwrap();
        let mut output = HostTensor::zeros(&[4, 4]);
        im2col(&input, &mut output, (2, 2), (1, 1), (0, 0)).unwrap();
        assert_eq!(output.as_slice()[0..4], [1.0, 2.0, 4.0, 5.0]);
    }

    #[test]
    fn attention_works() {
        let query = HostTensor::from_vec(&[1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let key = HostTensor::from_vec(&[1, 2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap();
        let value = HostTensor::from_vec(&[1, 2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut output = HostTensor::zeros(&[1, 2, 2]);
        attention(&query, &key, &value, &mut output).unwrap();
        assert_eq!(output.shape().dims(), &[1, 2, 2]);
        assert!(output.as_slice()[0] > 1.0 && output.as_slice()[0] < 3.0);
    }

    #[test]
    fn graph_executes() {
        let executor = PersistentExecutor::new(2, false).unwrap();
        let graph = ExecutionGraph::new()
            .with_input("a", HostTensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap())
            .with_input("b", HostTensor::from_vec(&[2, 2], vec![1.0, 0.0, 0.0, 1.0]).unwrap())
            .add_node(GraphNode::FusedMatmulGelu {
                lhs: "a".to_string(),
                rhs: "b".to_string(),
                output: "mm".to_string(),
            })
            .add_node(GraphNode::Transpose {
                input: "mm".to_string(),
                output: "t".to_string(),
            })
            .add_node(GraphNode::ReduceSum {
                input: "t".to_string(),
                output: "sum".to_string(),
            });
        let result = executor.execute_graph(&graph).unwrap();
        let sum = result.tensors.get("sum").unwrap();
        assert_eq!(sum.shape().dims(), &[2]);
    }
}
