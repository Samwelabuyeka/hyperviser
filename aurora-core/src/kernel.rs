//! Kernel abstraction for compute operations

use serde::{Deserialize, Serialize};
use std::fmt;
use crate::device::DeviceId;
use crate::tensor::{DataType, TensorShape};

/// Unique kernel identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct KernelId(pub u64);

impl KernelId {
    /// Create a new kernel ID
    pub fn new(id: u64) -> Self {
        Self(id)
    }
}

/// Kernel signature (input/output types)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelSignature {
    /// Input tensor shapes and types
    pub inputs: Vec<(TensorShape, DataType)>,
    /// Output tensor shapes and types
    pub outputs: Vec<(TensorShape, DataType)>,
    /// Scalar parameters
    pub scalars: Vec<DataType>,
}

impl KernelSignature {
    /// Create a new kernel signature
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            outputs: Vec::new(),
            scalars: Vec::new(),
        }
    }
    
    /// Add an input
    pub fn with_input(mut self, shape: TensorShape, dtype: DataType) -> Self {
        self.inputs.push((shape, dtype));
        self
    }
    
    /// Add an output
    pub fn with_output(mut self, shape: TensorShape, dtype: DataType) -> Self {
        self.outputs.push((shape, dtype));
        self
    }
    
    /// Add a scalar parameter
    pub fn with_scalar(mut self, dtype: DataType) -> Self {
        self.scalars.push(dtype);
        self
    }
}

impl Default for KernelSignature {
    fn default() -> Self {
        Self::new()
    }
}

/// Kernel launch configuration
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LaunchConfig {
    /// Grid dimensions (x, y, z)
    pub grid_dim: (u32, u32, u32),
    /// Block dimensions (x, y, z)
    pub block_dim: (u32, u32, u32),
    /// Shared memory size in bytes
    pub shared_mem: usize,
    /// Stream ID (0 for default)
    pub stream: u64,
}

impl LaunchConfig {
    /// Create a new launch configuration
    pub fn new(grid_x: u32, block_x: u32) -> Self {
        Self {
            grid_dim: (grid_x, 1, 1),
            block_dim: (block_x, 1, 1),
            shared_mem: 0,
            stream: 0,
        }
    }
    
    /// Create a 2D launch configuration
    pub fn new_2d(grid_x: u32, grid_y: u32, block_x: u32, block_y: u32) -> Self {
        Self {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (block_x, block_y, 1),
            shared_mem: 0,
            stream: 0,
        }
    }
    
    /// Create a 3D launch configuration
    pub fn new_3d(
        grid_x: u32, grid_y: u32, grid_z: u32,
        block_x: u32, block_y: u32, block_z: u32
    ) -> Self {
        Self {
            grid_dim: (grid_x, grid_y, grid_z),
            block_dim: (block_x, block_y, block_z),
            shared_mem: 0,
            stream: 0,
        }
    }
    
    /// Set shared memory size
    pub fn with_shared_mem(mut self, bytes: usize) -> Self {
        self.shared_mem = bytes;
        self
    }
    
    /// Set stream
    pub fn with_stream(mut self, stream: u64) -> Self {
        self.stream = stream;
        self
    }
    
    /// Get total number of threads
    pub fn total_threads(&self) -> u64 {
        let grid = self.grid_dim.0 as u64 * self.grid_dim.1 as u64 * self.grid_dim.2 as u64;
        let block = self.block_dim.0 as u64 * self.block_dim.1 as u64 * self.block_dim.2 as u64;
        grid * block
    }
    
    /// Get optimal config for element-wise operations
    pub fn for_elements(n: usize) -> Self {
        const BLOCK_SIZE: u32 = 256;
        let blocks = ((n as u32 + BLOCK_SIZE - 1) / BLOCK_SIZE).max(1);
        Self::new(blocks, BLOCK_SIZE)
    }
}

impl Default for LaunchConfig {
    fn default() -> Self {
        Self::new(1, 256)
    }
}

/// Kernel trait for compute operations
pub trait Kernel: Send + Sync {
    /// Get the kernel ID
    fn id(&self) -> KernelId;
    
    /// Get the kernel name
    fn name(&self) -> &str;
    
    /// Get the kernel signature
    fn signature(&self) -> &KernelSignature;
    
    /// Get the target device
    fn device(&self) -> DeviceId;
    
    /// Launch the kernel
    fn launch(&self, config: &LaunchConfig, args: &[&[u8]]) -> crate::error::Result<()>;
    
    /// Estimate execution time in microseconds (for scheduling)
    fn estimate_time_us(&self, elements: usize) -> u64;
}

/// Kernel registry for looking up kernels
#[derive(Debug, Default)]
pub struct KernelRegistry {
    kernels: std::collections::HashMap<KernelId, Box<dyn Kernel>>,
}

impl KernelRegistry {
    /// Create a new kernel registry
    pub fn new() -> Self {
        Self {
            kernels: std::collections::HashMap::new(),
        }
    }
    
    /// Register a kernel
    pub fn register<K: Kernel + 'static>(&mut self, kernel: K) {
        self.kernels.insert(kernel.id(), Box::new(kernel));
    }
    
    /// Get a kernel by ID
    pub fn get(&self, id: KernelId) -> Option<&dyn Kernel> {
        self.kernels.get(&id).map(|k| k.as_ref())
    }
    
    /// Find kernels by name
    pub fn find_by_name(&self, name: &str) -> Vec<&dyn Kernel> {
        self.kernels
            .values()
            .filter(|k| k.name() == name)
            .map(|k| k.as_ref())
            .collect()
    }
    
    /// List all registered kernels
    pub fn list(&self) -> Vec<KernelId> {
        self.kernels.keys().copied().collect()
    }
}

/// Built-in kernel types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KernelType {
    /// Element-wise unary operation
    Unary(UnaryOp),
    /// Element-wise binary operation
    Binary(BinaryOp),
    /// Reduction operation
    Reduce(ReduceOp),
    /// Matrix multiplication
    Matmul,
    /// Convolution
    Convolution,
    /// Pooling
    Pooling(PoolType),
    /// Softmax
    Softmax,
    /// Layer normalization
    LayerNorm,
    /// Attention
    Attention,
    /// Custom kernel
    Custom(&'static str),
}

/// Unary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UnaryOp {
    /// Identity
    Identity,
    /// Negation
    Neg,
    /// Absolute value
    Abs,
    /// Exponential
    Exp,
    /// Natural logarithm
    Log,
    /// Square root
    Sqrt,
    /// Reciprocal
    Reciprocal,
    /// Sine
    Sin,
    /// Cosine
    Cos,
    /// Tangent
    Tan,
    /// Hyperbolic tangent
    Tanh,
    /// Sigmoid
    Sigmoid,
    /// Rectified Linear Unit
    Relu,
    /// Gaussian Error Linear Unit
    Gelu,
    /// Swish/SiLU
    Swish,
    /// Exponential Linear Unit
    Elu,
    /// Leaky ReLU
    LeakyRelu,
    /// Softplus
    Softplus,
    /// Floor
    Floor,
    /// Ceiling
    Ceil,
    /// Round
    Round,
}

/// Binary operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BinaryOp {
    /// Addition
    Add,
    /// Subtraction
    Sub,
    /// Multiplication
    Mul,
    /// Division
    Div,
    /// Power
    Pow,
    /// Minimum
    Min,
    /// Maximum
    Max,
    /// Modulo
    Mod,
    /// Equal
    Eq,
    /// Not equal
    Ne,
    /// Less than
    Lt,
    /// Less than or equal
    Le,
    /// Greater than
    Gt,
    /// Greater than or equal
    Ge,
    /// Logical AND
    And,
    /// Logical OR
    Or,
    /// Logical XOR
    Xor,
}

/// Reduction operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReduceOp {
    /// Sum
    Sum,
    /// Product
    Product,
    /// Mean
    Mean,
    /// Maximum
    Max,
    /// Minimum
    Min,
    /// Argmax
    Argmax,
    /// Argmin
    Argmin,
    /// Standard deviation
    Std,
    /// Variance
    Var,
    /// Log sum exp
    LogSumExp,
}

/// Pooling types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PoolType {
    /// Maximum pooling
    Max,
    /// Average pooling
    Avg,
    /// Global average pooling
    GlobalAvg,
    /// Global maximum pooling
    GlobalMax,
}

impl fmt::Display for KernelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            KernelType::Unary(op) => write!(f, "Unary({:?})", op),
            KernelType::Binary(op) => write!(f, "Binary({:?})", op),
            KernelType::Reduce(op) => write!(f, "Reduce({:?})", op),
            KernelType::Matmul => write!(f, "Matmul"),
            KernelType::Convolution => write!(f, "Convolution"),
            KernelType::Pooling(t) => write!(f, "Pooling({:?})", t),
            KernelType::Softmax => write!(f, "Softmax"),
            KernelType::LayerNorm => write!(f, "LayerNorm"),
            KernelType::Attention => write!(f, "Attention"),
            KernelType::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}
