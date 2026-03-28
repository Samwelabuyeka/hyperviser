//! Common types and utilities

/// Generic scalar type for tensor operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Scalar {
    /// 32-bit float
    F32(f32),
    /// 16-bit float (stored as u16 bits)
    F16(u16),
    /// 64-bit float
    F64(f64),
    /// 8-bit signed integer
    I8(i8),
    /// 16-bit signed integer
    I16(i16),
    /// 32-bit signed integer
    I32(i32),
    /// 64-bit signed integer
    I64(i64),
    /// 8-bit unsigned integer
    U8(u8),
    /// 16-bit unsigned integer
    U16(u16),
    /// 32-bit unsigned integer
    U32(u32),
    /// 64-bit unsigned integer
    U64(u64),
    /// Boolean
    Bool(bool),
}

impl Scalar {
    /// Convert to f32
    pub fn as_f32(&self) -> f32 {
        match self {
            Scalar::F32(v) => *v,
            Scalar::F16(v) => f16::to_f32(*v),
            Scalar::F64(v) => *v as f32,
            Scalar::I8(v) => *v as f32,
            Scalar::I16(v) => *v as f32,
            Scalar::I32(v) => *v as f32,
            Scalar::I64(v) => *v as f32,
            Scalar::U8(v) => *v as f32,
            Scalar::U16(v) => *v as f32,
            Scalar::U32(v) => *v as f32,
            Scalar::U64(v) => *v as f32,
            Scalar::Bool(v) => if *v { 1.0 } else { 0.0 },
        }
    }
    
    /// Convert to f64
    pub fn as_f64(&self) -> f64 {
        match self {
            Scalar::F32(v) => *v as f64,
            Scalar::F16(v) => f16::to_f64(*v),
            Scalar::F64(v) => *v,
            Scalar::I8(v) => *v as f64,
            Scalar::I16(v) => *v as f64,
            Scalar::I32(v) => *v as f64,
            Scalar::I64(v) => *v as f64,
            Scalar::U8(v) => *v as f64,
            Scalar::U16(v) => *v as f64,
            Scalar::U32(v) => *v as f64,
            Scalar::U64(v) => *v as f64,
            Scalar::Bool(v) => if *v { 1.0 } else { 0.0 },
        }
    }
    
    /// Convert to i64
    pub fn as_i64(&self) -> i64 {
        match self {
            Scalar::F32(v) => *v as i64,
            Scalar::F16(v) => f16::to_f32(*v) as i64,
            Scalar::F64(v) => *v as i64,
            Scalar::I8(v) => *v as i64,
            Scalar::I16(v) => *v as i64,
            Scalar::I32(v) => *v as i64,
            Scalar::I64(v) => *v,
            Scalar::U8(v) => *v as i64,
            Scalar::U16(v) => *v as i64,
            Scalar::U32(v) => *v as i64,
            Scalar::U64(v) => *v as i64,
            Scalar::Bool(v) => if *v { 1 } else { 0 },
        }
    }
    
    /// Get zero value for this scalar type
    pub fn zero(dtype: super::tensor::DataType) -> Self {
        match dtype {
            super::tensor::DataType::F32 => Scalar::F32(0.0),
            super::tensor::DataType::F16 => Scalar::F16(0),
            super::tensor::DataType::BF16 => Scalar::F16(0),
            super::tensor::DataType::F64 => Scalar::F64(0.0),
            super::tensor::DataType::I8 => Scalar::I8(0),
            super::tensor::DataType::I16 => Scalar::I16(0),
            super::tensor::DataType::I32 => Scalar::I32(0),
            super::tensor::DataType::I64 => Scalar::I64(0),
            super::tensor::DataType::U8 => Scalar::U8(0),
            super::tensor::DataType::U16 => Scalar::U16(0),
            super::tensor::DataType::U32 => Scalar::U32(0),
            super::tensor::DataType::U64 => Scalar::U64(0),
            super::tensor::DataType::Bool => Scalar::Bool(false),
        }
    }
    
    /// Get one value for this scalar type
    pub fn one(dtype: super::tensor::DataType) -> Self {
        match dtype {
            super::tensor::DataType::F32 => Scalar::F32(1.0),
            super::tensor::DataType::F16 => Scalar::F16(f16::from_f32(1.0)),
            super::tensor::DataType::BF16 => Scalar::F16(half::bf16::from_f32(1.0).to_bits()),
            super::tensor::DataType::F64 => Scalar::F64(1.0),
            super::tensor::DataType::I8 => Scalar::I8(1),
            super::tensor::DataType::I16 => Scalar::I16(1),
            super::tensor::DataType::I32 => Scalar::I32(1),
            super::tensor::DataType::I64 => Scalar::I64(1),
            super::tensor::DataType::U8 => Scalar::U8(1),
            super::tensor::DataType::U16 => Scalar::U16(1),
            super::tensor::DataType::U32 => Scalar::U32(1),
            super::tensor::DataType::U64 => Scalar::U64(1),
            super::tensor::DataType::Bool => Scalar::Bool(true),
        }
    }
}

/// Dimension index type
pub type Dim = usize;

/// Index type for tensor access
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Index(pub Vec<usize>);

impl Index {
    /// Create a new index
    pub fn new(indices: &[usize]) -> Self {
        Self(indices.to_vec())
    }
    
    /// Get the index at a dimension
    pub fn get(&self, dim: usize) -> Option<usize> {
        self.0.get(dim).copied()
    }
    
    /// Number of dimensions
    pub fn ndim(&self) -> usize {
        self.0.len()
    }
}

/// Range for slicing tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Range {
    /// Start index (inclusive)
    pub start: usize,
    /// End index (exclusive)
    pub end: usize,
    /// Step size
    pub step: usize,
}

impl Range {
    /// Create a new range
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end, step: 1 }
    }
    
    /// Create a range with step
    pub fn with_step(start: usize, end: usize, step: usize) -> Self {
        Self { start, end, step }
    }
    
    /// Create a full range (0 to end)
    pub fn full(end: usize) -> Self {
        Self { start: 0, end, step: 1 }
    }
    
    /// Get the number of elements in this range
    pub fn len(&self) -> usize {
        if self.end <= self.start {
            return 0;
        }
        (self.end - self.start + self.step - 1) / self.step
    }
    
    /// Check if range is empty
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Padding configuration for convolution/pooling
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Padding {
    /// No padding
    Valid,
    /// Same output size as input
    Same,
    /// Custom padding for each dimension
    Custom { before: usize, after: usize },
}

/// Convolution parameters
#[derive(Debug, Clone)]
pub struct ConvParams {
    /// Stride for each dimension
    pub stride: Vec<usize>,
    /// Padding mode
    pub padding: Padding,
    /// Dilation for each dimension
    pub dilation: Vec<usize>,
    /// Number of groups
    pub groups: usize,
}

impl Default for ConvParams {
    fn default() -> Self {
        Self {
            stride: vec![1],
            padding: Padding::Valid,
            dilation: vec![1],
            groups: 1,
        }
    }
}

/// Pooling parameters
#[derive(Debug, Clone)]
pub struct PoolParams {
    /// Kernel size for each dimension
    pub kernel_size: Vec<usize>,
    /// Stride for each dimension
    pub stride: Vec<usize>,
    /// Padding mode
    pub padding: Padding,
}

impl Default for PoolParams {
    fn default() -> Self {
        Self {
            kernel_size: vec![2],
            stride: vec![2],
            padding: Padding::Valid,
        }
    }
}

/// f16 helper type
pub mod f16 {
    /// Convert f32 to f16 bits
    pub fn from_f32(v: f32) -> u16 {
        half::f16::from_f32(v).to_bits()
    }
    
    /// Convert f16 bits to f32
    pub fn to_f32(bits: u16) -> f32 {
        half::f16::from_bits(bits).to_f32()
    }
    
    /// Convert f16 bits to f64
    pub fn to_f64(bits: u16) -> f64 {
        half::f16::from_bits(bits).to_f64()
    }
}
