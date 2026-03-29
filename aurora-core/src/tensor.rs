//! Tensor types and operations

use serde::{Deserialize, Serialize};
use std::fmt;
use crate::device::DeviceId;
use crate::error::{AuroraError, Result};

/// Data type for tensor elements
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataType {
    /// 32-bit floating point
    F32,
    /// 16-bit floating point (IEEE 754-2008)
    F16,
    /// Brain floating point 16
    BF16,
    /// 64-bit floating point
    F64,
    /// 8-bit integer (signed)
    I8,
    /// 16-bit integer (signed)
    I16,
    /// 32-bit integer (signed)
    I32,
    /// 64-bit integer (signed)
    I64,
    /// 8-bit integer (unsigned)
    U8,
    /// 16-bit integer (unsigned)
    U16,
    /// 32-bit integer (unsigned)
    U32,
    /// 64-bit integer (unsigned)
    U64,
    /// Boolean
    Bool,
}

impl DataType {
    /// Get the size in bytes of this data type
    pub const fn size_bytes(&self) -> usize {
        match self {
            DataType::F32 => 4,
            DataType::F16 | DataType::BF16 => 2,
            DataType::F64 => 8,
            DataType::I8 | DataType::U8 | DataType::Bool => 1,
            DataType::I16 | DataType::U16 => 2,
            DataType::I32 | DataType::U32 => 4,
            DataType::I64 | DataType::U64 => 8,
        }
    }
    
    /// Check if this is a floating point type
    pub const fn is_float(&self) -> bool {
        matches!(self, DataType::F32 | DataType::F16 | DataType::BF16 | DataType::F64)
    }
    
    /// Check if this is an integer type
    pub const fn is_integer(&self) -> bool {
        matches!(self, DataType::I8 | DataType::I16 | DataType::I32 | DataType::I64 |
                      DataType::U8 | DataType::U16 | DataType::U32 | DataType::U64)
    }
    
    /// Check if this is a signed type
    pub const fn is_signed(&self) -> bool {
        matches!(self, DataType::I8 | DataType::I16 | DataType::I32 | DataType::I64 |
                      DataType::F32 | DataType::F16 | DataType::BF16 | DataType::F64)
    }
    
    /// Get the name of the data type
    pub const fn name(&self) -> &'static str {
        match self {
            DataType::F32 => "f32",
            DataType::F16 => "f16",
            DataType::BF16 => "bf16",
            DataType::F64 => "f64",
            DataType::I8 => "i8",
            DataType::I16 => "i16",
            DataType::I32 => "i32",
            DataType::I64 => "i64",
            DataType::U8 => "u8",
            DataType::U16 => "u16",
            DataType::U32 => "u32",
            DataType::U64 => "u64",
            DataType::Bool => "bool",
        }
    }
    
    /// Get the Rust type name
    pub const fn rust_type(&self) -> &'static str {
        match self {
            DataType::F32 => "f32",
            DataType::F16 => "half::f16",
            DataType::BF16 => "half::bf16",
            DataType::F64 => "f64",
            DataType::I8 => "i8",
            DataType::I16 => "i16",
            DataType::I32 => "i32",
            DataType::I64 => "i64",
            DataType::U8 => "u8",
            DataType::U16 => "u16",
            DataType::U32 => "u32",
            DataType::U64 => "u64",
            DataType::Bool => "bool",
        }
    }
}

impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name())
    }
}

/// Tensor shape (dimensions)
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TensorShape {
    /// Dimensions from outermost to innermost
    dims: Vec<usize>,
}

impl TensorShape {
    /// Create a new shape from dimensions
    pub fn new(dims: &[usize]) -> Self {
        Self {
            dims: dims.to_vec(),
        }
    }
    
    /// Create a scalar shape
    pub fn scalar() -> Self {
        Self { dims: vec![] }
    }
    
    /// Create a 1D shape
    pub fn from_1d(d0: usize) -> Self {
        Self { dims: vec![d0] }
    }
    
    /// Create a 2D shape
    pub fn from_2d(d0: usize, d1: usize) -> Self {
        Self { dims: vec![d0, d1] }
    }
    
    /// Create a 3D shape
    pub fn from_3d(d0: usize, d1: usize, d2: usize) -> Self {
        Self { dims: vec![d0, d1, d2] }
    }
    
    /// Create a 4D shape
    pub fn from_4d(d0: usize, d1: usize, d2: usize, d3: usize) -> Self {
        Self { dims: vec![d0, d1, d2, d3] }
    }
    
    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.dims.len()
    }
    
    /// Get the dimensions
    pub fn dims(&self) -> &[usize] {
        &self.dims
    }
    
    /// Get a specific dimension
    pub fn dim(&self, index: usize) -> Option<usize> {
        self.dims.get(index).copied()
    }
    
    /// Get the total number of elements
    pub fn numel(&self) -> usize {
        if self.dims.is_empty() {
            1
        } else {
            self.dims.iter().product()
        }
    }
    
    /// Check if this is a scalar
    pub fn is_scalar(&self) -> bool {
        self.dims.is_empty()
    }
    
    /// Check if this is a vector (1D)
    pub fn is_vector(&self) -> bool {
        self.dims.len() == 1
    }
    
    /// Check if this is a matrix (2D)
    pub fn is_matrix(&self) -> bool {
        self.dims.len() == 2
    }
    
    /// Get the size in bytes for a given data type
    pub fn size_bytes(&self, dtype: DataType) -> usize {
        self.numel() * dtype.size_bytes()
    }
    
    /// Broadcast this shape with another shape
    pub fn broadcast(&self, other: &TensorShape) -> Result<TensorShape> {
        let max_ndim = self.ndim().max(other.ndim());
        let mut result_dims = Vec::with_capacity(max_ndim);
        
        for i in 0..max_ndim {
            let self_idx = self.ndim() as isize - max_ndim as isize + i as isize;
            let other_idx = other.ndim() as isize - max_ndim as isize + i as isize;
            let d1 = if self_idx >= 0 {
                self.dim(self_idx as usize).unwrap_or(1)
            } else {
                1
            };
            let d2 = if other_idx >= 0 {
                other.dim(other_idx as usize).unwrap_or(1)
            } else {
                1
            };
            
            if d1 != d2 && d1 != 1 && d2 != 1 {
                return Err(AuroraError::shape_mismatch(self.dims.clone(), other.dims.clone()));
            }
            
            result_dims.push(d1.max(d2));
        }
        
        Ok(TensorShape::new(&result_dims))
    }
    
    /// Check if this shape can broadcast to another
    pub fn can_broadcast_to(&self, target: &TensorShape) -> bool {
        self.broadcast(target).is_ok()
    }
    
    /// Get the row count (for matrices)
    pub fn rows(&self) -> Option<usize> {
        if self.ndim() >= 2 {
            self.dim(self.ndim() - 2)
        } else {
            None
        }
    }
    
    /// Get the column count (for matrices)
    pub fn cols(&self) -> Option<usize> {
        if self.ndim() >= 1 {
            self.dim(self.ndim() - 1)
        } else {
            None
        }
    }
    
    /// Transpose (swap last two dimensions)
    pub fn transpose(&self) -> TensorShape {
        if self.ndim() < 2 {
            return self.clone();
        }
        
        let mut new_dims = self.dims.clone();
        let n = new_dims.len();
        new_dims.swap(n - 2, n - 1);
        
        TensorShape::new(&new_dims)
    }
    
    /// Slice along a dimension
    pub fn slice(&self, dim: usize, start: usize, end: usize) -> Result<TensorShape> {
        if dim >= self.ndim() {
            return Err(AuroraError::invalid_arg(format!(
                "Dimension {} out of range for shape with {} dimensions", dim, self.ndim()
            )));
        }
        
        let dim_size = self.dims[dim];
        if start > end || end > dim_size {
            return Err(AuroraError::invalid_arg(format!(
                "Invalid slice range [{}, {}) for dimension of size {}", start, end, dim_size
            )));
        }
        
        let mut new_dims = self.dims.clone();
        new_dims[dim] = end - start;
        
        Ok(TensorShape::new(&new_dims))
    }
}

impl fmt::Display for TensorShape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.dims.is_empty() {
            write!(f, "()")
        } else {
            write!(f, "[{}]", self.dims.iter().map(|d| d.to_string()).collect::<Vec<_>>().join(", "))
        }
    }
}

impl Default for TensorShape {
    fn default() -> Self {
        Self::scalar()
    }
}

/// Memory layout for tensors
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Layout {
    /// Row-major (C-style) layout
    RowMajor,
    /// Column-major (Fortran-style) layout
    ColumnMajor,
    /// Custom strided layout
    Strided,
}

impl Layout {
    /// Check if this is row-major
    pub const fn is_row_major(&self) -> bool {
        matches!(self, Layout::RowMajor)
    }
    
    /// Check if this is column-major
    pub const fn is_column_major(&self) -> bool {
        matches!(self, Layout::ColumnMajor)
    }
    
    /// Get default strides for a shape
    pub fn default_strides(&self, shape: &TensorShape) -> Vec<isize> {
        let ndim = shape.ndim();
        if ndim == 0 {
            return vec![];
        }
        
        let dims = shape.dims();
        let mut strides = vec![0isize; ndim];
        
        match self {
            Layout::RowMajor => {
                strides[ndim - 1] = 1;
                for i in (0..ndim - 1).rev() {
                    strides[i] = strides[i + 1] * dims[i + 1] as isize;
                }
            }
            Layout::ColumnMajor => {
                strides[0] = 1;
                for i in 1..ndim {
                    strides[i] = strides[i - 1] * dims[i - 1] as isize;
                }
            }
            Layout::Strided => {
                // Default to row-major
                strides[ndim - 1] = 1;
                for i in (0..ndim - 1).rev() {
                    strides[i] = strides[i + 1] * dims[i + 1] as isize;
                }
            }
        }
        
        strides
    }
}

impl Default for Layout {
    fn default() -> Self {
        Layout::RowMajor
    }
}

/// Stride information for each dimension
pub type Stride = Vec<isize>;

/// Tensor handle - lightweight reference to tensor data
#[derive(Debug, Clone)]
pub struct Tensor {
    /// Unique tensor ID
    pub id: u64,
    /// Tensor shape
    pub shape: TensorShape,
    /// Data type
    pub dtype: DataType,
    /// Memory layout
    pub layout: Layout,
    /// Strides for each dimension
    pub strides: Stride,
    /// Device where tensor is allocated
    pub device: DeviceId,
    /// Offset into the data buffer
    pub offset: usize,
    /// Total size in bytes
    pub size_bytes: usize,
}

impl Tensor {
    /// Create a new tensor descriptor
    pub fn new(
        id: u64,
        shape: TensorShape,
        dtype: DataType,
        device: DeviceId,
    ) -> Self {
        let strides = Layout::RowMajor.default_strides(&shape);
        let size_bytes = shape.size_bytes(dtype);
        
        Self {
            id,
            shape,
            dtype,
            layout: Layout::RowMajor,
            strides,
            device,
            offset: 0,
            size_bytes,
        }
    }
    
    /// Get the number of elements
    pub fn numel(&self) -> usize {
        self.shape.numel()
    }
    
    /// Get the number of dimensions
    pub fn ndim(&self) -> usize {
        self.shape.ndim()
    }
    
    /// Check if tensor is contiguous in memory
    pub fn is_contiguous(&self) -> bool {
        if self.shape.is_scalar() {
            return true;
        }
        
        let expected_strides = self.layout.default_strides(&self.shape);
        self.strides == expected_strides && self.offset == 0
    }
    
    /// Reshape the tensor (returns a new view if possible)
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self> {
        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(AuroraError::invalid_arg(
                format!("Cannot reshape tensor with {} elements to shape {:?}", self.numel(), new_shape)
            ));
        }
        
        if !self.is_contiguous() {
            return Err(AuroraError::invalid_arg(
                "Cannot reshape non-contiguous tensor"
            ));
        }
        
        let mut new_tensor = self.clone();
        new_tensor.shape = TensorShape::new(new_shape);
        new_tensor.strides = self.layout.default_strides(&new_tensor.shape);
        
        Ok(new_tensor)
    }
    
    /// Transpose the tensor (swap last two dimensions)
    pub fn transpose(&self) -> Self {
        let mut new_tensor = self.clone();
        new_tensor.shape = self.shape.transpose();
        
        // Update strides
        let ndim = new_tensor.ndim();
        if ndim >= 2 {
            new_tensor.strides.swap(ndim - 2, ndim - 1);
        }
        
        new_tensor.layout = Layout::Strided;
        new_tensor
    }
    
    /// Get a human-readable description
    pub fn describe(&self) -> String {
        format!(
            "Tensor({}, shape={}, dtype={}, device={}, {}contiguous)",
            self.id,
            self.shape,
            self.dtype,
            self.device,
            if self.is_contiguous() { "" } else { "non-" }
        )
    }
    
    /// Get byte offset for an index
    pub fn byte_offset(&self, indices: &[usize]) -> Result<usize> {
        if indices.len() != self.ndim() {
            return Err(AuroraError::invalid_arg(format!(
                "Expected {} indices, got {}", self.ndim(), indices.len()
            )));
        }
        
        let mut offset = self.offset;
        for (i, &idx) in indices.iter().enumerate() {
            if idx >= self.shape.dim(i).unwrap_or(0) {
                return Err(AuroraError::invalid_arg(format!(
                    "Index {} out of bounds for dimension {} of size {}",
                    idx, i, self.shape.dim(i).unwrap_or(0)
                )));
            }
            offset = offset.wrapping_add((idx as isize * self.strides[i]) as usize);
        }
        
        Ok(offset * self.dtype.size_bytes())
    }
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.describe())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::DeviceId;

    #[test]
    fn test_tensor_shape() {
        let shape = TensorShape::from_2d(3, 4);
        assert_eq!(shape.numel(), 12);
        assert_eq!(shape.ndim(), 2);
        assert!(shape.is_matrix());
        assert_eq!(shape.rows(), Some(3));
        assert_eq!(shape.cols(), Some(4));
    }

    #[test]
    fn test_tensor_shape_broadcast() {
        let a = TensorShape::from_2d(3, 4);
        let b = TensorShape::from_1d(4);
        let result = a.broadcast(&b).unwrap();
        assert_eq!(result.dims(), &[3, 4]);
        
        let c = TensorShape::from_2d(5, 6);
        assert!(a.broadcast(&c).is_err());
    }

    #[test]
    fn test_data_type_size() {
        assert_eq!(DataType::F32.size_bytes(), 4);
        assert_eq!(DataType::F16.size_bytes(), 2);
        assert_eq!(DataType::I8.size_bytes(), 1);
        assert_eq!(DataType::F64.size_bytes(), 8);
    }

    #[test]
    fn test_tensor_creation() {
        let tensor = Tensor::new(
            1,
            TensorShape::from_2d(3, 4),
            DataType::F32,
            DeviceId::CPU,
        );
        assert_eq!(tensor.numel(), 12);
        assert_eq!(tensor.size_bytes, 48);
        assert!(tensor.is_contiguous());
    }

    #[test]
    fn test_tensor_reshape() {
        let tensor = Tensor::new(
            1,
            TensorShape::from_2d(3, 4),
            DataType::F32,
            DeviceId::CPU,
        );
        
        let reshaped = tensor.reshape(&[2, 6]).unwrap();
        assert_eq!(reshaped.numel(), 12);
        assert_eq!(reshaped.shape.dims(), &[2, 6]);
    }

    #[test]
    fn test_tensor_transpose() {
        let tensor = Tensor::new(
            1,
            TensorShape::from_2d(3, 4),
            DataType::F32,
            DeviceId::CPU,
        );
        
        let transposed = tensor.transpose();
        assert_eq!(transposed.shape.dims(), &[4, 3]);
        assert_eq!(transposed.strides, vec![1, 4]);
    }

    #[test]
    fn test_layout_strides() {
        let shape = TensorShape::from_3d(2, 3, 4);
        
        let row_major = Layout::RowMajor.default_strides(&shape);
        assert_eq!(row_major, vec![12, 4, 1]);
        
        let col_major = Layout::ColumnMajor.default_strides(&shape);
        assert_eq!(col_major, vec![1, 2, 6]);
    }

    #[test]
    fn test_tensor_byte_offset() {
        let tensor = Tensor::new(
            1,
            TensorShape::from_2d(3, 4),
            DataType::F32,
            DeviceId::CPU,
        );
        
        assert_eq!(tensor.byte_offset(&[0, 0]).unwrap(), 0);
        assert_eq!(tensor.byte_offset(&[1, 0]).unwrap(), 16);
        assert_eq!(tensor.byte_offset(&[0, 1]).unwrap(), 4);
    }
}
