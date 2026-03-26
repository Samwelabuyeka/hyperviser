//! Error types for AURORA

use std::fmt;
use std::io;

/// Result type alias for AURORA operations
pub type Result<T> = std::result::Result<T, AuroraError>;

/// Main error type for AURORA runtime
#[derive(Debug, Clone)]
pub enum AuroraError {
    /// Invalid argument or configuration
    InvalidArgument(String),
    
    /// Device-related error
    DeviceError {
        device: String,
        message: String,
    },
    
    /// Out of memory
    OutOfMemory {
        requested: usize,
        available: usize,
    },
    
    /// Tensor shape mismatch
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    
    /// Unsupported operation
    Unsupported(String),
    
    /// Kernel execution error
    KernelError(String),
    
    /// Hardware detection error
    HardwareDetectionError(String),
    
    /// I/O error
    IoError(String),
    
    /// Not initialized
    NotInitialized,
    
    /// Already initialized
    AlreadyInitialized,
    
    /// Profiling error
    ProfilingError(String),
    
    /// Auto-tune error
    AutotuneError(String),
    
    /// External library error
    ExternalError {
        library: String,
        message: String,
    },
    
    /// Not found
    NotFound(String),
    
    /// Permission denied
    PermissionDenied(String),
    
    /// Timeout
    Timeout(String),
    
    /// Cancelled
    Cancelled,
    
    /// Other error
    Other(String),
}

impl fmt::Display for AuroraError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidArgument(msg) => write!(f, "Invalid argument: {}", msg),
            Self::DeviceError { device, message } => {
                write!(f, "Device error on {}: {}", device, message)
            }
            Self::OutOfMemory { requested, available } => {
                write!(f, "Out of memory: requested {} bytes, available {} bytes", requested, available)
            }
            Self::ShapeMismatch { expected, got } => {
                write!(f, "Shape mismatch: expected {:?}, got {:?}", expected, got)
            }
            Self::Unsupported(msg) => write!(f, "Unsupported operation: {}", msg),
            Self::KernelError(msg) => write!(f, "Kernel execution failed: {}", msg),
            Self::HardwareDetectionError(msg) => write!(f, "Hardware detection failed: {}", msg),
            Self::IoError(msg) => write!(f, "I/O error: {}", msg),
            Self::NotInitialized => write!(f, "AURORA runtime not initialized"),
            Self::AlreadyInitialized => write!(f, "AURORA runtime already initialized"),
            Self::ProfilingError(msg) => write!(f, "Profiling error: {}", msg),
            Self::AutotuneError(msg) => write!(f, "Auto-tune error: {}", msg),
            Self::ExternalError { library, message } => {
                write!(f, "External library error ({}): {}", library, message)
            }
            Self::NotFound(msg) => write!(f, "Not found: {}", msg),
            Self::PermissionDenied(msg) => write!(f, "Permission denied: {}", msg),
            Self::Timeout(msg) => write!(f, "Timeout: {}", msg),
            Self::Cancelled => write!(f, "Operation cancelled"),
            Self::Other(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for AuroraError {}

impl AuroraError {
    /// Create an invalid argument error
    pub fn invalid_arg<S: Into<String>>(msg: S) -> Self {
        Self::InvalidArgument(msg.into())
    }
    
    /// Create a device error
    pub fn device_error<D: Into<String>, M: Into<String>>(device: D, message: M) -> Self {
        Self::DeviceError {
            device: device.into(),
            message: message.into(),
        }
    }
    
    /// Create an out of memory error
    pub fn oom(requested: usize, available: usize) -> Self {
        Self::OutOfMemory { requested, available }
    }
    
    /// Create a shape mismatch error
    pub fn shape_mismatch(expected: Vec<usize>, got: Vec<usize>) -> Self {
        Self::ShapeMismatch { expected, got }
    }
    
    /// Create unsupported error
    pub fn unsupported<S: Into<String>>(msg: S) -> Self {
        Self::Unsupported(msg.into())
    }
    
    /// Create kernel error
    pub fn kernel_error<S: Into<String>>(msg: S) -> Self {
        Self::KernelError(msg.into())
    }
    
    /// Create not found error
    pub fn not_found<S: Into<String>>(msg: S) -> Self {
        Self::NotFound(msg.into())
    }
    
    /// Create permission denied error
    pub fn permission_denied<S: Into<String>>(msg: S) -> Self {
        Self::PermissionDenied(msg.into())
    }
    
    /// Create timeout error
    pub fn timeout<S: Into<String>>(msg: S) -> Self {
        Self::Timeout(msg.into())
    }
    
    /// Check if error is retryable
    pub fn is_retryable(&self) -> bool {
        matches!(self, 
            Self::DeviceError { .. } |
            Self::OutOfMemory { .. } |
            Self::Timeout { .. } |
            Self::IoError { .. }
        )
    }
    
    /// Check if error is fatal
    pub fn is_fatal(&self) -> bool {
        matches!(self,
            Self::InvalidArgument { .. } |
            Self::NotInitialized |
            Self::AlreadyInitialized |
            Self::Cancelled |
            Self::PermissionDenied { .. }
        )
    }
}

impl From<io::Error> for AuroraError {
    fn from(e: io::Error) -> Self {
        Self::IoError(e.to_string())
    }
}

impl From<std::num::TryFromIntError> for AuroraError {
    fn from(e: std::num::TryFromIntError) -> Self {
        Self::InvalidArgument(e.to_string())
    }
}

impl From<std::num::ParseIntError> for AuroraError {
    fn from(e: std::num::ParseIntError) -> Self {
        Self::InvalidArgument(e.to_string())
    }
}

impl From<std::num::ParseFloatError> for AuroraError {
    fn from(e: std::num::ParseFloatError) -> Self {
        Self::InvalidArgument(e.to_string())
    }
}

impl From<std::str::Utf8Error> for AuroraError {
    fn from(e: std::str::Utf8Error) -> Self {
        Self::InvalidArgument(format!("UTF-8 error: {}", e))
    }
}

impl From<std::string::FromUtf8Error> for AuroraError {
    fn from(e: std::string::FromUtf8Error) -> Self {
        Self::InvalidArgument(format!("UTF-8 error: {}", e))
    }
}

impl<T> From<std::sync::PoisonError<T>> for AuroraError {
    fn from(_: std::sync::PoisonError<T>) -> Self {
        Self::Other("Lock poisoned".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_creation() {
        let err = AuroraError::invalid_arg("test");
        assert!(matches!(err, AuroraError::InvalidArgument(_)));
        
        let err = AuroraError::oom(1024, 512);
        assert!(matches!(err, AuroraError::OutOfMemory { .. }));
    }

    #[test]
    fn test_error_display() {
        let err = AuroraError::invalid_arg("bad value");
        assert!(err.to_string().contains("bad value"));
        
        let err = AuroraError::oom(1024, 512);
        assert!(err.to_string().contains("Out of memory"));
    }

    #[test]
    fn test_is_retryable() {
        assert!(AuroraError::oom(1, 0).is_retryable());
        assert!(AuroraError::timeout("test").is_retryable());
        assert!(!AuroraError::invalid_arg("test").is_retryable());
    }

    #[test]
    fn test_is_fatal() {
        assert!(AuroraError::NotInitialized.is_fatal());
        assert!(AuroraError::permission_denied("test").is_fatal());
        assert!(!AuroraError::oom(1, 0).is_fatal());
    }

    #[test]
    fn test_from_io_error() {
        let io_err = io::Error::new(io::ErrorKind::NotFound, "file not found");
        let err: AuroraError = io_err.into();
        assert!(matches!(err, AuroraError::IoError(_)));
    }
}
