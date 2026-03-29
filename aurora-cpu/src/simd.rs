//! SIMD operations with multi-versioning

use aurora_core::device::SimdLevel;

/// SIMD vector operations trait
pub trait VectorOps {
    /// Get the SIMD level
    fn level(&self) -> SimdLevel;
    
    /// Vector addition: c[i] = a[i] + b[i]
    fn add_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]);
    
    /// Vector subtraction: c[i] = a[i] - b[i]
    fn sub_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]);
    
    /// Vector multiplication: c[i] = a[i] * b[i]
    fn mul_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]);
    
    /// Vector division: c[i] = a[i] / b[i]
    fn div_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]);
    
    /// Vector fused multiply-add: c[i] = a[i] * b[i] + c[i]
    fn fma_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]);
    
    /// Vector dot product: sum(a[i] * b[i])
    fn dot_f32(&self, a: &[f32], b: &[f32]) -> f32;
    
    /// Vector sum: sum(a[i])
    fn sum_f32(&self, a: &[f32]) -> f32;
    
    /// Vector ReLU: c[i] = max(0, a[i])
    fn relu_f32(&self, a: &[f32], c: &mut [f32]);
    
    /// Vector scale: c[i] = a[i] * scale
    fn scale_f32(&self, a: &[f32], scale: f32, c: &mut [f32]);
}

/// SIMD dispatcher - selects optimal implementation at runtime
pub struct SimdDispatcher {
    level: SimdLevel,
}

impl SimdDispatcher {
    /// Create a new SIMD dispatcher
    pub fn new(level: SimdLevel) -> Self {
        Self { level }
    }
    
    /// Get the SIMD level
    pub fn level(&self) -> SimdLevel {
        self.level
    }
    
    /// Get the best available implementation
    pub fn get_impl(&self) -> Box<dyn VectorOps> {
        match self.level {
            SimdLevel::Avx512 => Box::new(Avx512Ops),
            SimdLevel::Avx2 => Box::new(Avx2Ops),
            SimdLevel::Avx => Box::new(AvxOps),
            SimdLevel::Sse2 | SimdLevel::Sse4_2 => Box::new(Sse2Ops),
            _ => Box::new(ScalarOps),
        }
    }
}

/// Scalar (fallback) implementation
pub struct ScalarOps;

impl VectorOps for ScalarOps {
    fn level(&self) -> SimdLevel {
        SimdLevel::Scalar
    }
    
    fn add_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) {
        let len = a.len().min(b.len()).min(c.len());
        for i in 0..len {
            c[i] = a[i] + b[i];
        }
    }
    
    fn sub_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) {
        let len = a.len().min(b.len()).min(c.len());
        for i in 0..len {
            c[i] = a[i] - b[i];
        }
    }
    
    fn mul_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) {
        let len = a.len().min(b.len()).min(c.len());
        for i in 0..len {
            c[i] = a[i] * b[i];
        }
    }
    
    fn div_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) {
        let len = a.len().min(b.len()).min(c.len());
        for i in 0..len {
            c[i] = a[i] / b[i];
        }
    }
    
    fn fma_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) {
        let len = a.len().min(b.len()).min(c.len());
        for i in 0..len {
            c[i] = a[i] * b[i] + c[i];
        }
    }
    
    fn dot_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        let len = a.len().min(b.len());
        let mut sum = 0.0f32;
        for i in 0..len {
            sum += a[i] * b[i];
        }
        sum
    }
    
    fn sum_f32(&self, a: &[f32]) -> f32 {
        a.iter().sum()
    }
    
    fn relu_f32(&self, a: &[f32], c: &mut [f32]) {
        let len = a.len().min(c.len());
        for i in 0..len {
            c[i] = a[i].max(0.0);
        }
    }
    
    fn scale_f32(&self, a: &[f32], scale: f32, c: &mut [f32]) {
        let len = a.len().min(c.len());
        for i in 0..len {
            c[i] = a[i] * scale;
        }
    }
}

/// SSE2 implementation (128-bit vectors)
pub struct Sse2Ops;

#[cfg(target_arch = "x86_64")]
impl VectorOps for Sse2Ops {
    fn level(&self) -> SimdLevel {
        SimdLevel::Sse2
    }
    
    fn add_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) {
        use std::arch::x86_64::*;
        
        let len = a.len().min(b.len()).min(c.len());
        let mut i = 0;
        
        unsafe {
            while i + 4 <= len {
                let va = _mm_loadu_ps(a.as_ptr().add(i));
                let vb = _mm_loadu_ps(b.as_ptr().add(i));
                let vc = _mm_add_ps(va, vb);
                _mm_storeu_ps(c.as_mut_ptr().add(i), vc);
                i += 4;
            }
        }
        
        // Scalar remainder
        for j in i..len {
            c[j] = a[j] + b[j];
        }
    }
    
    fn sub_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) {
        use std::arch::x86_64::*;
        
        let len = a.len().min(b.len()).min(c.len());
        let mut i = 0;
        
        unsafe {
            while i + 4 <= len {
                let va = _mm_loadu_ps(a.as_ptr().add(i));
                let vb = _mm_loadu_ps(b.as_ptr().add(i));
                let vc = _mm_sub_ps(va, vb);
                _mm_storeu_ps(c.as_mut_ptr().add(i), vc);
                i += 4;
            }
        }
        
        for j in i..len {
            c[j] = a[j] - b[j];
        }
    }
    
    fn mul_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) {
        use std::arch::x86_64::*;
        
        let len = a.len().min(b.len()).min(c.len());
        let mut i = 0;
        
        unsafe {
            while i + 4 <= len {
                let va = _mm_loadu_ps(a.as_ptr().add(i));
                let vb = _mm_loadu_ps(b.as_ptr().add(i));
                let vc = _mm_mul_ps(va, vb);
                _mm_storeu_ps(c.as_mut_ptr().add(i), vc);
                i += 4;
            }
        }
        
        for j in i..len {
            c[j] = a[j] * b[j];
        }
    }
    
    fn div_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) {
        use std::arch::x86_64::*;
        
        let len = a.len().min(b.len()).min(c.len());
        let mut i = 0;
        
        unsafe {
            while i + 4 <= len {
                let va = _mm_loadu_ps(a.as_ptr().add(i));
                let vb = _mm_loadu_ps(b.as_ptr().add(i));
                let vc = _mm_div_ps(va, vb);
                _mm_storeu_ps(c.as_mut_ptr().add(i), vc);
                i += 4;
            }
        }
        
        for j in i..len {
            c[j] = a[j] / b[j];
        }
    }
    
    fn fma_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) {
        // SSE2 doesn't have FMA, use separate mul and add
        let len = a.len().min(b.len()).min(c.len());
        for i in 0..len {
            c[i] = a[i] * b[i] + c[i];
        }
    }
    
    fn dot_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::*;
        
        let len = a.len().min(b.len());
        let mut i = 0;
        let mut sum = unsafe { _mm_setzero_ps() };
        
        unsafe {
            while i + 4 <= len {
                let va = _mm_loadu_ps(a.as_ptr().add(i));
                let vb = _mm_loadu_ps(b.as_ptr().add(i));
                sum = _mm_add_ps(sum, _mm_mul_ps(va, vb));
                i += 4;
            }
        }
        
        // Horizontal sum
        let mut result = unsafe {
            let shuf = _mm_movehdup_ps(sum);
            let sums = _mm_add_ps(sum, shuf);
            let shuf2 = _mm_movehl_ps(shuf, sums);
            let sums2 = _mm_add_ss(sums, shuf2);
            _mm_cvtss_f32(sums2)
        };
        
        // Scalar remainder
        for j in i..len {
            result += a[j] * b[j];
        }
        
        result
    }
    
    fn sum_f32(&self, a: &[f32]) -> f32 {
        use std::arch::x86_64::*;
        
        let len = a.len();
        let mut i = 0;
        let mut sum = unsafe { _mm_setzero_ps() };
        
        unsafe {
            while i + 4 <= len {
                let va = _mm_loadu_ps(a.as_ptr().add(i));
                sum = _mm_add_ps(sum, va);
                i += 4;
            }
        }
        
        let mut result = unsafe {
            let shuf = _mm_movehdup_ps(sum);
            let sums = _mm_add_ps(sum, shuf);
            let shuf2 = _mm_movehl_ps(shuf, sums);
            let sums2 = _mm_add_ss(sums, shuf2);
            _mm_cvtss_f32(sums2)
        };
        
        for j in i..len {
            result += a[j];
        }
        
        result
    }
    
    fn relu_f32(&self, a: &[f32], c: &mut [f32]) {
        use std::arch::x86_64::*;
        
        let len = a.len().min(c.len());
        let mut i = 0;
        let zero = unsafe { _mm_setzero_ps() };
        
        unsafe {
            while i + 4 <= len {
                let va = _mm_loadu_ps(a.as_ptr().add(i));
                let vc = _mm_max_ps(va, zero);
                _mm_storeu_ps(c.as_mut_ptr().add(i), vc);
                i += 4;
            }
        }
        
        for j in i..len {
            c[j] = a[j].max(0.0);
        }
    }
    
    fn scale_f32(&self, a: &[f32], scale: f32, c: &mut [f32]) {
        use std::arch::x86_64::*;
        
        let len = a.len().min(c.len());
        let mut i = 0;
        let vscale = unsafe { _mm_set1_ps(scale) };
        
        unsafe {
            while i + 4 <= len {
                let va = _mm_loadu_ps(a.as_ptr().add(i));
                let vc = _mm_mul_ps(va, vscale);
                _mm_storeu_ps(c.as_mut_ptr().add(i), vc);
                i += 4;
            }
        }
        
        for j in i..len {
            c[j] = a[j] * scale;
        }
    }
}

#[cfg(not(target_arch = "x86_64"))]
impl VectorOps for Sse2Ops {
    fn level(&self) -> SimdLevel { SimdLevel::Sse2 }
    fn add_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) { ScalarOps.add_f32(a, b, c) }
    fn sub_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) { ScalarOps.sub_f32(a, b, c) }
    fn mul_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) { ScalarOps.mul_f32(a, b, c) }
    fn div_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) { ScalarOps.div_f32(a, b, c) }
    fn fma_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) { ScalarOps.fma_f32(a, b, c) }
    fn dot_f32(&self, a: &[f32], b: &[f32]) -> f32 { ScalarOps.dot_f32(a, b) }
    fn sum_f32(&self, a: &[f32]) -> f32 { ScalarOps.sum_f32(a) }
    fn relu_f32(&self, a: &[f32], c: &mut [f32]) { ScalarOps.relu_f32(a, c) }
    fn scale_f32(&self, a: &[f32], scale: f32, c: &mut [f32]) { ScalarOps.scale_f32(a, scale, c) }
}

/// AVX implementation (256-bit vectors)
pub struct AvxOps;

#[cfg(target_arch = "x86_64")]
impl VectorOps for AvxOps {
    fn level(&self) -> SimdLevel {
        SimdLevel::Avx
    }
    
    fn add_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) {
        use std::arch::x86_64::*;
        
        let len = a.len().min(b.len()).min(c.len());
        let mut i = 0;
        
        unsafe {
            while i + 8 <= len {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                let vc = _mm256_add_ps(va, vb);
                _mm256_storeu_ps(c.as_mut_ptr().add(i), vc);
                i += 8;
            }
        }
        
        // SSE2 remainder
        Sse2Ops.add_f32(&a[i..], &b[i..], &mut c[i..]);
    }
    
    fn sub_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) {
        use std::arch::x86_64::*;
        
        let len = a.len().min(b.len()).min(c.len());
        let mut i = 0;
        
        unsafe {
            while i + 8 <= len {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                let vc = _mm256_sub_ps(va, vb);
                _mm256_storeu_ps(c.as_mut_ptr().add(i), vc);
                i += 8;
            }
        }
        
        Sse2Ops.sub_f32(&a[i..], &b[i..], &mut c[i..]);
    }
    
    fn mul_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) {
        use std::arch::x86_64::*;
        
        let len = a.len().min(b.len()).min(c.len());
        let mut i = 0;
        
        unsafe {
            while i + 8 <= len {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                let vc = _mm256_mul_ps(va, vb);
                _mm256_storeu_ps(c.as_mut_ptr().add(i), vc);
                i += 8;
            }
        }
        
        Sse2Ops.mul_f32(&a[i..], &b[i..], &mut c[i..]);
    }
    
    fn div_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) {
        use std::arch::x86_64::*;
        
        let len = a.len().min(b.len()).min(c.len());
        let mut i = 0;
        
        unsafe {
            while i + 8 <= len {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                let vc = _mm256_div_ps(va, vb);
                _mm256_storeu_ps(c.as_mut_ptr().add(i), vc);
                i += 8;
            }
        }
        
        Sse2Ops.div_f32(&a[i..], &b[i..], &mut c[i..]);
    }
    
    fn fma_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) {
        // AVX without AVX2 doesn't have FMA
        let len = a.len().min(b.len()).min(c.len());
        let mut i = 0;
        
        use std::arch::x86_64::*;
        unsafe {
            while i + 8 <= len {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                let vc = _mm256_loadu_ps(c.as_ptr().add(i));
                let vmul = _mm256_mul_ps(va, vb);
                let vresult = _mm256_add_ps(vmul, vc);
                _mm256_storeu_ps(c.as_mut_ptr().add(i), vresult);
                i += 8;
            }
        }
        
        Sse2Ops.fma_f32(&a[i..], &b[i..], &mut c[i..]);
    }
    
    fn dot_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::*;
        
        let len = a.len().min(b.len());
        let mut i = 0;
        let mut sum = unsafe { _mm256_setzero_ps() };
        
        unsafe {
            while i + 8 <= len {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
                i += 8;
            }
        }
        
        // Horizontal sum of 256-bit vector
        let mut result = unsafe {
            let low = _mm256_castps256_ps128(sum);
            let high = _mm256_extractf128_ps(sum, 1);
            let combined = _mm_add_ps(low, high);
            let shuf = _mm_movehdup_ps(combined);
            let sums = _mm_add_ps(combined, shuf);
            let shuf2 = _mm_movehl_ps(shuf, sums);
            let sums2 = _mm_add_ss(sums, shuf2);
            _mm_cvtss_f32(sums2)
        };
        
        result += Sse2Ops.dot_f32(&a[i..], &b[i..]);
        result
    }
    
    fn sum_f32(&self, a: &[f32]) -> f32 {
        use std::arch::x86_64::*;
        
        let len = a.len();
        let mut i = 0;
        let mut sum = unsafe { _mm256_setzero_ps() };
        
        unsafe {
            while i + 8 <= len {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                sum = _mm256_add_ps(sum, va);
                i += 8;
            }
        }
        
        let mut result = unsafe {
            let low = _mm256_castps256_ps128(sum);
            let high = _mm256_extractf128_ps(sum, 1);
            let combined = _mm_add_ps(low, high);
            let shuf = _mm_movehdup_ps(combined);
            let sums = _mm_add_ps(combined, shuf);
            let shuf2 = _mm_movehl_ps(shuf, sums);
            let sums2 = _mm_add_ss(sums, shuf2);
            _mm_cvtss_f32(sums2)
        };
        
        result += Sse2Ops.sum_f32(&a[i..]);
        result
    }
    
    fn relu_f32(&self, a: &[f32], c: &mut [f32]) {
        use std::arch::x86_64::*;
        
        let len = a.len().min(c.len());
        let mut i = 0;
        let zero = unsafe { _mm256_setzero_ps() };
        
        unsafe {
            while i + 8 <= len {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vc = _mm256_max_ps(va, zero);
                _mm256_storeu_ps(c.as_mut_ptr().add(i), vc);
                i += 8;
            }
        }
        
        Sse2Ops.relu_f32(&a[i..], &mut c[i..]);
    }
    
    fn scale_f32(&self, a: &[f32], scale: f32, c: &mut [f32]) {
        use std::arch::x86_64::*;
        
        let len = a.len().min(c.len());
        let mut i = 0;
        let vscale = unsafe { _mm256_set1_ps(scale) };
        
        unsafe {
            while i + 8 <= len {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vc = _mm256_mul_ps(va, vscale);
                _mm256_storeu_ps(c.as_mut_ptr().add(i), vc);
                i += 8;
            }
        }
        
        Sse2Ops.scale_f32(&a[i..], scale, &mut c[i..]);
    }
}

#[cfg(not(target_arch = "x86_64"))]
impl VectorOps for AvxOps {
    fn level(&self) -> SimdLevel { SimdLevel::Avx }
    fn add_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) { ScalarOps.add_f32(a, b, c) }
    fn sub_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) { ScalarOps.sub_f32(a, b, c) }
    fn mul_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) { ScalarOps.mul_f32(a, b, c) }
    fn div_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) { ScalarOps.div_f32(a, b, c) }
    fn fma_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) { ScalarOps.fma_f32(a, b, c) }
    fn dot_f32(&self, a: &[f32], b: &[f32]) -> f32 { ScalarOps.dot_f32(a, b) }
    fn sum_f32(&self, a: &[f32]) -> f32 { ScalarOps.sum_f32(a) }
    fn relu_f32(&self, a: &[f32], c: &mut [f32]) { ScalarOps.relu_f32(a, c) }
    fn scale_f32(&self, a: &[f32], scale: f32, c: &mut [f32]) { ScalarOps.scale_f32(a, scale, c) }
}

/// AVX2 implementation (256-bit with FMA)
pub struct Avx2Ops;

#[cfg(target_arch = "x86_64")]
impl VectorOps for Avx2Ops {
    fn level(&self) -> SimdLevel {
        SimdLevel::Avx2
    }
    
    fn add_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) {
        AvxOps.add_f32(a, b, c)
    }
    
    fn sub_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) {
        AvxOps.sub_f32(a, b, c)
    }
    
    fn mul_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) {
        AvxOps.mul_f32(a, b, c)
    }
    
    fn div_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) {
        AvxOps.div_f32(a, b, c)
    }
    
    fn fma_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) {
        use std::arch::x86_64::*;
        
        let len = a.len().min(b.len()).min(c.len());
        let mut i = 0;
        
        unsafe {
            while i + 8 <= len {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                let vc = _mm256_loadu_ps(c.as_ptr().add(i));
                let vresult = _mm256_fmadd_ps(va, vb, vc);
                _mm256_storeu_ps(c.as_mut_ptr().add(i), vresult);
                i += 8;
            }
        }
        
        Sse2Ops.fma_f32(&a[i..], &b[i..], &mut c[i..]);
    }
    
    fn dot_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::*;
        
        let len = a.len().min(b.len());
        let mut i = 0;
        let mut sum = unsafe { _mm256_setzero_ps() };
        
        unsafe {
            while i + 8 <= len {
                let va = _mm256_loadu_ps(a.as_ptr().add(i));
                let vb = _mm256_loadu_ps(b.as_ptr().add(i));
                sum = _mm256_fmadd_ps(va, vb, sum);
                i += 8;
            }
        }
        
        let mut result = unsafe {
            let low = _mm256_castps256_ps128(sum);
            let high = _mm256_extractf128_ps(sum, 1);
            let combined = _mm_add_ps(low, high);
            let shuf = _mm_movehdup_ps(combined);
            let sums = _mm_add_ps(combined, shuf);
            let shuf2 = _mm_movehl_ps(shuf, sums);
            let sums2 = _mm_add_ss(sums, shuf2);
            _mm_cvtss_f32(sums2)
        };
        
        result += Sse2Ops.dot_f32(&a[i..], &b[i..]);
        result
    }
    
    fn sum_f32(&self, a: &[f32]) -> f32 {
        AvxOps.sum_f32(a)
    }
    
    fn relu_f32(&self, a: &[f32], c: &mut [f32]) {
        AvxOps.relu_f32(a, c)
    }
    
    fn scale_f32(&self, a: &[f32], scale: f32, c: &mut [f32]) {
        AvxOps.scale_f32(a, scale, c)
    }
}

#[cfg(not(target_arch = "x86_64"))]
impl VectorOps for Avx2Ops {
    fn level(&self) -> SimdLevel { SimdLevel::Avx2 }
    fn add_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) { ScalarOps.add_f32(a, b, c) }
    fn sub_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) { ScalarOps.sub_f32(a, b, c) }
    fn mul_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) { ScalarOps.mul_f32(a, b, c) }
    fn div_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) { ScalarOps.div_f32(a, b, c) }
    fn fma_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) { ScalarOps.fma_f32(a, b, c) }
    fn dot_f32(&self, a: &[f32], b: &[f32]) -> f32 { ScalarOps.dot_f32(a, b) }
    fn sum_f32(&self, a: &[f32]) -> f32 { ScalarOps.sum_f32(a) }
    fn relu_f32(&self, a: &[f32], c: &mut [f32]) { ScalarOps.relu_f32(a, c) }
    fn scale_f32(&self, a: &[f32], scale: f32, c: &mut [f32]) { ScalarOps.scale_f32(a, scale, c) }
}

/// AVX-512 implementation (512-bit vectors)
pub struct Avx512Ops;

#[cfg(target_arch = "x86_64")]
impl VectorOps for Avx512Ops {
    fn level(&self) -> SimdLevel {
        SimdLevel::Avx512
    }
    
    fn add_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) {
        use std::arch::x86_64::*;
        
        let len = a.len().min(b.len()).min(c.len());
        let mut i = 0;
        
        unsafe {
            while i + 16 <= len {
                let va = _mm512_loadu_ps(a.as_ptr().add(i));
                let vb = _mm512_loadu_ps(b.as_ptr().add(i));
                let vc = _mm512_add_ps(va, vb);
                _mm512_storeu_ps(c.as_mut_ptr().add(i), vc);
                i += 16;
            }
        }
        
        Avx2Ops.add_f32(&a[i..], &b[i..], &mut c[i..]);
    }
    
    fn sub_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) {
        use std::arch::x86_64::*;
        
        let len = a.len().min(b.len()).min(c.len());
        let mut i = 0;
        
        unsafe {
            while i + 16 <= len {
                let va = _mm512_loadu_ps(a.as_ptr().add(i));
                let vb = _mm512_loadu_ps(b.as_ptr().add(i));
                let vc = _mm512_sub_ps(va, vb);
                _mm512_storeu_ps(c.as_mut_ptr().add(i), vc);
                i += 16;
            }
        }
        
        Avx2Ops.sub_f32(&a[i..], &b[i..], &mut c[i..]);
    }
    
    fn mul_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) {
        use std::arch::x86_64::*;
        
        let len = a.len().min(b.len()).min(c.len());
        let mut i = 0;
        
        unsafe {
            while i + 16 <= len {
                let va = _mm512_loadu_ps(a.as_ptr().add(i));
                let vb = _mm512_loadu_ps(b.as_ptr().add(i));
                let vc = _mm512_mul_ps(va, vb);
                _mm512_storeu_ps(c.as_mut_ptr().add(i), vc);
                i += 16;
            }
        }
        
        Avx2Ops.mul_f32(&a[i..], &b[i..], &mut c[i..]);
    }
    
    fn div_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) {
        use std::arch::x86_64::*;
        
        let len = a.len().min(b.len()).min(c.len());
        let mut i = 0;
        
        unsafe {
            while i + 16 <= len {
                let va = _mm512_loadu_ps(a.as_ptr().add(i));
                let vb = _mm512_loadu_ps(b.as_ptr().add(i));
                let vc = _mm512_div_ps(va, vb);
                _mm512_storeu_ps(c.as_mut_ptr().add(i), vc);
                i += 16;
            }
        }
        
        Avx2Ops.div_f32(&a[i..], &b[i..], &mut c[i..]);
    }
    
    fn fma_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) {
        use std::arch::x86_64::*;
        
        let len = a.len().min(b.len()).min(c.len());
        let mut i = 0;
        
        unsafe {
            while i + 16 <= len {
                let va = _mm512_loadu_ps(a.as_ptr().add(i));
                let vb = _mm512_loadu_ps(b.as_ptr().add(i));
                let vc = _mm512_loadu_ps(c.as_ptr().add(i));
                let vresult = _mm512_fmadd_ps(va, vb, vc);
                _mm512_storeu_ps(c.as_mut_ptr().add(i), vresult);
                i += 16;
            }
        }
        
        Avx2Ops.fma_f32(&a[i..], &b[i..], &mut c[i..]);
    }
    
    fn dot_f32(&self, a: &[f32], b: &[f32]) -> f32 {
        use std::arch::x86_64::*;
        
        let len = a.len().min(b.len());
        let mut i = 0;
        let mut sum = unsafe { _mm512_setzero_ps() };
        
        unsafe {
            while i + 16 <= len {
                let va = _mm512_loadu_ps(a.as_ptr().add(i));
                let vb = _mm512_loadu_ps(b.as_ptr().add(i));
                sum = _mm512_fmadd_ps(va, vb, sum);
                i += 16;
            }
        }
        
        let result = unsafe { _mm512_reduce_add_ps(sum) };
        result + Avx2Ops.dot_f32(&a[i..], &b[i..])
    }
    
    fn sum_f32(&self, a: &[f32]) -> f32 {
        use std::arch::x86_64::*;
        
        let len = a.len();
        let mut i = 0;
        let mut sum = unsafe { _mm512_setzero_ps() };
        
        unsafe {
            while i + 16 <= len {
                let va = _mm512_loadu_ps(a.as_ptr().add(i));
                sum = _mm512_add_ps(sum, va);
                i += 16;
            }
        }
        
        let result = unsafe { _mm512_reduce_add_ps(sum) };
        result + Avx2Ops.sum_f32(&a[i..])
    }
    
    fn relu_f32(&self, a: &[f32], c: &mut [f32]) {
        use std::arch::x86_64::*;
        
        let len = a.len().min(c.len());
        let mut i = 0;
        let zero = unsafe { _mm512_setzero_ps() };
        
        unsafe {
            while i + 16 <= len {
                let va = _mm512_loadu_ps(a.as_ptr().add(i));
                let vc = _mm512_max_ps(va, zero);
                _mm512_storeu_ps(c.as_mut_ptr().add(i), vc);
                i += 16;
            }
        }
        
        Avx2Ops.relu_f32(&a[i..], &mut c[i..]);
    }
    
    fn scale_f32(&self, a: &[f32], scale: f32, c: &mut [f32]) {
        use std::arch::x86_64::*;
        
        let len = a.len().min(c.len());
        let mut i = 0;
        let vscale = unsafe { _mm512_set1_ps(scale) };
        
        unsafe {
            while i + 16 <= len {
                let va = _mm512_loadu_ps(a.as_ptr().add(i));
                let vc = _mm512_mul_ps(va, vscale);
                _mm512_storeu_ps(c.as_mut_ptr().add(i), vc);
                i += 16;
            }
        }
        
        Avx2Ops.scale_f32(&a[i..], scale, &mut c[i..]);
    }
}

#[cfg(not(target_arch = "x86_64"))]
impl VectorOps for Avx512Ops {
    fn level(&self) -> SimdLevel { SimdLevel::Avx512 }
    fn add_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) { ScalarOps.add_f32(a, b, c) }
    fn sub_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) { ScalarOps.sub_f32(a, b, c) }
    fn mul_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) { ScalarOps.mul_f32(a, b, c) }
    fn div_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) { ScalarOps.div_f32(a, b, c) }
    fn fma_f32(&self, a: &[f32], b: &[f32], c: &mut [f32]) { ScalarOps.fma_f32(a, b, c) }
    fn dot_f32(&self, a: &[f32], b: &[f32]) -> f32 { ScalarOps.dot_f32(a, b) }
    fn sum_f32(&self, a: &[f32]) -> f32 { ScalarOps.sum_f32(a) }
    fn relu_f32(&self, a: &[f32], c: &mut [f32]) { ScalarOps.relu_f32(a, c) }
    fn scale_f32(&self, a: &[f32], scale: f32, c: &mut [f32]) { ScalarOps.scale_f32(a, scale, c) }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_vector_ops<V: VectorOps + ?Sized>(ops: &V) {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0f32, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut c = vec![0.0f32; 8];
        
        // Test add
        ops.add_f32(&a, &b, &mut c);
        assert_eq!(c, vec![3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0]);
        
        // Test mul
        ops.mul_f32(&a, &b, &mut c);
        assert_eq!(c, vec![2.0, 6.0, 12.0, 20.0, 30.0, 42.0, 56.0, 72.0]);
        
        // Test dot
        let dot = ops.dot_f32(&a, &b);
        assert!((dot - 240.0).abs() < 0.001);
        
        // Test sum
        let sum = ops.sum_f32(&a);
        assert!((sum - 36.0).abs() < 0.001);
        
        // Test relu
        let d = vec![-1.0f32, 2.0, -3.0, 4.0];
        let mut e = vec![0.0f32; 4];
        ops.relu_f32(&d, &mut e);
        assert_eq!(e, vec![0.0, 2.0, 0.0, 4.0]);
        
        // Test scale
        ops.scale_f32(&a, 2.0, &mut c);
        assert_eq!(c, vec![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]);
    }

    #[test]
    fn test_scalar_ops() {
        test_vector_ops(&ScalarOps);
    }

    #[test]
    fn test_simd_dispatcher() {
        let dispatcher = SimdDispatcher::new(SimdLevel::Scalar);
        let ops = dispatcher.get_impl();
        test_vector_ops(ops.as_ref());
    }

    #[test]
    fn test_large_vectors() {
        let size = 1000;
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
        let mut c = vec![0.0f32; size];
        
        ScalarOps.add_f32(&a, &b, &mut c);
        
        for i in 0..size {
            assert!((c[i] - (i * 3) as f32).abs() < 0.001);
        }
    }
}
