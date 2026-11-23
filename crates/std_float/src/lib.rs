#![cfg_attr(
    feature = "as_crate",
    feature(core_intrinsics),
    feature(portable_simd),
    feature(f16),
    feature(impl_restriction),
    allow(internal_features)
)]
use core::intrinsics::simd as intrinsics;
#[cfg(not(feature = "as_crate"))]
use core::simd;

#[cfg(feature = "as_crate")]
use core_simd::simd;
use simd::Simd;

/// This trait provides a possibly-temporary implementation of float functions
/// that may, in the absence of hardware support, canonicalize to calling an
/// operating system's `math.h` dynamically-loaded library (also known as a
/// shared object). As these conditionally require runtime support, they
/// should only appear in binaries built assuming OS support: `std`.
///
/// However, there is no reason SIMD types, in general, need OS support,
/// as for many architectures an embedded binary may simply configure that
/// support itself. This means these types must be visible in `core`
/// but have these functions available in `std`.
///
/// [`f32`] and [`f64`] achieve a similar trick by using "lang items", but
/// due to compiler limitations, it is harder to implement this approach for
/// abstract data types like [`Simd`]. From that need, this trait is born.
///
/// It is possible this trait will be replaced in some manner in the future,
/// when either the compiler or its supporting runtime functions are improved.
/// For now this trait is available to permit experimentation with SIMD float
/// operations that may lack hardware support, such as `mul_add`.
pub impl(self) trait StdFloat: Sized {
    /// Elementwise fused multiply-add. Computes `(self * a) + b` with only one rounding error,
    /// yielding a more accurate result than an unfused multiply-add.
    ///
    /// Using `mul_add` *may* be more performant than an unfused multiply-add if the target
    /// architecture has a dedicated `fma` CPU instruction.  However, this is not always
    /// true, and will be heavily dependent on designing algorithms with specific target
    /// hardware in mind.
    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn mul_add(self, a: Self, b: Self) -> Self {
        // SAFETY: this trait is implemented only for supported floating-point SIMD vectors,
        // and all three operands have the same vector type and lane count.
        unsafe { intrinsics::simd_fma(self, a, b) }
    }

    /// Elementwise fused multiply-subtract. Computes `(self * a) - b` with one rounding
    /// step, avoiding the intermediate rounding of a separate multiplication and subtraction.
    ///
    /// This may use a dedicated fused multiply-add instruction when the target supports one.
    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn mul_sub(self, a: Self, b: Self) -> Self {
        // SAFETY: The operands are SIMD floating-point vectors, as required by simd_fma.
        unsafe { intrinsics::simd_fma(self, a, intrinsics::simd_neg(b)) }
    }

    /// Produces a vector where every element has the square root value
    /// of the equivalently-indexed element in `self`
    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn sqrt(self) -> Self {
        // SAFETY: `self` is a supported floating-point SIMD vector.
        unsafe { intrinsics::simd_fsqrt(self) }
    }

    /// Computes the cube root of each lane using the scalar floating-point implementation.
    fn cbrt(self) -> Self;

    /// Computes the hypotenuse of each pair of corresponding lanes.
    fn hypot(self, other: Self) -> Self;

    /// Computes e^x - 1 for each lane, accurately near zero.
    fn exp_m1(self) -> Self;

    /// Computes ln(1 + x) for each lane, accurately near zero.
    fn ln_1p(self) -> Self;

    /// Raises each lane to the same integer exponent.
    fn powi(self, n: i32) -> Self;

    /// Raises each lane to its corresponding floating-point exponent.
    fn powf(self, n: Self) -> Self;

    /// Computes the tangent of each lane.
    fn tan(self) -> Self;
    /// Computes the inverse sine of each lane.
    fn asin(self) -> Self;
    /// Computes the inverse cosine of each lane.
    fn acos(self) -> Self;
    /// Computes the inverse tangent of each lane.
    fn atan(self) -> Self;
    /// Computes the four-quadrant inverse tangent of each lane pair.
    fn atan2(self, other: Self) -> Self;
    /// Computes the hyperbolic sine of each lane.
    fn sinh(self) -> Self;
    /// Computes the hyperbolic cosine of each lane.
    fn cosh(self) -> Self;
    /// Computes the hyperbolic tangent of each lane.
    fn tanh(self) -> Self;
    /// Computes the inverse hyperbolic sine of each lane.
    fn asinh(self) -> Self;
    /// Computes the inverse hyperbolic cosine of each lane.
    fn acosh(self) -> Self;
    /// Computes the inverse hyperbolic tangent of each lane.
    fn atanh(self) -> Self;

    /// Produces a vector where every element has the sine of the value
    /// in the equivalently-indexed element in `self`.
    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn sin(self) -> Self {
        // SAFETY: `self` is a supported floating-point SIMD vector.
        unsafe { intrinsics::simd_fsin(self) }
    }

    /// Produces a vector where every element has the cosine of the value
    /// in the equivalently-indexed element in `self`.
    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn cos(self) -> Self {
        // SAFETY: `self` is a supported floating-point SIMD vector.
        unsafe { intrinsics::simd_fcos(self) }
    }

    /// Produces a vector where every element has the exponential (base e) of the value
    /// in the equivalently-indexed element in `self`.
    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn exp(self) -> Self {
        // SAFETY: `self` is a supported floating-point SIMD vector.
        unsafe { intrinsics::simd_fexp(self) }
    }

    /// Produces a vector where every element has the exponential (base 2) of the value
    /// in the equivalently-indexed element in `self`.
    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn exp2(self) -> Self {
        // SAFETY: `self` is a supported floating-point SIMD vector.
        unsafe { intrinsics::simd_fexp2(self) }
    }

    /// Produces a vector where every element has the natural logarithm of the value
    /// in the equivalently-indexed element in `self`.
    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn ln(self) -> Self {
        // SAFETY: `self` is a supported floating-point SIMD vector.
        unsafe { intrinsics::simd_flog(self) }
    }

    /// Produces a vector where every element has the logarithm with respect to an arbitrary
    /// in the equivalently-indexed elements in `self` and `base`.
    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn log(self, base: Self) -> Self {
        // SAFETY: both logarithms produce the same supported floating-point SIMD type.
        unsafe { intrinsics::simd_div(self.ln(), base.ln()) }
    }

    /// Produces a vector where every element has the base-2 logarithm of the value
    /// in the equivalently-indexed element in `self`.
    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn log2(self) -> Self {
        // SAFETY: `self` is a supported floating-point SIMD vector.
        unsafe { intrinsics::simd_flog2(self) }
    }

    /// Produces a vector where every element has the base-10 logarithm of the value
    /// in the equivalently-indexed element in `self`.
    #[inline]
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn log10(self) -> Self {
        // SAFETY: `self` is a supported floating-point SIMD vector.
        unsafe { intrinsics::simd_flog10(self) }
    }

    /// Returns the smallest integer greater than or equal to each element.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    #[inline]
    fn ceil(self) -> Self {
        // SAFETY: `self` is a supported floating-point SIMD vector.
        unsafe { intrinsics::simd_ceil(self) }
    }

    /// Returns the largest integer value less than or equal to each element.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    #[inline]
    fn floor(self) -> Self {
        // SAFETY: `self` is a supported floating-point SIMD vector.
        unsafe { intrinsics::simd_floor(self) }
    }

    /// Rounds to the nearest integer value. Ties round away from zero.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    #[inline]
    fn round(self) -> Self {
        // SAFETY: `self` is a supported floating-point SIMD vector.
        unsafe { intrinsics::simd_round(self) }
    }

    /// Returns the floating point's integer value, with its fractional part removed.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    #[inline]
    fn trunc(self) -> Self {
        // SAFETY: `self` is a supported floating-point SIMD vector.
        unsafe { intrinsics::simd_trunc(self) }
    }

    /// Rounds each element to the nearest integer-valued float.
    /// Ties are resolved by rounding to the number with an even least significant digit.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    #[inline]
    fn round_ties_even(self) -> Self {
        // SAFETY: `self` is a supported floating-point SIMD vector.
        unsafe { intrinsics::simd_round_ties_even(self) }
    }

    /// Rounds each element to the nearest integer-valued float using the platform's
    /// efficient SIMD rounding operation. Halfway cases are rounded to the nearest even
    /// integer, matching [`round_ties_even`](Self::round_ties_even).
    ///
    /// Unlike [`round`](Self::round), this method does not round halfway cases away from zero.
    /// Its performance advantage depends on the target architecture and compiler.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    #[inline]
    fn round_fast(self) -> Self {
        // SAFETY: `self` is a supported floating-point SIMD vector.
        unsafe { intrinsics::simd_round_ties_even(self) }
    }

    /// Returns the floating point's fractional value, with its integer part removed.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn fract(self) -> Self;
}

impl<const N: usize> StdFloat for Simd<f16, N> {
    #[inline]
    fn cbrt(self) -> Self {
        Self::from_array(self.to_array().map(f16::cbrt))
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        Self::from_array(core::array::from_fn(|i| {
            self.as_array()[i].hypot(other.as_array()[i])
        }))
    }

    #[inline]
    fn exp_m1(self) -> Self {
        Self::from_array(self.to_array().map(f16::exp_m1))
    }

    #[inline]
    fn ln_1p(self) -> Self {
        Self::from_array(self.to_array().map(f16::ln_1p))
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        Self::from_array(self.to_array().map(|x| x.powi(n)))
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        Self::from_array(core::array::from_fn(|i| {
            self.as_array()[i].powf(n.as_array()[i])
        }))
    }

    #[inline]
    fn tan(self) -> Self { Self::from_array(self.to_array().map(f16::tan)) }
    #[inline]
    fn asin(self) -> Self { Self::from_array(self.to_array().map(f16::asin)) }
    #[inline]
    fn acos(self) -> Self { Self::from_array(self.to_array().map(f16::acos)) }
    #[inline]
    fn atan(self) -> Self { Self::from_array(self.to_array().map(f16::atan)) }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        Self::from_array(core::array::from_fn(|i| self.as_array()[i].atan2(other.as_array()[i])))
    }
    #[inline]
    fn sinh(self) -> Self { Self::from_array(self.to_array().map(f16::sinh)) }
    #[inline]
    fn cosh(self) -> Self { Self::from_array(self.to_array().map(f16::cosh)) }
    #[inline]
    fn tanh(self) -> Self { Self::from_array(self.to_array().map(f16::tanh)) }
    #[inline]
    fn asinh(self) -> Self { Self::from_array(self.to_array().map(f16::asinh)) }
    #[inline]
    fn acosh(self) -> Self { Self::from_array(self.to_array().map(f16::acosh)) }
    #[inline]
    fn atanh(self) -> Self { Self::from_array(self.to_array().map(f16::atanh)) }

    #[inline]
    fn fract(self) -> Self {
        self - self.trunc()
    }
}

impl<const N: usize> StdFloat for Simd<f32, N> {
    #[inline]
    fn cbrt(self) -> Self {
        Self::from_array(self.to_array().map(f32::cbrt))
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        Self::from_array(core::array::from_fn(|i| {
            self.as_array()[i].hypot(other.as_array()[i])
        }))
    }

    #[inline]
    fn exp_m1(self) -> Self {
        Self::from_array(self.to_array().map(f32::exp_m1))
    }

    #[inline]
    fn ln_1p(self) -> Self {
        Self::from_array(self.to_array().map(f32::ln_1p))
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        Self::from_array(self.to_array().map(|x| x.powi(n)))
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        Self::from_array(core::array::from_fn(|i| {
            self.as_array()[i].powf(n.as_array()[i])
        }))
    }

    #[inline]
    fn tan(self) -> Self { Self::from_array(self.to_array().map(f32::tan)) }
    #[inline]
    fn asin(self) -> Self { Self::from_array(self.to_array().map(f32::asin)) }
    #[inline]
    fn acos(self) -> Self { Self::from_array(self.to_array().map(f32::acos)) }
    #[inline]
    fn atan(self) -> Self { Self::from_array(self.to_array().map(f32::atan)) }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        Self::from_array(core::array::from_fn(|i| self.as_array()[i].atan2(other.as_array()[i])))
    }
    #[inline]
    fn sinh(self) -> Self { Self::from_array(self.to_array().map(f32::sinh)) }
    #[inline]
    fn cosh(self) -> Self { Self::from_array(self.to_array().map(f32::cosh)) }
    #[inline]
    fn tanh(self) -> Self { Self::from_array(self.to_array().map(f32::tanh)) }
    #[inline]
    fn asinh(self) -> Self { Self::from_array(self.to_array().map(f32::asinh)) }
    #[inline]
    fn acosh(self) -> Self { Self::from_array(self.to_array().map(f32::acosh)) }
    #[inline]
    fn atanh(self) -> Self { Self::from_array(self.to_array().map(f32::atanh)) }

    #[inline]
    fn fract(self) -> Self {
        self - self.trunc()
    }
}

impl<const N: usize> StdFloat for Simd<f64, N> {
    #[inline]
    fn cbrt(self) -> Self {
        Self::from_array(self.to_array().map(f64::cbrt))
    }

    #[inline]
    fn hypot(self, other: Self) -> Self {
        Self::from_array(core::array::from_fn(|i| {
            self.as_array()[i].hypot(other.as_array()[i])
        }))
    }

    #[inline]
    fn exp_m1(self) -> Self {
        Self::from_array(self.to_array().map(f64::exp_m1))
    }

    #[inline]
    fn ln_1p(self) -> Self {
        Self::from_array(self.to_array().map(f64::ln_1p))
    }

    #[inline]
    fn powi(self, n: i32) -> Self {
        Self::from_array(self.to_array().map(|x| x.powi(n)))
    }

    #[inline]
    fn powf(self, n: Self) -> Self {
        Self::from_array(core::array::from_fn(|i| {
            self.as_array()[i].powf(n.as_array()[i])
        }))
    }

    #[inline]
    fn tan(self) -> Self { Self::from_array(self.to_array().map(f64::tan)) }
    #[inline]
    fn asin(self) -> Self { Self::from_array(self.to_array().map(f64::asin)) }
    #[inline]
    fn acos(self) -> Self { Self::from_array(self.to_array().map(f64::acos)) }
    #[inline]
    fn atan(self) -> Self { Self::from_array(self.to_array().map(f64::atan)) }
    #[inline]
    fn atan2(self, other: Self) -> Self {
        Self::from_array(core::array::from_fn(|i| self.as_array()[i].atan2(other.as_array()[i])))
    }
    #[inline]
    fn sinh(self) -> Self { Self::from_array(self.to_array().map(f64::sinh)) }
    #[inline]
    fn cosh(self) -> Self { Self::from_array(self.to_array().map(f64::cosh)) }
    #[inline]
    fn tanh(self) -> Self { Self::from_array(self.to_array().map(f64::tanh)) }
    #[inline]
    fn asinh(self) -> Self { Self::from_array(self.to_array().map(f64::asinh)) }
    #[inline]
    fn acosh(self) -> Self { Self::from_array(self.to_array().map(f64::acosh)) }
    #[inline]
    fn atanh(self) -> Self { Self::from_array(self.to_array().map(f64::atanh)) }

    #[inline]
    fn fract(self) -> Self {
        self - self.trunc()
    }
}
