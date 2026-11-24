use crate::simd::{
    Select, Simd, SimdCast, SimdElement,
    cmp::{SimdOrd, SimdPartialEq},
};

/// Operations on SIMD vectors of unsigned integers.
pub impl(self) trait SimdUint: Copy {
    /// Scalar type contained by this SIMD vector type.
    type Scalar;

    /// Converts each lane to `true` when it is not equal to zero.
    #[must_use]
    fn to_mask(self) -> <Self as SimdPartialEq>::Mask
    where
        Self: SimdPartialEq;

    /// A SIMD vector with a different element type.
    type Cast<T: SimdElement>;

    /// Performs elementwise conversion of this vector's elements to another SIMD-valid type.
    ///
    /// This follows Rust's `as` conversion semantics: integer-to-integer casts truncate to the
    /// destination width, while integer-to-float casts round to the nearest representable value.
    #[must_use]
    fn cast<T: SimdCast>(self) -> Self::Cast<T>;

    /// Computes the two's-complement negation of each lane, wrapping modulo the lane's
    /// integer range. Zero remains zero.
    fn wrapping_neg(self) -> Self;

    /// Adds corresponding lanes without checking for overflow.
    ///
    /// # Safety
    /// Every lane addition must be representable by the scalar element type.
    unsafe fn unchecked_add(self, second: Self) -> Self;

    /// Subtracts corresponding lanes without checking for underflow.
    ///
    /// # Safety
    /// Every lane subtraction must not underflow.
    unsafe fn unchecked_sub(self, second: Self) -> Self;

    /// Multiplies corresponding lanes without checking for overflow.
    ///
    /// # Safety
    /// Every lane product must be representable by the scalar element type.
    unsafe fn unchecked_mul(self, second: Self) -> Self;

    /// Divides corresponding lanes without checking for division by zero.
    ///
    /// # Safety
    /// Every lane in `second` must be nonzero.
    unsafe fn unchecked_div(self, second: Self) -> Self;

    /// Computes the remainder of corresponding lanes without checking for division by zero.
    ///
    /// # Safety
    /// Every lane in `second` must be nonzero.
    unsafe fn unchecked_rem(self, second: Self) -> Self;

    /// Shifts each lane left without masking the shift amount.
    ///
    /// Each lane's shift amount is converted to `u32`, matching the scalar
    /// integer shift operation, before it is checked against the lane width.
    ///
    /// # Safety
    /// After conversion to `u32`, every lane's shift amount must be less than
    /// the lane bit width.
    unsafe fn unchecked_shl(self, rhs: Self) -> Self;

    /// Shifts each lane right without masking the shift amount.
    ///
    /// Each lane's shift amount is converted to `u32`, matching the scalar
    /// integer shift operation, before it is checked against the lane width.
    ///
    /// # Safety
    /// After conversion to `u32`, every lane's shift amount must be less than
    /// the lane bit width.
    unsafe fn unchecked_shr(self, rhs: Self) -> Self;

    /// Rotates every lane's bits to the left by the same number of bits.
    ///
    /// The rotation amount is reduced modulo the lane bit width, matching the
    /// corresponding scalar integer method.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn rotate_left(self, rhs: u32) -> Self;

    /// Rotates every lane's bits to the right by the same number of bits.
    ///
    /// The rotation amount is reduced modulo the lane bit width, matching the
    /// corresponding scalar integer method.
    #[must_use = "method returns a new vector and does not mutate the original value"]
    fn rotate_right(self, rhs: u32) -> Self;

    /// Returns the high half of each unsigned lane's full-width product.
    ///
    /// The result is the full product shifted right by the lane bit width.
    #[must_use]
    fn mul_high(self, second: Self) -> Self;

    /// Lanewise saturating add.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// let x = Simd::from_array([2, 1, 0, u32::MAX]);
    /// let max = Simd::splat(u32::MAX);
    /// let unsat = x + max;
    /// let sat = x.saturating_add(max);
    /// assert_eq!(unsat, Simd::from_array([1, 0, u32::MAX, u32::MAX - 1]));
    /// assert_eq!(sat, max);
    /// ```
    fn saturating_add(self, second: Self) -> Self;

    /// Lanewise saturating multiplication.
    ///
    /// Each lane is multiplied independently. If the mathematical product exceeds
    /// the element type's maximum value, the result is clamped to that maximum.
    fn saturating_mul(self, second: Self) -> Self;

    /// Raises every lane to the same exponent, wrapping on overflow.
    ///
    /// An exponent of zero returns one for every lane, including zero lanes.
    fn wrapping_pow(self, exp: u32) -> Self;

    /// Raises every lane to the same exponent, saturating at the unsigned maximum.
    ///
    /// An exponent of zero returns one for every lane, including zero lanes.
    fn saturating_pow(self, exp: u32) -> Self;

    /// Returns the wrapped sum and a mask indicating lanes where addition overflowed.
    fn overflowing_add(self, second: Self) -> (Self, <Self as SimdPartialEq>::Mask)
    where
        Self: SimdPartialEq;

    /// Returns the wrapped difference and a mask indicating lanes where subtraction overflowed.
    fn overflowing_sub(self, second: Self) -> (Self, <Self as SimdPartialEq>::Mask)
    where
        Self: SimdPartialEq;

    /// Returns the wrapped product and a mask indicating lanes where multiplication overflowed.
    fn overflowing_mul(self, second: Self) -> (Self, <Self as SimdPartialEq>::Mask)
    where
        Self: SimdPartialEq;

    /// Raises every lane to the same exponent and returns a mask for lanes that overflowed.
    ///
    /// The returned values wrap on overflow, matching the scalar integer operation.
    fn overflowing_pow(self, exp: u32) -> (Self, <Self as SimdPartialEq>::Mask)
    where
        Self: SimdPartialEq;

    /// Returns the quotient and a mask indicating lanes where division overflowed.
    ///
    /// Division by zero panics, matching the scalar integer operation.
    fn overflowing_div(self, second: Self) -> (Self, <Self as SimdPartialEq>::Mask)
    where
        Self: SimdPartialEq;

    /// Returns the Euclidean quotient and a mask indicating lanes where division overflowed.
    ///
    /// Division by zero panics, matching the scalar integer operation.
    fn overflowing_div_euclid(self, rhs: Self) -> (Self, <Self as SimdPartialEq>::Mask)
    where
        Self: SimdPartialEq;

    /// Returns the Euclidean remainder and a mask indicating lanes where remainder overflowed.
    ///
    /// Division by zero panics, matching the scalar integer operation.
    fn overflowing_rem_euclid(self, rhs: Self) -> (Self, <Self as SimdPartialEq>::Mask)
    where
        Self: SimdPartialEq;

    /// Returns the quotient of each lane using Euclidean division.
    ///
    /// Division by zero panics, matching the scalar integer operation.
    fn div_euclid(self, rhs: Self) -> Self;

    /// Returns the remainder of each lane using Euclidean division.
    ///
    /// Division by zero panics, matching the scalar integer operation.
    fn rem_euclid(self, rhs: Self) -> Self;

    /// Returns the remainder and a mask indicating lanes where remainder overflowed.
    ///
    /// Division by zero panics, matching the scalar integer operation.
    fn overflowing_rem(self, second: Self) -> (Self, <Self as SimdPartialEq>::Mask)
    where
        Self: SimdPartialEq;

    /// Shifts each lane left and reports shift amounts greater than or equal to the lane width.
    ///
    /// Each lane's shift amount is first converted to `u32`, matching Rust's
    /// scalar integer shift operation. The overflow mask therefore reports
    /// whether that converted amount is at least the lane bit width.
    fn overflowing_shl(self, rhs: Self) -> (Self, <Self as SimdPartialEq>::Mask)
    where
        Self: SimdPartialEq;

    /// Shifts each lane right and reports shift amounts greater than or equal to the lane width.
    ///
    /// Each lane's shift amount is first converted to `u32`, matching Rust's
    /// scalar integer shift operation. The overflow mask therefore reports
    /// whether that converted amount is at least the lane bit width.
    fn overflowing_shr(self, rhs: Self) -> (Self, <Self as SimdPartialEq>::Mask)
    where
        Self: SimdPartialEq;

    /// Lanewise saturating subtract.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// let x = Simd::from_array([2, 1, 0, u32::MAX]);
    /// let max = Simd::splat(u32::MAX);
    /// let unsat = x - max;
    /// let sat = x.saturating_sub(max);
    /// assert_eq!(unsat, Simd::from_array([3, 2, 1, 0]));
    /// assert_eq!(sat, Simd::splat(0));
    /// ```
    fn saturating_sub(self, second: Self) -> Self;

    /// Lanewise absolute difference.
    /// Every element becomes the absolute difference of `self` and `second`.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// let a = Simd::from_array([0, u32::MAX, 100, 20]);
    /// let b = Simd::from_array([u32::MAX, 0, 80, 200]);
    /// assert_eq!(a.abs_diff(b), Simd::from_array([u32::MAX, u32::MAX, 20, 180]));
    /// ```
    fn abs_diff(self, second: Self) -> Self;

    /// Returns the sum of the elements of the vector, with wrapping addition.
    fn reduce_sum(self) -> Self::Scalar;

    /// Returns the product of the elements of the vector, with wrapping multiplication.
    fn reduce_product(self) -> Self::Scalar;

    /// Returns the maximum element in the vector.
    fn reduce_max(self) -> Self::Scalar;

    /// Returns the minimum element in the vector.
    fn reduce_min(self) -> Self::Scalar;

    /// Returns the index of the first lane containing the minimum value.
    ///
    /// If multiple lanes contain the minimum, the lowest lane index is returned.
    #[must_use]
    fn reduce_min_index(self) -> usize;

    /// Returns the index of the first lane containing the maximum value.
    ///
    /// If multiple lanes contain the maximum, the lowest lane index is returned.
    #[must_use]
    fn reduce_max_index(self) -> usize;

    /// Returns the cumulative bitwise "and" across the elements of the vector.
    fn reduce_and(self) -> Self::Scalar;

    /// Returns the cumulative bitwise "or" across the elements of the vector.
    fn reduce_or(self) -> Self::Scalar;

    /// Returns the cumulative bitwise "xor" across the elements of the vector.
    fn reduce_xor(self) -> Self::Scalar;

    /// Interprets each lane as a big-endian integer and converts it to native endianness.
    #[allow(clippy::wrong_self_convention)]
    fn from_be(self) -> Self;

    /// Interprets each lane as a little-endian integer and converts it to native endianness.
    #[allow(clippy::wrong_self_convention)]
    fn from_le(self) -> Self;

    /// Converts each lane from native endianness to big-endian representation.
    fn to_be(self) -> Self;

    /// Converts each lane from native endianness to little-endian representation.
    fn to_le(self) -> Self;

    /// Reverses the byte order of each element.
    fn swap_bytes(self) -> Self;

    /// Reverses the order of bits in each element.
    /// The least significant bit becomes the most significant bit, second least-significant bit becomes second most-significant bit, etc.
    fn reverse_bits(self) -> Self;

    /// Returns the number of ones in the binary representation of each element.
    fn count_ones(self) -> Self;

    /// Returns the number of zeros in the binary representation of each element.
    fn count_zeros(self) -> Self;

    /// Returns the number of leading zeros in the binary representation of each element.
    fn leading_zeros(self) -> Self;

    /// Returns the number of trailing zeros in the binary representation of each element.
    fn trailing_zeros(self) -> Self;

    /// Returns the number of leading ones in the binary representation of each element.
    fn leading_ones(self) -> Self;

    /// Returns the number of trailing ones in the binary representation of each element.
    fn trailing_ones(self) -> Self;
}

macro_rules! overflowing_mul_impl {
    ($ty:ty, $wide:ty, $lanes:ident, $lhs:ident, $rhs:ident, vectorized) => {{
        let lhs: Simd<$wide, $lanes> = $lhs.cast();
        let rhs: Simd<$wide, $lanes> = $rhs.cast();
        let product = lhs * rhs;
        let overflow_mask = product.simd_gt(Simd::splat(<$ty>::MAX as $wide));
        let overflow = overflow_mask
            .select(Simd::<$wide, $lanes>::splat(1), Simd::splat(0))
            .cast::<$ty>()
            .to_mask();
        (product.cast(), overflow)
    }};
    ($ty:ty, $wide:ty, $lanes:ident, $lhs:ident, $rhs:ident, scalar) => {{
        let mut overflow = [false; $lanes];
        let values = Simd::<$ty, $lanes>::from_array(core::array::from_fn(|lane| {
            let (value, did_overflow) =
                $lhs.as_array()[lane].overflowing_mul($rhs.as_array()[lane]);
            overflow[lane] = did_overflow;
            value
        }));
        let overflow = Simd::<$ty, $lanes>::from_array(
            overflow.map(|did_overflow| did_overflow as $ty),
        ).to_mask();
        (values, overflow)
    }};
}

macro_rules! saturating_mul_impl {
    ($ty:ty, $wide:ty, $lanes:ident, $lhs:ident, $rhs:ident, vectorized) => {{
        let lhs: Simd<$wide, $lanes> = $lhs.cast();
        let rhs: Simd<$wide, $lanes> = $rhs.cast();
        (lhs * rhs)
            .simd_min(Simd::splat(<$ty>::MAX as $wide))
            .cast()
    }};
    ($ty:ty, $wide:ty, $lanes:ident, $lhs:ident, $rhs:ident, scalar) => {{
        Simd::<$ty, $lanes>::from_array(core::array::from_fn(|lane| {
            $lhs.as_array()[lane].saturating_mul($rhs.as_array()[lane])
        }))
    }};
}

macro_rules! impl_trait {
    { $($ty:ident ($signed:ident, $wide:ident, $strategy:ident)),* } => {
        $(
        impl<const N: usize> SimdUint for Simd<$ty, N>
        {
            type Scalar = $ty;
            type Cast<T: SimdElement> = Simd<T, N>;

            #[inline]
            fn to_mask(self) -> <Self as SimdPartialEq>::Mask {
                self.simd_ne(Self::splat(0))
            }

            #[inline]
            fn cast<T: SimdCast>(self) -> Self::Cast<T> {
                // Safety: supported types are guaranteed by SimdCast
                unsafe { core::intrinsics::simd::simd_as(self) }
            }

            #[inline]
            fn wrapping_neg(self) -> Self {
                use crate::simd::num::SimdInt;
                (-self.cast::<$signed>()).cast()
            }

            #[inline]
            unsafe fn unchecked_add(self, second: Self) -> Self {
                Simd::<$ty, N>::from_array(core::array::from_fn(|lane| {
                    // SAFETY: The caller guarantees that every lane addition is representable.
                    unsafe { self.as_array()[lane].unchecked_add(second.as_array()[lane]) }
                }))
            }

            #[inline]
            unsafe fn unchecked_sub(self, second: Self) -> Self {
                Simd::<$ty, N>::from_array(core::array::from_fn(|lane| {
                    // SAFETY: The caller guarantees that every lane subtraction cannot underflow.
                    unsafe { self.as_array()[lane].unchecked_sub(second.as_array()[lane]) }
                }))
            }

            #[inline]
            unsafe fn unchecked_mul(self, second: Self) -> Self {
                Simd::<$ty, N>::from_array(core::array::from_fn(|lane| {
                    // SAFETY: The caller guarantees that every lane product is representable.
                    unsafe { self.as_array()[lane].unchecked_mul(second.as_array()[lane]) }
                }))
            }

            #[inline]
            unsafe fn unchecked_div(self, second: Self) -> Self {
                Simd::<$ty, N>::from_array(core::array::from_fn(|lane| {
                    // SAFETY: The caller guarantees every divisor is nonzero.
                    unsafe { core::intrinsics::unchecked_div(self.as_array()[lane], second.as_array()[lane]) }
                }))
            }

            #[inline]
            unsafe fn unchecked_rem(self, second: Self) -> Self {
                Simd::<$ty, N>::from_array(core::array::from_fn(|lane| {
                    // SAFETY: The caller guarantees every divisor is nonzero.
                    unsafe { core::intrinsics::unchecked_rem(self.as_array()[lane], second.as_array()[lane]) }
                }))
            }

            #[inline]
            unsafe fn unchecked_shl(self, rhs: Self) -> Self {
                Simd::<$ty, N>::from_array(core::array::from_fn(|lane| {
                    // SAFETY: The caller guarantees every shift amount is less than the lane width.
                    unsafe { self.as_array()[lane].unchecked_shl(rhs.as_array()[lane] as u32) }
                }))
            }

            #[inline]
            unsafe fn unchecked_shr(self, rhs: Self) -> Self {
                Simd::<$ty, N>::from_array(core::array::from_fn(|lane| {
                    // SAFETY: The caller guarantees every shift amount is less than the lane width.
                    unsafe { self.as_array()[lane].unchecked_shr(rhs.as_array()[lane] as u32) }
                }))
            }

            #[inline]
            fn rotate_left(self, rhs: u32) -> Self {
                Simd::<$ty, N>::from_array(core::array::from_fn(|lane| {
                    self.as_array()[lane].rotate_left(rhs)
                }))
            }

            #[inline]
            fn rotate_right(self, rhs: u32) -> Self {
                Simd::<$ty, N>::from_array(core::array::from_fn(|lane| {
                    self.as_array()[lane].rotate_right(rhs)
                }))
            }

            #[inline]
            fn mul_high(self, second: Self) -> Self {
                Simd::<$ty, N>::from_array(core::array::from_fn(|lane| {
                    let product = (self.as_array()[lane] as $wide)
                        .wrapping_mul(second.as_array()[lane] as $wide);
                    (product >> <$ty>::BITS) as $ty
                }))
            }

            #[inline]
            fn saturating_add(self, second: Self) -> Self {
                // Safety: `self` is a vector
                unsafe { core::intrinsics::simd::simd_saturating_add(self, second) }
            }

            #[inline]
            fn saturating_mul(self, second: Self) -> Self {
                saturating_mul_impl!($ty, $wide, N, self, second, $strategy)
            }

            #[inline]
            fn wrapping_pow(mut self, mut exp: u32) -> Self {
                let mut result = Self::splat(1);
                while exp != 0 {
                    if exp & 1 != 0 {
                        result = result * self;
                    }
                    exp >>= 1;
                    if exp != 0 {
                        self = self * self;
                    }
                }
                result
            }

            #[inline]
            fn saturating_pow(mut self, mut exp: u32) -> Self {
                let mut result = Self::splat(1);
                while exp != 0 {
                    if exp & 1 != 0 {
                        result = result.saturating_mul(self);
                    }
                    exp >>= 1;
                    if exp != 0 {
                        self = self.saturating_mul(self);
                    }
                }
                result
            }

            #[inline]
            fn overflowing_add(self, second: Self) -> (Self, <Self as SimdPartialEq>::Mask)
            where
                Self: SimdPartialEq
            {
                let mut overflow = [false; N];
                let values = Simd::<$ty, N>::from_array(core::array::from_fn(|lane| {
                    let (value, did_overflow) =
                        self.as_array()[lane].overflowing_add(second.as_array()[lane]);
                    overflow[lane] = did_overflow;
                    value
                }));
                let overflow = Simd::<$ty, N>::from_array(
                    overflow.map(|did_overflow| did_overflow as $ty),
                )
                .to_mask();
                (values, overflow)
            }

            #[inline]
            fn overflowing_sub(self, second: Self) -> (Self, <Self as SimdPartialEq>::Mask)
            where
                Self: SimdPartialEq
            {
                let mut overflow = [false; N];
                let values = Simd::<$ty, N>::from_array(core::array::from_fn(|lane| {
                    let (value, did_overflow) =
                        self.as_array()[lane].overflowing_sub(second.as_array()[lane]);
                    overflow[lane] = did_overflow;
                    value
                }));
                let overflow = Simd::<$ty, N>::from_array(
                    overflow.map(|did_overflow| did_overflow as $ty),
                )
                .to_mask();
                (values, overflow)
            }

            #[inline]
            fn overflowing_mul(self, second: Self) -> (Self, <Self as SimdPartialEq>::Mask)
            where
                Self: SimdPartialEq
            {
                overflowing_mul_impl!($ty, $wide, N, self, second, $strategy)
            }

            #[inline]
            fn overflowing_pow(self, exp: u32) -> (Self, <Self as SimdPartialEq>::Mask)
            where
                Self: SimdPartialEq
            {
                let mut overflow = [false; N];
                let values = Simd::<$ty, N>::from_array(core::array::from_fn(|lane| {
                    let (value, did_overflow) = self.as_array()[lane].overflowing_pow(exp);
                    overflow[lane] = did_overflow;
                    value
                }));
                let overflow = Simd::<$ty, N>::from_array(
                    overflow.map(|did_overflow| did_overflow as $ty),
                )
                .to_mask();
                (values, overflow)
            }

            #[inline]
            fn overflowing_div(self, second: Self) -> (Self, <Self as SimdPartialEq>::Mask)
            where
                Self: SimdPartialEq
            {
                let mut overflow = [false; N];
                let values = Simd::<$ty, N>::from_array(core::array::from_fn(|lane| {
                    let (value, did_overflow) =
                        self.as_array()[lane].overflowing_div(second.as_array()[lane]);
                    overflow[lane] = did_overflow;
                    value
                }));
                let overflow = Simd::<$ty, N>::from_array(
                    overflow.map(|did_overflow| did_overflow as $ty),
                )
                .to_mask();
                (values, overflow)
            }

            #[inline]
            fn overflowing_div_euclid(self, rhs: Self) -> (Self, <Self as SimdPartialEq>::Mask)
            where
                Self: SimdPartialEq
            {
                let mut overflow = [false; N];
                let values = Simd::<$ty, N>::from_array(core::array::from_fn(|lane| {
                    let (value, did_overflow) =
                        self.as_array()[lane].overflowing_div_euclid(rhs.as_array()[lane]);
                    overflow[lane] = did_overflow;
                    value
                }));
                let overflow = Simd::<$ty, N>::from_array(
                    overflow.map(|did_overflow| did_overflow as $ty),
                )
                .to_mask();
                (values, overflow)
            }

            #[inline]
            fn overflowing_rem_euclid(self, rhs: Self) -> (Self, <Self as SimdPartialEq>::Mask)
            where
                Self: SimdPartialEq
            {
                let mut overflow = [false; N];
                let values = Simd::<$ty, N>::from_array(core::array::from_fn(|lane| {
                    let (value, did_overflow) =
                        self.as_array()[lane].overflowing_rem_euclid(rhs.as_array()[lane]);
                    overflow[lane] = did_overflow;
                    value
                }));
                let overflow = Simd::<$ty, N>::from_array(
                    overflow.map(|did_overflow| did_overflow as $ty),
                )
                .to_mask();
                (values, overflow)
            }

            #[inline]
            fn div_euclid(self, rhs: Self) -> Self {
                Simd::<$ty, N>::from_array(core::array::from_fn(|lane| {
                    self.as_array()[lane].div_euclid(rhs.as_array()[lane])
                }))
            }

            #[inline]
            fn rem_euclid(self, rhs: Self) -> Self {
                Simd::<$ty, N>::from_array(core::array::from_fn(|lane| {
                    self.as_array()[lane].rem_euclid(rhs.as_array()[lane])
                }))
            }

            #[inline]
            fn overflowing_rem(self, second: Self) -> (Self, <Self as SimdPartialEq>::Mask)
            where
                Self: SimdPartialEq
            {
                let mut overflow = [false; N];
                let values = Simd::<$ty, N>::from_array(core::array::from_fn(|lane| {
                    let (value, did_overflow) =
                        self.as_array()[lane].overflowing_rem(second.as_array()[lane]);
                    overflow[lane] = did_overflow;
                    value
                }));
                let overflow = Simd::<$ty, N>::from_array(
                    overflow.map(|did_overflow| did_overflow as $ty),
                )
                .to_mask();
                (values, overflow)
            }

            #[inline]
            fn overflowing_shl(self, rhs: Self) -> (Self, <Self as SimdPartialEq>::Mask)
            where
                Self: SimdPartialEq
            {
                let mut overflow = [false; N];
                let values = Simd::<$ty, N>::from_array(core::array::from_fn(|lane| {
                    let shift = rhs.as_array()[lane];
                    let (value, did_overflow) = self.as_array()[lane].overflowing_shl(shift as u32);
                    // Match the scalar operation: shift counts are converted to u32
                    // before the lane-width check, including for 64-bit lane types.
                    overflow[lane] = did_overflow;
                    value
                }));
                let overflow = Simd::<$ty, N>::from_array(
                    overflow.map(|did_overflow| did_overflow as $ty),
                )
                .to_mask();
                (values, overflow)
            }

            #[inline]
            fn overflowing_shr(self, rhs: Self) -> (Self, <Self as SimdPartialEq>::Mask)
            where
                Self: SimdPartialEq
            {
                let mut overflow = [false; N];
                let values = Simd::<$ty, N>::from_array(core::array::from_fn(|lane| {
                    let shift = rhs.as_array()[lane];
                    let (value, did_overflow) = self.as_array()[lane].overflowing_shr(shift as u32);
                    // Match the scalar operation: shift counts are converted to u32
                    // before the lane-width check, including for 64-bit lane types.
                    overflow[lane] = did_overflow;
                    value
                }));
                let overflow = Simd::<$ty, N>::from_array(
                    overflow.map(|did_overflow| did_overflow as $ty),
                )
                .to_mask();
                (values, overflow)
            }

            #[inline]
            fn saturating_sub(self, second: Self) -> Self {
                // Safety: `self` is a vector
                unsafe { core::intrinsics::simd::simd_saturating_sub(self, second) }
            }

            #[inline]
            fn abs_diff(self, second: Self) -> Self {
                let max = self.simd_max(second);
                let min = self.simd_min(second);
                max - min
            }

            #[inline]
            fn reduce_sum(self) -> Self::Scalar {
                // Safety: `self` is an integer vector
                unsafe { core::intrinsics::simd::simd_reduce_add_ordered(self, 0) }
            }

            #[inline]
            fn reduce_product(self) -> Self::Scalar {
                // Safety: `self` is an integer vector
                unsafe { core::intrinsics::simd::simd_reduce_mul_ordered(self, 1) }
            }

            #[inline]
            fn reduce_max(self) -> Self::Scalar {
                // Safety: `self` is an integer vector
                unsafe { core::intrinsics::simd::simd_reduce_max(self) }
            }

            #[inline]
            fn reduce_min(self) -> Self::Scalar {
                // Safety: `self` is an integer vector
                unsafe { core::intrinsics::simd::simd_reduce_min(self) }
            }

            #[inline]
            fn reduce_min_index(self) -> usize {
                self.simd_eq(Self::splat(self.reduce_min()))
                    .first_set()
                    .expect("SIMD vectors always have at least one lane")
            }

            #[inline]
            fn reduce_max_index(self) -> usize {
                self.simd_eq(Self::splat(self.reduce_max()))
                    .first_set()
                    .expect("SIMD vectors always have at least one lane")
            }

            #[inline]
            fn reduce_and(self) -> Self::Scalar {
                // Safety: `self` is an integer vector
                unsafe { core::intrinsics::simd::simd_reduce_and(self) }
            }

            #[inline]
            fn reduce_or(self) -> Self::Scalar {
                // Safety: `self` is an integer vector
                unsafe { core::intrinsics::simd::simd_reduce_or(self) }
            }

            #[inline]
            fn reduce_xor(self) -> Self::Scalar {
                // Safety: `self` is an integer vector
                unsafe { core::intrinsics::simd::simd_reduce_xor(self) }
            }

            #[inline]
            fn from_be(self) -> Self {
                Simd::<$ty, N>::from_array(self.to_array().map($ty::from_be))
            }

            #[inline]
            fn from_le(self) -> Self {
                Simd::<$ty, N>::from_array(self.to_array().map($ty::from_le))
            }

            #[inline]
            fn to_be(self) -> Self {
                Simd::<$ty, N>::from_array(self.to_array().map($ty::to_be))
            }

            #[inline]
            fn to_le(self) -> Self {
                Simd::<$ty, N>::from_array(self.to_array().map($ty::to_le))
            }

            #[inline]
            fn swap_bytes(self) -> Self {
                // Safety: `self` is an integer vector
                unsafe { core::intrinsics::simd::simd_bswap(self) }
            }

            #[inline]
            fn reverse_bits(self) -> Self {
                // Safety: `self` is an integer vector
                unsafe { core::intrinsics::simd::simd_bitreverse(self) }
            }

            #[inline]
            fn count_ones(self) -> Self {
                // Safety: `self` is an integer vector
                unsafe { core::intrinsics::simd::simd_ctpop(self) }
            }

            #[inline]
            fn count_zeros(self) -> Self {
                (!self).count_ones()
            }

            #[inline]
            fn leading_zeros(self) -> Self {
                // Safety: `self` is an integer vector
                unsafe { core::intrinsics::simd::simd_ctlz(self) }
            }

            #[inline]
            fn trailing_zeros(self) -> Self {
                // Safety: `self` is an integer vector
                unsafe { core::intrinsics::simd::simd_cttz(self) }
            }

            #[inline]
            fn leading_ones(self) -> Self {
                (!self).leading_zeros()
            }

            #[inline]
            fn trailing_ones(self) -> Self {
                (!self).trailing_zeros()
            }
        }
        )*
    }
}

impl_trait! { u8 (i8, u16, vectorized), u16 (i16, u32, vectorized), u32 (i32, u64, vectorized), u64 (i64, u128, scalar), usize (isize, u128, scalar) }
