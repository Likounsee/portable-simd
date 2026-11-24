use crate::simd::{
    Mask, Select, Simd, SimdCast, SimdElement, cmp::SimdOrd, cmp::SimdPartialEq,
    cmp::SimdPartialOrd, num::SimdUint,
};

/// Operations on SIMD vectors of signed integers.
pub impl(self) trait SimdInt: Copy {
    /// Mask type used for manipulating this SIMD vector type.
    type Mask;

    /// Converts each lane to `true` when it is not equal to zero.
    #[must_use]
    fn to_mask(self) -> Self::Mask;

    /// Scalar type contained by this SIMD vector type.
    type Scalar;

    /// A SIMD vector of unsigned integers with the same element size.
    type Unsigned;

    /// A SIMD vector with a different element type.
    type Cast<T: SimdElement>;

    /// Performs elementwise conversion of this vector's elements to another SIMD-valid type.
    ///
    /// This follows Rust's `as` conversion semantics: integer-to-integer casts truncate to the
    /// destination width, while integer-to-float casts round to the nearest representable value.
    #[must_use]
    fn cast<T: SimdCast>(self) -> Self::Cast<T>;

    /// Adds corresponding lanes without checking for overflow.
    ///
    /// # Safety
    /// Every lane addition must be representable by the scalar element type.
    unsafe fn unchecked_add(self, second: Self) -> Self;

    /// Subtracts corresponding lanes without checking for overflow.
    ///
    /// # Safety
    /// Every lane subtraction must be representable by the scalar element type.
    unsafe fn unchecked_sub(self, second: Self) -> Self;

    /// Multiplies corresponding lanes without checking for overflow.
    ///
    /// # Safety
    /// Every lane product must be representable by the scalar element type.
    unsafe fn unchecked_mul(self, second: Self) -> Self;

    /// Divides corresponding lanes without checking for division by zero or signed overflow.
    ///
    /// # Safety
    /// Every divisor must be nonzero. For signed integers, no lane may divide the
    /// minimum value by -1.
    unsafe fn unchecked_div(self, second: Self) -> Self;

    /// Computes the remainder of corresponding lanes without checking for division by zero
    /// or signed overflow.
    ///
    /// # Safety
    /// Every divisor must be nonzero. For signed integers, no lane may compute the
    /// remainder of the minimum value by -1.
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

    /// Returns the high half of each signed lane's full-width product.
    ///
    /// The result is the product shifted right by the lane bit width, using
    /// arithmetic (sign-extending) shift semantics.
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
    /// let x = Simd::from_array([i32::MIN, 0, 1, i32::MAX]);
    /// let max = Simd::splat(i32::MAX);
    /// let unsat = x + max;
    /// let sat = x.saturating_add(max);
    /// assert_eq!(unsat, Simd::from_array([-1, i32::MAX, i32::MIN, -2]));
    /// assert_eq!(sat, Simd::from_array([-1, i32::MAX, i32::MAX, i32::MAX]));
    /// ```
    fn saturating_add(self, second: Self) -> Self;

    /// Lanewise saturating multiplication.
    ///
    /// Each lane is multiplied independently. If the mathematical product is outside
    /// the range of the element type, the result is clamped to the nearest bound.
    fn saturating_mul(self, second: Self) -> Self;

    /// Raises every lane to the same exponent, wrapping on overflow.
    ///
    /// An exponent of zero returns one for every lane, including zero lanes.
    fn wrapping_pow(self, exp: u32) -> Self;

    /// Raises every lane to the same exponent, saturating at the signed integer bounds.
    ///
    /// An exponent of zero returns one for every lane, including zero lanes.
    fn saturating_pow(self, exp: u32) -> Self;

    /// Returns the wrapped sum and a mask indicating lanes where addition overflowed.
    fn overflowing_add(self, second: Self) -> (Self, Self::Mask);

    /// Returns the wrapped difference and a mask indicating lanes where subtraction overflowed.
    fn overflowing_sub(self, second: Self) -> (Self, Self::Mask);

    /// Returns the wrapped product and a mask indicating lanes where multiplication overflowed.
    fn overflowing_mul(self, second: Self) -> (Self, Self::Mask);

    /// Raises every lane to the same exponent and returns a mask for lanes that overflowed.
    ///
    /// The returned values wrap on overflow, matching the scalar integer operation.
    fn overflowing_pow(self, exp: u32) -> (Self, Self::Mask);

    /// Returns the quotient and a mask indicating lanes where division overflowed.
    ///
    /// Division by zero panics, matching the scalar integer operation.
    fn overflowing_div(self, second: Self) -> (Self, Self::Mask);

    /// Returns the Euclidean quotient and a mask indicating lanes where division overflowed.
    ///
    /// Division by zero panics, matching the scalar integer operation.
    fn overflowing_div_euclid(self, rhs: Self) -> (Self, Self::Mask);

    /// Returns the Euclidean remainder and a mask indicating lanes where remainder overflowed.
    ///
    /// Division by zero panics, matching the scalar integer operation.
    fn overflowing_rem_euclid(self, rhs: Self) -> (Self, Self::Mask);

    /// Returns the Euclidean quotient of each lane.
    ///
    /// Division by zero panics, matching the scalar integer operation. The signed `MIN / -1`
    /// overflow case also panics, as it does for the scalar operation.
    fn div_euclid(self, rhs: Self) -> Self;

    /// Returns the Euclidean remainder of each lane.
    ///
    /// Division by zero panics, matching the scalar integer operation. The signed `MIN % -1`
    /// overflow case also panics, as it does for the scalar operation.
    fn rem_euclid(self, rhs: Self) -> Self;

    /// Returns the wrapped remainder and a mask indicating lanes where remainder overflowed.
    ///
    /// Division by zero panics, matching the scalar integer operation.
    fn overflowing_rem(self, second: Self) -> (Self, Self::Mask);

    /// Returns the wrapped negation and a mask indicating lanes where negation overflowed.
    ///
    /// The overflow mask is true for lanes containing the signed minimum value.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// let x = i32x4::from_array([i32::MIN, -2, 0, i32::MAX]);
    /// let (value, overflow) = x.overflowing_neg();
    /// assert_eq!(value, i32x4::from_array([i32::MIN, 2, 0, -i32::MAX]));
    /// assert_eq!(overflow.to_array(), [true, false, false, false]);
    /// ```
    fn overflowing_neg(self) -> (Self, Self::Mask);

    /// Returns the wrapped absolute value and a mask indicating lanes where absolute value overflowed.
    ///
    /// The overflow mask is true for lanes containing the signed minimum value.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// let x = i32x4::from_array([i32::MIN, -2, 0, i32::MAX]);
    /// let (value, overflow) = x.overflowing_abs();
    /// assert_eq!(value, i32x4::from_array([i32::MIN, 2, 0, i32::MAX]));
    /// assert_eq!(overflow.to_array(), [true, false, false, false]);
    /// ```
    fn overflowing_abs(self) -> (Self, Self::Mask);

    /// Shifts each lane left and returns a mask for shift amounts greater than or equal to the lane width.
    ///
    /// Each lane's shift count is converted to `u32`, matching the scalar
    /// `overflowing_shl` operation. Negative values therefore follow Rust's
    /// integer-cast semantics before the lane-width check.
    fn overflowing_shl(self, rhs: Self) -> (Self, Self::Mask);

    /// Shifts each lane right and returns a mask for shift amounts greater than or equal to the lane width.
    ///
    /// Each lane's shift count is converted to `u32`, matching the scalar
    /// `overflowing_shr` operation. Negative values therefore follow Rust's
    /// integer-cast semantics before the lane-width check.
    fn overflowing_shr(self, rhs: Self) -> (Self, Self::Mask);

    /// Lanewise saturating subtract.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// let x = Simd::from_array([i32::MIN, -2, -1, i32::MAX]);
    /// let max = Simd::splat(i32::MAX);
    /// let unsat = x - max;
    /// let sat = x.saturating_sub(max);
    /// assert_eq!(unsat, Simd::from_array([1, i32::MAX, i32::MIN, 0]));
    /// assert_eq!(sat, Simd::from_array([i32::MIN, i32::MIN, i32::MIN, 0]));
    /// ```
    fn saturating_sub(self, second: Self) -> Self;

    /// Lanewise absolute value, implemented in Rust.
    /// Every element becomes its absolute value.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// let xs = Simd::from_array([i32::MIN, i32::MIN + 1, -5, 0]);
    /// assert_eq!(xs.abs(), Simd::from_array([i32::MIN, i32::MAX, 5, 0]));
    /// ```
    fn abs(self) -> Self;

    /// Lanewise absolute difference.
    /// Every element becomes the absolute difference of `self` and `second`.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// let a = Simd::from_array([i32::MIN, i32::MAX, 100, -100]);
    /// let b = Simd::from_array([i32::MAX, i32::MIN, -80, -120]);
    /// assert_eq!(a.abs_diff(b), Simd::from_array([u32::MAX, u32::MAX, 180, 20]));
    /// ```
    fn abs_diff(self, second: Self) -> Self::Unsigned;

    /// Lanewise saturating absolute value, implemented in Rust.
    /// As abs(), except the MIN value becomes MAX instead of itself.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// let xs = Simd::from_array([i32::MIN, -2, 0, 3]);
    /// let unsat = xs.abs();
    /// let sat = xs.saturating_abs();
    /// assert_eq!(unsat, Simd::from_array([i32::MIN, 2, 0, 3]));
    /// assert_eq!(sat, Simd::from_array([i32::MAX, 2, 0, 3]));
    /// ```
    fn saturating_abs(self) -> Self;

    /// Lanewise saturating negation, implemented in Rust.
    /// As neg(), except the MIN value becomes MAX instead of itself.
    ///
    /// # Examples
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// let x = Simd::from_array([i32::MIN, -2, 3, i32::MAX]);
    /// let unsat = -x;
    /// let sat = x.saturating_neg();
    /// assert_eq!(unsat, Simd::from_array([i32::MIN, 2, -3, i32::MIN + 1]));
    /// assert_eq!(sat, Simd::from_array([i32::MAX, 2, -3, i32::MIN + 1]));
    /// ```
    fn saturating_neg(self) -> Self;

    /// Returns true for each positive element and false if it is zero or negative.
    fn is_positive(self) -> Self::Mask;

    /// Returns true for each negative element and false if it is zero or positive.
    fn is_negative(self) -> Self::Mask;

    /// Returns numbers representing the sign of each element.
    /// * `0` if the number is zero
    /// * `1` if the number is positive
    /// * `-1` if the number is negative
    fn signum(self) -> Self;

    /// Returns the sum of the elements of the vector, with wrapping addition.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// let v = i32x4::from_array([1, 2, 3, 4]);
    /// assert_eq!(v.reduce_sum(), 10);
    ///
    /// // SIMD integer addition is always wrapping
    /// let v = i32x4::from_array([i32::MAX, 1, 0, 0]);
    /// assert_eq!(v.reduce_sum(), i32::MIN);
    /// ```
    fn reduce_sum(self) -> Self::Scalar;

    /// Returns the product of the elements of the vector, with wrapping multiplication.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// let v = i32x4::from_array([1, 2, 3, 4]);
    /// assert_eq!(v.reduce_product(), 24);
    ///
    /// // SIMD integer multiplication is always wrapping
    /// let v = i32x4::from_array([i32::MAX, 2, 1, 1]);
    /// assert!(v.reduce_product() < i32::MAX);
    /// ```
    fn reduce_product(self) -> Self::Scalar;

    /// Returns the maximum element in the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// let v = i32x4::from_array([1, 2, 3, 4]);
    /// assert_eq!(v.reduce_max(), 4);
    /// ```
    fn reduce_max(self) -> Self::Scalar;

    /// Returns the minimum element in the vector.
    ///
    /// # Examples
    ///
    /// ```
    /// # #![feature(portable_simd)]
    /// # #[cfg(feature = "as_crate")] use core_simd::simd;
    /// # #[cfg(not(feature = "as_crate"))] use core::simd;
    /// # use simd::prelude::*;
    /// let v = i32x4::from_array([1, 2, 3, 4]);
    /// assert_eq!(v.reduce_min(), 1);
    /// ```
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
    fn count_ones(self) -> Self::Unsigned;

    /// Returns the number of zeros in the binary representation of each element.
    fn count_zeros(self) -> Self::Unsigned;

    /// Returns the number of leading zeros in the binary representation of each element.
    fn leading_zeros(self) -> Self::Unsigned;

    /// Returns the number of trailing zeros in the binary representation of each element.
    fn trailing_zeros(self) -> Self::Unsigned;

    /// Returns the number of leading ones in the binary representation of each element.
    fn leading_ones(self) -> Self::Unsigned;

    /// Returns the number of trailing ones in the binary representation of each element.
    fn trailing_ones(self) -> Self::Unsigned;
}

macro_rules! overflowing_add_sub_impl {
    ($ty:ty, $wide:ty, $lanes:ident, $lhs:ident, $rhs:ident, add, vectorized) => {{
        let lhs: Simd<$wide, $lanes> = $lhs.cast();
        let rhs: Simd<$wide, $lanes> = $rhs.cast();
        let result = lhs + rhs;
        let overflow_mask = result.simd_lt(Simd::splat(<$ty>::MIN as $wide))
            | result.simd_gt(Simd::splat(<$ty>::MAX as $wide));
        let overflow = overflow_mask
            .select(Simd::<$wide, $lanes>::splat(1), Simd::splat(0))
            .cast::<$ty>()
            .to_mask();
        (result.cast(), overflow)
    }};
    ($ty:ty, $wide:ty, $lanes:ident, $lhs:ident, $rhs:ident, sub, vectorized) => {{
        let lhs: Simd<$wide, $lanes> = $lhs.cast();
        let rhs: Simd<$wide, $lanes> = $rhs.cast();
        let result = lhs - rhs;
        let overflow_mask = result.simd_lt(Simd::splat(<$ty>::MIN as $wide))
            | result.simd_gt(Simd::splat(<$ty>::MAX as $wide));
        let overflow = overflow_mask
            .select(Simd::<$wide, $lanes>::splat(1), Simd::splat(0))
            .cast::<$ty>()
            .to_mask();
        (result.cast(), overflow)
    }};
    ($ty:ty, $wide:ty, $lanes:ident, $lhs:ident, $rhs:ident, add, scalar) => {{
        let mut overflow = [false; $lanes];
        let values = Simd::<$ty, $lanes>::from_array(core::array::from_fn(|lane| {
            let (value, did_overflow) =
                $lhs.as_array()[lane].overflowing_add($rhs.as_array()[lane]);
            overflow[lane] = did_overflow;
            value
        }));
        let overflow =
            Simd::<$ty, $lanes>::from_array(overflow.map(|did_overflow| did_overflow as $ty))
                .to_mask();
        (values, overflow)
    }};
    ($ty:ty, $wide:ty, $lanes:ident, $lhs:ident, $rhs:ident, sub, scalar) => {{
        let mut overflow = [false; $lanes];
        let values = Simd::<$ty, $lanes>::from_array(core::array::from_fn(|lane| {
            let (value, did_overflow) =
                $lhs.as_array()[lane].overflowing_sub($rhs.as_array()[lane]);
            overflow[lane] = did_overflow;
            value
        }));
        let overflow =
            Simd::<$ty, $lanes>::from_array(overflow.map(|did_overflow| did_overflow as $ty))
                .to_mask();
        (values, overflow)
    }};
}

macro_rules! overflowing_neg_abs_impl {
    ($ty:ty, $wide:ty, $lanes:ident, $value:ident, neg, vectorized) => {{
        let value: Simd<$wide, $lanes> = $value.cast();
        let overflow_mask = value.simd_eq(Simd::splat(<$ty>::MIN as $wide));
        let result = Simd::<$wide, $lanes>::splat(0) - value;
        let overflow = overflow_mask
            .select(Simd::<$wide, $lanes>::splat(1), Simd::splat(0))
            .cast::<$ty>()
            .to_mask();
        (result.cast(), overflow)
    }};
    ($ty:ty, $wide:ty, $lanes:ident, $value:ident, abs, vectorized) => {{
        let value: Simd<$wide, $lanes> = $value.cast();
        let overflow_mask = value.simd_eq(Simd::splat(<$ty>::MIN as $wide));
        let result = value
            .simd_lt(Simd::splat(0))
            .select(Simd::<$wide, $lanes>::splat(0) - value, value);
        let overflow = overflow_mask
            .select(Simd::<$wide, $lanes>::splat(1), Simd::splat(0))
            .cast::<$ty>()
            .to_mask();
        (result.cast(), overflow)
    }};
    ($ty:ty, $wide:ty, $lanes:ident, $value:ident, neg, scalar) => {{
        let mut overflow = [false; $lanes];
        let values = Simd::<$ty, $lanes>::from_array(core::array::from_fn(|lane| {
            let (value, did_overflow) = $value.as_array()[lane].overflowing_neg();
            overflow[lane] = did_overflow;
            value
        }));
        let overflow =
            Simd::<$ty, $lanes>::from_array(overflow.map(|did_overflow| did_overflow as $ty))
                .to_mask();
        (values, overflow)
    }};
    ($ty:ty, $wide:ty, $lanes:ident, $value:ident, abs, scalar) => {{
        let mut overflow = [false; $lanes];
        let values = Simd::<$ty, $lanes>::from_array(core::array::from_fn(|lane| {
            let (value, did_overflow) = $value.as_array()[lane].overflowing_abs();
            overflow[lane] = did_overflow;
            value
        }));
        let overflow =
            Simd::<$ty, $lanes>::from_array(overflow.map(|did_overflow| did_overflow as $ty))
                .to_mask();
        (values, overflow)
    }};
}

macro_rules! rotate_impl {
    ($ty:ty, $unsigned:ty, $lanes:ident, $value:ident, $rhs:ident, vectorized) => {{
        let shift = $rhs & (<$ty>::BITS - 1);
        let value: Simd<$unsigned, $lanes> = $value.cast();
        let left = value << Simd::<$unsigned, $lanes>::splat(shift as $unsigned);
        let right_shift = (<$ty>::BITS - shift) & (<$ty>::BITS - 1);
        let right = value >> Simd::<$unsigned, $lanes>::splat(right_shift as $unsigned);
        (left | right).cast()
    }};
    ($ty:ty, $unsigned:ty, $lanes:ident, $value:ident, $rhs:ident, scalar) => {{
        Simd::<$ty, $lanes>::from_array(core::array::from_fn(|lane| {
            $value.as_array()[lane].rotate_left($rhs)
        }))
    }};
}

macro_rules! rotate_right_impl {
    ($ty:ty, $unsigned:ty, $lanes:ident, $value:ident, $rhs:ident, vectorized) => {{
        let shift = $rhs & (<$ty>::BITS - 1);
        let value: Simd<$unsigned, $lanes> = $value.cast();
        let right = value >> Simd::<$unsigned, $lanes>::splat(shift as $unsigned);
        let left_shift = (<$ty>::BITS - shift) & (<$ty>::BITS - 1);
        let left = value << Simd::<$unsigned, $lanes>::splat(left_shift as $unsigned);
        (left | right).cast()
    }};
    ($ty:ty, $unsigned:ty, $lanes:ident, $value:ident, $rhs:ident, scalar) => {{
        Simd::<$ty, $lanes>::from_array(core::array::from_fn(|lane| {
            $value.as_array()[lane].rotate_right($rhs)
        }))
    }};
}

macro_rules! mul_high_impl {
    ($ty:ty, $wide:ty, $lanes:ident, $lhs:ident, $rhs:ident, vectorized) => {{
        let lhs: Simd<$wide, $lanes> = $lhs.cast();
        let rhs: Simd<$wide, $lanes> = $rhs.cast();
        ((lhs * rhs) >> Simd::<$wide, $lanes>::splat(<$ty>::BITS as $wide)).cast()
    }};
    ($ty:ty, $wide:ty, $lanes:ident, $lhs:ident, $rhs:ident, scalar) => {{
        Simd::<$ty, $lanes>::from_array(core::array::from_fn(|lane| {
            let product =
                ($lhs.as_array()[lane] as $wide).wrapping_mul($rhs.as_array()[lane] as $wide);
            (product >> <$ty>::BITS) as $ty
        }))
    }};
}

macro_rules! overflowing_mul_impl {
    ($ty:ty, $wide:ty, $lanes:ident, $lhs:ident, $rhs:ident, vectorized) => {{
        let lhs: Simd<$wide, $lanes> = $lhs.cast();
        let rhs: Simd<$wide, $lanes> = $rhs.cast();
        let product = lhs * rhs;
        let overflow_mask = product.simd_lt(Simd::splat(<$ty>::MIN as $wide))
            | product.simd_gt(Simd::splat(<$ty>::MAX as $wide));
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
        let overflow =
            Simd::<$ty, $lanes>::from_array(overflow.map(|did_overflow| did_overflow as $ty))
                .to_mask();
        (values, overflow)
    }};
}

macro_rules! saturating_mul_impl {
    ($ty:ty, $wide:ty, $lanes:ident, $lhs:ident, $rhs:ident, vectorized) => {{
        let lhs: Simd<$wide, $lanes> = $lhs.cast();
        let rhs: Simd<$wide, $lanes> = $rhs.cast();
        (lhs * rhs)
            .simd_max(Simd::splat(<$ty>::MIN as $wide))
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
    { $($ty:ident ($unsigned:ident, $wide:ident, $strategy:ident)),* } => {
        $(
        impl<const N: usize> SimdInt for Simd<$ty, N> {
            type Mask = Mask<<$ty as SimdElement>::Mask, N>;
            type Scalar = $ty;
            type Unsigned = Simd<$unsigned, N>;
            type Cast<T: SimdElement> = Simd<T, N>;

            #[inline]
            fn to_mask(self) -> Self::Mask {
                self.simd_ne(Self::splat(0))
            }

            #[inline]
            fn cast<T: SimdCast>(self) -> Self::Cast<T> {
                // Safety: supported types are guaranteed by SimdCast
                unsafe { core::intrinsics::simd::simd_as(self) }
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
                    // SAFETY: The caller guarantees that every lane subtraction is representable.
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
                    // SAFETY: The caller guarantees nonzero divisors and no MIN / -1 lanes.
                    unsafe { core::intrinsics::unchecked_div(self.as_array()[lane], second.as_array()[lane]) }
                }))
            }

            #[inline]
            unsafe fn unchecked_rem(self, second: Self) -> Self {
                Simd::<$ty, N>::from_array(core::array::from_fn(|lane| {
                    // SAFETY: The caller guarantees nonzero divisors and no MIN % -1 lanes.
                    unsafe { core::intrinsics::unchecked_rem(self.as_array()[lane], second.as_array()[lane]) }
                }))
            }

            #[inline]
            unsafe fn unchecked_shl(self, rhs: Self) -> Self {
                Simd::<$ty, N>::from_array(core::array::from_fn(|lane| {
                    // SAFETY: After conversion to u32, the caller guarantees the shift amount is less than the lane width.
                    unsafe { self.as_array()[lane].unchecked_shl(rhs.as_array()[lane] as u32) }
                }))
            }

            #[inline]
            unsafe fn unchecked_shr(self, rhs: Self) -> Self {
                Simd::<$ty, N>::from_array(core::array::from_fn(|lane| {
                    // SAFETY: After conversion to u32, the caller guarantees the shift amount is less than the lane width.
                    unsafe { self.as_array()[lane].unchecked_shr(rhs.as_array()[lane] as u32) }
                }))
            }

            #[inline]
            fn rotate_left(self, rhs: u32) -> Self {
                rotate_impl!($ty, $unsigned, N, self, rhs, $strategy)
            }

            #[inline]
            fn rotate_right(self, rhs: u32) -> Self {
                rotate_right_impl!($ty, $unsigned, N, self, rhs, $strategy)
            }

            #[inline]
            fn mul_high(self, second: Self) -> Self {
                mul_high_impl!($ty, $wide, N, self, second, $strategy)
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
            fn overflowing_add(self, second: Self) -> (Self, Self::Mask) {
                overflowing_add_sub_impl!($ty, $wide, N, self, second, add, $strategy)
            }

            #[inline]
            fn overflowing_sub(self, second: Self) -> (Self, Self::Mask) {
                overflowing_add_sub_impl!($ty, $wide, N, self, second, sub, $strategy)
            }

            #[inline]
            fn overflowing_mul(self, second: Self) -> (Self, Self::Mask) {
                overflowing_mul_impl!($ty, $wide, N, self, second, $strategy)
            }

            #[inline]
            fn overflowing_pow(self, exp: u32) -> (Self, Self::Mask) {
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
            fn overflowing_div(self, second: Self) -> (Self, Self::Mask) {
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
            fn overflowing_div_euclid(self, rhs: Self) -> (Self, Self::Mask) {
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
            fn overflowing_rem_euclid(self, rhs: Self) -> (Self, Self::Mask) {
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
            fn overflowing_rem(self, second: Self) -> (Self, Self::Mask) {
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
            fn overflowing_neg(self) -> (Self, Self::Mask) {
                overflowing_neg_abs_impl!($ty, $wide, N, self, neg, $strategy)
            }

            #[inline]
            fn overflowing_abs(self) -> (Self, Self::Mask) {
                overflowing_neg_abs_impl!($ty, $wide, N, self, abs, $strategy)
            }

            #[inline]
            fn overflowing_shl(self, rhs: Self) -> (Self, Self::Mask) {
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
            fn overflowing_shr(self, rhs: Self) -> (Self, Self::Mask) {
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
            fn abs(self) -> Self {
                const SHR: $ty = <$ty>::BITS as $ty - 1;
                let m = self >> Simd::splat(SHR);
                (self^m) - m
            }

            #[inline]
            fn abs_diff(self, second: Self) -> Self::Unsigned {
                let max = self.simd_max(second);
                let min = self.simd_min(second);
                (max - min).cast()
            }

            #[inline]
            fn saturating_abs(self) -> Self {
                // arith shift for -1 or 0 mask based on sign bit, giving 2s complement
                const SHR: $ty = <$ty>::BITS as $ty - 1;
                let m = self >> Simd::splat(SHR);
                (self^m).saturating_sub(m)
            }

            #[inline]
            fn saturating_neg(self) -> Self {
                Self::splat(0).saturating_sub(self)
            }

            #[inline]
            fn is_positive(self) -> Self::Mask {
                self.simd_gt(Self::splat(0))
            }

            #[inline]
            fn is_negative(self) -> Self::Mask {
                self.simd_lt(Self::splat(0))
            }

            #[inline]
            fn signum(self) -> Self {
                self.is_positive().select(
                    Self::splat(1),
                    self.is_negative().select(Self::splat(-1), Self::splat(0))
                )
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
            fn count_ones(self) -> Self::Unsigned {
                self.cast::<$unsigned>().count_ones()
            }

            #[inline]
            fn count_zeros(self) -> Self::Unsigned {
                self.cast::<$unsigned>().count_zeros()
            }

            #[inline]
            fn leading_zeros(self) -> Self::Unsigned {
                self.cast::<$unsigned>().leading_zeros()
            }

            #[inline]
            fn trailing_zeros(self) -> Self::Unsigned {
                self.cast::<$unsigned>().trailing_zeros()
            }

            #[inline]
            fn leading_ones(self) -> Self::Unsigned {
                self.cast::<$unsigned>().leading_ones()
            }

            #[inline]
            fn trailing_ones(self) -> Self::Unsigned {
                self.cast::<$unsigned>().trailing_ones()
            }
        }
        )*
    }
}

impl_trait! { i8 (u8, i16, vectorized), i16 (u16, i32, vectorized), i32 (u32, i64, vectorized), i64 (u64, i128, scalar), isize (usize, i128, scalar) }
