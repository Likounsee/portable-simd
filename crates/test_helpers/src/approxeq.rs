//! Compare numeric types approximately.

pub trait ApproxEq {
    fn approxeq(&self, other: &Self, _ulps: i64) -> bool;
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result;
}

impl ApproxEq for bool {
    fn approxeq(&self, other: &Self, _ulps: i64) -> bool {
        self == other
    }

    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        write!(f, "{:?}", self)
    }
}

macro_rules! impl_integer_approxeq {
    { $($type:ty),* } => {
        $(
        impl ApproxEq for $type {
            fn approxeq(&self, other: &Self, _ulps: i64) -> bool {
                self == other
            }

            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(f, "{:?} ({:x})", self, self)
            }
        }
        )*
    };
}

impl_integer_approxeq! { u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize }

macro_rules! impl_float_approxeq {
    { $($type:ty, $bits:ty, $sign_mask:expr);* $(;)? } => {
        $(
        impl ApproxEq for $type {
            fn approxeq(&self, other: &Self, ulps: i64) -> bool {
                // Equality handles equal finite values, both signed zeros, and
                // matching infinities without performing floating-point arithmetic.
                if self == other {
                    return true;
                }
                if self.is_nan() || other.is_nan() {
                    return self.is_nan() && other.is_nan();
                }
                // A finite value must never compare approximately equal to an
                // infinity, and a negative ULP allowance cannot accept anything.
                if !self.is_finite() || !other.is_finite() || ulps < 0 {
                    return false;
                }

                // Map IEEE-754 sign-magnitude encodings to monotonically ordered
                // integers. This gives an exact ULP distance without division by
                // a locally estimated ULP (which fails at zero and exponent edges).
                let self_bits = self.to_bits() as $bits;
                let other_bits = other.to_bits() as $bits;
                let ordered = |bits: $bits| {
                    let sign_mask = $sign_mask as $bits;
                    let magnitude = bits & !sign_mask;
                    if bits & sign_mask != 0 {
                        sign_mask - magnitude
                    } else {
                        sign_mask + magnitude
                    }
                };
                let left = ordered(self_bits) as u64;
                let right = ordered(other_bits) as u64;
                left.abs_diff(right) <= ulps as u64
            }

            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(f, "{:?} ({:x})", self, self.to_bits())
            }
        }
        )*
    };
}

impl_float_approxeq! {
    f16, u16, 0x8000;
    f32, u32, 0x8000_0000;
    f64, u64, 0x8000_0000_0000_0000;
}

impl<T: ApproxEq, const N: usize> ApproxEq for [T; N] {
    fn approxeq(&self, other: &Self, ulps: i64) -> bool {
        self.iter()
            .zip(other.iter())
            .fold(true, |value, (left, right)| {
                value && left.approxeq(right, ulps)
            })
    }

    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        #[repr(transparent)]
        struct Wrapper<'a, T: ApproxEq>(&'a T);

        impl<T: ApproxEq> core::fmt::Debug for Wrapper<'_, T> {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                self.0.fmt(f)
            }
        }

        f.debug_list()
            .entries(self.iter().map(|x| Wrapper(x)))
            .finish()
    }
}

#[doc(hidden)]
pub struct ApproxEqWrapper<'a, T>(pub &'a T, pub i64);

impl<T: ApproxEq> PartialEq<T> for ApproxEqWrapper<'_, T> {
    fn eq(&self, other: &T) -> bool {
        self.0.approxeq(other, self.1)
    }
}

impl<T: ApproxEq> core::fmt::Debug for ApproxEqWrapper<'_, T> {
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

#[macro_export]
macro_rules! prop_assert_approxeq {
    { $a:expr, $b:expr, $ulps:expr $(,)? } => {
        {
            use $crate::approxeq::ApproxEqWrapper;
            let a = $a;
            let b = $b;
            proptest::prop_assert_eq!(ApproxEqWrapper(&a, $ulps), b);
        }
    };
}

#[cfg(test)]
mod tests {
    use super::ApproxEq;

    #[test]
    fn finite_values_never_match_nan_or_infinity() {
        assert!(!1.0_f32.approxeq(&f32::NAN, i64::MAX));
        assert!(!f32::NAN.approxeq(&1.0_f32, i64::MAX));
        assert!(!1.0_f32.approxeq(&f32::INFINITY, i64::MAX));
        assert!(!f32::NEG_INFINITY.approxeq(&1.0_f32, i64::MAX));
        assert!(!f32::INFINITY.approxeq(&f32::NEG_INFINITY, i64::MAX));
    }

    #[test]
    fn matching_special_values_and_signed_zero_are_equal() {
        assert!(f32::NAN.approxeq(&f32::NAN, 0));
        assert!(f64::INFINITY.approxeq(&f64::INFINITY, 0));
        assert!(f64::NEG_INFINITY.approxeq(&f64::NEG_INFINITY, 0));
        assert!(0.0_f64.approxeq(&-0.0_f64, 0));
    }

    #[test]
    fn ulp_distance_is_correct_across_zero_and_exponent_boundaries() {
        assert!(0.0_f32.approxeq(&f32::from_bits(1), 1));
        assert!((-f32::from_bits(1)).approxeq(&0.0_f32, 1));
        assert!(f32::from_bits(0x007f_ffff).approxeq(&f32::MIN_POSITIVE, 1));
        assert!(!0.0_f32.approxeq(&f32::from_bits(2), 1));
        assert!(!1.0_f32.approxeq(&f32::from_bits(1.0_f32.to_bits() + 2), 1));
        assert!(f64::MAX.approxeq(&f64::MAX, 0));
    }

    #[test]
    fn negative_ulp_allowance_rejects_unequal_values() {
        assert!(!1.0_f32.approxeq(&1.0_f32.next_up(), -1));
    }
}
