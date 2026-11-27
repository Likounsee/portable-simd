pub trait FlushSubnormals: Sized {
    fn flush(self) -> Self {
        self
    }
}

impl<T> FlushSubnormals for *const T {}
impl<T> FlushSubnormals for *mut T {}

impl FlushSubnormals for i8 {}
impl FlushSubnormals for i16 {}
impl FlushSubnormals for i32 {}
impl FlushSubnormals for i64 {}
impl FlushSubnormals for isize {}

impl FlushSubnormals for u8 {}
impl FlushSubnormals for u16 {}
impl FlushSubnormals for u32 {}
impl FlushSubnormals for u64 {}
impl FlushSubnormals for usize {}

impl FlushSubnormals for f16 {}
impl FlushSubnormals for f64 {}

impl FlushSubnormals for f32 {
    fn flush(self) -> Self {
        // Correct for non-IEEE behavior for f32 on arm and powerpc.
        let ppc_flush = cfg!(all(
            any(
                target_arch = "powerpc",
                all(target_arch = "powerpc64", target_endian = "big")
            ),
            target_feature = "altivec",
            not(target_feature = "vsx"),
        ));
        let arm_flush = cfg!(all(target_arch = "arm", target_feature = "neon"));
        let flush = ppc_flush || arm_flush;

        if flush && self.is_subnormal() {
            f32::copysign(0.0, self)
        } else {
            self
        }
    }
}

/// NOTE: altivec had a subnormal flushing bug in older QEMU versions.
/// <https://gitlab.com/qemu-project/qemu/-/issues/1779>
pub fn flush_in<T: FlushSubnormals>(x: T) -> T {
    x.flush()
}

pub fn flush<T: FlushSubnormals>(x: T) -> T {
    x.flush()
}

#[cfg(test)]
mod tests {
    use super::{FlushSubnormals, flush, flush_in};

    const TARGET_FLUSHES_F32_SUBNORMALS: bool = cfg!(any(
        all(target_arch = "arm", target_feature = "neon"),
        all(
            any(
                target_arch = "powerpc",
                all(target_arch = "powerpc64", target_endian = "big")
            ),
            target_feature = "altivec",
            not(target_feature = "vsx")
        )
    ));

    #[test]
    fn f32_subnormals_follow_target_flush_behavior() {
        for bits in [1_u32, 0x007f_ffff] {
            let positive = f32::from_bits(bits);
            let negative = f32::from_bits(bits | (1 << 31));

            let expected_positive = if TARGET_FLUSHES_F32_SUBNORMALS {
                0.0_f32.to_bits()
            } else {
                bits
            };
            let expected_negative = if TARGET_FLUSHES_F32_SUBNORMALS {
                (-0.0_f32).to_bits()
            } else {
                bits | (1 << 31)
            };

            assert_eq!(positive.flush().to_bits(), expected_positive);
            assert_eq!(negative.flush().to_bits(), expected_negative);
            assert_eq!(flush(positive).to_bits(), expected_positive);
            assert_eq!(flush_in(negative).to_bits(), expected_negative);
        }
    }

    #[test]
    fn f32_normal_values_and_infinities_are_unchanged() {
        for value in [
            0.0_f32,
            -0.0,
            f32::MIN_POSITIVE,
            -f32::MIN_POSITIVE,
            1.0,
            -1.0,
            f32::INFINITY,
            f32::NEG_INFINITY,
        ] {
            assert_eq!(value.flush().to_bits(), value.to_bits());
        }
    }

    #[test]
    fn f32_nan_payload_and_sign_are_preserved() {
        for bits in [0x7fc0_0001_u32, 0xffc0_1234] {
            let value = f32::from_bits(bits);
            assert_eq!(value.flush().to_bits(), bits);
        }
    }
}
