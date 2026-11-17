#![feature(portable_simd)]
#![feature(f16)]

use core_simd::simd::{Mask, Simd, num::SimdFloat};

#[test]
fn to_mask_maps_zero_and_nonzero_signed_integers() {
    let values = Simd::from_array([0_i32, 4, -2, 0]);
    assert_eq!(values.to_mask().to_array(), [false, true, true, false]);
}

#[test]
fn to_mask_maps_zero_and_nonzero_unsigned_integers() {
    let values = Simd::from_array([0_u32, 4, u32::MAX, 0]);
    assert_eq!(values.to_mask().to_array(), [false, true, true, false]);
}

#[test]
fn to_mask_maps_zero_and_nonzero_floats() {
    let values = Simd::from_array([0.0_f32, -0.0, 1.5, -2.0]);
    assert_eq!(values.to_mask().to_array(), [false, false, true, true]);
}

#[test]
fn to_mask_preserves_f32_subnormals_and_classifies_special_values() {
    let values = Simd::from_array([
        f32::from_bits(0x0000_0001),
        f32::from_bits(0x8000_0001),
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        0.0,
        -0.0,
        1.0,
    ]);
    assert_eq!(
        values.to_mask().to_array(),
        [true, true, true, true, true, false, false, true]
    );
}

#[test]
fn to_mask_preserves_f64_subnormals_and_classifies_special_values() {
    let values = Simd::from_array([
        f64::from_bits(0x0000_0000_0000_0001),
        f64::from_bits(0x8000_0000_0000_0001),
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
        0.0,
        -0.0,
        1.0,
    ]);
    assert_eq!(
        values.to_mask().to_array(),
        [true, true, true, true, true, false, false, true]
    );
}

#[test]
fn float_classification_handles_f32_subnormals_and_boundaries() {
    let values = Simd::from_array([
        f32::from_bits(0x0000_0001),
        f32::from_bits(0x8000_0001),
        f32::from_bits(0x007f_ffff),
        f32::MIN_POSITIVE,
        -f32::MIN_POSITIVE,
        0.0,
        f32::INFINITY,
        f32::NAN,
    ]);

    assert_eq!(
        values.is_subnormal().to_array(),
        [true, true, true, false, false, false, false, false]
    );
    assert_eq!(
        values.is_normal().to_array(),
        [false, false, false, true, true, false, false, false]
    );
}

#[test]
fn float_classification_handles_f64_subnormals_and_boundaries() {
    let values = Simd::from_array([
        f64::from_bits(0x0000_0000_0000_0001),
        f64::from_bits(0x8000_0000_0000_0001),
        f64::from_bits(0x000f_ffff_ffff_ffff),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        0.0,
        f64::INFINITY,
        f64::NAN,
    ]);

    assert_eq!(
        values.is_subnormal().to_array(),
        [true, true, true, false, false, false, false, false]
    );
    assert_eq!(
        values.is_normal().to_array(),
        [false, false, false, true, true, false, false, false]
    );
}

#[test]
fn float_classification_handles_f32_infinities_nans_and_finite_values() {
    let values = Simd::from_array([
        f32::NEG_INFINITY,
        -f32::MAX,
        -1.0,
        -0.0,
        0.0,
        1.0,
        f32::MAX,
        f32::INFINITY,
        f32::NAN,
    ]);

    assert_eq!(
        values.is_nan().to_array(),
        [false, false, false, false, false, false, false, false, true]
    );
    assert_eq!(
        values.is_infinite().to_array(),
        [true, false, false, false, false, false, false, true, false]
    );
    assert_eq!(
        values.is_finite().to_array(),
        [false, true, true, true, true, true, true, false, false]
    );
}

#[test]
fn float_classification_handles_f64_infinities_nans_and_finite_values() {
    let values = Simd::from_array([
        f64::NEG_INFINITY,
        -f64::MAX,
        -1.0,
        -0.0,
        0.0,
        1.0,
        f64::MAX,
        f64::INFINITY,
        f64::NAN,
    ]);

    assert_eq!(
        values.is_nan().to_array(),
        [false, false, false, false, false, false, false, false, true]
    );
    assert_eq!(
        values.is_infinite().to_array(),
        [true, false, false, false, false, false, false, true, false]
    );
    assert_eq!(
        values.is_finite().to_array(),
        [false, true, true, true, true, true, true, false, false]
    );
}

#[test]
fn float_sign_predicates_preserve_f32_sign_bits_for_special_values() {
    let values = Simd::from_array([
        f32::from_bits(0x0000_0000),
        f32::from_bits(0x8000_0000),
        f32::from_bits(0x7f80_0000),
        f32::from_bits(0xff80_0000),
        f32::from_bits(0x7fc1_2345),
        f32::from_bits(0xffc5_4321),
        f32::from_bits(0x0000_0001),
        f32::from_bits(0x8000_0001),
    ]);

    assert_eq!(
        values.is_sign_positive().to_array(),
        [true, false, true, false, true, false, true, false]
    );
    assert_eq!(
        values.is_sign_negative().to_array(),
        [false, true, false, true, false, true, false, true]
    );
}

#[test]
fn float_sign_predicates_preserve_f64_sign_bits_for_special_values() {
    let values = Simd::from_array([
        f64::from_bits(0x0000_0000_0000_0000),
        f64::from_bits(0x8000_0000_0000_0000),
        f64::from_bits(0x7ff0_0000_0000_0000),
        f64::from_bits(0xfff0_0000_0000_0000),
        f64::from_bits(0x7ff8_1234_5678_9abc),
        f64::from_bits(0xfff8_4321_8765_cba9),
        f64::from_bits(0x0000_0000_0000_0001),
        f64::from_bits(0x8000_0000_0000_0001),
    ]);

    assert_eq!(
        values.is_sign_positive().to_array(),
        [true, false, true, false, true, false, true, false]
    );
    assert_eq!(
        values.is_sign_negative().to_array(),
        [false, true, false, true, false, true, false, true]
    );
}

#[test]
fn float_abs_preserves_f32_magnitude_bits_for_special_values() {
    let values = Simd::from_array([
        f32::from_bits(0x0000_0000),
        f32::from_bits(0x8000_0000),
        f32::from_bits(0x7f80_0000),
        f32::from_bits(0xff80_0000),
        f32::from_bits(0x7fc1_2345),
        f32::from_bits(0xffc5_4321),
        f32::from_bits(0x0000_0001),
        f32::from_bits(0x8000_0001),
    ]);
    let expected = values
        .to_array()
        .map(|value| f32::from_bits(value.to_bits() & 0x7fff_ffff));
    assert_eq!(
        values.abs().to_bits().to_array(),
        expected.map(f32::to_bits)
    );
}

#[test]
fn float_abs_preserves_f64_magnitude_bits_for_special_values() {
    let values = Simd::from_array([
        f64::from_bits(0x0000_0000_0000_0000),
        f64::from_bits(0x8000_0000_0000_0000),
        f64::from_bits(0x7ff0_0000_0000_0000),
        f64::from_bits(0xfff0_0000_0000_0000),
        f64::from_bits(0x7ff8_1234_5678_9abc),
        f64::from_bits(0xfff8_4321_8765_cba9),
        f64::from_bits(0x0000_0000_0000_0001),
        f64::from_bits(0x8000_0000_0000_0001),
    ]);
    let expected = values
        .to_array()
        .map(|value| f64::from_bits(value.to_bits() & 0x7fff_ffff_ffff_ffff));
    assert_eq!(
        values.abs().to_bits().to_array(),
        expected.map(f64::to_bits)
    );
}

#[test]
fn float_copysign_preserves_f32_payload_and_uses_each_sign_bit() {
    let magnitude = Simd::from_array([
        f32::from_bits(0x0000_0000),
        f32::from_bits(0x8000_0000),
        f32::from_bits(0x7f80_0000),
        f32::from_bits(0xff80_0000),
        f32::from_bits(0x7fc1_2345),
        f32::from_bits(0xffc5_4321),
        f32::from_bits(0x0000_0001),
        f32::from_bits(0x8000_0001),
    ]);
    let signs = Simd::from_array([-1.0_f32, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]);
    let magnitude_values = magnitude.to_array();
    let sign_values = signs.to_array();
    let expected = core::array::from_fn(|i| {
        (magnitude_values[i].to_bits() & 0x7fff_ffff) | (sign_values[i].to_bits() & 0x8000_0000)
    });
    assert_eq!(magnitude.copysign(signs).to_bits().to_array(), expected);
}

#[test]
fn float_copysign_preserves_f64_payload_and_uses_each_sign_bit() {
    let magnitude = Simd::from_array([
        f64::from_bits(0x0000_0000_0000_0000),
        f64::from_bits(0x8000_0000_0000_0000),
        f64::from_bits(0x7ff0_0000_0000_0000),
        f64::from_bits(0xfff0_0000_0000_0000),
        f64::from_bits(0x7ff8_1234_5678_9abc),
        f64::from_bits(0xfff8_4321_8765_cba9),
        f64::from_bits(0x0000_0000_0000_0001),
        f64::from_bits(0x8000_0000_0000_0001),
    ]);
    let signs = Simd::from_array([-1.0_f64, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]);
    let magnitude_values = magnitude.to_array();
    let sign_values = signs.to_array();
    let expected = core::array::from_fn(|i| {
        (magnitude_values[i].to_bits() & 0x7fff_ffff_ffff_ffff)
            | (sign_values[i].to_bits() & 0x8000_0000_0000_0000)
    });
    assert_eq!(magnitude.copysign(signs).to_bits().to_array(), expected);
}

#[test]
fn float_signum_preserves_f32_nan_bits_and_matches_scalar_semantics() {
    let values = Simd::from_array([
        f32::from_bits(0x7fc1_2345),
        f32::from_bits(0xffc5_4321),
        f32::NEG_INFINITY,
        -3.5,
        -0.0,
        0.0,
        2.5,
        f32::INFINITY,
    ]);
    let expected = values.to_array().map(|value| {
        if value.is_nan() {
            value.to_bits()
        } else {
            value.signum().to_bits()
        }
    });

    assert_eq!(values.signum().to_bits().to_array(), expected);
}

#[test]
fn float_signum_preserves_f64_nan_bits_and_matches_scalar_semantics() {
    let values = Simd::from_array([
        f64::from_bits(0x7ff8_1234_5678_9abc),
        f64::from_bits(0xfff8_4321_8765_cba9),
        f64::NEG_INFINITY,
        -3.5,
        -0.0,
        0.0,
        2.5,
        f64::INFINITY,
    ]);
    let expected = values.to_array().map(|value| {
        if value.is_nan() {
            value.to_bits()
        } else {
            value.signum().to_bits()
        }
    });

    assert_eq!(values.signum().to_bits().to_array(), expected);
}

#[test]
fn to_mask_classifies_f16_subnormals_and_special_values() {
    let values = Simd::from_array([
        f16::from_bits(0x0001),
        f16::from_bits(0x8001),
        f16::INFINITY,
        f16::NEG_INFINITY,
        f16::NAN,
        0.0_f16,
        -0.0_f16,
        1.0_f16,
    ]);

    let expected = [true, true, true, true, true, false, false, true];
    // Inspect the stored lane bits without invoking the affected SIMD bitcast.
    // Read each lane through a raw pointer so the test does not create a u16
    // reference over storage whose declared element type is f16.
    let raw_bits = values.as_array().as_ptr().cast::<u16>();
    let observed_bits: [u16; 8] = core::array::from_fn(|i| {
        // SAFETY: f16 and u16 have identical size and alignment, and every u16
        // bit pattern is valid. The pointer addresses an initialized SIMD lane.
        unsafe { raw_bits.add(i).read_unaligned() }
    });
    assert_eq!(
        observed_bits,
        [
            0x0001, 0x8001, 0x7c00, 0xfc00, 0x7e00, 0x0000, 0x8000, 0x3c00
        ],
        "raw f16 lane storage must preserve the input bit patterns",
    );
    let scalar_classification = observed_bits.map(|bits| bits & 0x7fff != 0);
    assert_eq!(
        scalar_classification, expected,
        "scalar f16 bit classification must distinguish both zero encodings",
    );
    assert_eq!(
        Mask::<i16, 8>::from_array(expected).to_array(),
        expected,
        "i16 mask construction and extraction must preserve false lanes",
    );
    assert_eq!(
        values.to_array().map(|value| value.to_bits()),
        [
            0x0001, 0x8001, 0x7c00, 0xfc00, 0x7e00, 0x0000, 0x8000, 0x3c00
        ],
        "the SIMD-to-array roundtrip must preserve every f16 bit pattern",
    );
    assert_eq!(values.to_mask().to_array(), expected);
}

#[test]
fn to_mask_covers_all_integer_width_families() {
    let signed8 = Simd::from_array([0_i8, 1, -1, 0]);
    assert_eq!(signed8.to_mask().to_array(), [false, true, true, false]);
    let signed16 = Simd::from_array([i16::MIN, 0, 1, -1]);
    assert_eq!(signed16.to_mask().to_array(), [true, false, true, true]);
    let signed64 = Simd::from_array([0_i64, i64::MIN, 0, 1]);
    assert_eq!(signed64.to_mask().to_array(), [false, true, false, true]);
    let signed_size = Simd::from_array([0_isize, -1, 1, 0]);
    assert_eq!(signed_size.to_mask().to_array(), [false, true, true, false]);

    let unsigned8 = Simd::from_array([0_u8, 1, u8::MAX, 0]);
    assert_eq!(unsigned8.to_mask().to_array(), [false, true, true, false]);
    let unsigned16 = Simd::from_array([0_u16, u16::MAX, 0, 1]);
    assert_eq!(unsigned16.to_mask().to_array(), [false, true, false, true]);
    let unsigned64 = Simd::from_array([0_u64, u64::MAX, 0, 1]);
    assert_eq!(unsigned64.to_mask().to_array(), [false, true, false, true]);
    let unsigned_size = Simd::from_array([0_usize, usize::MAX, 0, 1]);
    assert_eq!(
        unsigned_size.to_mask().to_array(),
        [false, true, false, true]
    );
}
