use crate::simd::{
    Simd,
    num::{SimdFloat, SimdInt, SimdUint},
};

/// Converts SIMD vectors to vectors of bytes
pub impl(self) trait ToBytes {
    /// This type, reinterpreted as bytes.
    type Bytes: Copy
        + Unpin
        + Send
        + Sync
        + AsRef<[u8]>
        + AsMut<[u8]>
        + SimdUint<Scalar = u8>
        + 'static;

    /// Returns the memory representation of this value as a byte array in native byte
    /// order.
    fn to_ne_bytes(self) -> Self::Bytes;

    /// Returns the memory representation of this value as a byte array in big-endian
    /// (network) byte order.
    fn to_be_bytes(self) -> Self::Bytes;

    /// Returns the memory representation of this value as a byte array in little-endian
    /// byte order.
    fn to_le_bytes(self) -> Self::Bytes;

    /// Creates a value from its memory representation in native byte order.
    fn from_ne_bytes(bytes: Self::Bytes) -> Self;

    /// Creates a value from its representation as a byte array in big-endian byte order.
    fn from_be_bytes(bytes: Self::Bytes) -> Self;

    /// Creates a value from its representation as a byte array in little-endian byte order.
    fn from_le_bytes(bytes: Self::Bytes) -> Self;
}

macro_rules! swap_bytes {
    { f16, $x:expr } => { Simd::from_bits($x.to_bits().swap_bytes()) };
    { f32, $x:expr } => { Simd::from_bits($x.to_bits().swap_bytes()) };
    { f64, $x:expr } => { Simd::from_bits($x.to_bits().swap_bytes()) };
    { $ty:ty, $x:expr } => { $x.swap_bytes() }
}

macro_rules! impl_to_bytes {
    { $ty:tt, 1  } => { impl_to_bytes! { $ty, 1  * [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64] } };
    { $ty:tt, 2  } => { impl_to_bytes! { $ty, 2  * [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32] } };
    { $ty:tt, 4  } => { impl_to_bytes! { $ty, 4  * [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16] } };
    { $ty:tt, 8  } => { impl_to_bytes! { $ty, 8  * [1, 2, 3, 4, 5, 6, 7, 8] } };
    { $ty:tt, 16 } => { impl_to_bytes! { $ty, 16 * [1, 2, 3, 4] } };
    { $ty:tt, 32 } => { impl_to_bytes! { $ty, 32 * [1, 2] } };
    { $ty:tt, 64 } => { impl_to_bytes! { $ty, 64 * [1] } };

    { $ty:tt, $size:literal * [$($elems:literal),*] } => {
        $(
        impl ToBytes for Simd<$ty, $elems> {
            type Bytes = Simd<u8, { $size * $elems }>;

            #[inline]
            fn to_ne_bytes(self) -> Self::Bytes {
                // Safety: transmuting between vectors is safe
                unsafe {
                    #![allow(clippy::useless_transmute)]
                    core::mem::transmute(self)
                }
            }

            #[inline]
            fn to_be_bytes(mut self) -> Self::Bytes {
                if !cfg!(target_endian = "big") {
                    self = swap_bytes!($ty, self);
                }
                self.to_ne_bytes()
            }

            #[inline]
            fn to_le_bytes(mut self) -> Self::Bytes {
                if !cfg!(target_endian = "little") {
                    self = swap_bytes!($ty, self);
                }
                self.to_ne_bytes()
            }

            #[inline]
            fn from_ne_bytes(bytes: Self::Bytes) -> Self {
                // Safety: transmuting between vectors is safe
                unsafe {
                    #![allow(clippy::useless_transmute)]
                    core::mem::transmute(bytes)
                }
            }

            #[inline]
            fn from_be_bytes(bytes: Self::Bytes) -> Self {
                let ret = Self::from_ne_bytes(bytes);
                if cfg!(target_endian = "big") {
                    ret
                } else {
                    swap_bytes!($ty, ret)
                }
            }

            #[inline]
            fn from_le_bytes(bytes: Self::Bytes) -> Self {
                let ret = Self::from_ne_bytes(bytes);
                if cfg!(target_endian = "little") {
                    ret
                } else {
                    swap_bytes!($ty, ret)
                }
            }
        }
        )*
    }
}

impl_to_bytes! { u8, 1 }
impl_to_bytes! { u16, 2 }
impl_to_bytes! { u32, 4 }
impl_to_bytes! { u64, 8 }
#[cfg(target_pointer_width = "32")]
impl_to_bytes! { usize, 4 }
#[cfg(target_pointer_width = "64")]
impl_to_bytes! { usize, 8 }

impl_to_bytes! { i8, 1 }
impl_to_bytes! { i16, 2 }
impl_to_bytes! { i32, 4 }
impl_to_bytes! { i64, 8 }
#[cfg(target_pointer_width = "32")]
impl_to_bytes! { isize, 4 }
#[cfg(target_pointer_width = "64")]
impl_to_bytes! { isize, 8 }

impl_to_bytes! { f16, 2 }
impl_to_bytes! { f32, 4 }
impl_to_bytes! { f64, 8 }
