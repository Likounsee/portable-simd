//! Traits for vectors of pointers.

mod const_ptr;
mod mut_ptr;

pub use const_ptr::*;
pub use mut_ptr::*;

use crate::simd::Simd;

/// Creates pointers with the given addresses and no provenance.
///
/// Equivalent to calling [`core::ptr::without_provenance`] on each element.
#[inline]
pub fn without_provenance<T, const N: usize>(addr: Simd<usize, N>) -> Simd<*const T, N> {
    Simd::from_array(addr.to_array().map(core::ptr::without_provenance::<T>))
}

/// Creates mutable pointers with the given addresses and no provenance.
///
/// Equivalent to calling [`core::ptr::without_provenance_mut`] on each element.
#[inline]
pub fn without_provenance_mut<T, const N: usize>(addr: Simd<usize, N>) -> Simd<*mut T, N> {
    Simd::from_array(addr.to_array().map(core::ptr::without_provenance_mut::<T>))
}

/// Converts addresses back to pointers, picking up some previously "exposed" provenance.
///
/// Equivalent to calling [`core::ptr::with_exposed_provenance`] on each element.
#[inline]
pub fn with_exposed_provenance<T, const N: usize>(addr: Simd<usize, N>) -> Simd<*const T, N> {
    // SAFETY: addr is a vector of usize
    unsafe { core::intrinsics::simd::simd_with_exposed_provenance(addr) }
}

/// Converts addresses back to mutable pointers, picking up some previously "exposed" provenance.
///
/// Equivalent to calling [`core::ptr::with_exposed_provenance_mut`] on each element.
#[inline]
pub fn with_exposed_provenance_mut<T, const N: usize>(addr: Simd<usize, N>) -> Simd<*mut T, N> {
    // SAFETY: addr is a vector of usize
    unsafe { core::intrinsics::simd::simd_with_exposed_provenance(addr) }
}
