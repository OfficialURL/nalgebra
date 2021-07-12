use std::any::TypeId;
use std::fmt::Debug;

/// The basic scalar trait for all structures of `nalgebra`.
///
/// This is by design a very loose trait, and does not make any assumption on
/// the algebraic properties of `Self`. Its main purpose is to allow for
/// specialization of methods for floating point types, for optimization
/// purposes.
pub trait Scalar: 'static + Clone + Debug {
    #[inline]
    /// Tests if `Self` is the same as the type `T`.
    ///
    /// Typically used to test of `Self` is an `f32` or an `f64`, which is
    /// important as it allows for specialization and certain optimizations to
    /// be made.
    ///
    /// If the need ever arose to get rid of the `'static` requirement
    fn is<T: Scalar>() -> bool {
        TypeId::of::<Self>() == TypeId::of::<T>()
    }

    /// Performance hack: Clone doesn't get inlined for Copy types in debug
    /// mode, so make it inline anyway.
    fn inlined_clone(&self) -> Self;
}

impl<T: 'static + Copy + Debug> Scalar for T {
    #[inline(always)]
    fn inlined_clone(&self) -> Self {
        *self
    }
}
