use std::any::Any;
use std::any::TypeId;
use std::fmt::Debug;

/// The basic scalar type for all structures of `nalgebra`.
///
/// This does not make any assumption on the algebraic properties of `Self`.
pub trait Scalar: Clone + PartialEq + Debug + Any {
    #[inline]
    /// Tests if `Self` the same as the type `T`
    ///
    /// Typically used to test of `Self` is a f32 or a f64 with `T::is::<f32>()`.
    fn is<T: Scalar>() -> bool {
        TypeId::of::<Self>() == TypeId::of::<T>()
    }
}

impl<T: Copy + PartialEq + Debug + Any> Scalar for T {}

/// Previously, the [`Scalar`] trait had an `inlined_clone` associated method, which just inlined `*self`.
/// Due to our trait restructuring, we're no longer able to use this method in its original form. However,
/// this new solution fails at solving the original problem. So we should probably rework it.
pub trait InlinedClone: Clone {
    #[inline(always)]
    /// (Disclaimer, does not currently work!)
    ///
    /// Performance hack: Clone doesn't get inlined for Copy types in debug mode, so make it inline anyway.
    fn inlined_clone(&self) -> Self {
        self.clone()
    }
}

impl<T: Copy> InlinedClone for T {
    #[inline(always)]
    fn inlined_clone(&self) -> Self {
        *self
    }
}
