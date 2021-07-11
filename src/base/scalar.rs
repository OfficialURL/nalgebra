/// Previously, the `Scalar` trait had an `inlined_clone` associated method, which just inlined `*self`.
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
