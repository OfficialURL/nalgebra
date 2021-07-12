#[cfg(not(all(feature = "debug", feature = "compare", feature = "rand",)))]
compile_error!(
    "Please enable the `debug`, `compare`, `rand` features in order to compile and run the tests.
     Example: `cargo test --features debug,compare,rand`"
);

#[cfg(feature = "abomonation-serialize")]
extern crate abomonation;
#[cfg(all(feature = "debug", feature = "compare", feature = "rand"))]
extern crate approx;
extern crate nalgebra as na;
extern crate num_traits as num;
#[cfg(feature = "rand")]
extern crate rand_package as rand;

#[cfg(all(feature = "debug", feature = "compare", feature = "rand"))]
mod core;
#[cfg(feature = "proptest-support")]
mod proptest;

//#[cfg(all(feature = "debug", feature = "compare", feature = "rand"))]
//#[cfg(feature = "sparse")]
//mod sparse;
