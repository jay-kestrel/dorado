//! dorado native routing kernel — scaffolding.
//!
//! Spec deliverable D2 moves the per-particle stepping loop
//! (`dorado.lagrangian_walker.particle_stepper` and its helpers) into this
//! crate. The module also exposes version metadata so that packaging,
//! CI, and the `dorado._native` import path are exercised end to end
//! (spec deliverable D6).

pub mod rng;

fn kernel_version_impl() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

#[pyo3::pymodule]
mod _native {
    use pyo3::prelude::*;

    /// Reports the native crate version so packaging/CI can verify the
    /// import path. The routing kernel (spec deliverable D2) lands alongside
    /// this surface.
    #[pyfunction]
    fn kernel_version() -> &'static str {
        super::kernel_version_impl()
    }
}

#[cfg(test)]
mod tests {
    use super::kernel_version_impl;

    #[test]
    fn kernel_version_matches_crate_version() {
        assert_eq!(kernel_version_impl(), env!("CARGO_PKG_VERSION"));
    }
}
