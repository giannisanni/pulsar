//! FFI to the pulsar CUDA kernel library. Linux + NVIDIA only; on other
//! hosts the crate compiles to nothing so the workspace still builds.

#[cfg(target_os = "linux")]
mod ffi {
    extern "C" {
        pub fn pulsar_gqa_selftest() -> i32;
        pub fn pulsar_q8_0_matmul_selftest() -> i32;
    }
}

/// Run the GQA kernel self-test (kernels vs a CPU reference, no model
/// file needed). Requires a CUDA device.
#[cfg(target_os = "linux")]
pub fn gqa_selftest() -> bool {
    unsafe { ffi::pulsar_gqa_selftest() != 0 }
}

/// Run the pulsar-native Q8_0 matmul self-test (GPU vs CPU reference on
/// host-quantized random weights). Requires a CUDA device.
#[cfg(target_os = "linux")]
pub fn q8_0_matmul_selftest() -> bool {
    unsafe { ffi::pulsar_q8_0_matmul_selftest() != 0 }
}

#[cfg(test)]
mod tests {
    /// GPU-required; run explicitly: cargo test -p kernels -- --ignored
    #[test]
    #[ignore = "requires a CUDA device"]
    #[cfg(target_os = "linux")]
    fn gqa_kernels_match_cpu_reference() {
        assert!(super::gqa_selftest());
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    #[cfg(target_os = "linux")]
    fn q8_0_matmul_matches_cpu_reference() {
        assert!(super::q8_0_matmul_selftest());
    }
}
