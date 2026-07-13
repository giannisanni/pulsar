//! FFI to the pulsar CUDA kernel library. Linux + NVIDIA only; on other
//! hosts the crate compiles to nothing so the workspace still builds.

#[cfg(target_os = "linux")]
mod ffi {
    extern "C" {
        pub fn pulsar_gqa_selftest() -> i32;
        pub fn pulsar_q8_0_matmul_selftest() -> i32;
        pub fn pulsar_router_selftest() -> i32;
        pub fn pulsar_moe_selftest() -> i32;
        pub fn pulsar_glue_selftest() -> i32;
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

/// Run the sigmoid router + top-k select self-test (GPU vs CPU reference
/// across Hy3-like and GLM-like shapes). Requires a CUDA device.
#[cfg(target_os = "linux")]
pub fn router_selftest() -> bool {
    unsafe { ffi::pulsar_router_selftest() != 0 }
}

/// Run the routed-expert MoE self-test (IQ2_XXS + Q2_K pair-swiglu and
/// down kernels vs a host dequant reference). Requires a CUDA device.
#[cfg(target_os = "linux")]
pub fn moe_selftest() -> bool {
    unsafe { ffi::pulsar_moe_selftest() != 0 }
}

/// Run the forward-graph glue self-test (rms-norm, f32 matmul, swiglu,
/// add, q8_0 embedding lookup vs CPU references). Requires a CUDA device.
#[cfg(target_os = "linux")]
pub fn glue_selftest() -> bool {
    unsafe { ffi::pulsar_glue_selftest() != 0 }
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

    #[test]
    #[ignore = "requires a CUDA device"]
    #[cfg(target_os = "linux")]
    fn router_select_matches_cpu_reference() {
        assert!(super::router_selftest());
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    #[cfg(target_os = "linux")]
    fn moe_kernels_match_cpu_reference() {
        assert!(super::moe_selftest());
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    #[cfg(target_os = "linux")]
    fn glue_kernels_match_cpu_reference() {
        assert!(super::glue_selftest());
    }
}
