//! GPU kernel selftests as a cargo test gate (see scripts/check.sh).
//! Each wrapper runs a device-vs-host reference comparison implemented in
//! pulsar_kernels.cu. Needs a CUDA GPU; run serially (--test-threads=1).
#![cfg(target_os = "linux")]

macro_rules! selftest {
    ($name:ident) => {
        #[test]
        fn $name() {
            assert!(kernels::$name(), stringify!($name));
        }
    };
}

selftest!(gqa_selftest);
selftest!(sconv_selftest);
selftest!(q8_0_matmul_selftest);
selftest!(router_selftest);
selftest!(moe_selftest);
selftest!(glue_selftest);
selftest!(mla_selftest);
