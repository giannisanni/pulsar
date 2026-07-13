fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return; // kernels are CUDA/Linux; other hosts get an empty crate
    }
    println!("cargo:rerun-if-changed=cuda/pulsar_kernels.cu");
    println!("cargo:rerun-if-changed=cuda/gqa_kernels.inc");
    cc::Build::new()
        .cuda(true)
        .flag("-O3")
        .flag("--use_fast_math")
        .flag("-arch=sm_89")
        .file("cuda/pulsar_kernels.cu")
        .compile("pulsar_kernels");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
}
