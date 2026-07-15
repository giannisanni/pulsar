fn main() {
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() != Ok("linux") {
        return; // kernels are CUDA/Linux; other hosts get an empty crate
    }
    println!("cargo:rerun-if-changed=cuda");
    // One fatbin for every NVIDIA generation the kernels can serve. The
    // floor is dp4a = sm_61 (Pascal / GTX 10-series); nothing newer is
    // required (no tensor cores, no async-copy, static <=48KB shared).
    //
    //   sm_61  SASS  GTX 10-series           + compute_61 PTX: JIT floor
    //   sm_75  SASS  GTX 16 / RTX 20-series    for anything unlisted
    //   sm_86  SASS  RTX 30-series             (sm_70 Volta, sm_80 A100,
    //   sm_89  SASS  RTX 40-series              Hopper, ...)
    //   compute_89 PTX: JIT for sm_90+ (Blackwell RTX 50 etc.) with the
    //   newest ISA the toolkit knows, instead of the sm_61 floor.
    //
    // PULSAR_CUDA_ARCH overrides (e.g. "89" for a fast dev build, or
    // "89,120" once the toolkit codegens Blackwell SASS natively).
    // 80 must stay in the list: the int8 mma prefill GEMM gates on
    // cc >= 8 at runtime, so every >= 8.0 device needs a fatbin entry
    // compiled with __CUDA_ARCH__ >= 800 (A100 falling back to the
    // compute_61 floor PTX would silently run the empty stub)
    let archs = std::env::var("PULSAR_CUDA_ARCH").unwrap_or_else(|_| "61,75,80,86,89".into());
    let mut build = cc::Build::new();
    build.cuda(true).flag("-O3").flag("--use_fast_math");
    // nvcc rejects host compilers newer than its toolkit supports (e.g.
    // CUDA 12.0 caps at gcc 12 while distro c++ is gcc 13). Probe a tiny
    // compile with candidate ccbins and take the first one nvcc accepts.
    if let Some(ccbin) = pick_ccbin() {
        build.flag(&format!("-ccbin={ccbin}"));
    }
    let list: Vec<&str> = archs.split(',').map(str::trim).filter(|s| !s.is_empty()).collect();
    for (i, a) in list.iter().enumerate() {
        let first = i == 0;
        let last = i + 1 == list.len();
        // lowest arch also embeds its PTX (universal JIT floor); highest
        // embeds its PTX too (best ISA for future GPUs); middles are SASS-only
        let code = if first || last {
            format!("arch=compute_{a},code=[sm_{a},compute_{a}]")
        } else {
            format!("arch=compute_{a},code=sm_{a}")
        };
        build.flag("-gencode").flag(&code);
    }
    build.file("cuda/pulsar_kernels.cu").compile("pulsar_kernels");
    println!("cargo:rustc-link-lib=cudart");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");
}

fn pick_ccbin() -> Option<String> {
    let mut candidates: Vec<String> = Vec::new();
    if let Ok(env) = std::env::var("NVCC_CCBIN") {
        return Some(env); // explicit override, no probe
    }
    candidates.push("c++".into());
    for v in ["14", "13", "12", "11", "10"] {
        candidates.push(format!("g++-{v}"));
    }
    let out = std::env::var("OUT_DIR").unwrap_or_else(|_| ".".into());
    let probe = format!("{out}/ccbin_probe.cu");
    std::fs::write(&probe, "int main(){return 0;}\n").ok()?;
    for cand in candidates {
        let ok = std::process::Command::new("nvcc")
            .args([&format!("-ccbin={cand}"), "-c", &probe, "-o"])
            .arg(format!("{out}/ccbin_probe.o"))
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if ok {
            return Some(cand);
        }
    }
    None // let nvcc use its default and report its own error
}
