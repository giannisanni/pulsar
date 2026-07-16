//! FFI to the pulsar CUDA kernel library. Linux + NVIDIA only; on other
//! hosts the crate compiles to nothing so the workspace still builds.

#[cfg(target_os = "linux")]
mod real {
    use std::ffi::c_void;

    pub type Result<T = ()> = std::result::Result<T, Error>;

    #[derive(Debug)]
    pub struct Error(pub &'static str);

    impl std::fmt::Display for Error {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "cuda kernel op failed: {}", self.0)
        }
    }

    impl std::error::Error for Error {}

    /// Matches `pulsar_expert_ptrs` in pulsar_kernels.cu: explicit device
    /// pointers for one (token, slot); NULL means "not routed".
    #[repr(C)]
    #[derive(Clone, Copy)]
    pub struct ExpertPtrs {
        pub gate: *const c_void,
        pub up: *const c_void,
        pub down: *const c_void,
    }

    unsafe impl Send for ExpertPtrs {}

    impl ExpertPtrs {
        pub const NULL: ExpertPtrs = ExpertPtrs {
            gate: std::ptr::null(),
            up: std::ptr::null(),
            down: std::ptr::null(),
        };
    }

    pub const QUANT_Q2_K: u32 = 0;
    pub const QUANT_IQ2_XXS: u32 = 1;
    pub const QUANT_Q4_K: u32 = 2;
    pub const QUANT_Q5_K: u32 = 3;
    pub const QUANT_Q6_K: u32 = 4;
    pub const QUANT_Q3_K: u32 = 5;
    pub const QUANT_IQ2_XS: u32 = 6;
    pub const QUANT_IQ3_XXS: u32 = 7;
    pub const QUANT_Q4_0: u32 = 8;
    pub const QUANT_Q5_1: u32 = 9;
    pub const QUANT_Q8_0: u32 = 10;
    pub const QUANT_IQ4_XS: u32 = 11;

    const H2D: i32 = 1;
    const D2H: i32 = 2;

    extern "C" {
        fn cudaSetDevice(dev: i32) -> i32;
        fn cudaGetDevice(dev: *mut i32) -> i32;
        fn cudaGetDeviceCount(count: *mut i32) -> i32;
        fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
        fn cudaDeviceGetAttribute(val: *mut i32, attr: i32, dev: i32) -> i32;
        fn cudaMalloc(ptr: *mut *mut c_void, bytes: usize) -> i32;
        fn cudaFree(ptr: *mut c_void) -> i32;
        fn cudaHostAlloc(ptr: *mut *mut c_void, bytes: usize, flags: u32) -> i32;
        fn cudaFreeHost(ptr: *mut c_void) -> i32;
        fn cudaHostGetDevicePointer(dev: *mut *mut c_void, host: *mut c_void, flags: u32) -> i32;
        fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> i32;
        fn cudaMemset(ptr: *mut c_void, value: i32, bytes: usize) -> i32;
        fn cudaDeviceSynchronize() -> i32;

        fn pulsar_embed_q8_0(out: *mut c_void, w: *const c_void, tokens: *const c_void, n_embd: u32, n_vocab: u32, n_tok: u32) -> i32;
        fn pulsar_rms_norm(out: *mut c_void, x: *const c_void, w: *const c_void, n: u32, rows: u32, eps: f32) -> i32;
        fn pulsar_q8_0_matmul(out: *mut c_void, w: *const c_void, x: *const c_void, in_dim: u32, out_dim: u32, n_tok: u32) -> i32;
        fn pulsar_matmul_f32(out: *mut c_void, w: *const c_void, x: *const c_void, in_dim: u32, out_dim: u32, n_tok: u32) -> i32;
        fn pulsar_matmul_kq(out: *mut c_void, w: *const c_void, xq: *const c_void, in_dim: u32, out_dim: u32, n_tok: u32, row_bytes: u64, quant: u32) -> i32;
        fn pulsar_idx_rope0(x: *mut c_void, n_tok: u32, n_head: u32, head_dim: u32, rot_dim: u32, pos0: u32, n_ctx_orig: u32, freq_base: f32, freq_scale: f32, ext_factor: f32, attn_factor: f32, beta_fast: f32, beta_slow: f32) -> i32;
        fn pulsar_idx_store_k(raw_k: *const c_void, w: *const c_void, b: *const c_void, cache: *mut c_void, pos0: u32, n_tok: u32, cache_cap: u32, head_dim: u32, rot_dim: u32, n_ctx_orig: u32, eps: f32, freq_base: f32, freq_scale: f32, ext_factor: f32, attn_factor: f32, beta_fast: f32, beta_slow: f32) -> i32;
        fn pulsar_idx_score_one(scores: *mut c_void, q: *const c_void, weights: *const c_void, cache: *const c_void, n_rows: u32, n_head: u32, head_dim: u32, scale: f32) -> i32;
        fn pulsar_idx_topk(selected: *mut c_void, scores: *const c_void, n_rows: u32, top_k: u32) -> i32;
        fn pulsar_idx_scores_batch(scores: *mut c_void, q: *const c_void, weights: *const c_void, cache: *const c_void, q16: *mut c_void, n_rows: u32, n_tokens: u32, pos0: u32, n_head: u32, head_dim: u32, scale: f32) -> i32;
        fn pulsar_idx_selftest() -> i32;
        fn pulsar_swiglu(out: *mut c_void, gate: *const c_void, up: *const c_void, n: u32, clamp: f32, weight: f32, act_op: u32) -> i32;
        fn pulsar_scale(x: *mut c_void, n: u32, c: f32) -> i32;
        fn pulsar_fill_row_tail(x: *mut c_void, rows: u32, row_w: u32, keep: u32, v: f32) -> i32;
        fn pulsar_softcap(x: *mut c_void, n: u32, cap: f32) -> i32;
        fn pulsar_router_scale_selected(w: *mut c_void, sel: *const c_void, scale: *const c_void, n: u32, n_expert: u32) -> i32;
        fn pulsar_add(out: *mut c_void, a: *const c_void, b: *const c_void, n: u32) -> i32;
        fn pulsar_router_select(selected: *mut c_void, weights: *mut c_void, logits: *const c_void, bias: *const c_void, n_expert: u32, k_used: u32, weight_scale: f32, n_tok: u32, softmax_mode: u32, n_shexp: u32) -> i32;
        fn pulsar_quantize_q8_K(out: *mut c_void, x: *const c_void, in_dim: u32, n_rows: u32) -> i32;
        fn pulsar_moe_pair_swiglu(mid: *mut c_void, ptrs: *const c_void, weights: *const c_void, x: *const c_void, in_dim: u32, mid_dim: u32, n_used: u32, n_tok: u32, row_bytes: u64, quant: u32, act_op: u32) -> i32;
        fn pulsar_moe_down(out: *mut c_void, ptrs: *const c_void, mid: *const c_void, mid_dim: u32, out_dim: u32, n_used: u32, n_tok: u32, row_bytes: u64, quant: u32) -> i32;
        fn pulsar_moe_pair_swiglu_grouped(mid: *mut c_void, gptrs: *const c_void, starts: *const c_void, pairs: *const c_void, weights: *const c_void, xq: *const c_void, in_dim: u32, mid_dim: u32, n_used: u32, n_group: u32, row_bytes: u64, quant: u32, act_op: u32) -> i32;
        fn pulsar_moe_down_grouped(partial: *mut c_void, gptrs: *const c_void, starts: *const c_void, pairs: *const c_void, midq: *const c_void, mid_dim: u32, out_dim: u32, n_used: u32, n_group: u32, row_bytes: u64, quant: u32) -> i32;
        fn pulsar_moe_slot_sum(out: *mut c_void, partial: *const c_void, out_dim: u32, n_used: u32, n_tok: u32) -> i32;
        fn pulsar_gqa_head_rms_norm(x: *mut c_void, w: *const c_void, rows: u32, head_dim: u32, eps: f32) -> i32;
        fn pulsar_gqa_rope(x: *mut c_void, n_tok: u32, n_head: u32, head_dim: u32, rot_dim: u32, pos0: u32, theta: f32, factors: *const c_void) -> i32;
        fn pulsar_gqa_kv_append(cache: *mut c_void, kv: *const c_void, n_tok: u32, n_kv_head: u32, head_dim: u32, cap: u32, pos0: u32, kvq: u32) -> i32;
        fn pulsar_gqa_attention(out: *mut c_void, q: *const c_void, k_cache: *const c_void, v_cache: *const c_void, n_tok: u32, n_head: u32, n_kv_head: u32, head_dim: u32, cap: u32, pos0: u32, scale: f32, window: u32, rel: *const c_void, rel_extent: u32, kvq: u32) -> i32;

        fn pulsar_sconv(out: *mut c_void, x: *const c_void, kern: *const c_void, state: *mut c_void, n_tok: u32, w: u32, k: u32) -> i32;

        fn pulsar_gqa_selftest() -> i32;
        fn pulsar_q8_0_matmul_selftest() -> i32;
        fn pulsar_router_selftest() -> i32;
        fn pulsar_moe_selftest() -> i32;
        fn pulsar_glue_selftest() -> i32;
        fn pulsar_mla_selftest() -> i32;
        fn pulsar_sconv_selftest() -> i32;

        fn pulsar_mla_rope_tail(x: *mut c_void, n_tok: u32, n_head: u32, head_dim: u32, rot_dim: u32, pos0: u32, n_ctx_orig: u32, freq_base: f32, freq_scale: f32, ext_factor: f32, attn_factor: f32, beta_fast: f32, beta_slow: f32) -> i32;
        fn pulsar_dsv4_rope_tail(x: *mut c_void, n_tok: u32, n_head: u32, head_dim: u32, rot_dim: u32, pos0: u32, n_ctx_orig: u32, freq_base: f32, freq_scale: f32, ext_factor: f32, attn_factor: f32, beta_fast: f32, beta_slow: f32, inverse: u32) -> i32;
        fn pulsar_dsv4_hc_mix(out: *mut c_void, streams: *const c_void, block: *const c_void, st_c: *const f32, blk_c: *const f32, n_embd: u32, n_hc: u32, n_out: u32) -> i32;
        fn pulsar_dsv4_attention(out: *mut c_void, q: *const c_void, raw: *const c_void, n_raw: u32, comp: *const c_void, n_comp: u32, allowed: *const c_void, sinks: *const c_void, n_head: u32, head_dim: u32, scale: f32) -> i32;
        fn pulsar_dsv4_fp8_sim(x: *mut c_void, n_rows: u32, head_dim: u32, n_rot: u32) -> i32;
        fn pulsar_dsv4_f16_round(x: *mut c_void, n: u32) -> i32;
        fn pulsar_dsv4_selftest() -> i32;
        fn pulsar_mla_kv_lora_rms_norm(out: *mut c_void, kv_raw: *const c_void, w: *const c_void, n_tok: u32, kv_raw_dim: u32, kv_lora_dim: u32, eps: f32) -> i32;
        fn pulsar_mla_store_compact_kv(kv_lora_cache: *mut c_void, k_rope_cache: *mut c_void, kv_norm: *const c_void, kv_raw: *const c_void, pos0: u32, n_tok: u32, cache_cap: u32, kv_raw_dim: u32, kv_lora_dim: u32, qk_rope: u32) -> i32;
        fn pulsar_mla_fill_selected_range(selected: *mut c_void, n_tok: u32, pos0: u32, n_selected: u32, pad_row: u32) -> i32;
        fn pulsar_mla_qk_lowrank(qk_low: *mut c_void, q: *const c_void, k_b: *const c_void, n_tok: u32, n_head: u32, kv_lora_dim: u32, qk_nope: u32, qk_dim: u32) -> i32;
        fn pulsar_mla_attention(heads: *mut c_void, q: *const c_void, qk_low: *const c_void, kv_lora_cache: *const c_void, k_rope_cache: *const c_void, v_b: *const c_void, selected: *const c_void, n_tok: u32, n_selected: u32, cache_cap: u32, n_head: u32, kv_lora_dim: u32, qk_nope: u32, qk_rope: u32, value_dim: u32, n_ctx_orig: u32, freq_base: f32, freq_scale: f32, ext_factor: f32, attn_factor: f32, beta_fast: f32, beta_slow: f32, kq_mult: f32) -> i32;
    }

    /// RoPE/YaRN configuration for the MLA family. GLM-5.2 ships
    /// ext_factor 0 (yarn off) but the parameters ride along.
    #[derive(Debug, Clone, Copy)]
    pub struct RopeCfg {
        pub n_ctx_orig: u32,
        pub freq_base: f32,
        pub freq_scale: f32,
        pub ext_factor: f32,
        pub attn_factor: f32,
        pub beta_fast: f32,
        pub beta_slow: f32,
        /// deepseek2 YaRN mscale^2, multiplied into the attention softmax
        /// scale (kq_scale = kq_mult / sqrt(qk_dim)); 1.0 = plain.
        pub kq_mult: f32,
    }

    fn check(ret: i32, op: &'static str) -> Result {
        if ret != 0 {
            Ok(())
        } else {
            Err(Error(op))
        }
    }

    fn check_rt(ret: i32, op: &'static str) -> Result {
        if ret == 0 {
            Ok(())
        } else {
            Err(Error(op))
        }
    }

    /// An owned device-visible allocation: VRAM (cudaMalloc) or mapped
    /// pinned host memory (weights too big for VRAM, read zero-copy over
    /// PCIe - ds4's trick for GLM-class backbones). Byte-oriented; callers
    /// track element layout themselves.
    pub struct DeviceBuf {
        ptr: *mut c_void,
        host: *mut c_void, // null for VRAM allocations
        bytes: usize,
        /// CUDA device the VRAM lives on (-1 for pinned host memory).
        dev: i32,
    }

    unsafe impl Send for DeviceBuf {}

    const ATTR_CC_MAJOR: i32 = 75;
    const ATTR_CC_MINOR: i32 = 76;

    /// Raw probe used during device selection (must not route through
    /// set_device - Once re-entrancy). Best-of-3 pinned 64MB H2D, GB/s.
    fn raw_h2d_probe(dev: i32) -> f64 {
        const MB64: usize = 64 << 20;
        if unsafe { cudaSetDevice(dev) } != 0 {
            return 0.0;
        }
        let mut host = std::ptr::null_mut();
        let mut dst = std::ptr::null_mut();
        if unsafe { cudaHostAlloc(&mut host, MB64, 0) } != 0 {
            return 0.0;
        }
        if unsafe { cudaMalloc(&mut dst, MB64) } != 0 {
            unsafe { cudaFreeHost(host) };
            return 0.0;
        }
        let mut best = 0f64;
        for _ in 0..3 {
            let t = std::time::Instant::now();
            if unsafe { cudaMemcpy(dst, host, MB64, H2D) } == 0 {
                best = best.max(MB64 as f64 / 1e9 / t.elapsed().as_secs_f64());
            }
        }
        unsafe {
            cudaFree(dst);
            cudaFreeHost(host);
        }
        best
    }

    /// Pick the primary GPU once, before the first allocation.
    ///
    /// CUDA's default device is index 0 under its own "fastest first" ordering,
    /// which is NOT PCI bus order and does not agree with nvidia-smi. Worse,
    /// static rankings lie about what matters: expert streaming is H2D-bound,
    /// and substrate's 4060 Ti sits in a slot that trains PCIe x1 (0.8 GB/s vs
    /// the 5060 Ti's 28.8) - a compute-capability heuristic can't see that, and
    /// neither can lspci at idle. So MEASURE: probe H2D bandwidth per device
    /// and take the fastest link (~100ms/device at startup, tie-break by
    /// compute capability). PULSAR_GPU overrides with a CUDA device index.
    fn ensure_device() {
        use std::sync::Once;
        static ONCE: Once = Once::new();
        ONCE.call_once(|| {
            let pick = std::env::var("PULSAR_GPU").ok().and_then(|s| s.trim().parse::<i32>().ok());
            let mut probed = 0.0;
            let dev = pick.unwrap_or_else(|| {
                let mut n = 0;
                if unsafe { cudaGetDeviceCount(&mut n) } != 0 || n <= 1 {
                    return 0;
                }
                let cc = |d: i32| -> i32 {
                    let (mut maj, mut min) = (0, 0);
                    unsafe {
                        cudaDeviceGetAttribute(&mut maj, ATTR_CC_MAJOR, d);
                        cudaDeviceGetAttribute(&mut min, ATTR_CC_MINOR, d);
                    }
                    maj * 10 + min
                };
                let best = (0..n)
                    .map(|d| (d, raw_h2d_probe(d)))
                    .max_by(|a, b| {
                        a.1.partial_cmp(&b.1)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| cc(a.0).cmp(&cc(b.0)))
                            .then_with(|| b.0.cmp(&a.0))
                    })
                    .unwrap_or((0, 0.0));
                probed = best.1;
                best.0
            });
            if unsafe { cudaSetDevice(dev) } != 0 {
                eprintln!("pulsar: cudaSetDevice({dev}) failed, falling back to CUDA default");
            } else if std::env::var_os("PULSAR_QUIET").is_none() {
                if probed > 0.0 {
                    eprintln!("pulsar: using CUDA device {dev} ({probed:.1} GB/s H2D measured)");
                } else {
                    eprintln!("pulsar: using CUDA device {dev}");
                }
            }
        });
    }

    /// Switch the calling thread's current CUDA device. Kernel wrappers
    /// launch on whatever device is current; the engine brackets its
    /// attn-GPU segments with this.
    pub fn set_device(dev: i32) -> Result {
        ensure_device();
        check_rt(unsafe { cudaSetDevice(dev) }, "cudaSetDevice")
    }

    pub fn get_device() -> i32 {
        let mut d = 0;
        unsafe { cudaGetDevice(&mut d) };
        d
    }

    pub fn device_count() -> i32 {
        let mut n = 0;
        unsafe { cudaGetDeviceCount(&mut n) };
        n
    }

    /// Measured H2D bandwidth to `dev` in GB/s (pinned 64MB, best of 3).
    /// Labels lie - a Gen5 card can train at Gen1, an x8 slot can run x1 -
    /// so role assignment trusts measurements only. Restores the device.
    pub fn h2d_bandwidth(dev: i32) -> Result<f64> {
        const MB64: usize = 64 << 20;
        let cur = get_device();
        set_device(dev)?;
        let mut host = std::ptr::null_mut();
        let mut dst = std::ptr::null_mut();
        check_rt(unsafe { cudaHostAlloc(&mut host, MB64, 0) }, "probe host alloc")?;
        if let Err(e) = check_rt(unsafe { cudaMalloc(&mut dst, MB64) }, "probe dev alloc") {
            unsafe { cudaFreeHost(host) };
            set_device(cur)?;
            return Err(e);
        }
        let mut best = 0f64;
        for _ in 0..3 {
            let t = std::time::Instant::now();
            let r = unsafe { cudaMemcpy(dst, host, MB64, H2D) };
            if r == 0 {
                best = best.max(MB64 as f64 / 1e9 / t.elapsed().as_secs_f64());
            }
        }
        unsafe {
            cudaFree(dst);
            cudaFreeHost(host);
        }
        set_device(cur)?;
        Ok(best)
    }

    /// True on unified-memory systems (GB10/DGX Spark, Jetson: GPU and CPU
    /// share one physical pool), where pinned host memory reads at full
    /// device speed and H2D staging is pure waste. Uses the `integrated`
    /// attribute - NOT pageableMemoryAccess, which HMM also reports on
    /// discrete x86 boxes where zero-copy would be a 50x regression.
    /// PULSAR_UNIFIED=1/0 overrides detection either way.
    pub fn unified_memory() -> bool {
        match std::env::var("PULSAR_UNIFIED").ok().as_deref() {
            Some("1") => return true,
            Some("0") => return false,
            _ => {}
        }
        ensure_device();
        const ATTR_INTEGRATED: i32 = 18;
        let mut v = 0;
        unsafe { cudaDeviceGetAttribute(&mut v, ATTR_INTEGRATED, get_device()) };
        v == 1
    }

    /// (free, total) VRAM in bytes on `dev`. Restores the current device.
    pub fn mem_info(dev: i32) -> Result<(usize, usize)> {
        let cur = get_device();
        set_device(dev)?;
        let (mut free, mut total) = (0usize, 0usize);
        let r = check_rt(unsafe { cudaMemGetInfo(&mut free, &mut total) }, "cudaMemGetInfo");
        set_device(cur)?;
        r?;
        Ok((free, total))
    }

    impl DeviceBuf {
        pub fn alloc(bytes: usize) -> Result<Self> {
            ensure_device();
            let mut ptr = std::ptr::null_mut();
            if let Err(e) = check_rt(unsafe { cudaMalloc(&mut ptr, bytes.max(1)) }, "cudaMalloc") {
                eprintln!(
                    "pulsar: cudaMalloc({:.2} GB) failed on device {}",
                    bytes as f64 / 1e9,
                    get_device()
                );
                return Err(e);
            }
            Ok(DeviceBuf { ptr, host: std::ptr::null_mut(), bytes, dev: get_device() })
        }

        /// Mapped pinned host memory; `ptr()` is device-visible. With UVA
        /// (64-bit Linux) the pointer is valid on every device.
        pub fn alloc_pinned(bytes: usize) -> Result<Self> {
            ensure_device();
            const MAPPED: u32 = 2; // cudaHostAllocMapped
            let mut host = std::ptr::null_mut();
            check_rt(unsafe { cudaHostAlloc(&mut host, bytes.max(1), MAPPED) }, "cudaHostAlloc")?;
            let mut dev = std::ptr::null_mut();
            check_rt(unsafe { cudaHostGetDevicePointer(&mut dev, host, 0) }, "cudaHostGetDevicePointer")?;
            Ok(DeviceBuf { ptr: dev, host, bytes, dev: -1 })
        }

        pub fn from_bytes(data: &[u8]) -> Result<Self> {
            let mut b = Self::alloc(data.len())?;
            b.write(0, data)?;
            Ok(b)
        }

        pub fn from_f32(data: &[f32]) -> Result<Self> {
            Self::from_bytes(as_bytes(data))
        }

        pub fn bytes(&self) -> usize {
            self.bytes
        }

        pub fn is_pinned(&self) -> bool {
            !self.host.is_null()
        }

        pub fn ptr(&self) -> *const c_void {
            self.ptr
        }

        pub fn ptr_mut(&mut self) -> *mut c_void {
            self.ptr
        }

        /// Device pointer at a byte offset (for slab arenas).
        pub fn ptr_at(&self, off: usize) -> *const c_void {
            debug_assert!(off <= self.bytes);
            unsafe { (self.ptr as *const u8).add(off) as *const c_void }
        }

        pub fn write(&mut self, off: usize, data: &[u8]) -> Result {
            assert!(off + data.len() <= self.bytes, "device write out of range");
            if !self.host.is_null() {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        data.as_ptr(),
                        (self.host as *mut u8).add(off),
                        data.len(),
                    )
                };
                return Ok(());
            }
            check_rt(
                unsafe {
                    cudaMemcpy(
                        (self.ptr as *mut u8).add(off) as *mut c_void,
                        data.as_ptr() as *const c_void,
                        data.len(),
                        H2D,
                    )
                },
                "cudaMemcpy h2d",
            )
        }

        pub fn read(&self, off: usize, out: &mut [u8]) -> Result {
            assert!(off + out.len() <= self.bytes, "device read out of range");
            check_rt(
                unsafe {
                    cudaMemcpy(
                        out.as_mut_ptr() as *mut c_void,
                        (self.ptr as *const u8).add(off) as *const c_void,
                        out.len(),
                        D2H,
                    )
                },
                "cudaMemcpy d2h",
            )
        }

        pub fn read_f32(&self, n: usize) -> Result<Vec<f32>> {
            let mut v = vec![0f32; n];
            self.read(0, as_bytes_mut(&mut v))?;
            Ok(v)
        }

        pub fn read_i32(&self, n: usize) -> Result<Vec<i32>> {
            let mut v = vec![0i32; n];
            self.read(0, as_bytes_mut(&mut v))?;
            Ok(v)
        }
    }

    impl Drop for DeviceBuf {
        fn drop(&mut self) {
            if self.host.is_null() {
                // free with the owning device current, restore after
                let cur = get_device();
                if self.dev >= 0 && self.dev != cur {
                    unsafe { cudaSetDevice(self.dev) };
                }
                unsafe { cudaFree(self.ptr) };
                if self.dev >= 0 && self.dev != cur {
                    unsafe { cudaSetDevice(cur) };
                }
            } else {
                unsafe { cudaFreeHost(self.host) };
            }
        }
    }

    /// Plain-function pinned allocator pair for injection into CUDA-free
    /// crates (fetch buffers that later feed cudaMemcpy). cudaHostAlloc
    /// costs milliseconds of page-pinning per call, so freed buffers are
    /// recycled through a size-keyed pool: at steady state (cache evicting
    /// as fast as it fills) no pinning syscalls happen at all. Returns
    /// null on failure so callers fall back to pageable memory.
    fn pinned_pool() -> &'static std::sync::Mutex<std::collections::HashMap<usize, Vec<usize>>> {
        static POOL: std::sync::OnceLock<
            std::sync::Mutex<std::collections::HashMap<usize, Vec<usize>>>,
        > = std::sync::OnceLock::new();
        POOL.get_or_init(Default::default)
    }

    pub fn pinned_alloc(bytes: usize) -> *mut u8 {
        ensure_device();
        if let Some(ptr) = pinned_pool()
            .lock()
            .unwrap()
            .get_mut(&bytes)
            .and_then(Vec::pop)
        {
            return ptr as *mut u8;
        }
        let mut host = std::ptr::null_mut();
        let rc = unsafe { cudaHostAlloc(&mut host, bytes.max(1), 0) };
        if rc == 0 {
            host as *mut u8
        } else {
            std::ptr::null_mut()
        }
    }

    pub fn pinned_free(ptr: *mut u8, bytes: usize) {
        pinned_pool()
            .lock()
            .unwrap()
            .entry(bytes)
            .or_default()
            .push(ptr as usize);
    }

    pub fn as_bytes<T: Copy>(v: &[T]) -> &[u8] {
        unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
    }

    fn as_bytes_mut<T: Copy>(v: &mut [T]) -> &mut [u8] {
        unsafe {
            std::slice::from_raw_parts_mut(v.as_mut_ptr() as *mut u8, std::mem::size_of_val(v))
        }
    }

    pub fn sync() -> Result {
        check_rt(unsafe { cudaDeviceSynchronize() }, "cudaDeviceSynchronize")
    }

    extern "C" {
        fn cudaStreamCreateWithFlags(s: *mut *mut c_void, flags: u32) -> i32;
        fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32, stream: *mut c_void) -> i32;
        fn cudaEventCreateWithFlags(e: *mut *mut c_void, flags: u32) -> i32;
        fn cudaEventRecord(e: *mut c_void, stream: *mut c_void) -> i32;
        fn cudaEventQuery(e: *mut c_void) -> i32;
        fn cudaStreamWaitEvent(stream: *mut c_void, e: *mut c_void, flags: u32) -> i32;
    }

    /// A side stream + event for best-effort background H2D staging.
    /// `copy_async` from PINNED sources overlaps default-stream kernels;
    /// `done()` polls without blocking.
    pub struct CopyStream {
        stream: *mut c_void,
        event: *mut c_void,
        gate: *mut c_void,
    }

    unsafe impl Send for CopyStream {}

    impl CopyStream {
        pub fn new() -> Result<CopyStream> {
            ensure_device();
            const NON_BLOCKING: u32 = 1;
            const DISABLE_TIMING: u32 = 2;
            let mut stream = std::ptr::null_mut();
            check_rt(unsafe { cudaStreamCreateWithFlags(&mut stream, NON_BLOCKING) }, "stream create")?;
            let mut event = std::ptr::null_mut();
            check_rt(unsafe { cudaEventCreateWithFlags(&mut event, DISABLE_TIMING) }, "event create")?;
            let mut gate = std::ptr::null_mut();
            check_rt(unsafe { cudaEventCreateWithFlags(&mut gate, DISABLE_TIMING) }, "gate create")?;
            Ok(CopyStream { stream, event, gate })
        }

        /// Make queued-after copies wait for all default-stream work
        /// submitted so far (the consumers of whatever the arena holds).
        pub fn gate_behind_default(&self) -> Result {
            check_rt(unsafe { cudaEventRecord(self.gate, std::ptr::null_mut()) }, "gate record")?;
            check_rt(unsafe { cudaStreamWaitEvent(self.stream, self.gate, 0) }, "gate wait")
        }

        /// Queue an async H2D copy of a whole pinned buffer into `dst` at
        /// `dst_off`. Record the event after the LAST copy of a batch.
        pub fn copy_from_pinned(&self, dst: &mut DeviceBuf, dst_off: usize, src: &DeviceBuf) -> Result {
            assert!(!src.host.is_null(), "source must be pinned");
            assert!(dst_off + src.bytes <= dst.bytes);
            check_rt(
                unsafe {
                    cudaMemcpyAsync(
                        (dst.ptr as *mut u8).add(dst_off) as *mut c_void,
                        src.host,
                        src.bytes,
                        H2D,
                        self.stream,
                    )
                },
                "async h2d",
            )
        }

        pub fn record(&self) -> Result {
            check_rt(unsafe { cudaEventRecord(self.event, self.stream) }, "event record")
        }

        /// True once every copy queued before the last `record` finished.
        pub fn done(&self) -> bool {
            unsafe { cudaEventQuery(self.event) == 0 }
        }
    }

    const D2D: i32 = 3;
    const MEMCPY_DEFAULT: i32 = 4; // UVA infers direction; works across devices

    /// Copy between buffers on ANY pair of devices (or pinned host).
    /// Blocking cudaMemcpy: legacy-stream ordered on the current device,
    /// so issue it with the producer's device current and the consumer
    /// device's later launches see the data.
    pub fn copy_across(dst: &mut DeviceBuf, src: &DeviceBuf, bytes: usize) -> Result {
        assert!(bytes <= dst.bytes() && bytes <= src.bytes());
        check_rt(
            unsafe { cudaMemcpy(dst.ptr_mut(), src.ptr(), bytes, MEMCPY_DEFAULT) },
            "cudaMemcpy across",
        )
    }

    /// Device-to-device copy between buffers (byte offsets).
    pub fn copy_d2d(dst: &mut DeviceBuf, dst_off: usize, src: &DeviceBuf, src_off: usize, bytes: usize) -> Result {
        assert!(dst_off + bytes <= dst.bytes() && src_off + bytes <= src.bytes());
        check_rt(
            unsafe {
                cudaMemcpy(
                    (dst.ptr_mut() as *mut u8).add(dst_off) as *mut c_void,
                    (src.ptr() as *const u8).add(src_off) as *const c_void,
                    bytes,
                    D2D,
                )
            },
            "cudaMemcpy d2d",
        )
    }

    pub fn embed_q8_0(out: &mut DeviceBuf, w: &DeviceBuf, tokens: &DeviceBuf, n_embd: u32, n_vocab: u32, n_tok: u32) -> Result {
        check(unsafe { pulsar_embed_q8_0(out.ptr_mut(), w.ptr(), tokens.ptr(), n_embd, n_vocab, n_tok) }, "embed_q8_0")
    }

    pub fn rms_norm(out: &mut DeviceBuf, x: &DeviceBuf, w: &DeviceBuf, n: u32, rows: u32, eps: f32) -> Result {
        check(unsafe { pulsar_rms_norm(out.ptr_mut(), x.ptr(), w.ptr(), n, rows, eps) }, "rms_norm")
    }

    /// In-place rms_norm (kernel reads each element before writing it).
    pub fn rms_norm_inplace(x: &mut DeviceBuf, w: &DeviceBuf, n: u32, rows: u32, eps: f32) -> Result {
        check(unsafe { pulsar_rms_norm(x.ptr_mut(), x.ptr(), w.ptr(), n, rows, eps) }, "rms_norm")
    }

    pub fn matmul_q8_0(out: &mut DeviceBuf, w: &DeviceBuf, x: &DeviceBuf, in_dim: u32, out_dim: u32, n_tok: u32) -> Result {
        check(unsafe { pulsar_q8_0_matmul(out.ptr_mut(), w.ptr(), x.ptr(), in_dim, out_dim, n_tok) }, "matmul_q8_0")
    }

    /// matmul_q8_0 with byte offsets into each buffer (deepseek4's
    /// grouped output projection launches one bank per offset triple).
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_q8_0_off(out: &mut DeviceBuf, out_off: usize, w: &DeviceBuf, w_off: usize, x: &DeviceBuf, x_off: usize, in_dim: u32, out_dim: u32, n_tok: u32) -> Result {
        debug_assert!(out_off < out.bytes() && w_off < w.bytes() && x_off < x.bytes());
        check(
            unsafe {
                pulsar_q8_0_matmul(
                    (out.ptr_mut() as *mut u8).add(out_off) as *mut c_void,
                    (w.ptr() as *const u8).add(w_off) as *const c_void,
                    (x.ptr() as *const u8).add(x_off) as *const c_void,
                    in_dim, out_dim, n_tok,
                )
            },
            "matmul_q8_0_off",
        )
    }

    /// Dense matmul over a K-quant weight matrix; `xq` holds q8_K-quantized
    /// activations (quantize_q8_k) - the lm-head path for K-quant ggufs.
    #[allow(clippy::too_many_arguments)]
    pub fn matmul_kq(out: &mut DeviceBuf, w: &DeviceBuf, xq: &DeviceBuf, in_dim: u32, out_dim: u32, n_tok: u32, row_bytes: u64, quant: u32) -> Result {
        check(unsafe { pulsar_matmul_kq(out.ptr_mut(), w.ptr(), xq.ptr(), in_dim, out_dim, n_tok, row_bytes, quant) }, "matmul_kq")
    }

    /// DSA indexer wrappers (GLM-5.2 lightning indexer).
    #[allow(clippy::too_many_arguments)]
    pub fn idx_rope0(x: &mut DeviceBuf, n_tok: u32, n_head: u32, head_dim: u32, rot_dim: u32, pos0: u32, r: &RopeCfg, ext_factor: f32, attn_factor: f32) -> Result {
        check(unsafe { pulsar_idx_rope0(x.ptr_mut(), n_tok, n_head, head_dim, rot_dim, pos0, r.n_ctx_orig, r.freq_base, r.freq_scale, ext_factor, attn_factor, r.beta_fast, r.beta_slow) }, "idx_rope0")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn idx_store_k(raw_k: &DeviceBuf, w: &DeviceBuf, b: &DeviceBuf, cache: &mut DeviceBuf, pos0: u32, n_tok: u32, cache_cap: u32, head_dim: u32, rot_dim: u32, eps: f32, r: &RopeCfg, ext_factor: f32, attn_factor: f32) -> Result {
        check(unsafe { pulsar_idx_store_k(raw_k.ptr(), w.ptr(), b.ptr(), cache.ptr_mut(), pos0, n_tok, cache_cap, head_dim, rot_dim, r.n_ctx_orig, eps, r.freq_base, r.freq_scale, ext_factor, attn_factor, r.beta_fast, r.beta_slow) }, "idx_store_k")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn idx_score_one(scores: &mut DeviceBuf, q: &DeviceBuf, weights: &DeviceBuf, cache: &DeviceBuf, n_rows: u32, n_head: u32, head_dim: u32, scale: f32) -> Result {
        check(unsafe { pulsar_idx_score_one(scores.ptr_mut(), q.ptr(), weights.ptr(), cache.ptr(), n_rows, n_head, head_dim, scale) }, "idx_score_one")
    }

    pub fn idx_topk(selected: &mut DeviceBuf, scores: &DeviceBuf, n_rows: u32, top_k: u32) -> Result {
        check(unsafe { pulsar_idx_topk(selected.ptr_mut(), scores.ptr(), n_rows, top_k) }, "idx_topk")
    }

    /// Per-token top-k over a batch score matrix: row list for token t
    /// lands at selected[t*top_k..]. One bitonic launch per token.
    pub fn idx_topk_batch(selected: &mut DeviceBuf, scores: &DeviceBuf, n_rows: u32, n_tok: u32, top_k: u32) -> Result {
        for t in 0..n_tok as usize {
            let sel = unsafe { (selected.ptr_mut() as *mut u8).add(t * top_k as usize * 4) };
            let sc = unsafe { (scores.ptr() as *const u8).add(t * n_rows as usize * 4) };
            check(unsafe { pulsar_idx_topk(sel as *mut c_void, sc as *const c_void, n_rows, top_k) }, "idx_topk_batch")?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    pub fn idx_scores_batch(scores: &mut DeviceBuf, q: &DeviceBuf, weights: &DeviceBuf, cache: &DeviceBuf, q16: Option<&mut DeviceBuf>, n_rows: u32, n_tok: u32, pos0: u32, n_head: u32, head_dim: u32, scale: f32) -> Result {
        check(unsafe { pulsar_idx_scores_batch(scores.ptr_mut(), q.ptr(), weights.ptr(), cache.ptr(), q16.map_or(std::ptr::null_mut(), |b| b.ptr_mut()), n_rows, n_tok, pos0, n_head, head_dim, scale) }, "idx_scores_batch")
    }

    pub fn matmul_f32(out: &mut DeviceBuf, w: &DeviceBuf, x: &DeviceBuf, in_dim: u32, out_dim: u32, n_tok: u32) -> Result {
        check(unsafe { pulsar_matmul_f32(out.ptr_mut(), w.ptr(), x.ptr(), in_dim, out_dim, n_tok) }, "matmul_f32")
    }

    pub fn scale(x: &mut DeviceBuf, n: u32, c: f32) -> Result {
        check(unsafe { pulsar_scale(x.ptr_mut(), n, c) }, "scale")
    }

    /// Fill columns keep..row_w of each row with v (inkling padded-vocab
    /// logit poison).
    pub fn fill_row_tail(x: &mut DeviceBuf, rows: u32, row_w: u32, keep: u32, v: f32) -> Result {
        check(unsafe { pulsar_fill_row_tail(x.ptr_mut(), rows, row_w, keep, v) }, "fill_row_tail")
    }

    pub fn softcap(x: &mut DeviceBuf, n: u32, cap: f32) -> Result {
        check(unsafe { pulsar_softcap(x.ptr_mut(), n, cap) }, "softcap")
    }

    pub fn router_scale_selected(w: &mut DeviceBuf, sel: &DeviceBuf, scale: &DeviceBuf, n: u32, n_expert: u32) -> Result {
        check(unsafe { pulsar_router_scale_selected(w.ptr_mut(), sel.ptr(), scale.ptr(), n, n_expert) }, "router_scale_selected")
    }

    /// act_op: 0 = silu (swiglu), 1 = gelu tanh (Gemma)
    pub fn swiglu(out: &mut DeviceBuf, gate: &DeviceBuf, up: &DeviceBuf, n: u32, clamp: f32, weight: f32, act_op: u32) -> Result {
        check(unsafe { pulsar_swiglu(out.ptr_mut(), gate.ptr(), up.ptr(), n, clamp, weight, act_op) }, "swiglu")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_pair_swiglu_grouped(mid: &mut DeviceBuf, gptrs: &DeviceBuf, starts: &DeviceBuf, pairs: &DeviceBuf, weights: &DeviceBuf, xq: &DeviceBuf, in_dim: u32, mid_dim: u32, n_used: u32, n_group: u32, row_bytes: u64, quant: u32, act_op: u32) -> Result {
        check(unsafe { pulsar_moe_pair_swiglu_grouped(mid.ptr_mut(), gptrs.ptr(), starts.ptr(), pairs.ptr(), weights.ptr(), xq.ptr(), in_dim, mid_dim, n_used, n_group, row_bytes, quant, act_op) }, "moe_pair_swiglu_grouped")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_down_grouped(partial: &mut DeviceBuf, gptrs: &DeviceBuf, starts: &DeviceBuf, pairs: &DeviceBuf, midq: &DeviceBuf, mid_dim: u32, out_dim: u32, n_used: u32, n_group: u32, row_bytes: u64, quant: u32) -> Result {
        check(unsafe { pulsar_moe_down_grouped(partial.ptr_mut(), gptrs.ptr(), starts.ptr(), pairs.ptr(), midq.ptr(), mid_dim, out_dim, n_used, n_group, row_bytes, quant) }, "moe_down_grouped")
    }

    pub fn zero(buf: &mut DeviceBuf, bytes: usize) -> Result {
        check_rt(unsafe { cudaMemset(buf.ptr_mut(), 0, bytes) }, "cudaMemset")
    }

    pub fn moe_slot_sum(out: &mut DeviceBuf, partial: &DeviceBuf, out_dim: u32, n_used: u32, n_tok: u32) -> Result {
        check(unsafe { pulsar_moe_slot_sum(out.ptr_mut(), partial.ptr(), out_dim, n_used, n_tok) }, "moe_slot_sum")
    }

    pub fn add(out: &mut DeviceBuf, a: &DeviceBuf, b: &DeviceBuf, n: u32) -> Result {
        check(unsafe { pulsar_add(out.ptr_mut(), a.ptr(), b.ptr(), n) }, "add")
    }

    /// out += b (elementwise kernel; aliasing out as input is safe).
    pub fn add_assign(out: &mut DeviceBuf, b: &DeviceBuf, n: u32) -> Result {
        let o = out.ptr_mut();
        check(unsafe { pulsar_add(o, o as *const c_void, b.ptr(), n) }, "add_assign")
    }

    /// mode: 0 = sigmoid+bias (Hy3/GLM/M3), 1 = softmax (qwen3moe/gemma4),
    /// 2 = inkling sink (n_shexp shared experts append as slots k..k+n_shexp
    /// with logsigmoid-softmax weights; selected/weights hold k+n_shexp).
    #[allow(clippy::too_many_arguments)]
    pub fn router_select(selected: &mut DeviceBuf, weights: &mut DeviceBuf, logits: &DeviceBuf, bias: &DeviceBuf, n_expert: u32, k_used: u32, weight_scale: f32, n_tok: u32, mode: u32, n_shexp: u32) -> Result {
        check(
            unsafe {
                pulsar_router_select(selected.ptr_mut(), weights.ptr_mut(), logits.ptr(), bias.ptr(), n_expert, k_used, weight_scale, n_tok, mode, n_shexp)
            },
            "router_select",
        )
    }

    /// GGML q8_K block: f32 scale + 256 int8 + 16 i16 block sums.
    pub const Q8_K_BLOCK_BYTES: usize = 292;
    pub const Q8_K_BLOCK_ELEMS: usize = 256;

    /// Quantize f32 rows to q8_K (the activation side of the expert dots).
    pub fn quantize_q8_k(out: &mut DeviceBuf, x: &DeviceBuf, in_dim: u32, n_rows: u32) -> Result {
        check(unsafe { pulsar_quantize_q8_K(out.ptr_mut(), x.ptr(), in_dim, n_rows) }, "quantize_q8_k")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_pair_swiglu(mid: &mut DeviceBuf, ptrs: &DeviceBuf, weights: &DeviceBuf, x: &DeviceBuf, in_dim: u32, mid_dim: u32, n_used: u32, n_tok: u32, row_bytes: u64, quant: u32, act_op: u32) -> Result {
        check(
            unsafe {
                pulsar_moe_pair_swiglu(mid.ptr_mut(), ptrs.ptr(), weights.ptr(), x.ptr(), in_dim, mid_dim, n_used, n_tok, row_bytes, quant, act_op)
            },
            "moe_pair_swiglu",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn moe_down(out: &mut DeviceBuf, ptrs: &DeviceBuf, mid: &DeviceBuf, mid_dim: u32, out_dim: u32, n_used: u32, n_tok: u32, row_bytes: u64, quant: u32) -> Result {
        check(
            unsafe {
                pulsar_moe_down(out.ptr_mut(), ptrs.ptr(), mid.ptr(), mid_dim, out_dim, n_used, n_tok, row_bytes, quant)
            },
            "moe_down",
        )
    }

    pub fn gqa_head_rms_norm(x: &mut DeviceBuf, w: Option<&DeviceBuf>, rows: u32, head_dim: u32, eps: f32) -> Result {
        check(unsafe { pulsar_gqa_head_rms_norm(x.ptr_mut(), w.map_or(std::ptr::null(), |b| b.ptr()), rows, head_dim, eps) }, "gqa_head_rms_norm")
    }

    pub fn gqa_rope(x: &mut DeviceBuf, n_tok: u32, n_head: u32, head_dim: u32, rot_dim: u32, pos0: u32, theta: f32, factors: Option<&DeviceBuf>) -> Result {
        check(unsafe { pulsar_gqa_rope(x.ptr_mut(), n_tok, n_head, head_dim, rot_dim, pos0, theta, factors.map_or(std::ptr::null(), |b| b.ptr())) }, "gqa_rope")
    }

    /// KV cache storage format: 0 = f32 (exact), 1 = fp8 e4m3 + per-row
    /// scale (row stride head_dim+4 bytes, opt-in via PULSAR_KV=fp8).
    #[allow(clippy::too_many_arguments)]
    pub fn gqa_kv_append(cache: &mut DeviceBuf, kv: &DeviceBuf, n_tok: u32, n_kv_head: u32, head_dim: u32, cap: u32, pos0: u32, kvq: u32) -> Result {
        check(unsafe { pulsar_gqa_kv_append(cache.ptr_mut(), kv.ptr(), n_tok, n_kv_head, head_dim, cap, pos0, kvq) }, "gqa_kv_append")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gqa_attention(out: &mut DeviceBuf, q: &DeviceBuf, k_cache: &DeviceBuf, v_cache: &DeviceBuf, n_tok: u32, n_head: u32, n_kv_head: u32, head_dim: u32, cap: u32, pos0: u32, scale: f32, window: u32) -> Result {
        gqa_attention_rel(out, q, k_cache, v_cache, n_tok, n_head, n_kv_head, head_dim, cap, pos0, scale, window, None, 0, 0)
    }

    /// GQA attention with an optional inkling relative-position bias:
    /// rel is [n_tok][n_head][rel_extent], score(i,j) += rel[i-j] in-band.
    #[allow(clippy::too_many_arguments)]
    pub fn gqa_attention_rel(out: &mut DeviceBuf, q: &DeviceBuf, k_cache: &DeviceBuf, v_cache: &DeviceBuf, n_tok: u32, n_head: u32, n_kv_head: u32, head_dim: u32, cap: u32, pos0: u32, scale: f32, window: u32, rel: Option<&DeviceBuf>, rel_extent: u32, kvq: u32) -> Result {
        check(
            unsafe {
                pulsar_gqa_attention(out.ptr_mut(), q.ptr(), k_cache.ptr(), v_cache.ptr(), n_tok, n_head, n_kv_head, head_dim, cap, pos0, scale, window, rel.map_or(std::ptr::null(), |r| r.ptr()), rel_extent, kvq)
            },
            "gqa_attention",
        )
    }

    /// Inkling shortconv: out = x + causal depthwise conv over the last K
    /// inputs; state [w][K-1] rolls forward (zero it at pos 0). out != x.
    pub fn sconv(out: &mut DeviceBuf, x: &DeviceBuf, kern: &DeviceBuf, state: &mut DeviceBuf, n_tok: u32, w: u32, k: u32) -> Result {
        check(
            unsafe { pulsar_sconv(out.ptr_mut(), x.ptr(), kern.ptr(), state.ptr_mut(), n_tok, w, k) },
            "sconv",
        )
    }

    pub fn gqa_selftest() -> bool {
        unsafe { pulsar_gqa_selftest() != 0 }
    }

    pub fn idx_selftest() -> bool {
        unsafe { pulsar_idx_selftest() != 0 }
    }

    pub fn sconv_selftest() -> bool {
        unsafe { pulsar_sconv_selftest() != 0 }
    }

    pub fn q8_0_matmul_selftest() -> bool {
        unsafe { pulsar_q8_0_matmul_selftest() != 0 }
    }

    pub fn router_selftest() -> bool {
        unsafe { pulsar_router_selftest() != 0 }
    }

    pub fn moe_selftest() -> bool {
        unsafe { pulsar_moe_selftest() != 0 }
    }

    pub fn glue_selftest() -> bool {
        unsafe { pulsar_glue_selftest() != 0 }
    }

    pub fn mla_selftest() -> bool {
        unsafe { pulsar_mla_selftest() != 0 }
    }

    pub fn mla_rope_tail(x: &mut DeviceBuf, n_tok: u32, n_head: u32, head_dim: u32, rot_dim: u32, pos0: u32, r: &RopeCfg) -> Result {
        check(
            unsafe {
                pulsar_mla_rope_tail(x.ptr_mut(), n_tok, n_head, head_dim, rot_dim, pos0, r.n_ctx_orig, r.freq_base, r.freq_scale, r.ext_factor, r.attn_factor, r.beta_fast, r.beta_slow)
            },
            "mla_rope_tail",
        )
    }

    /// deepseek4 rope tail: mla_rope_tail plus inverse mode (heads get
    /// un-rotated before the grouped output projection).
    #[allow(clippy::too_many_arguments)]
    pub fn dsv4_rope_tail(x: &mut DeviceBuf, n_tok: u32, n_head: u32, head_dim: u32, rot_dim: u32, pos0: u32, r: &RopeCfg, inverse: bool) -> Result {
        check(
            unsafe {
                pulsar_dsv4_rope_tail(x.ptr_mut(), n_tok, n_head, head_dim, rot_dim, pos0, r.n_ctx_orig, r.freq_base, r.freq_scale, r.ext_factor, r.attn_factor, r.beta_fast, r.beta_slow, inverse as u32)
            },
            "dsv4_rope_tail",
        )
    }

    /// deepseek4 hyper-connection combine: out[j] = blk_c[j]*block +
    /// sum_src st_c[j*n_hc+src]*streams[src]. Serves the pre reduce
    /// (n_out 1), post expand (n_out 4) and the output-head merge.
    #[allow(clippy::too_many_arguments)]
    pub fn dsv4_hc_mix(out: &mut DeviceBuf, streams: &DeviceBuf, block: Option<&DeviceBuf>, st_c: &[f32], blk_c: Option<&[f32]>, n_embd: u32, n_hc: u32, n_out: u32) -> Result {
        debug_assert_eq!(st_c.len(), (n_out * n_hc) as usize);
        check(
            unsafe {
                pulsar_dsv4_hc_mix(out.ptr_mut(), streams.ptr(), block.map_or(std::ptr::null(), |b| b.ptr()), st_c.as_ptr(), blk_c.map_or(std::ptr::null(), |b| b.as_ptr()), n_embd, n_hc, n_out)
            },
            "dsv4_hc_mix",
        )
    }

    /// deepseek4 decode attention: sinks + raw SWA ring + compressed
    /// rows (K == V), optional per-comp-row visibility mask.
    #[allow(clippy::too_many_arguments)]
    pub fn dsv4_attention(out: &mut DeviceBuf, q: &DeviceBuf, raw: &DeviceBuf, n_raw: u32, comp: Option<&DeviceBuf>, n_comp: u32, allowed: Option<&DeviceBuf>, sinks: &DeviceBuf, n_head: u32, head_dim: u32, scale: f32) -> Result {
        check(
            unsafe {
                pulsar_dsv4_attention(out.ptr_mut(), q.ptr(), raw.ptr(), n_raw, comp.map_or(std::ptr::null(), |b| b.ptr()), n_comp, allowed.map_or(std::ptr::null(), |b| b.ptr()), sinks.ptr(), n_head, head_dim, scale)
            },
            "dsv4_attention",
        )
    }

    /// ds4's fp8 e4m3 round-trip on the non-rope dims of each row
    /// (64-wide blocks, power-of-2 scale, clamp +-448). In place.
    pub fn dsv4_fp8_sim(x: &mut DeviceBuf, n_rows: u32, head_dim: u32, n_rot: u32) -> Result {
        check(unsafe { pulsar_dsv4_fp8_sim(x.ptr_mut(), n_rows, head_dim, n_rot) }, "dsv4_fp8_sim")
    }

    /// Round f32 values through f16 storage in place (V4 cache rows).
    pub fn dsv4_f16_round(x: &mut DeviceBuf, n: u32) -> Result {
        check(unsafe { pulsar_dsv4_f16_round(x.ptr_mut(), n) }, "dsv4_f16_round")
    }

    pub fn dsv4_selftest() -> bool {
        unsafe { pulsar_dsv4_selftest() != 0 }
    }

    pub fn mla_kv_lora_rms_norm(out: &mut DeviceBuf, kv_raw: &DeviceBuf, w: &DeviceBuf, n_tok: u32, kv_raw_dim: u32, kv_lora_dim: u32, eps: f32) -> Result {
        check(
            unsafe {
                pulsar_mla_kv_lora_rms_norm(out.ptr_mut(), kv_raw.ptr(), w.ptr(), n_tok, kv_raw_dim, kv_lora_dim, eps)
            },
            "mla_kv_lora_rms_norm",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mla_store_compact_kv(kv_lora_cache: &mut DeviceBuf, k_rope_cache: &mut DeviceBuf, kv_norm: &DeviceBuf, kv_raw: &DeviceBuf, pos0: u32, n_tok: u32, cache_cap: u32, kv_raw_dim: u32, kv_lora_dim: u32, qk_rope: u32) -> Result {
        check(
            unsafe {
                pulsar_mla_store_compact_kv(kv_lora_cache.ptr_mut(), k_rope_cache.ptr_mut(), kv_norm.ptr(), kv_raw.ptr(), pos0, n_tok, cache_cap, kv_raw_dim, kv_lora_dim, qk_rope)
            },
            "mla_store_compact_kv",
        )
    }

    pub fn mla_fill_selected_range(selected: &mut DeviceBuf, n_tok: u32, pos0: u32, n_selected: u32, pad_row: u32) -> Result {
        check(
            unsafe { pulsar_mla_fill_selected_range(selected.ptr_mut(), n_tok, pos0, n_selected, pad_row) },
            "mla_fill_selected_range",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mla_qk_lowrank(qk_low: &mut DeviceBuf, q: &DeviceBuf, k_b: &DeviceBuf, n_tok: u32, n_head: u32, kv_lora_dim: u32, qk_nope: u32, qk_dim: u32) -> Result {
        check(
            unsafe {
                pulsar_mla_qk_lowrank(qk_low.ptr_mut(), q.ptr(), k_b.ptr(), n_tok, n_head, kv_lora_dim, qk_nope, qk_dim)
            },
            "mla_qk_lowrank",
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mla_attention(heads: &mut DeviceBuf, q: &DeviceBuf, qk_low: &DeviceBuf, kv_lora_cache: &DeviceBuf, k_rope_cache: &DeviceBuf, v_b: &DeviceBuf, selected: &DeviceBuf, n_tok: u32, n_selected: u32, cache_cap: u32, n_head: u32, kv_lora_dim: u32, qk_nope: u32, qk_rope: u32, value_dim: u32, r: &RopeCfg) -> Result {
        check(
            unsafe {
                pulsar_mla_attention(heads.ptr_mut(), q.ptr(), qk_low.ptr(), kv_lora_cache.ptr(), k_rope_cache.ptr(), v_b.ptr(), selected.ptr(), n_tok, n_selected, cache_cap, n_head, kv_lora_dim, qk_nope, qk_rope, value_dim, r.n_ctx_orig, r.freq_base, r.freq_scale, r.ext_factor, r.attn_factor, r.beta_fast, r.beta_slow, r.kq_mult)
            },
            "mla_attention",
        )
    }
}

#[cfg(target_os = "linux")]
pub use real::*;

#[cfg(test)]
#[cfg(target_os = "linux")]
mod tests {
    /// GPU-required; run explicitly: cargo test -p kernels -- --ignored
    #[test]
    #[ignore = "requires a CUDA device"]
    fn gqa_kernels_match_cpu_reference() {
        assert!(super::gqa_selftest());
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn q8_0_matmul_matches_cpu_reference() {
        assert!(super::q8_0_matmul_selftest());
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn router_select_matches_cpu_reference() {
        assert!(super::router_selftest());
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn moe_kernels_match_cpu_reference() {
        assert!(super::moe_selftest());
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn sconv_matches_cpu_reference() {
        assert!(super::sconv_selftest());
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn glue_kernels_match_cpu_reference() {
        assert!(super::glue_selftest());
    }

    #[test]
    #[ignore = "requires a CUDA device"]
    fn mla_kernels_match_cpu_reference() {
        assert!(super::mla_selftest());
    }

    /// End-to-end DeviceBuf + rust-side wrapper smoke test: y = a + b.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn device_buf_roundtrip_and_add() {
        let a: Vec<f32> = (0..1024).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..1024).map(|i| 2.0 * i as f32).collect();
        let da = super::DeviceBuf::from_f32(&a).unwrap();
        let db = super::DeviceBuf::from_f32(&b).unwrap();
        let mut dy = super::DeviceBuf::alloc(1024 * 4).unwrap();
        super::add(&mut dy, &da, &db, 1024).unwrap();
        super::sync().unwrap();
        let y = dy.read_f32(1024).unwrap();
        for i in 0..1024 {
            assert_eq!(y[i], 3.0 * i as f32);
        }
    }
}
