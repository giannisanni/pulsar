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
        /// Per-expert f32 bias vectors, null when the architecture has
        /// none. gpt-oss is the only shipped model that carries them; every
        /// other MoE here leaves these null and the kernels skip the add.
        /// gate/up are mid_dim long, down is out_dim long.
        pub gate_b: *const c_void,
        pub up_b: *const c_void,
        pub down_b: *const c_void,
    }

    unsafe impl Send for ExpertPtrs {}

    impl ExpertPtrs {
        pub const NULL: ExpertPtrs = ExpertPtrs {
            gate: std::ptr::null(),
            up: std::ptr::null(),
            down: std::ptr::null(),
            gate_b: std::ptr::null(),
            up_b: std::ptr::null(),
            down_b: std::ptr::null(),
        };
    }

    /// OR into  to permit W4A4 at that call site (NVFP4 only).
    /// The calibrated recipe applies fp4 activations to the FFN and the
    /// full-attention projections, never to linear_attn/GDN, lm_head or
    /// the MTP block.
    pub const QUANT_A4_OK: u32 = 0x8000_0000;
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
    pub const QUANT_MXFP4: u32 = 12;
    pub const QUANT_IQ4_NL: u32 = 13;
    pub const QUANT_IQ3_S: u32 = 14;
    pub const QUANT_IQ2_S: u32 = 15;
    pub const QUANT_IQ1_S: u32 = 16;
    pub const QUANT_NVFP4: u32 = 17;

    const H2D: i32 = 1;
    const D2H: i32 = 2;

    extern "C" {
        fn cudaSetDevice(dev: i32) -> i32;
        fn cudaGetDevice(dev: *mut i32) -> i32;
        fn cudaGetDeviceCount(count: *mut i32) -> i32;
        fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> i32;
        fn cudaDeviceGetAttribute(val: *mut i32, attr: i32, dev: i32) -> i32;
        fn cudaDeviceCanAccessPeer(can: *mut i32, dev: i32, peer: i32) -> i32;
        fn cudaDeviceEnablePeerAccess(peer: i32, flags: u32) -> i32;
        fn cudaMalloc(ptr: *mut *mut c_void, bytes: usize) -> i32;
        fn pulsar_pack_bf16(out: *mut c_void, inp: *const c_void, n: u32) -> i32;
        fn pulsar_unpack_bf16(out: *mut c_void, inp: *const c_void, n: u32) -> i32;
        fn cudaFree(ptr: *mut c_void) -> i32;
        fn cudaHostAlloc(ptr: *mut *mut c_void, bytes: usize, flags: u32) -> i32;
        fn cudaFreeHost(ptr: *mut c_void) -> i32;
        fn cudaHostGetDevicePointer(dev: *mut *mut c_void, host: *mut c_void, flags: u32) -> i32;
        fn cudaMemcpy(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32) -> i32;
        fn cudaMemset(ptr: *mut c_void, value: i32, bytes: usize) -> i32;
        fn cudaDeviceSynchronize() -> i32;

        fn pulsar_embed_q8_0(out: *mut c_void, w: *const c_void, tokens: *const c_void, n_embd: u32, n_vocab: u32, n_tok: u32) -> i32;
        fn pulsar_dspark_markov_argmax(logits: *const c_void, w2: *const c_void, state: *const c_void, vocab: u32, rank: u32, scratch: *mut c_void, out: *mut c_void) -> i32;
        fn pulsar_rms_norm(out: *mut c_void, x: *const c_void, w: *const c_void, n: u32, rows: u32, eps: f32) -> i32;
        fn pulsar_q8_0_matmul(out: *mut c_void, w: *const c_void, x: *const c_void, in_dim: u32, out_dim: u32, n_tok: u32) -> i32;
        fn pulsar_preq_scratch_reserve(bytes: u64) -> i32;
        fn pulsar_set_lane(lane: i32);
        fn pulsar_lane_prewarm();
        fn pulsar_q8_0_matmul_banked(out: *mut c_void, w: *const c_void, x: *const c_void, in_dim: u32, out_dim: u32, n_bank: u32, n_tok: u32) -> i32;
        fn pulsar_matmul_f32(out: *mut c_void, w: *const c_void, x: *const c_void, in_dim: u32, out_dim: u32, n_tok: u32) -> i32;
        fn pulsar_matmul_kq(out: *mut c_void, w: *const c_void, xq: *const c_void, in_dim: u32, out_dim: u32, n_tok: u32, row_bytes: u64, quant: u32) -> i32;
        fn pulsar_bw_read(buf: *mut c_void, bytes: u64, sink: *mut c_void) -> i32;
        fn pulsar_idx_rope0(x: *mut c_void, n_tok: u32, n_head: u32, head_dim: u32, rot_dim: u32, pos0: u32, n_ctx_orig: u32, freq_base: f32, freq_scale: f32, ext_factor: f32, attn_factor: f32, beta_fast: f32, beta_slow: f32) -> i32;
        fn pulsar_idx_store_k(raw_k: *const c_void, w: *const c_void, b: *const c_void, cache: *mut c_void, pos0: u32, n_tok: u32, cache_cap: u32, head_dim: u32, rot_dim: u32, n_ctx_orig: u32, eps: f32, freq_base: f32, freq_scale: f32, ext_factor: f32, attn_factor: f32, beta_fast: f32, beta_slow: f32, fp8: u32) -> i32;
        fn pulsar_idx_score_one(scores: *mut c_void, q: *const c_void, weights: *const c_void, cache: *const c_void, n_rows: u32, n_head: u32, head_dim: u32, scale: f32, fp8: u32) -> i32;
        fn pulsar_idx_topk(selected: *mut c_void, scores: *const c_void, n_rows: u32, top_k: u32) -> i32;
        fn pulsar_idx_scores_batch(scores: *mut c_void, q: *const c_void, weights: *const c_void, cache: *const c_void, q16: *mut c_void, n_rows: u32, n_tokens: u32, pos0: u32, n_head: u32, head_dim: u32, scale: f32, fp8: u32) -> i32;
        fn pulsar_idx_selftest() -> i32;
        fn pulsar_swiglu(out: *mut c_void, gate: *const c_void, up: *const c_void, n: u32, clamp: f32, weight: f32, act_op: u32) -> i32;
        fn pulsar_scale(x: *mut c_void, n: u32, c: f32) -> i32;
        fn pulsar_fill_row_tail(x: *mut c_void, rows: u32, row_w: u32, keep: u32, v: f32) -> i32;
        fn pulsar_softcap(x: *mut c_void, n: u32, cap: f32) -> i32;
        fn pulsar_router_scale_selected(w: *mut c_void, sel: *const c_void, scale: *const c_void, n: u32, n_expert: u32) -> i32;
        fn pulsar_add(out: *mut c_void, a: *const c_void, b: *const c_void, n: u32) -> i32;
        fn pulsar_router_select(selected: *mut c_void, weights: *mut c_void, logits: *const c_void, bias: *const c_void, n_expert: u32, k_used: u32, weight_scale: f32, n_tok: u32, softmax_mode: u32, n_shexp: u32) -> i32;
        fn pulsar_quantize_q8_K(out: *mut c_void, x: *const c_void, in_dim: u32, n_rows: u32) -> i32;
        fn pulsar_quantize_nvfp4(out: *mut c_void, x: *const c_void, in_dim: u32, n_rows: u32) -> i32;
        fn pulsar_nvfp4_from_q8k(out: *mut c_void, xq: *const c_void, in_dim: u32, n_tok: u32) -> i32;
        fn pulsar_matmul_nvfp4_a4(out: *mut c_void, w: *const c_void, xq: *const c_void, in_dim: u32, out_dim: u32, n_tok: u32, row_bytes: u64) -> i32;
        fn pulsar_cc_major() -> i32;
        fn pulsar_moe_pair_swiglu(mid: *mut c_void, ptrs: *const c_void, weights: *const c_void, x: *const c_void, in_dim: u32, mid_dim: u32, n_used: u32, n_tok: u32, row_bytes: u64, quant: u32, act_op: u32) -> i32;
        fn pulsar_moe_down(out: *mut c_void, ptrs: *const c_void, mid: *const c_void, mid_dim: u32, out_dim: u32, n_used: u32, n_tok: u32, row_bytes: u64, quant: u32) -> i32;
        fn pulsar_moe_pair_swiglu_grouped(mid: *mut c_void, gptrs: *const c_void, starts: *const c_void, pairs: *const c_void, weights: *const c_void, xq: *const c_void, in_dim: u32, mid_dim: u32, n_used: u32, n_group: u32, row_bytes: u64, quant: u32, act_op: u32) -> i32;
        fn pulsar_moe_down_grouped(partial: *mut c_void, gptrs: *const c_void, starts: *const c_void, pairs: *const c_void, midq: *const c_void, mid_dim: u32, out_dim: u32, n_used: u32, n_group: u32, row_bytes: u64, quant: u32) -> i32;
        fn pulsar_moe_slot_sum(out: *mut c_void, partial: *const c_void, out_dim: u32, n_used: u32, n_tok: u32) -> i32;
        fn pulsar_moe_down_bias(out: *mut c_void, ptrs: *const c_void, weights: *const c_void, out_dim: u32, n_used: u32, n_tok: u32) -> i32;
        fn pulsar_add_bias_rows(x: *mut c_void, bias: *const c_void, dim: u32, rows: u32) -> i32;
        fn pulsar_gqa_head_rms_norm(x: *mut c_void, w: *const c_void, rows: u32, head_dim: u32, eps: f32) -> i32;
        fn pulsar_gqa_rope_dev(x: *mut c_void, n_tok: u32, n_head: u32, head_dim: u32, rot_dim: u32, pos_dev: *const c_void, theta: f32) -> i32;
        fn pulsar_gqa_kv_append_dev(cache: *mut c_void, kv: *const c_void, n_tok: u32, n_kv_head: u32, head_dim: u32, cap: u32, pos_dev: *const c_void) -> i32;
        fn pulsar_gqa_attention_dev(out: *mut c_void, q: *const c_void, k: *const c_void, v: *const c_void, n_tok: u32, n_head: u32, n_kv_head: u32, head_dim: u32, cap: u32, pos_dev: *const c_void, scale: f32) -> i32;
        fn pulsar_set_u32(dst: *mut c_void, v: u32) -> i32;
        fn pulsar_argmax_rows(out: *mut c_void, x: *const c_void, n: u32, rows: u32) -> i32;
         fn pulsar_gqa_rope(x: *mut c_void, n_tok: u32, n_head: u32, head_dim: u32, rot_dim: u32, pos0: u32, theta: f32, factors: *const c_void) -> i32;
        fn pulsar_gqa_kv_append(cache: *mut c_void, kv: *const c_void, n_tok: u32, n_kv_head: u32, head_dim: u32, cap: u32, pos0: u32, kvq: u32) -> i32;
        fn pulsar_gqa_attention(out: *mut c_void, q: *const c_void, k_cache: *const c_void, v_cache: *const c_void, n_tok: u32, n_head: u32, n_kv_head: u32, head_dim: u32, cap: u32, pos0: u32, scale: f32, window: u32, rel: *const c_void, rel_extent: u32, kvq: u32, sinks: *const c_void) -> i32;

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
        fn pulsar_dsv4_attention(out: *mut c_void, q: *const c_void, raw: *const c_void, n_raw: u32, comp: *const c_void, n_comp: u32, allowed: *const c_void, sinks: *const c_void, n_head: u32, head_dim: u32, scale: f32, kvq: u32, turbo: u32, pi: *const c_void) -> i32;
        fn pulsar_dsv4_fp8_sim(x: *mut c_void, n_rows: u32, head_dim: u32, n_rot: u32) -> i32;
        fn pulsar_dsv4_f16_round(x: *mut c_void, n: u32) -> i32;
        fn pulsar_dsv4_kv_store(dst: *mut c_void, src: *const c_void, head_dim: u32, kvq: u32, turbo: u32, pi: *const c_void) -> i32;
        fn pulsar_dsv4_sinkhorn(coef: *mut c_void, mix: *const c_void, scale: *const c_void, base: *const c_void, n_hc: u32, iters: u32, eps: f32, n_tok: u32) -> i32;
        fn pulsar_dsv4_hc_mix_dev(out: *mut c_void, streams: *const c_void, block: *const c_void, coef: *const c_void, st_off: u32, blk_off: i32, n_embd: u32, n_hc: u32, n_out: u32, n_tok: u32) -> i32;
        fn pulsar_dsv4_comp_step(state_kv: *mut c_void, state_sc: *mut c_void, cache_row: *mut c_void, kv_cur: *const c_void, sc_cur: *const c_void, ape: *const c_void, norm: *const c_void, width: u32, head_dim: u32, ratio: u32, pos: u32, emit: u32, is_idx: u32, rms_eps: f32, n_rot: u32, n_ctx_orig: u32, freq_base: f32, freq_scale: f32, ext_factor: f32, attn_factor: f32, kvq: u32, turbo: u32, pi: *const c_void) -> i32;
        fn pulsar_dsv4_selftest() -> i32;
        fn pulsar_qwen35_conv_step(out: *mut c_void, x: *const c_void, kern: *const c_void, state: *mut c_void, n_chan: u32, k: u32) -> i32;
        fn pulsar_qwen35_l2_norm(x: *mut c_void, rows: u32, dim: u32, eps: f32) -> i32;
        fn pulsar_qwen35_gdn_step(out: *mut c_void, state: *mut c_void, q: *const c_void, k: *const c_void, v: *const c_void, g: *const c_void, beta: *const c_void, h_v: u32, h_k: u32, dim: u32) -> i32;
        fn pulsar_qwen35_split_gate(q: *mut c_void, gate: *mut c_void, fused: *const c_void, n_head: u32, dim: u32) -> i32;
        fn pulsar_qwen35_sigmoid_gate(x: *mut c_void, gate: *const c_void, n: u32) -> i32;
        fn pulsar_laguna_head_gate(x: *mut c_void, gate: *const c_void, n_tok: u32, n_head: u32, head_dim: u32) -> i32;
        #[allow(clippy::too_many_arguments)]
        fn pulsar_rope_yarn_partial(x: *mut c_void, n_tok: u32, n_head: u32, head_dim: u32, rot_dim: u32, pos0: u32, freq_base: f32, freq_scale: f32, ext_factor: f32, attn_factor: f32, beta_fast: f32, beta_slow: f32, n_ctx_orig: u32) -> i32;
        fn pulsar_qwen35_conv_batch(out: *mut c_void, x: *const c_void, kern: *const c_void, state: *mut c_void, n_chan: u32, k: u32, n_tok: u32) -> i32;
    fn pulsar_q4e_hc_norm(y: *mut c_void, x: *const c_void, gamma: *const c_void, n_embd: u32, n_hc: u32, rows: u32, eps: f32) -> i32;
    fn pulsar_q4e_hc_collapse(mixed: *mut c_void, xn: *const c_void, gl: *const c_void, n_embd: u32, n_hc: u32, rows: u32) -> i32;
    fn pulsar_q4e_stream_scatter(res: *mut c_void, out: *const c_void, coef: *const c_void, n_embd: u32, n_hc: u32, rows: u32, mode: u32) -> i32;
    fn pulsar_q4e_replicate(res: *mut c_void, emb: *const c_void, n_embd: u32, n_hc: u32, rows: u32) -> i32;
    fn pulsar_q4e_silu_scale(x: *mut c_void, scale: f32, n: u64) -> i32;
    fn pulsar_q4e_ple_gate(coef: *mut c_void, keyn: *const c_void, queryn: *const c_void, n_embd: u32, n_hc: u32, rows: u32) -> i32;
    fn pulsar_q4e_conv_dil(out: *mut c_void, x: *const c_void, kern: *const c_void, state: *mut c_void, n_chan: u32, k: u32, dil: u32, n_tok: u32) -> i32;
        fn pulsar_qwen35_gdn_batch(out: *mut c_void, state: *mut c_void, q: *const c_void, k: *const c_void, v: *const c_void, g: *const c_void, beta: *const c_void, h_v: u32, h_k: u32, dim: u32, n_tok: u32) -> i32;
        fn pulsar_qwen35_row_scale(x: *mut c_void, s: *const c_void, n_rows: u32, dim: u32) -> i32;
        fn pulsar_qwen35_draft_attn(out: *mut c_void, q: *const c_void, k: *const c_void, v: *const c_void, n_q: u32, n_kv: u32, n_head: u32, n_kv_head: u32, dim: u32, scale: f32) -> i32;
        fn pulsar_qwen35_dflash2_conv(out: *mut c_void, h: *const c_void, delta: *const c_void, base: *const c_void, n_tok: u32, n_embd: u32, taps: u32, group: u32, side: u32) -> i32;
        fn pulsar_qwen35_dflash2_topk_step(logits: *mut c_void, n_rows: u32, vocab: u32, it: u32, k: u32, ids: *mut c_void, vals: *mut c_void) -> i32;
        fn pulsar_qwen35_dflash2_gather(out: *mut c_void, book: *const c_void, ids: *const c_void, n: u32, rank: u32) -> i32;
        fn pulsar_qwen35_rope_yarn(x: *mut c_void, n_tok: u32, n_head: u32, head_dim: u32, pos0: u32, freq_base: f32, freq_scale: f32, ext_factor: f32, attn_factor: f32, beta_fast: f32, beta_slow: f32, n_ctx_orig: u32) -> i32;
        fn pulsar_qwen35_split_qkv(q: *mut c_void, k: *mut c_void, v: *mut c_void, x: *const c_void, n_tok: u32, key_dim: u32, value_dim: u32) -> i32;
        fn pulsar_qwen35_ring_scatter(ring: *mut c_void, src: *const c_void, pos: u32, cap: u32, n_rows: u32, row_elems: u32, ring_stride: u32, ring_off: u32) -> i32;
        fn pulsar_qwen35_ring_gather(dst: *mut c_void, ring: *const c_void, start: u32, cap: u32, n_rows: u32, row_elems: u32) -> i32;
        fn pulsar_qwen35_gdn_coeffs(g_alpha: *mut c_void, beta: *mut c_void, a: *const c_void, dt: *const c_void, n_tok: u32, n_head: u32) -> i32;
        fn pulsar_qwen35_row_sigmoid_scale(x: *mut c_void, s: *const c_void, n_rows: u32, dim: u32) -> i32;
        fn pulsar_qwen35_selftest() -> i32;
        #[allow(clippy::too_many_arguments)]
        fn pulsar_k3_kda_coeffs(g: *mut c_void, beta: *mut c_void, a: *const c_void, dt: *const c_void, n_tok: u32, n_head: u32, head_dim: u32, g_min: f32) -> i32;
        fn pulsar_k3_kda_step(out: *mut c_void, state: *mut c_void, q: *const c_void, k: *const c_void, v: *const c_void, g: *const c_void, beta: *const c_void, n_head: u32, dim: u32) -> i32;
        fn pulsar_k3_attn_res(out: *mut c_void, cur: *const c_void, ckpt: *const c_void, w: *const c_void, n_tok: u32, n_embd: u32, n_ckpt: u32, eps: f32) -> i32;
        fn pulsar_k3_selftest() -> i32;
        fn pulsar_mla_kv_lora_rms_norm(out: *mut c_void, kv_raw: *const c_void, w: *const c_void, n_tok: u32, kv_raw_dim: u32, kv_lora_dim: u32, eps: f32) -> i32;
        fn pulsar_mla_store_compact_kv(kv_lora_cache: *mut c_void, k_rope_cache: *mut c_void, kv_norm: *const c_void, kv_raw: *const c_void, pos0: u32, n_tok: u32, cache_cap: u32, kv_raw_dim: u32, kv_lora_dim: u32, qk_rope: u32, kvq: u32) -> i32;
        fn pulsar_mla_fill_selected_range(selected: *mut c_void, n_tok: u32, pos0: u32, n_selected: u32, pad_row: u32) -> i32;
        fn pulsar_mla_qk_lowrank(qk_low: *mut c_void, q: *const c_void, k_b: *const c_void, n_tok: u32, n_head: u32, kv_lora_dim: u32, qk_nope: u32, qk_dim: u32) -> i32;
        fn pulsar_mla_attention(heads: *mut c_void, q: *const c_void, qk_low: *const c_void, kv_lora_cache: *const c_void, k_rope_cache: *const c_void, v_b: *const c_void, selected: *const c_void, n_tok: u32, n_selected: u32, cache_cap: u32, n_head: u32, kv_lora_dim: u32, qk_nope: u32, qk_rope: u32, value_dim: u32, n_ctx_orig: u32, freq_base: f32, freq_scale: f32, ext_factor: f32, attn_factor: f32, beta_fast: f32, beta_slow: f32, kq_mult: f32, kvq: u32) -> i32;
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
    /// The primary (stream) device ensure_device picked; get_device()
    /// drifts with per-layer set_device calls, this does not.
    static PRIMARY_DEV: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(0);
    static PRIMARY_H2D_BITS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    /// The measured-best primary device chosen at startup. Unlike
    /// get_device() this is stable across the engine's per-layer device
    /// switching, so late allocations (draft models, probes) can pin
    /// themselves back to the fast card.
    pub fn primary_device() -> i32 {
        ensure_device();
        PRIMARY_DEV.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// Pinned 64MB H2D probe of the primary stream device, GB/s. Zero until
    /// `ensure_device` has run. Used by q* so the miss split matches the
    /// link that was actually measured, not a spec-sheet default.
    pub fn primary_h2d_gbs() -> f64 {
        ensure_device();
        f64::from_bits(PRIMARY_H2D_BITS.load(std::sync::atomic::Ordering::Relaxed))
    }

    /// Default-ON env flag: unset or any value but "0" means on. Both TP
    /// hop levers (PULSAR_P2P, PULSAR_TP_BF16) measured strictly faster
    /// (11.15 -> 10.12s on the 7.4k TP prefill) and p2p self-detects
    /// driver support, so opt-OUT is the safe polarity.
    fn env_on(name: &str) -> bool {
        std::env::var(name).map_or(true, |v| v != "0")
    }

    /// P2P's effective default: ON, EXCEPT when the two-lane prefill
    /// pipe is armed. The patched consumer-P2P kernel module faults
    /// (libcuda SIGSEGV in cuStreamWaitEvent) when replayed decode
    /// graphs carrying peer copies mix with the pipe's event/alloc
    /// churn - the repro needs p2p+bf16+graphs+MTP+growing prompts
    /// together (2026-08-23). Under the pipe p2p measured ZERO prefill
    /// win (hops hide behind the other lane), so off is free. An
    /// explicit PULSAR_P2P=1 still forces it on.
    fn p2p_on() -> bool {
        match std::env::var("PULSAR_P2P") {
            Ok(v) => v != "0",
            Err(_) => std::env::var("PULSAR_PIPE").ok().as_deref() != Some("1"),
        }
    }

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
            PRIMARY_DEV.store(dev, std::sync::atomic::Ordering::Relaxed);
            // P2P defaults ON (PULSAR_P2P=0 disables), and the peer access must
            // be established HERE, inside the same

            // once-block that picks the device, i.e. before a single
            // DeviceBuf exists. Enabling it later (from TpLink::new, after
            // the weight pool is allocated) surfaced as a failing kernel
            // launch two calls downstream, which is what a deferred CUDA
            // error looks like. Consumer cards report can-access only with
            // a patched driver; where they do not, this is a no-op.
            if p2p_on() {
                let mut n = 0;
                unsafe { cudaGetDeviceCount(&mut n) };
                let mut pairs = 0;
                for a in 0..n {
                    if unsafe { cudaSetDevice(a) } != 0 {
                        continue;
                    }
                    for b in 0..n {
                        if a == b {
                            continue;
                        }
                        let mut can = 0;
                        unsafe { cudaDeviceCanAccessPeer(&mut can, a, b) };
                        if can != 1 {
                            continue;
                        }
                        let e = unsafe { cudaDeviceEnablePeerAccess(b, 0) };
                        // 704 = cudaErrorPeerAccessAlreadyEnabled
                        if e == 0 || e == 704 {
                            pairs += 1;
                        }
                    }
                }
                if pairs > 0 && std::env::var_os("PULSAR_QUIET").is_none() {
                    eprintln!("pulsar: p2p peer access enabled on {pairs} ordered pair(s)");
                }
            }
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

    /// Measured VRAM bandwidth of `dev` in GB/s: a 64MB on-card D2D copy
    /// (reads + writes VRAM, so ~2x the copy rate; reported as copy rate
    /// x2). H2D probes measure the PCIe LINK, which is the wrong axis
    /// for placing dense-resident weights - those are read from the
    /// card's own VRAM every token. Measured, not derived: the
    /// memoryClockRate attribute is deprecated and reads 0 on Blackwell,
    /// which silently tied a 448GB/s card with a 288GB/s one. Restores
    /// the current device; returns 0.0 on any probe failure so a broken
    /// card ranks last instead of erroring placement.
    pub fn vram_bandwidth(dev: i32) -> f64 {
        // 512MB, not 64MB: Ada/Blackwell L2 runs to 32MB, so a 64MB
        // buffer is served largely from cache and every card reports the
        // same ~390 GB/s regardless of its actual DRAM rate. Sized to
        // dwarf any L2 on this generation.
        const MB64: usize = 512 << 20;
        const D2D_KIND: i32 = 3;
        let cur = get_device();
        if set_device(dev).is_err() {
            return 0.0;
        }
        let mut a = std::ptr::null_mut();
        let mut b = std::ptr::null_mut();
        let mut best = 0f64;
        if unsafe { cudaMalloc(&mut a, MB64) } == 0 {
            if unsafe { cudaMalloc(&mut b, 16) } == 0 {
                // one warmup, then best of 3 timed streaming reads
                unsafe { pulsar_bw_read(a, MB64 as u64, b) };
                unsafe { cudaDeviceSynchronize() };
                for _ in 0..3 {
                    let t = std::time::Instant::now();
                    if unsafe { pulsar_bw_read(a, MB64 as u64, b) } != 0
                        && unsafe { cudaDeviceSynchronize() } == 0
                    {
                        best = best.max(MB64 as f64 / 1e9 / t.elapsed().as_secs_f64());
                    }
                }
                unsafe { cudaFree(b) };
            }
            unsafe { cudaFree(a) };
        }
        let _ = set_device(cur);
        best
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

    /// Per-device free-VRAM budget for WEIGHT uploads (DeviceBuf::from_bytes
    /// callers only). i64::MAX = untracked; 0 = pin everything new.
    static WEIGHT_BUDGET: [std::sync::atomic::AtomicI64; 16] =
        [const { std::sync::atomic::AtomicI64::new(i64::MAX) }; 16];

    /// Cap weight VRAM on `dev` at `bytes` (engine passes free minus the
    /// 3GiB state reserve for KV/activations/scratch/graph pools);
    /// from_bytes
    /// spends it and spills the overflow to pinned host RAM.
    pub fn set_weight_vram_budget(dev: i32, bytes: i64) {
        WEIGHT_BUDGET[dev as usize].store(bytes.clamp(0, i64::MAX), std::sync::atomic::Ordering::Relaxed);
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
            // PORTABLE matters on multi-GPU: without it the allocation is
            // only registered in the context that made it, so another
            // device copying out of the same host buffer is undefined.
            // The TP staging buffer is allocated on card A and read by an
            // H2D on card B, and the missing flag corrupted it
            // deterministically - byte 0 survived while the middle of a
            // 20KB transfer came back wrong, which is what made tensor
            // parallel emit wrong tokens.
            const PORTABLE: u32 = 1; // cudaHostAllocPortable
            const MAPPED: u32 = 2; // cudaHostAllocMapped
            let mut host = std::ptr::null_mut();
            check_rt(unsafe { cudaHostAlloc(&mut host, bytes.max(1), PORTABLE | MAPPED) }, "cudaHostAlloc")?;
            let mut dev = std::ptr::null_mut();
            check_rt(unsafe { cudaHostGetDevicePointer(&mut dev, host, 0) }, "cudaHostGetDevicePointer")?;
            Ok(DeviceBuf { ptr: dev, host, bytes, dev: -1 })
        }

        pub fn from_bytes(data: &[u8]) -> Result<Self> {
            // Weights only (every engine caller is a load-time weight).
            // VRAM while the device's weight budget lasts, then mapped
            // pinned host memory - zero-copy PCIe reads, the same path
            // budgeted Mla/dsv4 attn weights already take every token.
            // Activations and KV never come through here, so a full card
            // degrades to host residency instead of a fatal cudaMalloc.
            let dev = get_device();
            use std::sync::atomic::Ordering;
            let mut b = if WEIGHT_BUDGET[dev as usize].load(Ordering::Relaxed) >= data.len() as i64 {
                match Self::alloc(data.len()) {
                    Ok(b) => {
                        WEIGHT_BUDGET[dev as usize].fetch_sub(data.len() as i64, Ordering::Relaxed);
                        b
                    }
                    Err(_) => Self::weight_pinned(data)?,
                }
            } else {
                Self::weight_pinned(data)?
            };
            b.write(0, data)?;
            Ok(b)
        }

        fn weight_pinned(data: &[u8]) -> Result<Self> {
            let dev = get_device();
            use std::sync::atomic::Ordering;
            if WEIGHT_BUDGET[dev as usize].swap(0, Ordering::Relaxed) != 0 {
                eprintln!(
                    "pulsar: weights exceed VRAM on device {dev} - overflow resident in pinned host RAM (zero-copy PCIe reads)"
                );
            }
            let mut b = Self::alloc_pinned(data.len())?;
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

        /// Read `n` f32s starting at element offset `off`.
        pub fn read_f32_at(&self, off: usize, n: usize) -> Result<Vec<f32>> {
            let mut v = vec![0f32; n];
            self.read(off * 4, as_bytes_mut(&mut v))?;
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
        fn cudaStreamBeginCapture(stream: *mut c_void, mode: u32) -> i32;
        fn cudaStreamEndCapture(stream: *mut c_void, graph: *mut *mut c_void) -> i32;
        fn cudaGraphInstantiateWithFlags(exec: *mut *mut c_void, graph: *mut c_void, flags: u64) -> i32;
        fn cudaGraphLaunch(exec: *mut c_void, stream: *mut c_void) -> i32;
        fn cudaGraphDestroy(graph: *mut c_void) -> i32;
        fn cudaGraphExecDestroy(exec: *mut c_void) -> i32;
    }

    /// The calling thread's default stream (kernels compile with
    /// --default-stream=per-thread, so <<<>>> launches land here and are
    /// capturable; the legacy NULL stream cannot begin capture).
    const STREAM_PER_THREAD: *mut c_void = 0x2 as *mut c_void;
    /// cudaStreamCaptureModeThreadLocal: only THIS thread's illegal
    /// calls invalidate the capture (background copy threads unaffected).
    const CAPTURE_THREAD_LOCAL: u32 = 1;

    /// An instantiated CUDA graph: a recorded launch chain replayed with
    /// one API call. Capture and launch with the SAME device current
    /// (single-device graphs).
    pub struct Graph {
        exec: *mut c_void,
    }

    unsafe impl Send for Graph {}

    impl Graph {
        /// Record every kernel the closure launches on this thread's
        /// default stream into a graph (nothing executes during capture)
        /// and instantiate it. On closure error the capture is unwound.
        pub fn capture<F: FnOnce() -> Result>(f: F) -> Result<Graph> {
            ensure_device();
            check_rt(
                unsafe { cudaStreamBeginCapture(STREAM_PER_THREAD, CAPTURE_THREAD_LOCAL) },
                "graph begin capture",
            )?;
            let r = f();
            let mut graph = std::ptr::null_mut();
            let end = check_rt(
                unsafe { cudaStreamEndCapture(STREAM_PER_THREAD, &mut graph) },
                "graph end capture",
            );
            r?;
            end?;
            let mut exec = std::ptr::null_mut();
            let inst = check_rt(
                unsafe { cudaGraphInstantiateWithFlags(&mut exec, graph, 0) },
                "graph instantiate",
            );
            unsafe { cudaGraphDestroy(graph) };
            inst?;
            Ok(Graph { exec })
        }

        /// Replay the recorded chain on this thread's default stream.
        pub fn launch(&self) -> Result {
            check_rt(unsafe { cudaGraphLaunch(self.exec, STREAM_PER_THREAD) }, "graph launch")
        }
    }

    impl Drop for Graph {
        fn drop(&mut self) {
            unsafe { cudaGraphExecDestroy(self.exec) };
        }
    }

    extern "C" {
        fn cudaStreamCreateWithFlags(s: *mut *mut c_void, flags: u32) -> i32;
        fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, bytes: usize, kind: i32, stream: *mut c_void) -> i32;
        fn cudaMemcpyPeerAsync(dst: *mut c_void, dst_dev: i32, src: *const c_void, src_dev: i32, bytes: usize, stream: *mut c_void) -> i32;
        fn cudaEventCreateWithFlags(e: *mut *mut c_void, flags: u32) -> i32;
        fn cudaEventRecord(e: *mut c_void, stream: *mut c_void) -> i32;
        fn cudaEventQuery(e: *mut c_void) -> i32;
        fn cudaEventDestroy(e: *mut c_void) -> i32;
        fn cudaStreamWaitEvent(stream: *mut c_void, e: *mut c_void, flags: u32) -> i32;
    }

    /// Cross-device handoff for tensor parallel: an async D2H into a
    /// PINNED staging buffer on the source device's null stream, an
    /// event, and an async H2D on the consumer device's null stream
    /// gated on that event. No host syncs anywhere, both DMA engines,
    /// fully known behavior. NOT cudaMemcpyPeerAsync: without P2P
    /// access (GeForce) the driver stages peer copies with implicit
    /// synchronization on BOTH devices - measured 26.4 -> 18.8 tok/s,
    /// worse than v1's plain sync bounces.
    ///
    /// Create with the SOURCE device current (the event records there).
    /// Host-call order per use: `send` on the source device, then
    /// `recv` on the consumer BEFORE the same link's next `send`.
    pub struct TpLink {
        ev: *mut c_void,
        pin: DeviceBuf,
        /// this link's source and consumer devices
        dev_src: i32,
        dev_dst: i32,
        /// real peer access both ways (needs PULSAR_P2P + a driver that
        /// allows it). When set, the pinned bounce is skipped entirely.
        p2p: bool,
        /// source parked by `send` for the peer `recv` to read from
        pending: std::cell::Cell<(*const c_void, usize)>,
        /// bf16 stage on the SENDER's device (pack target)
        s16: DeviceBuf,
        /// bf16 stage on the RECEIVER's device (unpack source)
        r16: DeviceBuf,
        /// bf16 hop staging, on by default (PULSAR_TP_BF16=0 for exact
        /// f32 hops): halves the bytes that cross the bus
        bf16: bool,
    }

    unsafe impl Send for TpLink {}

    impl TpLink {
        /// `peer` is the device on the OTHER end of this link. Both bf16
        /// stages are allocated here, at setup: the hop runs inside CUDA
        /// graph capture, where cudaMalloc is illegal, so nothing may be
        /// allocated lazily on the hot path.
        pub fn new(bytes: usize, peer: i32) -> Result<TpLink> {
            ensure_device();
            const DISABLE_TIMING: u32 = 2;
            let mut ev = std::ptr::null_mut();
            check_rt(unsafe { cudaEventCreateWithFlags(&mut ev, DISABLE_TIMING) }, "tplink event")?;
            let pin = DeviceBuf::alloc_pinned(bytes)?;
            let s16 = DeviceBuf::alloc(bytes / 2 + 4)?;
            let here = get_device();
            set_device(peer)?;
            let r16 = DeviceBuf::alloc(bytes / 2 + 4)?;
            set_device(here)?;
            let bf16 = env_on("PULSAR_TP_BF16");
            // Peer access itself is established once in ensure_device, before
            // any allocation; here we only ASK whether it took, both ways.
            let mut p2p = false;
            if p2p_on() && here != peer {
                let (mut ab, mut ba) = (0i32, 0i32);
                unsafe {
                    cudaDeviceCanAccessPeer(&mut ab, here, peer);
                    cudaDeviceCanAccessPeer(&mut ba, peer, here);
                }
                p2p = ab == 1 && ba == 1;
            }
            Ok(TpLink {
                ev, pin, s16, r16, bf16,
                dev_src: here, dev_dst: peer, p2p,
                pending: std::cell::Cell::new((std::ptr::null(), 0)),
            })
        }

        /// Send `n` f32s. With bf16 staging on, packs to bf16 first so
        /// half the bytes cross; otherwise byte-identical to `send`.
        pub fn send_f32(&mut self, src: &DeviceBuf, n: usize) -> Result {
            if !self.bf16 {
                return self.send(src, n * 4);
            }
            // bf16 COMPOSES with p2p: peer access removed the host round trip,
            // it did not shrink what crosses the bus, and PCIe is still the
            // slow link. Pack once on the source card either way.
            check(unsafe { pulsar_pack_bf16(self.s16.ptr_mut(), src.ptr(), n as u32) }, "pack_bf16")?;
            if self.p2p {
                self.pending.set((self.s16.ptr(), n * 2));
                return check_rt(unsafe { cudaEventRecord(self.ev, STREAM_PER_THREAD) }, "tplink record");
            }
            check_rt(
                unsafe { cudaMemcpyAsync(self.pin.host, self.s16.ptr(), n * 2, D2H, STREAM_PER_THREAD) },
                "tplink d2h bf16",
            )?;
            check_rt(unsafe { cudaEventRecord(self.ev, STREAM_PER_THREAD) }, "tplink record")
        }

        /// Receive `n` f32s written by `send_f32`.
        pub fn recv_f32(&mut self, dst: &mut DeviceBuf, n: usize) -> Result {
            if !self.bf16 {
                return self.recv(dst, n * 4);
            }
            check_rt(unsafe { cudaStreamWaitEvent(STREAM_PER_THREAD, self.ev, 0) }, "tplink wait")?;
            if self.p2p {
                let (psrc, pn) = self.pending.get();
                if psrc.is_null() {
                    return Err(Error("tplink: p2p bf16 recv with no parked send"));
                }
                check_rt(
                    unsafe {
                        cudaMemcpyAsync(self.r16.ptr_mut(), psrc, pn, MEMCPY_DEFAULT, STREAM_PER_THREAD)
                    },
                    "tplink peer copy bf16",
                )?;
            } else {
                check_rt(
                    unsafe { cudaMemcpyAsync(self.r16.ptr_mut(), self.pin.host, n * 2, H2D, STREAM_PER_THREAD) },
                    "tplink h2d bf16",
                )?;
            }
            check(unsafe { pulsar_unpack_bf16(dst.ptr_mut(), self.r16.ptr(), n as u32) }, "unpack_bf16")
        }

        /// Async D2H of `bytes` from `src` into the pinned stage, on the
        /// CURRENT (source) device's per-thread default stream (the
        /// stream the kernels launch on, and the only stream CUDA graph
        /// capture may record - the legacy null stream is capture-
        /// illegal), event behind it.
        pub fn send(&self, src: &DeviceBuf, bytes: usize) -> Result {
            assert!(bytes <= self.pin.bytes() && bytes <= src.bytes());
            if self.p2p {
                // Nothing crosses here: park the source and fence, so the
                // consumer's peer read orders after this card's kernels.
                self.pending.set((src.ptr(), bytes));
                return check_rt(unsafe { cudaEventRecord(self.ev, STREAM_PER_THREAD) }, "tplink record");
            }
            check_rt(
                unsafe { cudaMemcpyAsync(self.pin.host, src.ptr(), bytes, D2H, STREAM_PER_THREAD) },
                "tplink d2h",
            )?;
            check_rt(unsafe { cudaEventRecord(self.ev, STREAM_PER_THREAD) }, "tplink record")
        }

        /// Gate the CURRENT (consumer) device's per-thread default stream
        /// on the last send, then async H2D from the pinned stage into
        /// `dst`. During graph capture this wait is the edge that JOINS
        /// the consumer device's stream into the capture DAG.
        pub fn recv(&self, dst: &mut DeviceBuf, bytes: usize) -> Result {
            assert!(bytes <= self.pin.bytes() && bytes <= dst.bytes());
            check_rt(unsafe { cudaStreamWaitEvent(STREAM_PER_THREAD, self.ev, 0) }, "tplink wait")?;
            if self.p2p {
                let (src, n) = self.pending.get();
                if src.is_null() {
                    return Err(Error("tplink: p2p recv with no parked send"));
                }
                // NOT cudaMemcpyPeerAsync: it returns 900
                // (cudaErrorStreamCaptureUnsupported) inside graph capture,
                // which is where every decode hop runs. A plain async memcpy
                // with cudaMemcpyDefault resolves both pointers through
                // unified addressing and, with peer access enabled, moves the
                // bytes card-to-card directly - and it captures fine.
                return check_rt(
                    unsafe {
                        cudaMemcpyAsync(dst.ptr_mut(), src, n, MEMCPY_DEFAULT, STREAM_PER_THREAD)
                    },
                    "tplink peer copy",
                );
            }
            check_rt(
                unsafe { cudaMemcpyAsync(dst.ptr_mut(), self.pin.host, bytes, H2D, STREAM_PER_THREAD) },
                "tplink h2d",
            )
        }
    }

    impl Drop for TpLink {
        fn drop(&mut self) {
            unsafe { cudaEventDestroy(self.ev) };
        }
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

        /// Queue async H2D from an arbitrary host pointer. The bytes must
        /// remain valid until a matching `wait_default` / `synchronize`
        /// (expert host-cache slabs: pinned, owned by StreamingStore).
        pub fn copy_h2d_raw(
            &self,
            dst: &mut DeviceBuf,
            dst_off: usize,
            src: *const u8,
            bytes: usize,
        ) -> Result {
            assert!(dst_off + bytes <= dst.bytes);
            if bytes == 0 {
                return Ok(());
            }
            check_rt(
                unsafe {
                    cudaMemcpyAsync(
                        (dst.ptr as *mut u8).add(dst_off) as *mut c_void,
                        src as *const c_void,
                        bytes,
                        H2D,
                        self.stream,
                    )
                },
                "async h2d raw",
            )
        }

        pub fn record(&self) -> Result {
            check_rt(unsafe { cudaEventRecord(self.event, self.stream) }, "event record")
        }

        /// True once every copy queued before the last `record` finished.
        pub fn done(&self) -> bool {
            unsafe { cudaEventQuery(self.event) == 0 }
        }

        /// Make the default stream wait for the last `record` (kernels
        /// launched afterward see completed H2D without a full device sync).
        pub fn wait_default(&self) -> Result {
            check_rt(
                unsafe { cudaStreamWaitEvent(std::ptr::null_mut(), self.event, 0) },
                "default wait event",
            )
        }

        /// Block the host until the last `record` completes.
        pub fn synchronize(&self) -> Result {
            extern "C" {
                fn cudaEventSynchronize(e: *mut c_void) -> i32;
            }
            check_rt(unsafe { cudaEventSynchronize(self.event) }, "event synchronize")
        }
    }

    const D2D: i32 = 3;
    const MEMCPY_DEFAULT: i32 = 4; // UVA infers direction; works across devices

    /// Copy between buffers on ANY pair of devices (or pinned host).
    /// Blocking cudaMemcpy: legacy-stream ordered on the current device,
    /// so issue it with the producer's device current and the consumer
    /// device's later launches see the data.
    /// An event for ordering work across cards. Recordable only on the
    /// device that was current when it was created; waiting on it from
    /// another device's stream is legal (and is the whole point).
    pub struct XEvent(*mut c_void);
    unsafe impl Send for XEvent {}
    impl XEvent {
        pub fn new() -> Result<XEvent> {
            const DISABLE_TIMING: u32 = 2;
            let mut e: *mut c_void = std::ptr::null_mut();
            check_rt(unsafe { cudaEventCreateWithFlags(&mut e, DISABLE_TIMING) }, "xevent create")?;
            Ok(XEvent(e))
        }
        pub fn record(&self) -> Result {
            check_rt(unsafe { cudaEventRecord(self.0, STREAM_PER_THREAD) }, "xevent record")
        }
        pub fn wait(&self) -> Result {
            check_rt(unsafe { cudaStreamWaitEvent(STREAM_PER_THREAD, self.0, 0) }, "xevent wait")
        }
    }

    /// Async cross-device copy on the CURRENT device's stream. Issue it on
    /// the CONSUMER's stream after waiting on the producer's event: that
    /// orders the consumer's own prior reads of dst before the overwrite,
    /// which a copy issued on the producer's stream would race. The point
    /// of async here is that the host stops blocking at every bank
    /// boundary, which is what serialized the three cards.
    pub fn copy_across_async(dst: &mut DeviceBuf, src: &DeviceBuf, bytes: usize) -> Result {
        assert!(bytes <= dst.bytes() && bytes <= src.bytes());
        check_rt(
            unsafe { cudaMemcpyAsync(dst.ptr_mut(), src.ptr(), bytes, MEMCPY_DEFAULT, STREAM_PER_THREAD) },
            "copy across async",
        )
    }

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

    /// embed_q8_0 starting at token index `tok_off` of `tokens`. Lets a
    /// prefill upload the whole prompt's ids ONCE and walk it per chunk:
    /// the per-chunk 512B `write` was a synchronous cudaMemcpy, and each
    /// one drained the device - 11.4s of a 12s 7k-token prefill.
    pub fn embed_q8_0_at(out: &mut DeviceBuf, w: &DeviceBuf, tokens: &DeviceBuf, tok_off: u32, n_embd: u32, n_vocab: u32, n_tok: u32) -> Result {
        assert!(((tok_off + n_tok) as usize) * 4 <= tokens.bytes());
        let p = unsafe { (tokens.ptr() as *const u8).add(tok_off as usize * 4) as *const c_void };
        check(unsafe { pulsar_embed_q8_0(out.ptr_mut(), w.ptr(), p, n_embd, n_vocab, n_tok) }, "embed_q8_0_at")
    }

    /// DSpark fused markov argmax over one logits row: returns
    /// argmax_v logits[row_off + v] + q8dot(w2[v], state). `scratch` needs
    /// 128 * 8 bytes; `out` 4 bytes (device); the id is read back here.
    #[allow(clippy::too_many_arguments)] // kernel launch mirrors the CUDA signature
    pub fn dspark_markov_argmax(
        logits: &DeviceBuf,
        row_off_elems: usize,
        w2: &DeviceBuf,
        state: &DeviceBuf,
        vocab: u32,
        rank: u32,
        scratch: &mut DeviceBuf,
        out: &mut DeviceBuf,
    ) -> Result<u32> {
        check(
            unsafe {
                pulsar_dspark_markov_argmax(
                    (logits.ptr() as *const u8).add(row_off_elems * 4) as *const c_void,
                    w2.ptr(),
                    state.ptr(),
                    vocab,
                    rank,
                    scratch.ptr_mut(),
                    out.ptr_mut(),
                )
            },
            "dspark_markov_argmax",
        )?;
        sync()?;
        let id = out.read_i32(1)?;
        Ok(id[0].max(0) as u32)
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

    /// Pre-allocate the q8_0 prequant scratch on the CURRENT device so a
    /// later matmul_q8_0 inside a CUDA-graph capture hits the cached
    /// pointer: cudaMalloc is illegal mid-capture.
    /// Selects this HOST THREAD's scratch-arena lane (0 or 1) for the
    /// process-global preq/octet/nvfp4-act/gqa-split arenas. The
    /// PULSAR_PIPE prefill sets 1 on its second worker thread so the two
    /// lanes never share a staging buffer mid-flight.
    pub fn set_lane(lane: u32) {
        unsafe { pulsar_set_lane(lane as i32) }
    }

    /// Allocates the current thread-lane's scratch arenas on the CURRENT
    /// device at their floors (first-use cudaMalloc stalls move out of
    /// the timed prefill). Call once per (device, lane).
    pub fn lane_prewarm() {
        unsafe { pulsar_lane_prewarm() }
    }

    pub fn preq_scratch_reserve(bytes: u64) -> Result {
        check(unsafe { pulsar_preq_scratch_reserve(bytes) }, "preq_scratch_reserve")
    }

    /// Banked matmul: x is n_tok*n_bank contiguous pseudo-rows of in_dim,
    /// w is n_bank stacked [out_dim x in_dim] q8_0 matrices, pseudo-row j
    /// multiplies bank j % n_bank (deepseek4's grouped output projection
    /// in ONE launch, bitwise identical to the per-bank loop).
    pub fn matmul_q8_0_banked(out: &mut DeviceBuf, w: &DeviceBuf, x: &DeviceBuf, in_dim: u32, out_dim: u32, n_bank: u32, n_tok: u32) -> Result {
        check(unsafe { pulsar_q8_0_matmul_banked(out.ptr_mut(), w.ptr(), x.ptr(), in_dim, out_dim, n_bank, n_tok) }, "matmul_q8_0_banked")
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
    pub fn idx_store_k(raw_k: &DeviceBuf, w: &DeviceBuf, b: &DeviceBuf, cache: &mut DeviceBuf, pos0: u32, n_tok: u32, cache_cap: u32, head_dim: u32, rot_dim: u32, eps: f32, r: &RopeCfg, ext_factor: f32, attn_factor: f32, fp8: u32) -> Result {
        check(unsafe { pulsar_idx_store_k(raw_k.ptr(), w.ptr(), b.ptr(), cache.ptr_mut(), pos0, n_tok, cache_cap, head_dim, rot_dim, r.n_ctx_orig, eps, r.freq_base, r.freq_scale, ext_factor, attn_factor, r.beta_fast, r.beta_slow, fp8) }, "idx_store_k")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn idx_score_one(scores: &mut DeviceBuf, q: &DeviceBuf, weights: &DeviceBuf, cache: &DeviceBuf, n_rows: u32, n_head: u32, head_dim: u32, scale: f32, fp8: u32) -> Result {
        check(unsafe { pulsar_idx_score_one(scores.ptr_mut(), q.ptr(), weights.ptr(), cache.ptr(), n_rows, n_head, head_dim, scale, fp8) }, "idx_score_one")
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
    pub fn idx_scores_batch(scores: &mut DeviceBuf, q: &DeviceBuf, weights: &DeviceBuf, cache: &DeviceBuf, q16: Option<&mut DeviceBuf>, n_rows: u32, n_tok: u32, pos0: u32, n_head: u32, head_dim: u32, scale: f32, fp8: u32) -> Result {
        check(unsafe { pulsar_idx_scores_batch(scores.ptr_mut(), q.ptr(), weights.ptr(), cache.ptr(), q16.map_or(std::ptr::null_mut(), |b| b.ptr_mut()), n_rows, n_tok, pos0, n_head, head_dim, scale, fp8) }, "idx_scores_batch")
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

    /// Adds sum_s w_s * b_down_s to a finished down-projection output.
    /// No-op unless some expert carries a down bias; only gpt-oss does.
    pub fn moe_down_bias(out: &mut DeviceBuf, ptrs: &DeviceBuf, weights: &DeviceBuf, out_dim: u32, n_used: u32, n_tok: u32) -> Result {
        check(unsafe { pulsar_moe_down_bias(out.ptr_mut(), ptrs.ptr(), weights.ptr(), out_dim, n_used, n_tok) }, "moe_down_bias")
    }

    /// Adds a [dim] bias vector to each of `rows` rows, in place.
    pub fn add_bias_rows(x: &mut DeviceBuf, bias: &DeviceBuf, dim: u32, rows: u32) -> Result {
        check(unsafe { pulsar_add_bias_rows(x.ptr_mut(), bias.ptr(), dim, rows) }, "add_bias_rows")
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

    /// Bytes per 256 values of NVFP4: 4 blocks of (4 ue4m3 scales + 32
    /// nibble bytes). One block is exactly one m16n8k64 mma.
    pub const NVFP4_SB_BYTES: usize = 144;

    /// Compute-capability major of the current device. 12 is Blackwell,
    /// the only tier carrying the FP4 tensor ops.
    pub fn cc_major() -> i32 {
        unsafe { pulsar_cc_major() }
    }

    /// Bytes an NVFP4 activation buffer needs. Rows round up to a whole
    /// octet: the GEMM reads 8 tokens per tile, and the quantizer fills
    /// the padding by repeating the last real token.
    pub fn nvfp4_act_bytes(in_dim: u32, n_tok: u32) -> usize {
        (((n_tok + 7) & !7) as usize) * (in_dim as usize / 256) * NVFP4_SB_BYTES
    }

    /// f32 -> NVFP4. Same 4-bit encoding as the weights but an
    /// octet-major tile, so a warp's B fragment is one contiguous
    /// region; see PULSAR_NVFP4_ACT_TILE in the kernels.
    pub fn quantize_nvfp4(out: &mut DeviceBuf, x: &DeviceBuf, in_dim: u32, n_rows: u32) -> Result {
        check(unsafe { pulsar_quantize_nvfp4(out.ptr_mut(), x.ptr(), in_dim, n_rows) }, "quantize_nvfp4")
    }

    /// q8_K activations -> NVFP4, the route matmul_kq takes when
    /// PULSAR_FP4 is set. Quantizes twice by construction.
    pub fn nvfp4_from_q8k(out: &mut DeviceBuf, xq: &DeviceBuf, in_dim: u32, n_tok: u32) -> Result {
        check(unsafe { pulsar_nvfp4_from_q8k(out.ptr_mut(), xq.ptr(), in_dim, n_tok) }, "nvfp4_from_q8k")
    }

    /// W4A4 prefill GEMM on Blackwell tensor cores: NVFP4 on both sides
    /// with hardware per-16 block scaling and no dequantization at all.
    /// Requires sm_120a; see docs/blackwell-fp4-mma.md.
    pub fn matmul_nvfp4_a4(out: &mut DeviceBuf, w: &DeviceBuf, xq: &DeviceBuf, in_dim: u32, out_dim: u32, n_tok: u32, row_bytes: u64) -> Result {
        check(unsafe { pulsar_matmul_nvfp4_a4(out.ptr_mut(), w.ptr(), xq.ptr(), in_dim, out_dim, n_tok, row_bytes) }, "matmul_nvfp4_a4")
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

    #[allow(clippy::too_many_arguments)] // kernel launch mirrors the CUDA signature
    /// Device-position rope (multi-device graph capture): position read
    /// from a device word instead of a baked argument.
    pub fn gqa_rope_dev(x: &mut DeviceBuf, n_tok: u32, n_head: u32, head_dim: u32, rot_dim: u32, pos_dev: &DeviceBuf, theta: f32) -> Result {
        check(unsafe { pulsar_gqa_rope_dev(x.ptr_mut(), n_tok, n_head, head_dim, rot_dim, pos_dev.ptr(), theta) }, "gqa_rope_dev")
    }

    pub fn gqa_kv_append_dev(cache: &mut DeviceBuf, kv: &DeviceBuf, n_tok: u32, n_kv_head: u32, head_dim: u32, cap: u32, pos_dev: &DeviceBuf) -> Result {
        check(unsafe { pulsar_gqa_kv_append_dev(cache.ptr_mut(), kv.ptr(), n_tok, n_kv_head, head_dim, cap, pos_dev.ptr()) }, "gqa_kv_append_dev")
    }

    /// f32 KV, plain single-pass kernel (no split-K): the engine gates
    /// this to contexts below the split threshold.
    #[allow(clippy::too_many_arguments)]
    pub fn gqa_attention_dev(out: &mut DeviceBuf, q: &DeviceBuf, k: &DeviceBuf, v: &DeviceBuf, n_tok: u32, n_head: u32, n_kv_head: u32, head_dim: u32, cap: u32, pos_dev: &DeviceBuf, scale: f32) -> Result {
        check(unsafe { pulsar_gqa_attention_dev(out.ptr_mut(), q.ptr(), k.ptr(), v.ptr(), n_tok, n_head, n_kv_head, head_dim, cap, pos_dev.ptr(), scale) }, "gqa_attention_dev")
    }

    /// Row-wise argmax on device: 8 bytes back per row - a (value,
    /// index) pair, so vocab-split halves merge host-side exactly like
    /// one full scan. First index wins ties (matches the host scan).
    pub fn argmax_rows_pairs(out: &mut DeviceBuf, x: &DeviceBuf, n: u32, rows: u32) -> Result<Vec<(f32, u32)>> {
        check(unsafe { pulsar_argmax_rows(out.ptr_mut(), x.ptr(), n, rows) }, "argmax_rows")?;
        sync()?;
        let raw = out.read_f32(rows as usize * 2)?;
        Ok((0..rows as usize).map(|i| (raw[i * 2], raw[i * 2 + 1].to_bits())).collect())
    }

    pub fn argmax_rows(out: &mut DeviceBuf, x: &DeviceBuf, n: u32, rows: u32) -> Result<Vec<u32>> {
        Ok(argmax_rows_pairs(out, x, n, rows)?.into_iter().map(|(_, i)| i).collect())
    }

    /// Launch-only argmax (no sync): the vocab-split head launches one
    /// per card and reads both AFTER both chains are in flight.
    pub fn argmax_rows_launch(out: &mut DeviceBuf, x: &DeviceBuf, n: u32, rows: u32) -> Result {
        check(unsafe { pulsar_argmax_rows(out.ptr_mut(), x.ptr(), n, rows) }, "argmax_rows")
    }

    /// Read back (value, index) pairs from an argmax_rows_launch (syncs
    /// the buffer's device).
    pub fn argmax_pairs_read(out: &DeviceBuf, rows: u32) -> Result<Vec<(f32, u32)>> {
        let raw = out.read_f32(rows as usize * 2)?;
        Ok((0..rows as usize).map(|i| (raw[i * 2], raw[i * 2 + 1].to_bits())).collect())
    }

    /// Async one-thread device write (per-token position cells).
    pub fn set_u32(dst: &mut DeviceBuf, v: u32) -> Result {
        check(unsafe { pulsar_set_u32(dst.ptr_mut(), v) }, "set_u32")
    }

    pub fn gqa_rope(x: &mut DeviceBuf, n_tok: u32, n_head: u32, head_dim: u32, rot_dim: u32, pos0: u32, theta: f32, factors: Option<&DeviceBuf>) -> Result {
        check(unsafe { pulsar_gqa_rope(x.ptr_mut(), n_tok, n_head, head_dim, rot_dim, pos0, theta, factors.map_or(std::ptr::null(), |b| b.ptr())) }, "gqa_rope")
    }

    /// KV cache storage format (kvq). 0 = f32 (exact), 1 = fp8 e4m3 +
    /// per-row scale, 2 = fp16, 3 = int8 + per-row scale, 4 = q8_0,
    /// 5 = q4_0, 6-10 = dsv4-only turbo3/2 + tcq. Opt-in via
    /// PULSAR_KV=<fmt>; row stride per format.
    #[allow(clippy::too_many_arguments)]
    pub fn gqa_kv_append(cache: &mut DeviceBuf, kv: &DeviceBuf, n_tok: u32, n_kv_head: u32, head_dim: u32, cap: u32, pos0: u32, kvq: u32) -> Result {
        check(unsafe { pulsar_gqa_kv_append(cache.ptr_mut(), kv.ptr(), n_tok, n_kv_head, head_dim, cap, pos0, kvq) }, "gqa_kv_append")
    }

    #[allow(clippy::too_many_arguments)]
    pub fn gqa_attention(out: &mut DeviceBuf, q: &DeviceBuf, k_cache: &DeviceBuf, v_cache: &DeviceBuf, n_tok: u32, n_head: u32, n_kv_head: u32, head_dim: u32, cap: u32, pos0: u32, scale: f32, window: u32) -> Result {
        gqa_attention_rel(out, q, k_cache, v_cache, n_tok, n_head, n_kv_head, head_dim, cap, pos0, scale, window, None, 0, 0, None)
    }

    /// GQA attention with an optional inkling relative-position bias:
    /// rel is [n_tok][n_head][rel_extent], score(i,j) += rel[i-j] in-band.
    #[allow(clippy::too_many_arguments)]
    pub fn gqa_attention_rel(out: &mut DeviceBuf, q: &DeviceBuf, k_cache: &DeviceBuf, v_cache: &DeviceBuf, n_tok: u32, n_head: u32, n_kv_head: u32, head_dim: u32, cap: u32, pos0: u32, scale: f32, window: u32, rel: Option<&DeviceBuf>, rel_extent: u32, kvq: u32, sinks: Option<&DeviceBuf>) -> Result {
        check(
            unsafe {
                pulsar_gqa_attention(out.ptr_mut(), q.ptr(), k_cache.ptr(), v_cache.ptr(), n_tok, n_head, n_kv_head, head_dim, cap, pos0, scale, window, rel.map_or(std::ptr::null(), |r| r.ptr()), rel_extent, kvq, sinks.map_or(std::ptr::null(), |b| b.ptr()))
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
    pub fn dsv4_attention(out: &mut DeviceBuf, q: &DeviceBuf, raw: &DeviceBuf, n_raw: u32, comp: Option<&DeviceBuf>, n_comp: u32, allowed: Option<&DeviceBuf>, sinks: &DeviceBuf, n_head: u32, head_dim: u32, scale: f32, kvq: u32, turbo: u32, pi: Option<&DeviceBuf>) -> Result {
        dsv4_attention_at(out, 0, q, 0, raw, n_raw, comp, n_comp, allowed, sinks, n_head, head_dim, scale, kvq, turbo, pi)
    }

    /// dsv4_attention with byte offsets into out/q (chunked prefill's
    /// per-token interleave).
    #[allow(clippy::too_many_arguments)]
    pub fn dsv4_attention_at(out: &mut DeviceBuf, out_off: usize, q: &DeviceBuf, q_off: usize, raw: &DeviceBuf, n_raw: u32, comp: Option<&DeviceBuf>, n_comp: u32, allowed: Option<&DeviceBuf>, sinks: &DeviceBuf, n_head: u32, head_dim: u32, scale: f32, kvq: u32, turbo: u32, pi: Option<&DeviceBuf>) -> Result {
        let out_ptr = unsafe { (out.ptr_mut() as *mut u8).add(out_off) as *mut c_void };
        let q_ptr = unsafe { (q.ptr() as *const u8).add(q_off) as *const c_void };
        check(
            unsafe {
                pulsar_dsv4_attention(out_ptr, q_ptr, raw.ptr(), n_raw, comp.map_or(std::ptr::null(), |b| b.ptr()), n_comp, allowed.map_or(std::ptr::null(), |b| b.ptr()), sinks.ptr(), n_head, head_dim, scale, kvq, turbo, pi.map_or(std::ptr::null(), |b| b.ptr()))
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

    /// Device-side Sinkhorn HC gate split: mix[6*n_hc] -> coef buffer
    /// (pre | post | comb in kernel layout), zero host round-trips.
    #[allow(clippy::too_many_arguments)]
    pub fn dsv4_sinkhorn(coef: &mut DeviceBuf, mix: &DeviceBuf, scale: &DeviceBuf, base: &DeviceBuf, n_hc: u32, iters: u32, eps: f32, n_tok: u32) -> Result {
        check(unsafe { pulsar_dsv4_sinkhorn(coef.ptr_mut(), mix.ptr(), scale.ptr(), base.ptr(), n_hc, iters, eps, n_tok) }, "dsv4_sinkhorn")
    }

    /// hc_mix with device-resident coefficients (offsets into the
    /// sinkhorn coef buffer; blk_off -1 = no block gains).
    #[allow(clippy::too_many_arguments)]
    pub fn dsv4_hc_mix_dev(out: &mut DeviceBuf, streams: &DeviceBuf, block: Option<&DeviceBuf>, coef: &DeviceBuf, st_off: u32, blk_off: i32, n_embd: u32, n_hc: u32, n_out: u32, n_tok: u32) -> Result {
        check(unsafe { pulsar_dsv4_hc_mix_dev(out.ptr_mut(), streams.ptr(), block.map_or(std::ptr::null(), |b| b.ptr()), coef.ptr(), st_off, blk_off, n_embd, n_hc, n_out, n_tok) }, "dsv4_hc_mix_dev")
    }

    /// Streaming compressor step fully on-device; cache_row is the
    /// emit target (pass a dummy when emit is false).
    #[allow(clippy::too_many_arguments)]
    pub fn dsv4_comp_step(state_kv: &mut DeviceBuf, state_sc: &mut DeviceBuf, cache_row: &mut DeviceBuf, cache_off: usize, kv_cur: &DeviceBuf, sc_cur: &DeviceBuf, cur_off: usize, ape: &DeviceBuf, norm: &DeviceBuf, width: u32, head_dim: u32, ratio: u32, pos: u32, emit: bool, is_idx: bool, rms_eps: f32, r: &RopeCfg, kvq: u32, turbo: u32, pi: Option<&DeviceBuf>) -> Result {
        let row_ptr = unsafe { (cache_row.ptr_mut() as *mut u8).add(cache_off) as *mut c_void };
        let kv_ptr = unsafe { (kv_cur.ptr() as *const u8).add(cur_off) as *const c_void };
        let sc_ptr = unsafe { (sc_cur.ptr() as *const u8).add(cur_off) as *const c_void };
        check(
            unsafe {
                pulsar_dsv4_comp_step(state_kv.ptr_mut(), state_sc.ptr_mut(), row_ptr, kv_ptr, sc_ptr, ape.ptr(), norm.ptr(), width, head_dim, ratio, pos, emit as u32, is_idx as u32, rms_eps, 64, r.n_ctx_orig, r.freq_base, r.freq_scale, r.ext_factor, r.attn_factor, kvq, turbo, pi.map_or(std::ptr::null(), |b| b.ptr()))
            },
            "dsv4_comp_step",
        )
    }

    /// Store one dsv4 latent row into a ring/comp slot, quantized per kvq
    /// (0 f32, 1 fp8, 2 fp16, 3 int8, 4 q8_0, 5 q4_0) with optional turbo
    /// (pi rotation for block formats).
    #[allow(clippy::too_many_arguments)] // kernel launch mirrors the CUDA signature
    pub fn dsv4_kv_store(dst: &mut DeviceBuf, dst_off: usize, src: &DeviceBuf, src_off: usize, head_dim: u32, kvq: u32, turbo: u32, pi: Option<&DeviceBuf>) -> Result {
        let dst_ptr = unsafe { (dst.ptr_mut() as *mut u8).add(dst_off) as *mut c_void };
        let src_ptr = unsafe { (src.ptr() as *const u8).add(src_off) as *const c_void };
        check(
            unsafe { pulsar_dsv4_kv_store(dst_ptr, src_ptr, head_dim, kvq, turbo, pi.map_or(std::ptr::null(), |b| b.ptr())) },
            "dsv4_kv_store",
        )
    }

    pub fn dsv4_selftest() -> bool {
        unsafe { pulsar_dsv4_selftest() != 0 }
    }

    /// qwen35 GDN: depthwise conv (K taps) + silu over one token's qkv
    /// row; state [K-1][n_chan] rolls forward in place.
    pub fn qwen35_conv_step(out: &mut DeviceBuf, x: &DeviceBuf, kern: &DeviceBuf, state: &mut DeviceBuf, n_chan: u32, k: u32) -> Result {
        check(unsafe { pulsar_qwen35_conv_step(out.ptr_mut(), x.ptr(), kern.ptr(), state.ptr_mut(), n_chan, k) }, "qwen35_conv_step")
    }

    /// L2-normalize rows in place (ggml_l2_norm: x / sqrt(sum x^2 + eps)).
    pub fn qwen35_l2_norm(x: &mut DeviceBuf, rows: u32, dim: u32, eps: f32) -> Result {
        check(unsafe { pulsar_qwen35_l2_norm(x.ptr_mut(), rows, dim, eps) }, "qwen35_l2_norm")
    }

    /// Gated DeltaNet autoregressive step (one token, all heads):
    /// decay + delta-rule rank-1 state update + output, in place on
    /// state [h_v][dim][dim]. q/k are h_k-headed (repeat h_v/h_k).
    #[allow(clippy::too_many_arguments)]
    pub fn qwen35_gdn_step(out: &mut DeviceBuf, state: &mut DeviceBuf, q: &DeviceBuf, k: &DeviceBuf, v: &DeviceBuf, g: &DeviceBuf, beta: &DeviceBuf, h_v: u32, h_k: u32, dim: u32) -> Result {
        check(unsafe { pulsar_qwen35_gdn_step(out.ptr_mut(), state.ptr_mut(), q.ptr(), k.ptr(), v.ptr(), g.ptr(), beta.ptr(), h_v, h_k, dim) }, "qwen35_gdn_step")
    }

    /// Split the fused per-head [q dim | gate dim] projection into
    /// contiguous q and gate buffers.
    pub fn qwen35_split_gate(q: &mut DeviceBuf, gate: &mut DeviceBuf, fused: &DeviceBuf, n_head: u32, dim: u32) -> Result {
        check(unsafe { pulsar_qwen35_split_gate(q.ptr_mut(), gate.ptr_mut(), fused.ptr(), n_head, dim) }, "qwen35_split_gate")
    }

    /// x *= sigmoid(gate) elementwise.
    /// Laguna per-head output gate: x [n_tok][n_head][head_dim] scaled by
    /// softplus(gate[n_tok][n_head]), one scalar per head row.
    pub fn laguna_head_gate(x: &mut DeviceBuf, gate: &DeviceBuf, n_tok: u32, n_head: u32, head_dim: u32) -> Result {
        check(
            unsafe { pulsar_laguna_head_gate(x.ptr_mut(), gate.ptr(), n_tok, n_head, head_dim) },
            "laguna_head_gate",
        )
    }

    pub fn qwen35_sigmoid_gate(x: &mut DeviceBuf, gate: &DeviceBuf, n: u32) -> Result {
        check(unsafe { pulsar_qwen35_sigmoid_gate(x.ptr_mut(), gate.ptr(), n) }, "qwen35_sigmoid_gate")
    }

    /// Batched conv+silu: n_tok tokens sequentially in one launch.
    pub fn qwen35_conv_batch(out: &mut DeviceBuf, x: &DeviceBuf, kern: &DeviceBuf, state: &mut DeviceBuf, n_chan: u32, k: u32, n_tok: u32) -> Result {
        check(unsafe { pulsar_qwen35_conv_batch(out.ptr_mut(), x.ptr(), kern.ptr(), state.ptr_mut(), n_chan, k, n_tok) }, "qwen35_conv_batch")
    }

    pub fn q4e_hc_norm(y: &mut DeviceBuf, x: &DeviceBuf, gamma: &DeviceBuf, n_embd: u32, n_hc: u32, rows: u32, eps: f32) -> Result {
        check(unsafe { pulsar_q4e_hc_norm(y.ptr_mut(), x.ptr(), gamma.ptr(), n_embd, n_hc, rows, eps) }, "q4e_hc_norm")
    }
    pub fn q4e_hc_collapse(mixed: &mut DeviceBuf, xn: &DeviceBuf, gl: &DeviceBuf, n_embd: u32, n_hc: u32, rows: u32) -> Result {
        check(unsafe { pulsar_q4e_hc_collapse(mixed.ptr_mut(), xn.ptr(), gl.ptr(), n_embd, n_hc, rows) }, "q4e_hc_collapse")
    }
    pub fn q4e_stream_scatter(res: &mut DeviceBuf, out: &DeviceBuf, coef: &DeviceBuf, n_embd: u32, n_hc: u32, rows: u32, mode: u32) -> Result {
        check(unsafe { pulsar_q4e_stream_scatter(res.ptr_mut(), out.ptr(), coef.ptr(), n_embd, n_hc, rows, mode) }, "q4e_stream_scatter")
    }
    pub fn q4e_replicate(res: &mut DeviceBuf, emb: &DeviceBuf, n_embd: u32, n_hc: u32, rows: u32) -> Result {
        check(unsafe { pulsar_q4e_replicate(res.ptr_mut(), emb.ptr(), n_embd, n_hc, rows) }, "q4e_replicate")
    }
    pub fn q4e_silu_scale(x: &mut DeviceBuf, scale: f32, n: u64) -> Result {
        check(unsafe { pulsar_q4e_silu_scale(x.ptr_mut(), scale, n) }, "q4e_silu_scale")
    }
    pub fn q4e_ple_gate(coef: &mut DeviceBuf, keyn: &DeviceBuf, queryn: &DeviceBuf, n_embd: u32, n_hc: u32, rows: u32) -> Result {
        check(unsafe { pulsar_q4e_ple_gate(coef.ptr_mut(), keyn.ptr(), queryn.ptr(), n_embd, n_hc, rows) }, "q4e_ple_gate")
    }
    pub fn q4e_conv_dil(out: &mut DeviceBuf, x: &DeviceBuf, kern: &DeviceBuf, state: &mut DeviceBuf, n_chan: u32, k: u32, dil: u32, n_tok: u32) -> Result {
        check(unsafe { pulsar_q4e_conv_dil(out.ptr_mut(), x.ptr(), kern.ptr(), state.ptr_mut(), n_chan, k, dil, n_tok) }, "q4e_conv_dil")
    }

    /// Batched GDN delta rule: state columns ride registers across n_tok
    /// sequential steps (dim <= 128).
    #[allow(clippy::too_many_arguments)]
    pub fn qwen35_gdn_batch(out: &mut DeviceBuf, state: &mut DeviceBuf, q: &DeviceBuf, k: &DeviceBuf, v: &DeviceBuf, g: &DeviceBuf, beta: &DeviceBuf, h_v: u32, h_k: u32, dim: u32, n_tok: u32) -> Result {
        check(unsafe { pulsar_qwen35_gdn_batch(out.ptr_mut(), state.ptr_mut(), q.ptr(), k.ptr(), v.ptr(), g.ptr(), beta.ptr(), h_v, h_k, dim, n_tok) }, "qwen35_gdn_batch")
    }

    /// x[row] *= s[row] (batched per-token scalar gates).
    pub fn qwen35_row_scale(x: &mut DeviceBuf, s: &DeviceBuf, n_rows: u32, dim: u32) -> Result {
        check(unsafe { pulsar_qwen35_row_scale(x.ptr_mut(), s.ptr(), n_rows, dim) }, "qwen35_row_scale")
    }

    /// Non-causal GQA attention over contiguous K/V rows (DFlash draft).
    #[allow(clippy::too_many_arguments)]
    /// DFlash2 grouped block-conv (side 0 = prepare, 1 = finish). `delta`
    /// is the kernel-projection output [n_tok][2][taps][n_embd/group].
    pub fn qwen35_dflash2_conv(out: &mut DeviceBuf, h: &DeviceBuf, delta: &DeviceBuf, base: &DeviceBuf, n_tok: u32, n_embd: u32, taps: u32, group: u32, side: u32) -> Result {
        check(unsafe { pulsar_qwen35_dflash2_conv(out.ptr_mut(), h.ptr(), delta.ptr(), base.ptr(), n_tok, n_embd, taps, group, side) }, "qwen35_dflash2_conv")
    }

    /// One row-wise top-k extraction step: record slot `it` and mask it.
    /// Destructive on `logits`. k calls give each row's top-k ids/values.
    pub fn qwen35_dflash2_topk_step(logits: &mut DeviceBuf, n_rows: u32, vocab: u32, it: u32, k: u32, ids: &mut DeviceBuf, vals: &mut DeviceBuf) -> Result {
        check(unsafe { pulsar_qwen35_dflash2_topk_step(logits.ptr_mut(), n_rows, vocab, it, k, ids.ptr_mut(), vals.ptr_mut()) }, "qwen35_dflash2_topk_step")
    }

    /// Gather f16 codebook rows by token id into f32: out[i] = book[ids[i]].
    pub fn qwen35_dflash2_gather(out: &mut DeviceBuf, book: &DeviceBuf, ids: &DeviceBuf, n: u32, rank: u32) -> Result {
        check(unsafe { pulsar_qwen35_dflash2_gather(out.ptr_mut(), book.ptr(), ids.ptr(), n, rank) }, "qwen35_dflash2_gather")
    }

    pub fn qwen35_draft_attn(out: &mut DeviceBuf, q: &DeviceBuf, k: &DeviceBuf, v: &DeviceBuf, n_q: u32, n_kv: u32, n_head: u32, n_kv_head: u32, dim: u32, scale: f32) -> Result {
        check(unsafe { pulsar_qwen35_draft_attn(out.ptr_mut(), q.ptr(), k.ptr(), v.ptr(), n_q, n_kv, n_head, n_kv_head, dim, scale) }, "qwen35_draft_attn")
    }

    /// NEOX-paired YaRN rope over the full head (DFlash draft: trained
    /// with rope_scaling yarn; ggml semantics via RopeCfg).
    /// NEOX yarn rope over a partial rotation width (lanes >= rot_dim
    /// pass through). rot_dim == head_dim reproduces the full rotation.
    #[allow(clippy::too_many_arguments)]
    pub fn rope_yarn_partial(x: &mut DeviceBuf, n_tok: u32, n_head: u32, head_dim: u32, rot_dim: u32, pos0: u32, r: &RopeCfg) -> Result {
        check(
            unsafe {
                pulsar_rope_yarn_partial(x.ptr_mut(), n_tok, n_head, head_dim, rot_dim, pos0,
                    r.freq_base, r.freq_scale, r.ext_factor, r.attn_factor,
                    r.beta_fast, r.beta_slow, r.n_ctx_orig)
            },
            "rope_yarn_partial",
        )
    }

    pub fn qwen35_rope_yarn(x: &mut DeviceBuf, n_tok: u32, n_head: u32, head_dim: u32, pos0: u32, r: &RopeCfg) -> Result {
        check(unsafe { pulsar_qwen35_rope_yarn(x.ptr_mut(), n_tok, n_head, head_dim, pos0, r.freq_base, r.freq_scale, r.ext_factor, r.attn_factor, r.beta_fast, r.beta_slow, r.n_ctx_orig) }, "qwen35_rope_yarn")
    }

    /// Split conv rows [t][2*key+value] into contiguous q/k/v buffers.
    pub fn qwen35_split_qkv(q: &mut DeviceBuf, k: &mut DeviceBuf, v: &mut DeviceBuf, x: &DeviceBuf, n_tok: u32, key_dim: u32, value_dim: u32) -> Result {
        check(unsafe { pulsar_qwen35_split_qkv(q.ptr_mut(), k.ptr_mut(), v.ptr_mut(), x.ptr(), n_tok, key_dim, value_dim) }, "qwen35_split_qkv")
    }

    /// Scatter t rows into ring[(pos+i)%cap] at column offset ring_off.
    #[allow(clippy::too_many_arguments)]
    pub fn qwen35_ring_scatter(ring: &mut DeviceBuf, src: &DeviceBuf, pos: u32, cap: u32, n_rows: u32, row_elems: u32, ring_stride: u32, ring_off: u32) -> Result {
        check(unsafe { pulsar_qwen35_ring_scatter(ring.ptr_mut(), src.ptr(), pos, cap, n_rows, row_elems, ring_stride, ring_off) }, "qwen35_ring_scatter")
    }

    /// Gather n rows from ring[(start+i)%cap] into a contiguous dst.
    pub fn qwen35_ring_gather(dst: &mut DeviceBuf, ring: &DeviceBuf, start: u32, cap: u32, n_rows: u32, row_elems: u32) -> Result {
        check(unsafe { pulsar_qwen35_ring_gather(dst.ptr_mut(), ring.ptr(), start, cap, n_rows, row_elems) }, "qwen35_ring_gather")
    }

    /// In place over [n_tok][n_head]: g = a*softplus(g+dt), beta = sigmoid(beta).
    /// Coeffs over a PACKED [g | beta] row (the concatenated alpha/beta
    /// matmul at n_tok == 1): same kernel, offset pointers.
    pub fn qwen35_gdn_coeffs_packed(gb: &mut DeviceBuf, beta_off: usize, a: &DeviceBuf, dt: &DeviceBuf, n_head: u32) -> Result {
        check(
            unsafe {
                pulsar_qwen35_gdn_coeffs(
                    gb.ptr_mut(),
                    (gb.ptr_mut() as *mut u8).add(beta_off) as *mut c_void,
                    a.ptr(), dt.ptr(), 1, n_head,
                )
            },
            "qwen35_gdn_coeffs",
        )
    }

    /// gdn_batch reading g/beta from one packed row (n_tok == 1).
    #[allow(clippy::too_many_arguments)]
    pub fn qwen35_gdn_batch_packed(out: &mut DeviceBuf, state: &mut DeviceBuf, q: &DeviceBuf, k: &DeviceBuf, v: &DeviceBuf, gb: &DeviceBuf, beta_off: usize, h_v: u32, h_k: u32, dim: u32) -> Result {
        check(
            unsafe {
                pulsar_qwen35_gdn_batch(
                    out.ptr_mut(), state.ptr_mut(), q.ptr(), k.ptr(), v.ptr(),
                    gb.ptr(),
                    (gb.ptr() as *const u8).add(beta_off) as *const c_void,
                    h_v, h_k, dim, 1,
                )
            },
            "qwen35_gdn_batch",
        )
    }

    /// matmul_f32 with a byte offset into the weight (the concatenated
    /// alpha/beta tensor's halves at n_tok > 1).
    pub fn matmul_f32_off(out: &mut DeviceBuf, w: &DeviceBuf, w_off: usize, x: &DeviceBuf, in_dim: u32, out_dim: u32, n_tok: u32) -> Result {
        check(
            unsafe {
                pulsar_matmul_f32(
                    out.ptr_mut(),
                    (w.ptr() as *const u8).add(w_off) as *const c_void,
                    x.ptr(), in_dim, out_dim, n_tok,
                )
            },
            "matmul_f32",
        )
    }

    pub fn qwen35_gdn_coeffs(g_alpha: &mut DeviceBuf, beta: &mut DeviceBuf, a: &DeviceBuf, dt: &DeviceBuf, n_tok: u32, n_head: u32) -> Result {
        check(unsafe { pulsar_qwen35_gdn_coeffs(g_alpha.ptr_mut(), beta.ptr_mut(), a.ptr(), dt.ptr(), n_tok, n_head) }, "qwen35_gdn_coeffs")
    }

    /// x[row] *= sigmoid(s[row]).
    pub fn qwen35_row_sigmoid_scale(x: &mut DeviceBuf, s: &DeviceBuf, n_rows: u32, dim: u32) -> Result {
        check(unsafe { pulsar_qwen35_row_sigmoid_scale(x.ptr_mut(), s.ptr(), n_rows, dim) }, "qwen35_row_sigmoid_scale")
    }

    pub fn qwen35_selftest() -> bool {
        unsafe { pulsar_qwen35_selftest() != 0 }
    }

    /// KDA mixing coefficients in place: g becomes the lower-bounded
    /// channel-wise log-decay `g_min * sigmoid(exp(A_log) * (z + dt))`
    /// (g holds [n_tok][n_head][head_dim], `a` holds -exp(A_log)), and
    /// beta becomes sigmoid(beta) per head.
    #[allow(clippy::too_many_arguments)]
    pub fn k3_kda_coeffs(g: &mut DeviceBuf, beta: &mut DeviceBuf, a: &DeviceBuf, dt: &DeviceBuf, n_tok: u32, n_head: u32, head_dim: u32, g_min: f32) -> Result {
        check(unsafe { pulsar_k3_kda_coeffs(g.ptr_mut(), beta.ptr_mut(), a.ptr(), dt.ptr(), n_tok, n_head, head_dim, g_min) }, "k3_kda_coeffs")
    }

    /// Kimi Delta Attention autoregressive step (one token, all heads).
    /// Like `qwen35_gdn_step` but the forget gate is per key channel, so
    /// `g` is [n_head][dim] rather than [n_head]; q/k/v are all n_head-wide.
    #[allow(clippy::too_many_arguments)]
    pub fn k3_kda_step(out: &mut DeviceBuf, state: &mut DeviceBuf, q: &DeviceBuf, k: &DeviceBuf, v: &DeviceBuf, g: &DeviceBuf, beta: &DeviceBuf, n_head: u32, dim: u32) -> Result {
        check(unsafe { pulsar_k3_kda_step(out.ptr_mut(), state.ptr_mut(), q.ptr(), k.ptr(), v.ptr(), g.ptr(), beta.ptr(), n_head, dim) }, "k3_kda_step")
    }

    /// Attention Residuals: softmax over depth (the banked block
    /// checkpoints plus the live stream) scored by `w`, mixing the raw
    /// vectors. `n_ckpt == 0` copies `cur` through unchanged.
    #[allow(clippy::too_many_arguments)]
    pub fn k3_attn_res(out: &mut DeviceBuf, cur: &DeviceBuf, ckpt: Option<&DeviceBuf>, w: &DeviceBuf, n_tok: u32, n_embd: u32, n_ckpt: u32, eps: f32) -> Result {
        let ck = ckpt.map(|b| b.ptr()).unwrap_or(std::ptr::null());
        check(unsafe { pulsar_k3_attn_res(out.ptr_mut(), cur.ptr(), ck, w.ptr(), n_tok, n_embd, n_ckpt, eps) }, "k3_attn_res")
    }

    pub fn k3_selftest() -> bool {
        unsafe { pulsar_k3_selftest() != 0 }
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
    pub fn mla_store_compact_kv(kv_lora_cache: &mut DeviceBuf, k_rope_cache: &mut DeviceBuf, kv_norm: &DeviceBuf, kv_raw: &DeviceBuf, pos0: u32, n_tok: u32, cache_cap: u32, kv_raw_dim: u32, kv_lora_dim: u32, qk_rope: u32, kvq: u32) -> Result {
        check(
            unsafe {
                pulsar_mla_store_compact_kv(kv_lora_cache.ptr_mut(), k_rope_cache.ptr_mut(), kv_norm.ptr(), kv_raw.ptr(), pos0, n_tok, cache_cap, kv_raw_dim, kv_lora_dim, qk_rope, kvq)
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
    pub fn mla_attention(heads: &mut DeviceBuf, q: &DeviceBuf, qk_low: &DeviceBuf, kv_lora_cache: &DeviceBuf, k_rope_cache: &DeviceBuf, v_b: &DeviceBuf, selected: &DeviceBuf, n_tok: u32, n_selected: u32, cache_cap: u32, n_head: u32, kv_lora_dim: u32, qk_nope: u32, qk_rope: u32, value_dim: u32, r: &RopeCfg, kvq: u32) -> Result {
        check(
            unsafe {
                pulsar_mla_attention(heads.ptr_mut(), q.ptr(), qk_low.ptr(), kv_lora_cache.ptr(), k_rope_cache.ptr(), v_b.ptr(), selected.ptr(), n_tok, n_selected, cache_cap, n_head, kv_lora_dim, qk_nope, qk_rope, value_dim, r.n_ctx_orig, r.freq_base, r.freq_scale, r.ext_factor, r.attn_factor, r.beta_fast, r.beta_slow, r.kq_mult, kvq)
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

    /// Effective bandwidth of matmul_kq per quant on a dense-27B FFN
    /// shape - a probe, not a correctness test (weights are pseudorandom
    /// bytes; every wdot path is branchless so timing is data-blind).
    /// TpLink moves a whole buffer between two devices, not just its head.
    /// A partial transfer here is what made tensor parallel emit wrong
    /// tokens: byte 0 arrived, the middle did not.
    #[test]
    fn tplink_transfers_whole_buffer() {
        use super::*;
        if device_count() < 2 {
            eprintln!("tplink test: needs 2 devices, skipping");
            return;
        }
        let n = 5120usize; // one qwen35 row
        let src_host: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 - 2.0).collect();
        let primary = get_device();
        let other = (0..device_count()).find(|&d| d != primary).unwrap();

        set_device(primary).unwrap();
        let mut a = DeviceBuf::alloc(n * 4).unwrap();
        a.write(0, as_bytes(&src_host)).unwrap();
        let link = TpLink::new(n * 4, other).unwrap();

        set_device(other).unwrap();
        let mut b = DeviceBuf::alloc(n * 4).unwrap();
        let zero = vec![0u8; n * 4];
        b.write(0, &zero).unwrap();

        set_device(primary).unwrap();
        link.send(&a, n * 4).unwrap();
        set_device(other).unwrap();
        link.recv(&mut b, n * 4).unwrap();
        sync().unwrap();
        let got = b.read_f32(n).unwrap();
        set_device(primary).unwrap();

        let bad: Vec<usize> = (0..n).filter(|&i| got[i] != src_host[i]).collect();
        if !bad.is_empty() {
            eprintln!(
                "tplink: {} of {n} elements wrong; first {:?}, at {} got {} want {}",
                bad.len(), &bad[..bad.len().min(8)], bad[0], got[bad[0]], src_host[bad[0]]
            );
        }
        assert!(bad.is_empty(), "TpLink delivered a partial buffer");
    }

    #[test]
    fn kq_gemm_matches_reference() {
        use super::*;
        let out_dim = 100u32; // not a multiple of 32: exercises the row tail
        let in_dim = 512u32;
        let n_tok = 70u32; // >= 32 takes the gemm; 70 exercises the token tail
        let blocks = (in_dim / 256) as usize;
        for &(quant, bpb, d_off, name) in
            &[(QUANT_Q4_K, 144usize, 0usize, "q4_K"), (QUANT_Q6_K, 210, 208, "q6_K"),
              (QUANT_NVFP4, 144, 0, "nvfp4")]
        {
        let rb = blocks * bpb;
        let wbytes = out_dim as usize * rb;
        let mut host: Vec<u8> = (0..wbytes).map(|i| (i.wrapping_mul(2654435761) >> 7) as u8).collect();
        // pin every block's f16 scale fields to finite values
        for b in 0..out_dim as usize * blocks {
            if quant == QUANT_NVFP4 {
                // 4 sub-blocks of 36B; bytes 0..4 of each are UE4M3
                // scales, and 0 / 0x7F both mean "zero block" - pin a
                // finite one or the test compares zeros to zeros.
                for sb in 0..4 {
                    for k in 0..4 {
                        host[b * bpb + sb * 36 + k] = 0x40;
                    }
                }
            } else {
                host[b * bpb + d_off..b * bpb + d_off + 2].copy_from_slice(&0x3400u16.to_le_bytes()); // d = 0.25
                if quant == QUANT_Q4_K {
                    host[b * bpb + 2..b * bpb + 4].copy_from_slice(&0x3000u16.to_le_bytes()); // dmin = 0.125
                }
            }
        }
        let mut w = DeviceBuf::alloc(wbytes).unwrap();
        w.write(0, &host).unwrap();
        let x: Vec<f32> = (0..(in_dim * n_tok) as usize).map(|i| ((i * 37) % 97) as f32 * 0.01 - 0.5).collect();
        let mut xf = DeviceBuf::alloc(x.len() * 4).unwrap();
        xf.write(0, as_bytes(&x)).unwrap();
        let mut xq = DeviceBuf::alloc(n_tok as usize * blocks * Q8_K_BLOCK_BYTES).unwrap();
        quantize_q8_k(&mut xq, &xf, in_dim, n_tok).unwrap();
        let mut out = DeviceBuf::alloc((n_tok * out_dim) as usize * 4).unwrap();
        // reference: the proven grouped-16 path, same inputs
        std::env::set_var("PULSAR_NO_GEMM", "1");
        matmul_kq(&mut out, &w, &xq, in_dim, out_dim, n_tok, rb as u64, quant).unwrap();
        sync().unwrap();
        let want = out.read_f32((n_tok * out_dim) as usize).unwrap();
        std::env::remove_var("PULSAR_NO_GEMM");
        // both gemm flavors against the grouped reference: default (mma
        // on cc>=8) and the dp4a fallback
        let mut worst = 0f32;
        for force_dp4a in [false, true] {
            if force_dp4a { std::env::set_var("PULSAR_NO_MMA", "1"); }
            matmul_kq(&mut out, &w, &xq, in_dim, out_dim, n_tok, rb as u64, quant).unwrap();
            sync().unwrap();
            if force_dp4a { std::env::remove_var("PULSAR_NO_MMA"); }
            let got = out.read_f32((n_tok * out_dim) as usize).unwrap();
            for i in 0..got.len() {
                let d = (got[i] - want[i]).abs() / want[i].abs().max(1.0);
                if d > worst { worst = d; }
            }
        }
        eprintln!("kq gemm {name} vs reference: worst rel diff {worst:.2e}");
        // q6_K reads noisier than q4_K: signed scales cancel, so the
        // accumulation-order difference between the two paths surfaces
        // as ~3e-4 worst. Layout bugs measure in percent, not 1e-4;
        // greedy-ids equality at the engine level is the hard gate.
        assert!(worst < 2e-3, "{name} gemm diverges from the grouped path: {worst}");
        }
    }

    /// The FP4 GEMM is checked against a CPU decode of the very same
    /// bytes the hardware reads, so a wrong fragment layout or a
    /// misrouted scale lane shows up as a gross mismatch rather than
    /// hiding under a tolerance. Skips on non-Blackwell devices.
    #[test]
    fn nvfp4_a4_gemm_matches_reference() {
        use super::*;
        if cc_major() < 12 {
            eprintln!("nvfp4 a4: needs sm_120a, device is cc {}", cc_major());
            return;
        }
        fn e2m1(c: u8) -> f32 {
            const M: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
            let v = M[(c & 7) as usize];
            if c & 8 != 0 { -v } else { v }
        }
        fn ue4m3(x: u8) -> f32 {
            if x == 0 || x == 0x7F { return 0.0; }
            let e = ((x >> 3) & 0xF) as i32;
            let m = (x & 7) as f32;
            if e == 0 { m * 2f32.powi(-9) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) }
        }
        // 36B block: 4 scale bytes then 32 nibble bytes, low nibbles
        // values 0..7 of a 16-value sub-block and high nibbles 8..15
        fn decode(bytes: &[u8], n: usize) -> Vec<f32> {
            let mut out = vec![0f32; n];
            for sb in 0..n / 256 {
                for b in 0..4 {
                    let base = sb * 144 + b * 36;
                    for s in 0..4 {
                        let sc = ue4m3(bytes[base + s]);
                        for i in 0..8 {
                            let by = bytes[base + 4 + s * 8 + i];
                            out[sb * 256 + b * 64 + s * 16 + i] = e2m1(by & 0xF) * sc;
                            out[sb * 256 + b * 64 + s * 16 + i + 8] = e2m1(by >> 4) * sc;
                        }
                    }
                }
            }
            out
        }

        let in_dim = 512u32;
        let out_dim = 96u32;
        let n_tok = 70u32; // deliberately not a multiple of 64: tail path
        let blocks = (in_dim / 256) as usize;
        let rb = blocks * NVFP4_SB_BYTES;

        let mut host: Vec<u8> = (0..out_dim as usize * rb)
            .map(|i| (i.wrapping_mul(2654435761) >> 7) as u8)
            .collect();
        // pin every scale byte finite and nonzero: 0 is a zero block and
        // 0x7F is NaN in hardware even though ggml reads it as zero
        for b in 0..out_dim as usize * blocks * 4 {
            for s in 0..4 {
                host[b * 36 + s] = 0x30 + ((b + s) % 12) as u8;
            }
        }
        let mut w = DeviceBuf::alloc(host.len()).unwrap();
        w.write(0, &host).unwrap();

        let x: Vec<f32> = (0..(in_dim * n_tok) as usize)
            .map(|i| ((i * 37) % 97) as f32 * 0.01 - 0.5)
            .collect();
        let mut xf = DeviceBuf::alloc(x.len() * 4).unwrap();
        xf.write(0, as_bytes(&x)).unwrap();
        let abytes = nvfp4_act_bytes(in_dim, n_tok);
        let mut xq = DeviceBuf::alloc(abytes).unwrap();
        quantize_nvfp4(&mut xq, &xf, in_dim, n_tok).unwrap();
        sync().unwrap();
        let mut aq = vec![0u8; abytes];
        xq.read(0, &mut aq).unwrap();
        // activations land octet-major; rebuild the plain per-token
        // layout so one decoder serves both operands
        let mut aplain = vec![0u8; n_tok as usize * rb];
        for t in 0..n_tok as usize {
            let (o, r) = (t / 8, t % 8);
            for sb in 0..blocks {
                for kb in 0..4 {
                    let src = (o * blocks + sb) * 1152 + kb * 288;
                    let dst = t * rb + sb * 144 + kb * 36;
                    aplain[dst..dst + 4]
                        .copy_from_slice(&aq[src + 256 + r * 4..src + 260 + r * 4]);
                    aplain[dst + 4..dst + 36]
                        .copy_from_slice(&aq[src + r * 32..src + r * 32 + 32]);
                }
            }
        }
        let aq = aplain;

        // the quantizer stands on its own: e2m1 with a scale of absmax/6
        // cannot err by more than about absmax/6 anywhere
        let adec = decode(&aq, (in_dim * n_tok) as usize);
        for t in 0..n_tok as usize {
            for sub in 0..(in_dim / 16) as usize {
                let o = t * in_dim as usize + sub * 16;
                let amax = (0..16).fold(0f32, |m, i| m.max(x[o + i].abs()));
                for i in 0..16 {
                    let e = (adec[o + i] - x[o + i]).abs();
                    assert!(e <= 0.25 * amax + 1e-6,
                        "nvfp4 quantize err {e} at t{t} sub{sub} i{i} (amax {amax})");
                }
            }
        }

        let wdec = decode(&host, out_dim as usize * in_dim as usize);
        let mut want = vec![0f32; (n_tok * out_dim) as usize];
        for t in 0..n_tok as usize {
            for r in 0..out_dim as usize {
                let mut acc = 0f64;
                for k in 0..in_dim as usize {
                    acc += (wdec[r * in_dim as usize + k] * adec[t * in_dim as usize + k]) as f64;
                }
                want[t * out_dim as usize + r] = acc as f32;
            }
        }

        let mut out = DeviceBuf::alloc((n_tok * out_dim) as usize * 4).unwrap();
        matmul_nvfp4_a4(&mut out, &w, &xq, in_dim, out_dim, n_tok, rb as u64).unwrap();
        sync().unwrap();
        let got = out.read_f32((n_tok * out_dim) as usize).unwrap();

        let mut worst = 0f32;
        for i in 0..got.len() {
            let d = (got[i] - want[i]).abs() / want[i].abs().max(1.0);
            if d > worst { worst = d; }
        }
        eprintln!("nvfp4 a4 gemm vs cpu decode: worst rel diff {worst:.2e}");
        // both sides sum the identical decoded values, so only f32
        // accumulation order differs; a layout bug measures in percent
        assert!(worst < 1e-4, "nvfp4 a4 gemm diverges from the decode: {worst}");
    }

    /// Where does the W4A4 error actually come from? Compares three
    /// activation routes against an f32 reference over identical NVFP4
    /// weights, so the fp4-activation cost is separated from the
    /// double-quantization cost of going through q8_K.
    ///
    /// cargo test --release -p kernels nvfp4_activation_error -- --ignored --nocapture
    #[test]
    #[ignore = "diagnostic, requires a Blackwell CUDA device"]
    fn nvfp4_activation_error() {
        use super::*;
        if cc_major() < 12 {
            eprintln!("needs sm_120a, device is cc {}", cc_major());
            return;
        }
        fn e2m1(c: u8) -> f32 {
            const M: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
            let v = M[(c & 7) as usize];
            if c & 8 != 0 { -v } else { v }
        }
        fn ue4m3(x: u8) -> f32 {
            if x == 0 || x == 0x7F { return 0.0; }
            let e = ((x >> 3) & 0xF) as i32;
            let m = (x & 7) as f32;
            if e == 0 { m * 2f32.powi(-9) } else { (1.0 + m / 8.0) * 2f32.powi(e - 7) }
        }
        fn decode(bytes: &[u8], n: usize) -> Vec<f32> {
            let mut out = vec![0f32; n];
            for sb in 0..n / 256 {
                for b in 0..4 {
                    let base = sb * 144 + b * 36;
                    for sq in 0..4 {
                        let sc = ue4m3(bytes[base + sq]);
                        for i in 0..8 {
                            let by = bytes[base + 4 + sq * 8 + i];
                            out[sb * 256 + b * 64 + sq * 16 + i] = e2m1(by & 0xF) * sc;
                            out[sb * 256 + b * 64 + sq * 16 + i + 8] = e2m1(by >> 4) * sc;
                        }
                    }
                }
            }
            out
        }

        let in_dim = 2048u32;
        let out_dim = 256u32;
        let n_tok = 64u32;
        let blocks = (in_dim / 256) as usize;
        let rb = blocks * NVFP4_SB_BYTES;

        let mut host: Vec<u8> = (0..out_dim as usize * rb)
            .map(|i| (i.wrapping_mul(2654435761) >> 7) as u8)
            .collect();
        for b in 0..out_dim as usize * blocks * 4 {
            for sq in 0..4 { host[b * 36 + sq] = 0x30 + ((b + sq) % 12) as u8; }
        }
        let mut w = DeviceBuf::alloc(host.len()).unwrap();
        w.write(0, &host).unwrap();
        let wdec = decode(&host, out_dim as usize * in_dim as usize);

        // gaussian-ish activations: a flat ramp hides how badly fp4 handles
        // the heavy tail that real hidden states have
        let x: Vec<f32> = (0..(in_dim * n_tok) as usize)
            .map(|i| {
                let a = ((i * 2654435761usize) % 10007) as f32 / 10007.0 - 0.5;
                let b = ((i * 40503usize) % 9973) as f32 / 9973.0 - 0.5;
                (a + b) * 2.0
            })
            .collect();
        let mut xf = DeviceBuf::alloc(x.len() * 4).unwrap();
        xf.write(0, as_bytes(&x)).unwrap();

        let mut want = vec![0f32; (n_tok * out_dim) as usize];
        for t in 0..n_tok as usize {
            for r in 0..out_dim as usize {
                let mut acc = 0f64;
                for k in 0..in_dim as usize {
                    acc += (wdec[r * in_dim as usize + k] * x[t * in_dim as usize + k]) as f64;
                }
                want[t * out_dim as usize + r] = acc as f32;
            }
        }
        let rms_ref = (want.iter().map(|v| (v * v) as f64).sum::<f64>() / want.len() as f64).sqrt();

        let rel = |got: &[f32]| -> f64 {
            let e = got.iter().zip(&want).map(|(g, w)| ((g - w) as f64).powi(2)).sum::<f64>();
            (e / got.len() as f64).sqrt() / rms_ref
        };

        let mut out = DeviceBuf::alloc((n_tok * out_dim) as usize * 4).unwrap();
        let abytes = nvfp4_act_bytes(in_dim, n_tok);

        // A: f32 -> NVFP4 directly
        let mut a4 = DeviceBuf::alloc(abytes).unwrap();
        quantize_nvfp4(&mut a4, &xf, in_dim, n_tok).unwrap();
        matmul_nvfp4_a4(&mut out, &w, &a4, in_dim, out_dim, n_tok, rb as u64).unwrap();
        sync().unwrap();
        let ea = rel(&out.read_f32((n_tok * out_dim) as usize).unwrap());

        // B: f32 -> q8_K -> NVFP4, what matmul_kq does under PULSAR_FP4
        let mut x8 = DeviceBuf::alloc(n_tok as usize * blocks * Q8_K_BLOCK_BYTES).unwrap();
        quantize_q8_k(&mut x8, &xf, in_dim, n_tok).unwrap();
        let mut b4 = DeviceBuf::alloc(abytes).unwrap();
        nvfp4_from_q8k(&mut b4, &x8, in_dim, n_tok).unwrap();
        matmul_nvfp4_a4(&mut out, &w, &b4, in_dim, out_dim, n_tok, rb as u64).unwrap();
        sync().unwrap();
        let eb = rel(&out.read_f32((n_tok * out_dim) as usize).unwrap());

        // C: q8_K activations through the int8 mma, the shipping path
        matmul_kq(&mut out, &w, &x8, in_dim, out_dim, n_tok, rb as u64, QUANT_NVFP4).unwrap();
        sync().unwrap();
        let ec = rel(&out.read_f32((n_tok * out_dim) as usize).unwrap());

        eprintln!("activation error vs f32 reference (relative RMS):");
        eprintln!("  A  f32 -> nvfp4        {ea:.4}");
        eprintln!("  B  f32 -> q8_K -> nvfp4 {eb:.4}");
        eprintln!("  C  q8_K, int8 mma      {ec:.4}   <- shipping path");
        eprintln!("  fp4 activation cost over int8: {:.1}x", ea / ec);
        eprintln!("  double-quantization cost:      {:.2}x", eb / ea);
    }

    /// TP shard contract: matmul_kq over a matw_col_half-style shard
    /// (first half of every row's bytes, row_bytes halved, in_dim halved)
    /// must equal matmul_kq over the full row with the upper half of the
    /// activations zeroed. n_tok 1 / 8 / 64 cover the dot, token-tile
    /// and GEMM dispatch paths. If this fails, the sliced-row_bytes
    /// kernel path is the TP bug; if it passes, the fault is in how the
    /// engine builds the shard at load.
    ///
    /// cargo test --release -p kernels nvfp4_col_shard -- --ignored --nocapture
    #[test]
    #[ignore = "requires a CUDA device"]
    fn nvfp4_col_shard_matches_full() {
        use super::*;
        let in_dim = 6144u32; // ssm_out's real value_dim
        let half = in_dim / 2;
        let out_dim = 64u32;
        let blocks = (in_dim / 256) as usize; // 24 superblocks per row
        let rb = blocks * NVFP4_SB_BYTES; // 3456
        let hb = rb / 2; // 1728, superblock-aligned

        let mut host: Vec<u8> = (0..out_dim as usize * rb)
            .map(|i| (i.wrapping_mul(2654435761) >> 7) as u8)
            .collect();
        for b in 0..out_dim as usize * blocks * 4 {
            for sq in 0..4 {
                host[b * 36 + sq] = 0x30 + ((b + sq) % 12) as u8;
            }
        }
        let mut w_full = DeviceBuf::alloc(host.len()).unwrap();
        w_full.write(0, &host).unwrap();
        // matw_col_half, verbatim: first hb bytes of every row, contiguous
        let mut sl = Vec::with_capacity(out_dim as usize * hb);
        for r in 0..out_dim as usize {
            sl.extend_from_slice(&host[r * rb..r * rb + hb]);
        }
        let mut w_half = DeviceBuf::alloc(sl.len()).unwrap();
        w_half.write(0, &sl).unwrap();

        for &n_tok in &[1u32, 8, 64] {
            let x: Vec<f32> = (0..(in_dim * n_tok) as usize)
                .map(|i| {
                    if (i as u32 % in_dim) >= half {
                        0.0
                    } else {
                        ((i * 2654435761usize) % 10007) as f32 / 10007.0 - 0.5
                    }
                })
                .collect();
            let xh: Vec<f32> = (0..n_tok as usize)
                .flat_map(|t| {
                    let b = t * in_dim as usize;
                    x[b..b + half as usize].to_vec()
                })
                .collect();

            let mut xf = DeviceBuf::alloc(x.len() * 4).unwrap();
            xf.write(0, as_bytes(&x)).unwrap();
            let mut xq = DeviceBuf::alloc(n_tok as usize * blocks * Q8_K_BLOCK_BYTES).unwrap();
            quantize_q8_k(&mut xq, &xf, in_dim, n_tok).unwrap();

            let mut xfh = DeviceBuf::alloc(xh.len() * 4).unwrap();
            xfh.write(0, as_bytes(&xh)).unwrap();
            let mut xqh = DeviceBuf::alloc(n_tok as usize * (blocks / 2) * Q8_K_BLOCK_BYTES).unwrap();
            quantize_q8_k(&mut xqh, &xfh, half, n_tok).unwrap();

            let n_out = (n_tok * out_dim) as usize;
            let mut out_f = DeviceBuf::alloc(n_out * 4).unwrap();
            matmul_kq(&mut out_f, &w_full, &xq, in_dim, out_dim, n_tok, rb as u64, QUANT_NVFP4).unwrap();
            let mut out_h = DeviceBuf::alloc(n_out * 4).unwrap();
            matmul_kq(&mut out_h, &w_half, &xqh, half, out_dim, n_tok, hb as u64, QUANT_NVFP4).unwrap();
            sync().unwrap();

            let a = out_f.read_f32(n_out).unwrap();
            let b = out_h.read_f32(n_out).unwrap();
            let mut worst = 0f32;
            let mut wi = 0usize;
            for i in 0..n_out {
                let d = (a[i] - b[i]).abs();
                if d > worst {
                    worst = d;
                    wi = i;
                }
            }
            eprintln!(
                "n_tok {n_tok:>2}: max |full - shard| = {worst:.3e} at [{wi}] (full {} shard {})",
                a[wi], b[wi]
            );
            assert!(worst < 1e-4, "n_tok {n_tok}: shard diverges from full at [{wi}]");
        }
    }

    /// cargo test --release -p kernels nvfp4_a4_bench -- --ignored --nocapture
    ///
    /// The int8 mma path and the FP4 path over the IDENTICAL weight
    /// bytes, so the only difference measured is how they are consumed.
    #[test]
    #[ignore = "perf probe, requires a Blackwell CUDA device"]
    fn nvfp4_a4_bench() {
        use super::*;
        use std::time::Instant;
        if cc_major() < 12 {
            eprintln!("nvfp4 a4 bench: needs sm_120a, device is cc {}", cc_major());
            return;
        }
        let out_dim = 16384u32;
        let in_dim = 4096u32;
        let blocks = (in_dim / 256) as usize;
        let rb = blocks * NVFP4_SB_BYTES;
        for n_tok in [128u32, 512, 2048] {
        eprintln!(" n_tok {n_tok}:");

        let mut host: Vec<u8> = (0..out_dim as usize * rb)
            .map(|i| (i.wrapping_mul(2654435761) >> 7) as u8)
            .collect();
        for b in 0..out_dim as usize * blocks * 4 {
            for sc in 0..4 {
                host[b * 36 + sc] = 0x30 + ((b + sc) % 12) as u8;
            }
        }
        let mut w = DeviceBuf::alloc(host.len()).unwrap();
        w.write(0, &host).unwrap();

        let x: Vec<f32> = (0..(in_dim * n_tok) as usize)
            .map(|i| ((i * 37) % 97) as f32 * 0.01 - 0.5)
            .collect();
        let mut xf = DeviceBuf::alloc(x.len() * 4).unwrap();
        xf.write(0, as_bytes(&x)).unwrap();

        let mut x8 = DeviceBuf::alloc(n_tok as usize * blocks * Q8_K_BLOCK_BYTES).unwrap();
        let mut x4 = DeviceBuf::alloc(nvfp4_act_bytes(in_dim, n_tok)).unwrap();
        let mut out = DeviceBuf::alloc((n_tok * out_dim) as usize * 4).unwrap();

        let iters = 50;
        for (name, fp4) in [("int8 mma (q8_K acts)", false), ("fp4  mma (nvfp4 acts)", true)] {
            // warm up and include the activation quantize, which the
            // engine pays once per matmul either way
            for _ in 0..3 {
                if fp4 {
                    quantize_nvfp4(&mut x4, &xf, in_dim, n_tok).unwrap();
                    matmul_nvfp4_a4(&mut out, &w, &x4, in_dim, out_dim, n_tok, rb as u64).unwrap();
                } else {
                    quantize_q8_k(&mut x8, &xf, in_dim, n_tok).unwrap();
                    matmul_kq(&mut out, &w, &x8, in_dim, out_dim, n_tok, rb as u64, QUANT_NVFP4).unwrap();
                }
            }
            sync().unwrap();
            let t0 = Instant::now();
            for _ in 0..iters {
                if fp4 {
                    matmul_nvfp4_a4(&mut out, &w, &x4, in_dim, out_dim, n_tok, rb as u64).unwrap();
                } else {
                    matmul_kq(&mut out, &w, &x8, in_dim, out_dim, n_tok, rb as u64, QUANT_NVFP4).unwrap();
                }
            }
            sync().unwrap();
            let ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;
            let gflop = 2.0 * out_dim as f64 * in_dim as f64 * n_tok as f64 / 1e9;
            eprintln!("  {name}  {ms:8.3} ms  {:8.1} GFLOP/s", gflop / (ms * 1e-3));
        }
        }
    }

    /// Prefill attention at the qwen35 TP-shard shape (12 heads / 2 kv /
    /// head_dim 256, one 128-token chunk against a deep f32 cache).
    /// PULSAR_ATTN_TILE2=0/1 picks v1/v2 for A/B.
    /// cargo test --release -p kernels gqa_tile_bench -- --ignored --nocapture
    #[test]
    #[ignore = "perf probe, requires a CUDA device"]
    fn gqa_tile_bench() {
        use super::*;
        let (n_head, n_kv, hd, cap) = (12u32, 2u32, 256u32, 8192u32);
        let (n_tok, pos0) = (128u32, 3500u32);
        let scale = 1.0 / (hd as f32).sqrt();
        let c_elems = (n_kv * cap * hd) as usize;
        let q_elems = (n_tok * n_head * hd) as usize;
        let rnd = |i: usize| ((i * 37) % 97) as f32 * 0.01 - 0.5;
        let kc_h: Vec<f32> = (0..c_elems).map(rnd).collect();
        let q_h: Vec<f32> = (0..q_elems).map(|i| rnd(i + 13)).collect();
        let mut kc = DeviceBuf::alloc(c_elems * 4).unwrap();
        let mut vc = DeviceBuf::alloc(c_elems * 4).unwrap();
        let mut q = DeviceBuf::alloc(q_elems * 4).unwrap();
        kc.write(0, as_bytes(&kc_h)).unwrap();
        vc.write(0, as_bytes(&kc_h)).unwrap();
        q.write(0, as_bytes(&q_h)).unwrap();
        let mut out = DeviceBuf::alloc(q_elems * 4).unwrap();
        /* PULSAR_BENCH_L2BUST=1: cycle 6 distinct cache pairs so L2 never
         * retains K/V between launches - the production regime, where the
         * FFN GEMMs stream GBs of weights between attention launches. */
        let bust = std::env::var("PULSAR_BENCH_L2BUST").ok().as_deref() == Some("1");
        let n_sets = if bust { 6 } else { 1 };
        let mut sets: Vec<(DeviceBuf, DeviceBuf)> = Vec::new();
        for _ in 0..n_sets {
            let mut k2 = DeviceBuf::alloc(c_elems * 4).unwrap();
            let mut v2 = DeviceBuf::alloc(c_elems * 4).unwrap();
            k2.write(0, as_bytes(&kc_h)).unwrap();
            v2.write(0, as_bytes(&kc_h)).unwrap();
            sets.push((k2, v2));
        }
        let _ = (&kc, &vc);
        for _ in 0..5 {
            gqa_attention(&mut out, &q, &sets[0].0, &sets[0].1, n_tok, n_head, n_kv, hd, cap, pos0, scale, 0).unwrap();
        }
        sync().unwrap();
        let iters = 60;
        let t0 = std::time::Instant::now();
        for i in 0..iters {
            let (kx, vx) = &sets[i % n_sets];
            gqa_attention(&mut out, &q, kx, vx, n_tok, n_head, n_kv, hd, cap, pos0, scale, 0).unwrap();
        }
        sync().unwrap();
        let dt = t0.elapsed().as_secs_f64() / iters as f64;
        let rows = (pos0 + n_tok / 2) as f64; // avg visible rows per query
        let flops = 4.0 * rows * hd as f64 * n_tok as f64 * n_head as f64; // qk + pv
        eprintln!(
            "gqa tile bench (tile2={}): {:7.0} us, {:5.2} TFLOP/s",
            std::env::var("PULSAR_ATTN_TILE2").unwrap_or_else(|_| "default".into()),
            dt * 1e6,
            flops / dt / 1e12
        );
    }

    /// cargo test --release -p kernels kq_gemm_bench -- --ignored --nocapture
    #[test]
    #[ignore = "perf probe, requires a CUDA device"]
    fn kq_gemm_bench() {
        use super::*;
        let out_dim = 16384u32;
        let in_dim = 4096u32;
        let n_tok = 128u32;
        let blocks = (in_dim / 256) as usize;
        for &(quant, bpb, d_off, name) in
            &[(QUANT_Q4_K, 144usize, 0usize, "q4_K"), (QUANT_Q6_K, 210, 208, "q6_K")]
        {
            let rb = blocks * bpb;
            let wbytes = out_dim as usize * rb;
            let mut host: Vec<u8> = (0..wbytes).map(|i| (i.wrapping_mul(2654435761) >> 7) as u8).collect();
            for b in 0..out_dim as usize * blocks {
                host[b * bpb + d_off..b * bpb + d_off + 2].copy_from_slice(&0x3400u16.to_le_bytes());
                if quant == QUANT_Q4_K {
                    host[b * bpb + 2..b * bpb + 4].copy_from_slice(&0x3000u16.to_le_bytes());
                }
            }
            let mut w = DeviceBuf::alloc(wbytes).unwrap();
            w.write(0, &host).unwrap();
            let x: Vec<f32> = (0..(in_dim * n_tok) as usize).map(|i| ((i * 37) % 97) as f32 * 0.01 - 0.5).collect();
            let mut xf = DeviceBuf::alloc(x.len() * 4).unwrap();
            xf.write(0, as_bytes(&x)).unwrap();
            let mut xq = DeviceBuf::alloc(n_tok as usize * blocks * Q8_K_BLOCK_BYTES).unwrap();
            quantize_q8_k(&mut xq, &xf, in_dim, n_tok).unwrap();
            let mut out = DeviceBuf::alloc((n_tok * out_dim) as usize * 4).unwrap();
            for _ in 0..5 {
                matmul_kq(&mut out, &w, &xq, in_dim, out_dim, n_tok, rb as u64, quant).unwrap();
            }
            sync().unwrap();
            let iters = 200;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                matmul_kq(&mut out, &w, &xq, in_dim, out_dim, n_tok, rb as u64, quant).unwrap();
            }
            sync().unwrap();
            let dt = t0.elapsed().as_secs_f64() / iters as f64;
            let macs = in_dim as f64 * out_dim as f64 * n_tok as f64;
            eprintln!("kq gemm bench {name}: {:6.0} us, {:5.1} TOPS, weights {:5.1} GB/s",
                    dt * 1e6, 2.0 * macs / dt / 1e12, wbytes as f64 / dt / 1e9);
        }
    }

    /// cargo test --release -p kernels kq_bandwidth -- --ignored --nocapture
    #[test]
    #[ignore = "perf probe, requires a CUDA device"]
    fn kq_bandwidth_probe() {
        use super::*;
        let rows = 17408u32;
        let in_dim = 5120u32;
        let blocks = (in_dim / 256) as usize;
        for &(q, name, bpb) in &[
            (QUANT_Q4_K, "q4_K", 144usize),
            (QUANT_Q5_K, "q5_K", 176),
            (QUANT_Q6_K, "q6_K", 210),
            (QUANT_IQ4_XS, "iq4_xs", 136),
            (QUANT_NVFP4, "nvfp4", 144),
        ] {
            let rb = blocks * bpb;
            let wbytes = rows as usize * rb;
            let mut w = DeviceBuf::alloc(wbytes).unwrap();
            let host: Vec<u8> = (0..wbytes).map(|i| (i.wrapping_mul(2654435761) >> 7) as u8).collect();
            w.write(0, &host).unwrap();
            let x: Vec<f32> = (0..in_dim as usize).map(|i| ((i * 37) % 97) as f32 * 0.01 - 0.5).collect();
            let mut xf = DeviceBuf::alloc(in_dim as usize * 4).unwrap();
            xf.write(0, as_bytes(&x)).unwrap();
            let mut xq = DeviceBuf::alloc(blocks * Q8_K_BLOCK_BYTES).unwrap();
            quantize_q8_k(&mut xq, &xf, in_dim, 1).unwrap();
            let mut out = DeviceBuf::alloc(rows as usize * 4).unwrap();
            for _ in 0..3 {
                matmul_kq(&mut out, &w, &xq, in_dim, rows, 1, rb as u64, q).unwrap();
            }
            sync().unwrap();
            let iters = 200;
            let t0 = std::time::Instant::now();
            for _ in 0..iters {
                matmul_kq(&mut out, &w, &xq, in_dim, rows, 1, rb as u64, q).unwrap();
            }
            sync().unwrap();
            let dt = t0.elapsed().as_secs_f64() / iters as f64;
            eprintln!("kq {name}: {:6.1} GB/s ({:.0} us, {}MB)", wbytes as f64 / dt / 1e9, dt * 1e6, wbytes >> 20);
        }
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
        for (i, &v) in y.iter().enumerate() {
            assert_eq!(v, 3.0 * i as f32);
        }
    }

    /// Weight VRAM budget: exhausted -> pinned host fallback; refilled ->
    /// VRAM again. Same bytes either way.
    #[test]
    #[ignore = "requires a CUDA device"]
    fn weight_budget_pins_overflow() {
        const DEV: i32 = 0;
        super::set_device(DEV).unwrap();
        let data = vec![7u8; 4096];
        super::set_weight_vram_budget(DEV, 0);
        let p = super::DeviceBuf::from_bytes(&data).unwrap();
        assert!(p.is_pinned());
        let mut back = vec![0u8; 4096];
        p.read(0, &mut back).unwrap();
        assert_eq!(back[0], 7);
        super::set_weight_vram_budget(DEV, 1 << 20);
        let v = super::DeviceBuf::from_bytes(&data).unwrap();
        assert!(!v.is_pinned());
        super::set_weight_vram_budget(DEV, i64::MAX);
    }
}
