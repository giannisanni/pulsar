//! hy-v3 (Hy3) forward graph over the pulsar CUDA kernels.
//!
//! Op sequence is ds4's `hy3_forward_token`, the decode-parity reference:
//! embed -> per layer [rms-norm, qkv (q8_0), per-head q/k norm, neox rope,
//! kv append, gqa attention, out-proj, residual; rms-norm, dense FFN (layer
//! 0) or sigmoid-router MoE (shared expert + streamed routed experts)] ->
//! final norm -> lm head.
//!
//! Expert streaming: three tiers per layer step. A VRAM hot-set cache
//! (touch-count admission, so it never thrashes even though one token's
//! working set exceeds the pool), then an LFU host cache, then io_uring
//! batch reads whose completions overlap the H2D uploads. The MoE kernels
//! always receive explicit per-slot device pointers, wherever the bytes
//! ended up.

#[cfg(target_os = "linux")]
mod real {
    mod dsv4;
    mod qwen35;
    pub use qwen35::{generate_dflash, DraftModel};

    use std::fs::File;
    use std::os::unix::fs::FileExt;
    use std::path::Path;

    use gguf::{Gguf, TensorInfo, TensorType, Value};
    use kernels::{DeviceBuf, ExpertPtrs};

    pub type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

    fn meta_err(key: &str) -> Box<dyn std::error::Error> {
        format!("gguf metadata missing/bad: {key}").into()
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Family {
        /// Plain GQA attention (Hy3 / hy-v3).
        Gqa,
        /// Multi-head latent attention, compact-KV path (GLM-5.2 /
        /// glm-dsa; no DSA indexer, so contexts up to indexer_top_k only).
        Mla,
        /// DeepSeek-V4-Flash (deepseek4): 4-stream hyper-connection
        /// residual, sink attention over a raw SWA ring + streaming
        /// compressed KV, tid2eid hash routing on the first layers.
        /// Decode-only graph; prefill loops tokens (the compressor and
        /// SWA ring are sequential state machines).
        Dsv4,
        /// Qwen3.5/3.6 MoE hybrid (qwen35moe): Gated DeltaNet linear
        /// attention on 3 of 4 layers (O(1) recurrent state, no KV),
        /// sigmoid-gated full attention on the rest. Decode-only graph;
        /// prefill loops tokens (conv window + delta state are
        /// sequential).
        Qwen35,
    }

    #[derive(Debug, Clone, Copy)]
    pub struct Shape {
        pub family: Family,
        pub n_embd: u32,
        pub n_head: u32,
        pub n_head_kv: u32,
        pub head_dim: u32,
        pub n_layer: u32,
        pub n_exec_layer: u32,
        pub n_leading_dense: u32,
        pub n_expert: u32,
        pub n_expert_used: u32,
        pub n_ff_exp: u32,
        pub n_ff_dense: u32,
        pub n_vocab: u32,
        pub expert_weight_scale: f32,
        /// qwen3moe: softmax router, no bias, normalize top-k probs.
        /// false = sigmoid router (Hy3/GLM/DeepSeek/MiniMax lineage).
        pub router_softmax: bool,
        /// expert gate activation: 0 = silu, 1 = gelu tanh (gemma4)
        pub moe_act_op: u32,
        pub rope_freq_base: f32,
        pub rms_eps: f32,
        // MLA only (zero for Gqa)
        pub n_lora_q: u32,
        pub n_kv_lora: u32,
        pub qk_nope: u32,
        pub qk_rope: u32,
        pub value_mla: u32,
        pub rope_orig_ctx: u32,
        /// GQA rotary width (partial rotary when < head_dim; MiniMax M3
        /// rotates 64 of 128). Hy3 rotates the full head.
        pub rot_dim: u32,
        // DSA lightning indexer (zero when absent -> ctx capped at 2048)
        pub n_idx_head: u32,
        pub n_idx_dim: u32,
        pub n_idx_topk: u32,
        // YaRN (deepseek2/Kimi: factor 32, log_mult 0.1; GLM ships 1.0/off)
        pub rope_scale_factor: f32,
        pub rope_yarn_log_mult: f32,
        // Inkling (zero elsewhere). n_shexp_sink shared experts ride the
        // router as always-selected slots: n_expert_used INCLUDES them
        // (gguf expert_used_count + n_shexp_sink), expert ids >= n_expert
        // resolve into the shexp bank.
        pub n_shexp_sink: u32,
        pub d_rel: u32,
        pub rel_ext: u32,
        pub rel_ext_swa: u32,
        pub sconv_k: u32,
        // deepseek4 (zero elsewhere)
        pub n_swa: u32,
        pub n_hash_layer: u32,
        pub n_hc: u32,
        pub hc_sinkhorn: u32,
        pub hc_eps: f32,
        pub compress_rope_base: f32,
        pub n_out_group: u32,
        // qwen35moe GDN (zero elsewhere)
        pub ssm_conv_k: u32,
        pub ssm_state: u32,
        pub ssm_k_heads: u32,
        pub ssm_v_heads: u32,
        pub ssm_inner: u32,
        pub full_attn_interval: u32,
        /// SwiGLU clamp for routed AND shared experts (10.0 on V4;
        /// the per-layer metadata array is constant per model)
        pub clamp_exp: f32,
    }

    impl Shape {
        pub fn qk_dim(&self) -> u32 {
            self.qk_nope + self.qk_rope
        }

        /// Attention output width (input of attn_output).
        fn heads_dim(&self) -> u32 {
            match self.family {
                Family::Gqa | Family::Dsv4 | Family::Qwen35 => self.n_head * self.head_dim,
                Family::Mla => self.n_head * self.value_mla,
            }
        }

        fn rope_cfg(&self) -> kernels::RopeCfg {
            // GLM-5.2 ships yarn off (scale 1.0); deepseek2/Kimi runs real
            // YaRN: freq_scale = 1/factor, attn scaled by the log-mult
            // mscale (llama.cpp deepseek2 convention). NEEDS teacher-forced
            // parity validation on the first Kimi run.
            if self.rope_scale_factor > 1.0 {
                // llama.cpp deepseek2 YaRN (validated vs the fork's
                // deepseek2.cpp [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]): the rope
                // kernel internally multiplies mscale by (1 + 0.1 ln f), so
                // pass its reciprocal - rotated dims stay UNIT-scaled - and
                // apply the real magnitude correction mscale^2 on the whole
                // qk product (kq_mult), nope and rope dims alike, where
                // mscale = 1 + 0.1 * yarn_log_multiplier * ln f.
                let f = self.rope_scale_factor;
                let mscale = 1.0 + 0.1 * self.rope_yarn_log_mult * f.ln();
                kernels::RopeCfg {
                    n_ctx_orig: self.rope_orig_ctx,
                    freq_base: self.rope_freq_base,
                    freq_scale: 1.0 / f,
                    ext_factor: 1.0,
                    attn_factor: 1.0 / (1.0 + 0.1 * f.ln()),
                    beta_fast: 32.0,
                    beta_slow: 1.0,
                    kq_mult: mscale * mscale,
                }
            } else {
                kernels::RopeCfg {
                    n_ctx_orig: self.rope_orig_ctx,
                    freq_base: self.rope_freq_base,
                    freq_scale: 1.0,
                    ext_factor: 0.0,
                    attn_factor: 1.0,
                    beta_fast: 0.0,
                    beta_slow: 0.0,
                    kq_mult: 1.0,
                }
            }
        }
    }

    impl Shape {
        fn from_gguf(g: &Gguf) -> Result<Shape> {
            let u = |k: &str| -> Result<u32> {
                Ok(g.arch_meta(k).and_then(Value::as_u64).ok_or_else(|| meta_err(k))? as u32)
            };
            let f = |k: &str| -> Result<f32> {
                g.arch_meta(k).and_then(Value::as_f32).ok_or_else(|| meta_err(k))
            };
            let family = match g.architecture() {
                // hyphen vs underscore: the original ds4-lineage ggufs say
                // "hy-v3"; upstream llama.cpp (and AngelSlim's converter)
                // write "hy_v3". Same model either way.
                Some("hy-v3") | Some("hy_v3") => Family::Gqa,
                // MiniMax M3: Hy3-shaped GQA MoE (shexp, sigmoid router)
                // with partial rotary (rope.dimension_count < head_dim)
                Some("minimax-m3") | Some("minimax-m2") => Family::Gqa,
                // Qwen3 MoE (235B-A22B / 30B-A3B): GQA + per-head qk norm,
                // softmax router, no shared expert, no leading dense
                Some("qwen3moe") => Family::Gqa,
                // Gemma 4 (26B-A4B): interleaved SWA/full GQA, dual FFN
                // (GELU shared MLP + GELU MoE), per-layer geometry
                Some("gemma4") => Family::Gqa,
                Some("glm-dsa") | Some("glm_dsa") => Family::Mla,
                // DeepSeek-V3 family (Kimi K2 etc.): plain MLA, no indexer
                Some("deepseek2") => Family::Mla,
                // TML Inkling 1T: GQA without rope (learned rel-pos bias),
                // shortconv streams, sink router (llama.cpp PR 25731)
                Some("inkling") => Family::Gqa,
                // DeepSeek-V4-Flash: hyper-connections + sink attention +
                // compressed KV + hash routing (task #22)
                Some("deepseek4") => Family::Dsv4,
                // Qwen3.6-35B-A3B hybrid GDN (task #21)
                Some("qwen35moe") => Family::Qwen35,
                // Qwen3.6 dense (27B lineage, task #37): same GDN hybrid
                // stack; the dense FFN loads as a single always-on expert
                // so placement/caching/tiering machinery applies unchanged
                Some("qwen35") => Family::Qwen35,
                other => return Err(format!("unsupported architecture {other:?}").into()),
            };
            let inkling = g.architecture() == Some("inkling");
            let qwen35_dense = g.architecture() == Some("qwen35");
            let n_layer = u("block_count")?;
            // deepseek4 ships its MTP block as a SEPARATE gguf: the main
            // file's nextn_predict_layers=1 does not shrink block_count
            let nextn = if family == Family::Dsv4 {
                0
            } else {
                u("nextn_predict_layers").unwrap_or(0)
            };
            let n_vocab = match g.metadata.get("tokenizer.ggml.tokens") {
                Some(Value::Array(a)) => a.len() as u32,
                _ => return Err(meta_err("tokenizer.ggml.tokens")),
            };
            let mut s = Shape {
                family,
                n_embd: u("embedding_length")?,
                n_head: u("attention.head_count")?,
                // gemma4 ships head_count_kv as a per-layer array; the
                // scalar Shape field takes the max (buffer sizing), the
                // per-layer truth lives in Model::geom
                n_head_kv: u("attention.head_count_kv").unwrap_or_else(|_| {
                    match g.arch_meta("attention.head_count_kv") {
                        Some(Value::Array(a)) => a
                            .iter()
                            .filter_map(Value::as_u64)
                            .max()
                            .unwrap_or(1) as u32,
                        _ => 1,
                    }
                }),
                head_dim: u("attention.key_length")?,
                n_layer,
                n_exec_layer: n_layer - nextn,
                // the ds4-lineage converter writes this KV; upstream
                // llama.cpp (AngelSlim ggufs) omits it - infer it from
                // where routed-expert tensors start
                // qwen35-dense: every FFN is the one-expert synthesis,
                // so no layer is "leading dense" (that path re-quantizes
                // to q8_0 resident, 1.7x the bytes of the native K-quants)
                n_leading_dense: if qwen35_dense {
                    0
                } else {
                    match u("leading_dense_block_count") {
                        Ok(v) => v,
                        Err(_) => (0..u("block_count")?)
                            .find(|il| {
                                g.tensor(&format!("blk.{il}.ffn_gate_exps.weight")).is_some()
                                    || g.tensor(&format!("blk.{il}.ffn_gate_up_exps.weight")).is_some()
                            })
                            .ok_or_else(|| meta_err("no MoE layers found"))?,
                    }
                },
                n_expert: if qwen35_dense { 1 } else { u("expert_count")? },
                n_expert_used: if qwen35_dense { 1 } else { u("expert_used_count")? },
                n_ff_exp: if qwen35_dense {
                    u("feed_forward_length")?
                } else {
                    u("expert_feed_forward_length")?
                },
                // deepseek4/qwen35moe have no dense FFN layers and omit the key
                n_ff_dense: match family {
                    // note: or_else, not unwrap_or - the eager fallback
                    // arg would ? on files that only ship the plain key
                    Family::Dsv4 | Family::Qwen35 => u("feed_forward_length")
                        .or_else(|_| u("expert_feed_forward_length"))?,
                    _ => u("feed_forward_length")?,
                },
                n_vocab,
                // absent on qwen3moe (no scaling) - default 1.0
                expert_weight_scale: f("expert_weights_scale").unwrap_or(1.0),
                router_softmax: matches!(
                    g.architecture(),
                    Some("qwen3moe") | Some("gemma4") | Some("qwen35moe")
                ),
                // gated-FFN op: 1 = gelu (gemma4), 2 = swiglu_oai (MiniMax
                // M3: clamp 7, alpha 1.702, up+1 - llama.cpp PR 24523),
                // 0 = plain silu everywhere else (inkling included)
                moe_act_op: match g.architecture() {
                    Some("gemma4") => 1,
                    Some("minimax-m3") => 2,
                    _ => 0,
                },
                // inkling has no rope at all - the key may be absent
                rope_freq_base: if inkling {
                    f("rope.freq_base").unwrap_or(10_000.0)
                } else {
                    f("rope.freq_base")?
                },
                rms_eps: f("attention.layer_norm_rms_epsilon")?,
                n_lora_q: 0,
                n_kv_lora: 0,
                qk_nope: 0,
                qk_rope: 0,
                value_mla: 0,
                rope_orig_ctx: 0,
                rot_dim: 0,
                n_idx_head: 0,
                n_idx_dim: 0,
                n_idx_topk: 0,
                rope_scale_factor: 1.0,
                rope_yarn_log_mult: 0.0,
                n_shexp_sink: 0,
                d_rel: 0,
                rel_ext: 0,
                rel_ext_swa: 0,
                sconv_k: 0,
                n_swa: 0,
                n_hash_layer: 0,
                n_hc: 0,
                hc_sinkhorn: 0,
                hc_eps: 0.0,
                compress_rope_base: 0.0,
                n_out_group: 0,
                clamp_exp: 0.0,
                ssm_conv_k: 0,
                ssm_state: 0,
                ssm_k_heads: 0,
                ssm_v_heads: 0,
                ssm_inner: 0,
                full_attn_interval: 0,
            };
            if family == Family::Gqa {
                // partial rotary: MiniMax rotates rope.dimension_count of
                // head_dim; absent (Hy3) = full head
                s.rot_dim = u("rope.dimension_count").unwrap_or(s.head_dim);
            }
            if inkling {
                s.rot_dim = 0; // no rope: rel-pos bias carries position
                s.n_shexp_sink = u("expert_shared_count")?;
                // shared experts execute as always-selected router slots
                s.n_expert_used += s.n_shexp_sink;
                s.d_rel = u("d_rel")?;
                s.rel_ext = u("rel_extent")?;
                s.rel_ext_swa = u("rel_extent_swa")?;
                s.sconv_k = u("shortconv_kernel")?;
            }
            if family == Family::Mla {
                // GLM-5.2 MLA split from the gguf's own keys (verified
                // against the production glm-dsa file + DS4_SHAPE_GLM52):
                // per-head qk = key_length_mla (256) = nope (192) + rope
                // dims (64); value_length_mla (256) is the MLA value width
                // - attention.value_length (512) is NOT it.
                s.n_lora_q = u("attention.q_lora_rank").unwrap_or(2048);
                s.n_kv_lora = u("attention.kv_lora_rank").unwrap_or(512);
                s.qk_rope = u("rope.dimension_count").unwrap_or(64);
                let qk_mla = u("attention.key_length_mla").unwrap_or(256);
                s.qk_nope = qk_mla - s.qk_rope;
                s.value_mla = u("attention.value_length_mla").unwrap_or(256);
                s.rope_orig_ctx = u("rope.scaling.original_context_length").unwrap_or(1_048_576);
                s.n_idx_head = u("attention.indexer.head_count").unwrap_or(0);
                s.n_idx_dim = u("attention.indexer.key_length").unwrap_or(0);
                s.n_idx_topk = u("attention.indexer.top_k").unwrap_or(0);
                s.rope_scale_factor = f("rope.scaling.factor").unwrap_or(1.0);
                s.rope_yarn_log_mult = f("rope.scaling.yarn_log_multiplier").unwrap_or(0.0);
            }
            if family == Family::Qwen35 {
                s.rot_dim = u("rope.dimension_count").unwrap_or(64);
                s.ssm_conv_k = u("ssm.conv_kernel").unwrap_or(4);
                s.ssm_state = u("ssm.state_size").unwrap_or(128);
                s.ssm_k_heads = u("ssm.group_count").unwrap_or(16);
                s.ssm_v_heads = u("ssm.time_step_rank").unwrap_or(32);
                s.ssm_inner = u("ssm.inner_size").unwrap_or(4096);
                s.full_attn_interval = u("full_attention_interval").unwrap_or(4);
            }
            if family == Family::Dsv4 {
                s.n_lora_q = u("attention.q_lora_rank").unwrap_or(1024);
                s.rot_dim = u("rope.dimension_count").unwrap_or(64);
                s.rope_orig_ctx =
                    u("rope.scaling.original_context_length").unwrap_or(65_536);
                s.rope_scale_factor = f("rope.scaling.factor").unwrap_or(16.0);
                s.n_idx_head = u("attention.indexer.head_count").unwrap_or(64);
                s.n_idx_dim = u("attention.indexer.key_length").unwrap_or(128);
                s.n_idx_topk = u("attention.indexer.top_k").unwrap_or(512);
                s.n_swa = u("attention.sliding_window").unwrap_or(128);
                s.n_hash_layer = u("hash_layer_count").unwrap_or(3);
                s.n_hc = u("hyper_connection.count").unwrap_or(4);
                s.hc_sinkhorn = u("hyper_connection.sinkhorn_iterations").unwrap_or(20);
                s.hc_eps = f("hyper_connection.epsilon").unwrap_or(1.0e-6);
                s.compress_rope_base =
                    f("attention.compress_rope_freq_base").unwrap_or(160_000.0);
                s.n_out_group = u("attention.output_group_count").unwrap_or(8);
                // per-layer float array, constant across layers on V4
                s.clamp_exp = match g.arch_meta("swiglu_clamp_exp") {
                    Some(Value::Array(a)) => {
                        a.first().and_then(Value::as_f32).unwrap_or(10.0)
                    }
                    _ => 10.0,
                };
            }
            Ok(s)
        }
    }

    /// File location of one routed expert tensor: uniform per-expert slabs.
    #[derive(Clone)]
    struct ExpertTensor {
        abs_offset: u64,
        expert_bytes: u64,
        row_bytes: u64,
        quant: u32,
    }

    /// gguf tensor type -> kernel quant code (expert-dot coverage).
    fn quant_code(ty: TensorType) -> Option<u32> {
        Some(match ty {
            TensorType::IQ2XXS => kernels::QUANT_IQ2_XXS,
            TensorType::Q2K => kernels::QUANT_Q2_K,
            TensorType::Q4K => kernels::QUANT_Q4_K,
            TensorType::Q5K => kernels::QUANT_Q5_K,
            TensorType::Q6K => kernels::QUANT_Q6_K,
            TensorType::Q3K => kernels::QUANT_Q3_K,
            TensorType::IQ2XS => kernels::QUANT_IQ2_XS,
            TensorType::IQ3XXS => kernels::QUANT_IQ3_XXS,
            TensorType::Q4_0 => kernels::QUANT_Q4_0,
            TensorType::Q5_1 => kernels::QUANT_Q5_1,
            TensorType::Q8_0 => kernels::QUANT_Q8_0,
            TensorType::IQ4XS => kernels::QUANT_IQ4_XS,
            _ => return None,
        })
    }

    impl ExpertTensor {
        fn new(g: &Gguf, t: &TensorInfo, n_expert: u32) -> Result<ExpertTensor> {
            let quant = quant_code(t.ty)
                .ok_or_else(|| format!("{}: unsupported expert type {:?}", t.name, t.ty))?;
            let row_elems = t.dims[0];
            let rows_per_expert = t.dims[1];
            let row_bytes = t.ty.row_bytes(row_elems).unwrap();
            Ok(ExpertTensor {
                abs_offset: g.data_offset + t.offset,
                expert_bytes: row_bytes * rows_per_expert,
                row_bytes,
                quant: {
                    debug_assert_eq!(t.dims[2], n_expert as u64);
                    quant
                },
            })
        }
    }

    /// Tail slack after every expert slab: quants with sub-256 blocks
    /// (q8_0/q5_1/q4_0) on non-256-multiple rows (gemma4's 704) let the
    /// dot read past the last row - up to 7 phantom sub-blocks x 34 bytes
    /// (q8_0) = 238 for a dim = 32 mod 256. The math is exact (the q8
    /// tail is zero-quantized) - the slack only keeps the READ in bounds.
    const SLAB_SLACK: usize = 256;

    /// Byte-offset a device pointer (fused gate_up: up rows sit
    /// fused_up_off bytes into the gate slab).
    fn byte_off(p: *const std::ffi::c_void, off: u64) -> *const std::ffi::c_void {
        (p as *const u8).wrapping_add(off as usize) as *const std::ffi::c_void
    }

    /// A resident K-quant matmul weight (matmul_kq path).
    struct KqW {
        w: DeviceBuf,
        row_bytes: u64,
        quant: u32,
    }

    /// A matmul weight in whichever encoding the file made cheap to run:
    /// q8_0 (matmul_q8_0 on f32 activations) or native K-quant
    /// (matmul_kq on q8_K activations - half the bytes of the q8_0
    /// requant for a Q4_K file). qwen35 only.
    enum MatW {
        Q8(DeviceBuf),
        Kq(KqW),
    }

    impl MatW {
        /// True when this tensor should stay native: a K-quant with a
        /// warp-cooperative dot and a 256-divisible contraction dim.
        fn keep_native(t: &TensorInfo) -> bool {
            matches!(t.ty, TensorType::Q4K | TensorType::Q6K) && t.dims[0] % 256 == 0
        }

        fn load(file: &VFile, g: &Gguf, name: &str) -> Result<MatW> {
            let t = g.tensor(name).ok_or_else(|| meta_err(name))?;
            if Self::keep_native(t) {
                Ok(MatW::Kq(upload_kq(file, g, name)?))
            } else {
                Ok(MatW::Q8(upload(file, g, name)?))
            }
        }
    }

    enum Ffn {
        Dense {
            gate: DeviceBuf,
            up: DeviceBuf,
            down: DeviceBuf,
        },
        /// Dense qwen35 (27B): the whole FFN triple resident on the
        /// layer's owning card in native K-quant - no expert machinery,
        /// no tiers, no streaming (the model fits in combined VRAM).
        DenseKq {
            gate: KqW,
            up: KqW,
            down: KqW,
        },
        Moe {
            gate_inp: DeviceBuf,
            probs_b: DeviceBuf,
            /// shared expert; None on qwen3moe (routed experts only)
            shexp: Option<(DeviceBuf, DeviceBuf, DeviceBuf)>,
            gate_exps: ExpertTensor,
            up_exps: ExpertTensor,
            down_exps: ExpertTensor,
            /// fused ffn_gate_up_exps (gemma4): gate and up share one slab,
            /// up rows start this many bytes into it (0 = separate tensors)
            fused_up_off: u64,
            /// per-expert output scale [n_expert] (gemma4 down_exps.scale),
            /// folded into the route weights after selection
            down_scale: Option<DeviceBuf>,
            /// inkling shexp bank [gate, up, down] as n_shexp_sink-wide
            /// ExpertTensors: router slots with ids >= n_expert resolve
            /// here, so the offset-keyed cache/census/tier machinery
            /// serves shared experts like any other slab
            sink: Option<[ExpertTensor; 3]>,
        },
    }

    /// Gemma 4 per-layer extras (norm sandwich + scales); other families
    /// leave this None and take the classic residual path.
    struct GemmaW {
        attn_post_norm: DeviceBuf,
        /// router input norm weight, pre-scaled gate_inp_s / sqrt(n_embd)
        router_norm: DeviceBuf,
        pre_ffw_norm_2: DeviceBuf,
        post_ffw_norm_1: DeviceBuf,
        post_ffw_norm_2: DeviceBuf,
        post_ffw_norm: DeviceBuf,
        out_scale: f32,
    }

    /// Inkling per-layer weights (llama.cpp PR 25731): relative-position
    /// attention bias + four shortconv streams + the ffn global scale.
    struct InkW {
        /// attn_r projection (q8_0 matmul, n_embd -> n_head * d_rel)
        wr: DeviceBuf,
        /// rel_proj TRANSPOSED at load to [rel_extent][d_rel] row-major
        /// (gguf stores ne = [rel_extent, d_rel])
        rel_proj: DeviceBuf,
        /// this layer's band: rel_ext_swa on window layers, rel_ext global
        rel_extent: u32,
        /// f32 [w][K] depthwise kernels, tap K-1 = current token
        sconv_k: DeviceBuf,
        sconv_v: DeviceBuf,
        sconv_attn: DeviceBuf,
        sconv_mlp: DeviceBuf,
        /// ffn_gscale scalar: scales dense ffn output / folds into the
        /// route-weight scale for MoE layers
        gscale: f32,
    }

    /// Per-layer attention geometry (gemma4 interleaved SWA/full); empty
    /// for uniform-geometry families.
    #[derive(Clone, Copy)]
    struct Geom {
        n_head_kv: u32,
        head_dim: u32,
        theta: f32,
        window: u32,   /* 0 = full causal */
        factors: bool, /* proportional rope via rope_freqs */
    }

    enum Attn {
        Gqa {
            attn_q: DeviceBuf,
            /// None = k reused as v (gemma E-series attention_k_eq_v)
            attn_v: Option<DeviceBuf>,
            attn_k: DeviceBuf,
            q_norm: DeviceBuf,
            k_norm: DeviceBuf,
        },
        Mla {
            q_a: DeviceBuf,
            q_a_norm: DeviceBuf,
            q_b: DeviceBuf,
            kv_a_mqa: DeviceBuf,
            kv_a_norm: DeviceBuf,
            k_b: DeviceBuf,
            v_b: DeviceBuf,
            indexer: Option<IdxW>,
        },
        Dsv4(Box<Dsv4W>),
        Qwen35(Box<Qwen35W>),
    }

    /// qwen35moe per-layer stack: exactly one of attn/gdn is Some.
    /// The MoE half reuses Ffn::Moe; LayerW.attn_norm doubles as the
    /// pre-attention norm and LayerW.ffn_norm as post_attention_norm.
    struct Qwen35W {
        attn: Option<Qwen35Attn>,
        gdn: Option<Qwen35Gdn>,
        /// shared-expert scalar gate weight, f32 [n_embd -> 1]
        shexp_gate: DeviceBuf,
    }

    /// Full-attention layer (every full_attn_interval-th): the q
    /// projection is fused per head [q head_dim | gate head_dim].
    struct Qwen35Attn {
        wq: MatW, // [n_embd -> 2*n_head*head_dim]
        wk: MatW, // [n_embd -> n_kv*head_dim]
        wv: MatW,
        /// output projection [n_head*head_dim -> n_embd] (LayerW's
        /// attn_output slot stays a dummy for qwen35)
        out: MatW,
        q_norm: DeviceBuf, // f32 [head_dim]
        k_norm: DeviceBuf,
    }

    /// Gated DeltaNet layer: conv window + delta-rule state, no KV.
    struct Qwen35Gdn {
        wqkv: MatW, // [n_embd -> 2*key_dim + value_dim]
        wz: MatW,   // [n_embd -> value_dim] (attn_gate)
        conv: DeviceBuf, // f32 [conv_dim][ssm_conv_k]
        alpha_w: DeviceBuf, // f32 [n_embd -> ssm_v_heads]
        beta_w: DeviceBuf,
        /// g = a * softplus(alpha + dt_bias); a stored as -exp(A_log)
        a: DeviceBuf,
        dt_bias: DeviceBuf,
        ssm_norm: DeviceBuf, // f32 [ssm_state] per-v-head gated rms weight
        ssm_out: MatW,  // [value_dim -> n_embd]
    }

    /// deepseek4 per-layer stack: V4 attention, hyper-connection
    /// controls, streaming compressor, indexer, and the host-router
    /// extras. The MoE half reuses Ffn::Moe (LayerW.attn_output = the
    /// grouped projection's second stage attn_output_b).
    struct Dsv4W {
        q_a: DeviceBuf,      // q8_0 [n_embd -> n_lora_q]
        q_a_norm: DeviceBuf, // f32 [n_lora_q]
        q_b: DeviceBuf,      // q8_0 [n_lora_q -> n_head*head_dim]
        kv: DeviceBuf,       // q8_0 [n_embd -> head_dim] (K == V latent)
        kv_a_norm: DeviceBuf,
        /// attn_output_a: n_out_group banks of [group_dim -> rank] (q8_0)
        out_a: DeviceBuf,
        sinks: DeviceBuf, // f32 [n_head] per-head sink logits
        hc_attn_fn: DeviceBuf, // f32 [n_hc*n_embd -> 6*n_hc] (f16 converted)
        hc_ffn_fn: DeviceBuf,
        hc_attn_scale: DeviceBuf, // f32 [3]
        hc_attn_base: DeviceBuf,  // f32 [6*n_hc]
        hc_ffn_scale: DeviceBuf,
        hc_ffn_base: DeviceBuf,
        /// host router bias (selection only, like the noaux V3 router)
        probs_b: Vec<f32>,
        /// hash-routing table [n_vocab][n_expert_used] (first
        /// n_hash_layer layers replace top-k SELECTION with this)
        tid2eid: Option<Vec<i32>>,
        comp: Option<Dsv4CompW>, // compress_ratio != 0
        idx: Option<Dsv4IdxW>,   // compress_ratio == 4
        ratio: u32,
    }

    /// One compressor lane (attention 512-wide or indexer 128-wide).
    struct Dsv4CompW {
        kv_w: DeviceBuf,   // q8_0 [n_embd -> width] (f16 requantized)
        gate_w: DeviceBuf, // q8_0 [n_embd -> width]
        /// additive PE, f32 [ratio-mod slots][width]
        ape: DeviceBuf,
        norm: DeviceBuf, // f32 RMS weight [head_dim]
        width: u32,
    }

    struct Dsv4IdxW {
        q_b: DeviceBuf,  // q8_0 [n_lora_q -> n_idx_head*n_idx_dim]
        proj: DeviceBuf, // f32 [n_embd -> n_idx_head]
        comp: Dsv4CompW, // indexer lane (width 2*128, head_dim 128)
    }

    /// DSA lightning-indexer weights (small; resident beside the attn stack).
    struct IdxW {
        q_b: DeviceBuf,   // q8_0 [n_lora_q][idx_head*idx_dim]
        k: DeviceBuf,     // q8_0 [n_embd][idx_dim]
        k_norm: DeviceBuf, // f32 LayerNorm weight [idx_dim]
        k_norm_b: DeviceBuf, // f32 LayerNorm bias
        proj: DeviceBuf,  // f32 [n_embd][idx_head]
    }

    /// GLM-5.2 DSA layer policy: leading dense layers plus every 4th from
    /// layer 6 run the full indexer; the layers between reuse the last
    /// indexer layer's selection (verbatim from ds4).
    fn uses_full_indexer(il: usize, n_leading_dense: u32) -> bool {
        il < n_leading_dense as usize || (il >= 6 && (il - 6) % 4 == 0)
    }

    struct LayerW {
        attn_norm: DeviceBuf,
        attn: Attn,
        attn_output: DeviceBuf,
        ffn_norm: DeviceBuf,
        ffn: Ffn,
        gemma: Option<GemmaW>,
        ink: Option<InkW>,
    }

    /// The nextn/MTP draft block: predicts token t+2 from (hidden of
    /// t, embedding of t+1) through one extra transformer layer.
    struct MtpLayer {
        layer: LayerW,
        eh_proj: DeviceBuf, // q8_0 [n_embd][2*n_embd]
        enorm: DeviceBuf,
        hnorm: DeviceBuf,
        head_norm: DeviceBuf,
        /// ALL of the draft layer's expert slabs resident on the primary
        /// (~1.4GB Hy3 / ~2.5GB GLM): every draft pass routes through this
        /// one layer, so streaming its experts made drafting expensive -
        /// the main reason depth-1 MTP measured net-slower. Keyed by
        /// absolute file offset -> byte offset in the pool; empty map =
        /// residency didn't fit, resolve falls back to the caches.
        res_pool: DeviceBuf,
        res_map: std::collections::HashMap<u64, usize>,
    }

    pub struct Model {
        path: std::path::PathBuf,
        /// (virtual base, path) per shard; single file = one entry, base 0.
        shards: Vec<(u64, std::path::PathBuf)>,
        pub shape: Shape,
        pub gguf: Gguf,
        token_embd: DeviceBuf,
        output_norm: DeviceBuf,
        output: DeviceBuf,
        layers: Vec<LayerW>,
        /// PULSAR_ATTN_GPU: second CUDA device holding ALL attn weights +
        /// KV resident (Mla only). Attention weights are read every layer
        /// every token, so residency is the one job a bandwidth-crippled
        /// PCIe link can still do: only activations cross per layer.
        pub attn_dev: Option<i32>,
        /// Per-exec-layer owner device (dense split); all-primary
        /// everywhere else. Weights, KV, and GDN state live on the owner
        /// and the layer evals there.
        layer_dev: Vec<i32>,
        mtp: Option<MtpLayer>,
        /// Draft-chain depth (PULSAR_MTP_DEPTH, default 3): tokens
        /// speculated per round, verified together in one forward.
        pub mtp_depth: u32,
        /// (row_bytes, quant) when output.weight is a K-quant (AngelSlim
        /// ggufs keep the lm-head q6_K); None = the q8_0 fast path.
        output_kq: Option<(u64, u32)>,
        /// per-layer attention geometry; empty = uniform from Shape
        geom: Vec<Geom>,
        /// rope_freqs.weight [head_dim/2] frequency divisors (gemma4 full
        /// attention layers)
        rope_factors: Option<DeviceBuf>,
        /// residual-stream embedding multiplier (gemma: sqrt(n_embd))
        embd_scale: f32,
        /// final-logit softcap (gemma: 30.0); 0 = off
        logit_softcap: f32,
        /// post-embed rms norm weight (inkling token_embd_norm)
        tok_norm: Option<DeviceBuf>,
        /// final-logit multiplier (inkling muP: 1/logit_scale_denom); 1 = off
        logit_scale: f32,
        /// argmax/sampling cap (inkling pads the vocab: rows past
        /// unpadded_vocab_size are garbage); == n_vocab when unpadded
        pub n_vocab_out: u32,
        /// deepseek4 per-layer compression ratios (0 = raw SWA only,
        /// 4 = compressed + indexer, 128 = compressed); empty elsewhere
        compress_ratios: Vec<u32>,
        /// unit weight [n_hc*n_embd] for the weightless HC flat norm
        ones_hc: Option<DeviceBuf>,
        /// deepseek4 output-head HC merge
        dsv4_out: Option<Dsv4OutW>,
    }

    /// deepseek4 output_hc_*: collapse the final HC streams before
    /// output_norm and the lm head.
    struct Dsv4OutW {
        fn_w: DeviceBuf, // f32 [n_hc*n_embd -> n_hc]
        scale: f32,
        base: Vec<f32>, // [n_hc]
    }

    /// v1 StreamingStore (DESIGN-expert-store.md): io_uring batch fetch of
    /// cache misses + LFU host cache of expert slabs, keyed by absolute
    /// file offset (unique per layer/tensor/expert).
    pub struct StreamingStore {
        fetcher: stream::fetch::Fetcher,
        cache: std::collections::HashMap<u64, CacheEntry>,
        used: usize,
        budget: usize,
        tick: u64,
        pub hits: u64,
        pub misses: u64,
        /// offsets the CPU expert lane is reading right now - the evictors
        /// must not free them mid-dot (cleared after the pool joins)
        pinned: Vec<u64>,
    }

    struct CacheEntry {
        slab: stream::fetch::Slab,
        freq: u64,
        tick: u64,
    }

    /// Decode-loop stage timers. `sync` is the blocking wait for the GPU
    /// at the router readback (== all attention/router kernel time),
    /// `resolve` the expert resolve wall time, of which `h2d` is spent in
    /// uploads to the device.
    #[derive(Default)]
    pub struct Prof {
        pub sync: std::time::Duration,
        pub resolve: std::time::Duration,
        /// D2H of router_selected / pred_selected inside resolve
        pub resolve_d2h: std::time::Duration,
        /// distinct/offsets/wants/tier placement list building
        pub resolve_lists: std::time::Duration,
        /// host LFU ensure_with wall minus nested h2d (disk wait + cache hits)
        pub resolve_host: std::time::Duration,
        pub h2d: std::time::Duration,
        /// CPU expert lane wall time after the stage-A overlap (mid
        /// quantize + down-proj fan-out + join)
        pub cpu: std::time::Duration,
        pub tail: std::time::Duration,
        pub calls: u64,
    }

    impl Prof {
        pub fn report(&self) -> String {
            let s = |d: std::time::Duration| d.as_secs_f64();
            let accounted = self.resolve_d2h + self.resolve_lists + self.resolve_host + self.h2d;
            let other = self.resolve.saturating_sub(accounted);
            format!(
                "gpu-wait {:.2}s, resolve {:.2}s (d2h {:.2}s, lists {:.2}s, host {:.2}s, h2d {:.2}s, other {:.2}s), cpu-lane {:.2}s, logits-tail {:.2}s over {} layer steps",
                s(self.sync),
                s(self.resolve),
                s(self.resolve_d2h),
                s(self.resolve_lists),
                s(self.resolve_host),
                s(self.h2d),
                s(other),
                s(self.cpu),
                s(self.tail),
                self.calls
            )
        }
    }

    /// Ping-pong staging arena for one parity of MLA layers: layer N+1's
    /// PINNED attn tensors are cudaMemcpyAsync'd here (2x the bandwidth of
    /// zero-copy kernel reads, and overlapped under layer N's compute).
    /// Best-effort: if the copy hasn't landed when the layer runs, kernels
    /// fall back to the zero-copy pinned pointers - same bytes either way.
    struct AttnStage {
        q_a: DeviceBuf,
        q_b: DeviceBuf,
        kv_a: DeviceBuf,
        k_b: DeviceBuf,
        v_b: DeviceBuf,
        attn_output: DeviceBuf,
        stream: kernels::CopyStream,
        layer: Option<usize>,
    }

    impl AttnStage {
        fn new(l: &LayerW) -> Result<AttnStage> {
            let Attn::Mla { q_a, q_b, kv_a_mqa, k_b, v_b, .. } = &l.attn else {
                return Err("attn stage needs an Mla layer".into());
            };
            Ok(AttnStage {
                q_a: DeviceBuf::alloc(q_a.bytes())?,
                q_b: DeviceBuf::alloc(q_b.bytes())?,
                kv_a: DeviceBuf::alloc(kv_a_mqa.bytes())?,
                k_b: DeviceBuf::alloc(k_b.bytes())?,
                v_b: DeviceBuf::alloc(v_b.bytes())?,
                attn_output: DeviceBuf::alloc(l.attn_output.bytes())?,
                stream: kernels::CopyStream::new()?,
                layer: None,
            })
        }

        /// Queue copies of `l`'s pinned attn tensors for layer `il`.
        fn kick(&mut self, l: &LayerW, il: usize) -> Result {
            let Attn::Mla { q_a, q_b, kv_a_mqa, k_b, v_b, .. } = &l.attn else {
                return Ok(());
            };
            self.layer = None;
            // arena may still be read by in-flight default-stream kernels
            self.stream.gate_behind_default()?;
            let mut any = false;
            for (dst, src) in [
                (&mut self.q_a, q_a),
                (&mut self.q_b, q_b),
                (&mut self.kv_a, kv_a_mqa),
                (&mut self.k_b, k_b),
                (&mut self.v_b, v_b),
                (&mut self.attn_output, &l.attn_output),
            ] {
                if src.is_pinned() {
                    self.stream.copy_from_pinned(dst, 0, src)?;
                    any = true;
                }
            }
            if any {
                self.stream.record()?;
                self.layer = Some(il);
            }
            Ok(())
        }

        fn ready_for(&self, il: usize) -> bool {
            self.layer == Some(il) && self.stream.done()
        }
    }

    /// Cross-layer prefetcher: a background thread with its own io_uring
    /// fd fetches predicted next-layer expert slabs while the main thread
    /// resolves the current layer and the GPU computes. Slabs come back
    /// over a channel (ownership moves; no shared cache locking) and are
    /// absorbed into the host cache at the next resolve.
    pub struct Prefetcher {
        req_tx: std::sync::mpsc::Sender<Vec<stream::Read>>,
        done_rx: std::sync::mpsc::Receiver<(u64, stream::fetch::Slab)>,
    }

    impl Prefetcher {
        fn spawn(shards: &[(u64, std::path::PathBuf)]) -> Result<Prefetcher> {
            let mut fetcher = stream::fetch::Fetcher::open_split(shards, 16, fetch_buf_alloc())?;
            let (req_tx, req_rx) = std::sync::mpsc::channel::<Vec<stream::Read>>();
            let (done_tx, done_rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                while let Ok(first) = req_rx.recv() {
                    // stale requests are useless; keep only the newest
                    let reads = req_rx.try_iter().last().unwrap_or(first);
                    let _ = fetcher.fetch_each(&reads, |i, slab| {
                        let _ = done_tx.send((reads[i].offset, slab));
                        Ok(())
                    });
                }
            });
            Ok(Prefetcher { req_tx, done_rx })
        }
    }

    /// Static resident expert tier on a leftover GPU: the hottest expert
    /// TRIPLES (gate+up+down must colocate - the mid activations never
    /// leave the card) parked permanently in that card's VRAM, placed by
    /// warm-census heat at load. The MoE kernels run on the card that
    /// holds the weights and only activations cross PCIe, so - like attn
    /// residency - a bandwidth-crippled link serves a tier at full speed.
    /// No eviction: a tier is placement, not a cache.
    pub struct ExpertTier {
        dev: i32,
        pool: DeviceBuf,
        /// slab file offset -> pool ptr (all 3 slabs of a triple present)
        map: std::collections::HashMap<u64, *const std::ffi::c_void>,
        // per-card scratch, sized like the primary's
        xin: DeviceBuf,
        xq: DeviceBuf,
        mid: DeviceBuf,
        midq: DeviceBuf,
        out: DeviceBuf,
        ptrs: DeviceBuf,
        weights: DeviceBuf,
        /// inkling sink slots on a differently-quantized bank run as a
        /// second launch pair into their own output (mid/midq reuse is
        /// stream-ordered); 1-byte dummies elsewhere
        ptrs_sink: DeviceBuf,
        out_sink: DeviceBuf,
        /// grouped batch-MoE CSR scratch (hybrid-family verify/prefill
        /// chunks run the tensor-core kernels ON the tier card)
        grp_ptrs: DeviceBuf,
        grp_starts: DeviceBuf,
        grp_pairs: DeviceBuf,
        grp_partial: DeviceBuf,
        pub hits: u64,
    }

    unsafe impl Send for ExpertTier {}

    fn read_census(path: &Path) -> Vec<(u64, u64, u64)> {
        let Ok(bytes) = std::fs::read(warm_path(path)) else {
            return Vec::new();
        };
        let mut entries = Vec::with_capacity(bytes.len() / 24);
        for c in bytes.chunks_exact(24) {
            let off = u64::from_le_bytes(c[0..8].try_into().unwrap());
            let len = u64::from_le_bytes(c[8..16].try_into().unwrap());
            let count = u64::from_le_bytes(c[16..24].try_into().unwrap());
            entries.push((off, len, count));
        }
        entries
    }

    /// Device-side expert slab cache: a uniform-slot VRAM pool holding a
    /// STABLE hot set. The pool is smaller than one token's slab working
    /// set, so plain LFU would evict everything every token; instead every
    /// requested offset gets a global touch count, and a slab is admitted
    /// only when it is strictly hotter than the coldest resident. Cold
    /// slabs stream through the staging arena and never enter the pool.
    ///
    /// Gate/up/down triples that enter via `maybe_insert_triple` or warm
    /// load share a `group` id so eviction frees the whole triple (avoids
    /// half-resident experts that still force H2D of siblings).
    pub struct DeviceSlabCache {
        pool: DeviceBuf,
        slab_bytes: usize,
        map: std::collections::HashMap<u64, u32>,
        /// per slot: (touch count at admission, offset, group); offset
        /// u64::MAX = free; group u32::MAX = ungrouped singleton
        meta: Vec<(u64, u64, u32)>,
        /// free slot indices (O(1) take; rebuild not required)
        free_list: Vec<u32>,
        /// occupied slots with group == u32::MAX (singleton admits only)
        ungrouped: u32,
        /// global (touch count, slab len) per requested offset, cached or not
        touch: std::collections::HashMap<u64, (u64, u64)>,
        next_group: u32,
        pub hits: u64,
        pub misses: u64,
    }

    impl DeviceSlabCache {
        fn new(budget_bytes: usize, slab_bytes: usize) -> Result<DeviceSlabCache> {
            let slots = (budget_bytes / slab_bytes.max(1)).max(1);
            Ok(DeviceSlabCache {
                pool: DeviceBuf::alloc(slots * slab_bytes + SLAB_SLACK)?,
                slab_bytes,
                map: std::collections::HashMap::with_capacity(slots),
                meta: vec![(0, u64::MAX, u32::MAX); slots],
                free_list: (0..slots as u32).collect(),
                ungrouped: 0,
                touch: std::collections::HashMap::new(),
                next_group: 1,
                hits: 0,
                misses: 0,
            })
        }

        fn slot_ptr(&self, slot: u32) -> *const std::ffi::c_void {
            self.pool.ptr_at(slot as usize * self.slab_bytes)
        }

        fn free_slot(&mut self, slot: u32) {
            let off = self.meta[slot as usize].1;
            if off == u64::MAX {
                return; // already free
            }
            let g = self.meta[slot as usize].2;
            if g == u32::MAX {
                self.ungrouped = self.ungrouped.saturating_sub(1);
            }
            self.map.remove(&off);
            self.meta[slot as usize] = (0, u64::MAX, u32::MAX);
            self.free_list.push(slot);
        }

        /// Free `slot` and every other slot sharing its group (whole triple).
        /// Only for triple-unit admit/evict — never for single-slab runtime admits.
        fn free_group_of(&mut self, slot: u32) {
            let g = self.meta[slot as usize].2;
            if g == u32::MAX {
                self.free_slot(slot);
                return;
            }
            let members: Vec<u32> = self
                .meta
                .iter()
                .enumerate()
                .filter(|(_, m)| m.2 == g)
                .map(|(i, _)| i as u32)
                .collect();
            for s in members {
                self.free_slot(s);
            }
        }

        fn get(&mut self, offset: u64, len: u64) -> Option<*const std::ffi::c_void> {
            let t = self.touch.entry(offset).or_insert((0, len));
            t.0 += 1;
            let freq = t.0;
            match self.map.get(&offset).copied() {
                Some(slot) => {
                    self.meta[slot as usize].0 = freq;
                    self.hits += 1;
                    Some(self.slot_ptr(slot))
                }
                None => {
                    self.misses += 1;
                    None
                }
            }
        }

        /// Peek without bumping touch / hit counters (prefetch staging).
        fn peek(&self, offset: u64) -> Option<*const std::ffi::c_void> {
            self.map.get(&offset).map(|&slot| self.slot_ptr(slot))
        }

        /// Admit `payload` if it is hotter than the coldest *ungrouped*
        /// resident (or a free slot exists). Returns None when the slab is
        /// not worthy - the caller streams it through staging instead.
        ///
        /// Critical: never evict a warm-loaded triple member for a singleton
        /// admit. Breaking triples filled VRAM with incomplete experts and
        /// collapsed slab hit rate 72% -> 53% (measured). Triple groups are
        /// only replaced by `maybe_insert_triple`.
        ///
        /// After warm fill the pool is usually all triples: free_list empty
        /// and ungrouped==0 → O(1) early-out (stage path).
        fn maybe_insert(
            &mut self,
            offset: u64,
            payload: &[u8],
            in_use: &[u64],
        ) -> Result<Option<*const std::ffi::c_void>> {
            if let Some(slot) = self.map.get(&offset).copied() {
                return Ok(Some(self.slot_ptr(slot)));
            }
            // O(1): nothing singleton-admittable left (warm pool is all triples)
            if self.free_list.is_empty() && self.ungrouped == 0 {
                return Ok(None);
            }
            let freq = self.touch.get(&offset).map(|t| t.0).unwrap_or(0);
            let slot = if let Some(free) = self.free_list.pop() {
                free
            } else {
                // only steal UNGROUPED slots (group == u32::MAX)
                let Some((victim, vmeta)) = self
                    .meta
                    .iter()
                    .enumerate()
                    .filter(|(_, m)| {
                        m.1 != u64::MAX && m.2 == u32::MAX && !in_use.contains(&m.1)
                    })
                    .min_by_key(|(_, m)| m.0)
                else {
                    return Ok(None);
                };
                if vmeta.0 >= freq {
                    return Ok(None);
                }
                let victim = victim as u32;
                self.free_slot(victim); // pushes victim onto free_list
                self.free_list.pop().ok_or("free_list empty after free_slot")?
            };
            debug_assert_eq!(self.meta[slot as usize].1, u64::MAX);
            let base = slot as usize * self.slab_bytes;
            self.pool.write(base, payload)?;
            self.meta[slot as usize] = (freq, offset, u32::MAX);
            self.ungrouped += 1;
            self.map.insert(offset, slot);
            Ok(Some(self.slot_ptr(slot)))
        }

        /// Admit gate+up+down as one unit (all-or-nothing). Heat is the sum
        /// of per-slab touch counts; eviction picks the coldest freeable
        /// *groups* (or singletons) until three slots are free.
        fn maybe_insert_triple(
            &mut self,
            parts: &[(u64, &[u8]); 3],
            in_use: &[u64],
        ) -> Result<Option<[*const std::ffi::c_void; 3]>> {
            let mut ptrs = [std::ptr::null(); 3];
            let mut need: Vec<(usize, u64, &[u8])> = Vec::new();
            for (i, &(off, payload)) in parts.iter().enumerate() {
                if let Some(p) = self.map.get(&off).map(|&s| self.slot_ptr(s)) {
                    ptrs[i] = p;
                } else {
                    need.push((i, off, payload));
                }
            }
            if need.is_empty() {
                return Ok(Some(ptrs));
            }
            let heat: u64 = parts
                .iter()
                .map(|(off, _)| self.touch.get(off).map(|t| t.0).unwrap_or(0))
                .sum();
            // free slots already available
            while self.free_list.len() < need.len() {
                let mut cands: Vec<(u32, u64, u32)> = self
                    .meta
                    .iter()
                    .enumerate()
                    .filter(|(_, m)| m.1 != u64::MAX && !in_use.contains(&m.1))
                    .map(|(i, m)| (i as u32, m.0, m.2))
                    .collect();
                if cands.is_empty() {
                    return Ok(None);
                }
                cands.sort_by_key(|c| c.1);
                let (victim, vfreq, _) = cands[0];
                let g = self.meta[victim as usize].2;
                let group_heat: u64 = if g == u32::MAX {
                    vfreq
                } else {
                    self.meta
                        .iter()
                        .filter(|m| m.2 == g)
                        .map(|m| m.0)
                        .sum()
                };
                if group_heat >= heat {
                    return Ok(None);
                }
                self.free_group_of(victim);
            }
            let gid = self.next_group;
            self.next_group = self.next_group.wrapping_add(1).max(1);
            for (j, (i, off, payload)) in need.iter().enumerate() {
                let slot = self.free_list.pop().ok_or("triple admit: free_list empty")?;
                let base = slot as usize * self.slab_bytes;
                self.pool.write(base, payload)?;
                let freq = self.touch.get(off).map(|t| t.0).unwrap_or(0);
                debug_assert_eq!(self.meta[slot as usize].1, u64::MAX);
                self.meta[slot as usize] = (freq, *off, gid);
                self.map.insert(*off, slot);
                ptrs[*i] = self.slot_ptr(slot);
            }
            Ok(Some(ptrs))
        }
    }

    /// Fetch buffers in CUDA pinned memory (H2D at full PCIe rate; they
    /// live on as host-cache slabs, so cache-hit uploads benefit too).
    /// PULSAR_NO_PINNED=1 reverts to pageable.
    fn fetch_buf_alloc() -> Option<stream::uring::BufAlloc> {
        if std::env::var_os("PULSAR_NO_PINNED").is_some() {
            return None;
        }
        Some(stream::uring::BufAlloc {
            alloc: kernels::pinned_alloc,
            free: kernels::pinned_free,
        })
    }

    impl StreamingStore {
        fn open(shards: &[(u64, std::path::PathBuf)], budget: usize) -> Result<StreamingStore> {
            Ok(StreamingStore {
                fetcher: stream::fetch::Fetcher::open_split(shards, 32, fetch_buf_alloc())?,
                cache: std::collections::HashMap::new(),
                used: 0,
                budget,
                tick: 0,
                hits: 0,
                misses: 0,
                pinned: Vec::new(),
            })
        }

        /// Cache-hit payload for the CPU expert lane: bumps LFU heat like
        /// an ensure_with hit and returns the slab bytes as a raw span
        /// (valid until the entry is evicted - pin it across any evictor).
        fn peek_ptr(&mut self, offset: u64) -> Option<(*const u8, usize)> {
            let tick = self.tick;
            let e = self.cache.get_mut(&offset)?;
            e.freq += 1;
            e.tick = tick;
            self.hits += 1;
            let p = e.slab.payload();
            Some((p.as_ptr(), p.len()))
        }

        /// Resolve every read: cached payloads go to `place(offset, bytes)`
        /// immediately, disk misses as each io_uring completion lands - so
        /// the caller's H2D uploads overlap the remaining reads. Fetched
        /// slabs enter the LFU cache afterwards.
        fn ensure_with(
            &mut self,
            wants: &[stream::Read],
            mut place: impl FnMut(u64, &[u8]) -> Result,
        ) -> Result {
            self.tick += 1;
            let mut missing = Vec::new();
            for r in wants {
                if let Some(e) = self.cache.get_mut(&r.offset) {
                    e.freq += 1;
                    e.tick = self.tick;
                    self.hits += 1;
                    place(r.offset, e.slab.payload())?;
                } else {
                    self.misses += 1;
                    missing.push(*r);
                }
            }
            if missing.is_empty() {
                return Ok(());
            }
            // Evict lowest (freq, tick) among a strided SAMPLE of eligible
            // entries. Full min scans over ~40k host slabs burned seconds
            // into resolve "disk/host"; take(64) alone was iteration-order
            // biased and thrashy.
            let incoming: usize = missing.iter().map(|r| r.len as usize).sum();
            const EVICT_SAMPLE: usize = 64;
            while self.used + incoming > self.budget && !self.cache.is_empty() {
                let n = self.cache.len().max(1);
                let step = (n / EVICT_SAMPLE).max(1);
                let victim = self
                    .cache
                    .iter()
                    .filter(|(k, _)| {
                        !wants.iter().any(|w| w.offset == **k) && !self.pinned.contains(k)
                    })
                    .enumerate()
                    .filter(|(i, _)| i % step == 0)
                    .map(|(_, kv)| kv)
                    .take(EVICT_SAMPLE)
                    .min_by_key(|(_, e)| (e.freq, e.tick))
                    .map(|(k, _)| *k);
                let Some(k) = victim else { break };
                if let Some(e) = self.cache.remove(&k) {
                    self.used -= e.slab.bytes();
                }
            }
            let Self { fetcher, cache, used, tick, .. } = self;
            let mut place_err = None;
            fetcher.fetch_each(&missing, |i, slab| {
                if place_err.is_none() {
                    if let Err(e) = place(missing[i].offset, slab.payload()) {
                        place_err = Some(e);
                    }
                }
                *used += slab.bytes();
                cache.insert(
                    missing[i].offset,
                    CacheEntry { slab, freq: 1, tick: *tick },
                );
                Ok(())
            })?;
            match place_err {
                Some(e) => Err(e),
                None => Ok(()),
            }
        }

        /// Fetch without caching - warm-start uses this to route slabs
        /// straight to the device tier.
        fn fetch_direct(
            &mut self,
            reads: &[stream::Read],
            mut place: impl FnMut(u64, &[u8]) -> Result,
        ) -> Result {
            let mut place_err = None;
            self.fetcher.fetch_each(reads, |i, slab| {
                if place_err.is_none() {
                    if let Err(e) = place(reads[i].offset, slab.payload()) {
                        place_err = Some(e);
                    }
                }
                Ok(())
            })?;
            match place_err {
                Some(e) => Err(e),
                None => Ok(()),
            }
        }

        fn reset_stats(&mut self) {
            self.hits = 0;
            self.misses = 0;
        }

        fn contains(&self, offset: u64) -> bool {
            self.cache.contains_key(&offset)
        }

        /// Borrow a host-cache slab payload (pinned when fetch used CUDA host alloc).
        fn payload(&self, offset: u64) -> Option<&[u8]> {
            self.cache.get(&offset).map(|e| e.slab.payload())
        }

        /// Take ownership of a prefetched slab (evicting to budget).
        fn absorb(&mut self, offset: u64, slab: stream::fetch::Slab) {
            if self.cache.contains_key(&offset) {
                return;
            }
            let incoming = slab.bytes();
            const EVICT_SAMPLE: usize = 64;
            while self.used + incoming > self.budget && !self.cache.is_empty() {
                let n = self.cache.len().max(1);
                let step = (n / EVICT_SAMPLE).max(1);
                let victim = self
                    .cache
                    .iter()
                    .filter(|(k, _)| !self.pinned.contains(k))
                    .enumerate()
                    .filter(|(i, _)| i % step == 0)
                    .map(|(_, kv)| kv)
                    .take(EVICT_SAMPLE)
                    .min_by_key(|(_, e)| (e.freq, e.tick))
                    .map(|(k, _)| *k);
                let Some(k) = victim else { break };
                if let Some(e) = self.cache.remove(&k) {
                    self.used -= e.slab.bytes();
                }
            }
            self.used += incoming;
            self.cache.insert(offset, CacheEntry { slab, freq: 1, tick: self.tick });
        }
    }

    /// CPU expert lane: host-cache-hit experts compute where their bytes
    /// live (RAM at ~42GB/s on the 9900X via the AVX2 iq2_xxs dot) instead
    /// of crossing PCIe (~29GB/s), freeing H2D for disk-miss staging. The
    /// pool is persistent - per-layer thread spawns would cost more than
    /// the dots. Opt-in: PULSAR_CPU=1 (or =N for N worker threads).
    mod cpu_tier {
        pub type Job = Box<dyn FnOnce() + Send>;

        /// raw-pointer smuggler for jobs; soundness = caller keeps the
        /// pointee alive and unmutated until wait() returns
        #[derive(Clone, Copy)]
        pub struct SendPtr(pub *const u8);
        unsafe impl Send for SendPtr {}
        #[derive(Clone, Copy)]
        pub struct SendMut(pub *mut f32);
        unsafe impl Send for SendMut {}
        // accessors, not .0: edition-2021 closures capture the raw-ptr
        // FIELD on .0 (not Send); a method call captures the wrapper
        impl SendPtr {
            pub fn get(self) -> *const u8 {
                self.0
            }
        }
        impl SendMut {
            pub fn get(self) -> *mut f32 {
                self.0
            }
        }

        pub struct Pool {
            tx: std::sync::mpsc::Sender<Job>,
            done_rx: std::sync::mpsc::Receiver<()>,
            pub threads: usize,
        }

        impl Pool {
            pub fn from_env() -> Option<Pool> {
                let v = std::env::var("PULSAR_CPU").ok()?;
                let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8);
                let threads = match v.as_str() {
                    "" | "0" | "off" => return None,
                    // physical cores minus main + fetcher headroom
                    "1" | "on" => (cores / 2).saturating_sub(2).max(1),
                    n => n.parse().ok()?,
                };
                let (tx, rx) = std::sync::mpsc::channel::<Job>();
                let (done_tx, done_rx) = std::sync::mpsc::channel();
                let rx = std::sync::Arc::new(std::sync::Mutex::new(rx));
                for _ in 0..threads {
                    let rx = rx.clone();
                    let done_tx = done_tx.clone();
                    std::thread::spawn(move || loop {
                        let job = match rx.lock().unwrap().recv() {
                            Ok(j) => j,
                            Err(_) => return,
                        };
                        job();
                        let _ = done_tx.send(());
                    });
                }
                Some(Pool { tx, done_rx, threads })
            }

            pub fn submit(&self, jobs: Vec<Job>) -> usize {
                let n = jobs.len();
                for j in jobs {
                    let _ = self.tx.send(j);
                }
                n
            }

            pub fn wait(&self, n: usize) {
                for _ in 0..n {
                    let _ = self.done_rx.recv();
                }
            }
        }

        /// joins outstanding jobs on drop - an early `?` return between
        /// submit and the explicit join must not free buffers the workers
        /// still write
        pub struct WaitGuard<'a> {
            pub pool: &'a Pool,
            pub n: usize,
        }
        impl Drop for WaitGuard<'_> {
            fn drop(&mut self) {
                self.pool.wait(self.n);
            }
        }

        /// One MoE layer's worth of CPU-lane work, shared by both resolve
        /// paths (eval_layer's full resolve and the lean dsv4_moe). The
        /// caller does site-specific eligibility + slab peeking + pinning;
        /// Lane owns the two compute stages. Buffers are raw-pointer-
        /// shared with the pool, so between submit_a() and the WaitGuard
        /// join nothing may push into a Lane field (heap blocks are
        /// stable across moves of the Lane itself).
        pub struct Lane {
            pub idx: std::collections::HashMap<i32, usize>,
            ptrs: Vec<[SendPtr; 3]>,
            pairs: Vec<Vec<(usize, f32)>>,
            xqs: Vec<quant::cpu_dot::Q8KRow>,
            mids: Vec<f32>,
            gq: u32,
            dq: u32,
            grb: usize,
            drb: usize,
            ne: usize,
            nf: usize,
            act_op: u32,
        }

        impl Lane {
            #[allow(clippy::too_many_arguments)]
            pub fn new(gq: u32, dq: u32, grb: usize, drb: usize, ne: usize, nf: usize, act_op: u32) -> Lane {
                Lane {
                    idx: std::collections::HashMap::new(),
                    ptrs: Vec::new(),
                    pairs: Vec::new(),
                    xqs: Vec::new(),
                    mids: Vec::new(),
                    gq, dq, grb, drb, ne, nf, act_op,
                }
            }

            /// register expert e; up must already include any fused offset
            pub fn add(&mut self, e: i32, gate: *const u8, up: *const u8, down: *const u8) {
                self.idx.insert(e, self.ptrs.len());
                self.ptrs.push([SendPtr(gate), SendPtr(up), SendPtr(down)]);
                self.pairs.push(Vec::new());
            }

            pub fn is_empty(&self) -> bool {
                self.idx.is_empty()
            }

            /// attach (token, weight) pairs from the routed slots, quantize
            /// activations, fan out gate/up + glu jobs. Returns the job
            /// count for a WaitGuard.
            pub fn submit_a(
                &mut self,
                pool: &Pool,
                selected: &[i32],
                n_used: usize,
                normed: &[f32],
                rw: &[f32],
                n_tok: usize,
            ) -> usize {
                for (si, &e) in selected.iter().enumerate() {
                    if let Some(&ci) = self.idx.get(&e) {
                        self.pairs[ci].push((si / n_used, rw[si]));
                    }
                }
                let ne = self.ne;
                self.xqs = (0..n_tok)
                    .map(|t| quant::cpu_dot::quantize_row_q8_k(&normed[t * ne..(t + 1) * ne]))
                    .collect();
                let npairs: usize = self.pairs.iter().map(|p| p.len()).sum();
                self.mids = vec![0f32; npairs * self.nf];
                if std::env::var_os("PULSAR_LANE_DBG").is_some() {
                    for (ci, pairs) in self.pairs.iter().enumerate() {
                        let [gp, up_, _] = self.ptrs[ci];
                        let g_row = unsafe { std::slice::from_raw_parts(gp.get(), self.grb) };
                        let u_row = unsafe { std::slice::from_raw_parts(up_.get(), self.grb) };
                        let g = dot(self.gq, g_row, &self.xqs[0], self.ne);
                        let u = dot(self.gq, u_row, &self.xqs[0], self.ne);
                        let gs = quant::cpu_dot::vec_dot_iq2_xxs_q8_k_scalar(g_row, &self.xqs[0], self.ne);
                        eprintln!(
                            "lane dbg ci={ci} gq={} act={} g={g:.5} g_scalar={gs:.5} u={u:.5} w={:.5} mid0={:.6}",
                            self.gq, self.act_op,
                            pairs.first().map(|p| p.1).unwrap_or(0.0),
                            glu(g, u, self.act_op) * pairs.first().map(|p| p.1).unwrap_or(0.0)
                        );
                    }
                }
                let (nf, grb, gq, act_op) = (self.nf, self.grb, self.gq, self.act_op);
                let xq_ptr = SendPtr(self.xqs.as_ptr() as *const u8);
                let mut jobs: Vec<Job> = Vec::new();
                let mut mid_base = 0usize;
                for (ci, pairs) in self.pairs.iter().enumerate() {
                    let [gp, up_, _] = self.ptrs[ci];
                    let mid = SendMut(unsafe { self.mids.as_mut_ptr().add(mid_base * nf) });
                    mid_base += pairs.len();
                    for lo in (0..nf).step_by(256) {
                        let hi = (lo + 256).min(nf);
                        let pairs = pairs.clone();
                        jobs.push(Box::new(move || unsafe {
                            for j in lo..hi {
                                let g_row = std::slice::from_raw_parts(gp.get().add(j * grb), grb);
                                let u_row = std::slice::from_raw_parts(up_.get().add(j * grb), grb);
                                for (pi, &(tok, w)) in pairs.iter().enumerate() {
                                    let xq = &*(xq_ptr.get() as *const quant::cpu_dot::Q8KRow).add(tok);
                                    let g = dot(gq, g_row, xq, ne);
                                    let u = dot(gq, u_row, xq, ne);
                                    *mid.get().add(pi * nf + j) = glu(g, u, act_op) * w;
                                }
                            }
                        }));
                    }
                }
                pool.submit(jobs)
            }

            /// debug: (first mids, per-expert slot weights, expert order)
            pub fn dbg(&self) -> (Vec<f32>, Vec<Vec<f32>>, Vec<i32>) {
                let mut order: Vec<(i32, usize)> = self.idx.iter().map(|(&e, &ci)| (e, ci)).collect();
                order.sort_by_key(|&(_, ci)| ci);
                (
                    self.mids.iter().take(4).copied().collect(),
                    self.pairs.iter().map(|p| p.iter().map(|x| x.1).collect()).collect(),
                    order.into_iter().map(|(e, _)| e).collect(),
                )
            }

            /// after the stage-A join: quantize mids, run the down-proj
            /// fan-out, return the per-token f32 partial [n_tok * ne]
            pub fn finish(&self, pool: &Pool, n_tok: usize) -> Vec<f32> {
                let (ne, nf, drb, dq) = (self.ne, self.nf, self.drb, self.dq);
                let npairs: usize = self.pairs.iter().map(|p| p.len()).sum();
                let midq: Vec<quant::cpu_dot::Q8KRow> = (0..npairs)
                    .map(|p| quant::cpu_dot::quantize_row_q8_k(&self.mids[p * nf..(p + 1) * nf]))
                    .collect();
                let mut per_tok: Vec<Vec<(SendPtr, usize)>> = vec![Vec::new(); n_tok];
                let mut mid_base = 0usize;
                for (ci, pairs) in self.pairs.iter().enumerate() {
                    for (pi, &(tok, _)) in pairs.iter().enumerate() {
                        per_tok[tok].push((self.ptrs[ci][2], mid_base + pi));
                    }
                    mid_base += pairs.len();
                }
                let mut acc = vec![0f32; n_tok * ne];
                let acc_ptr = SendMut(acc.as_mut_ptr());
                let midq_ptr = SendPtr(midq.as_ptr() as *const u8);
                let mut jobs: Vec<Job> = Vec::new();
                for (t, list) in per_tok.iter().enumerate() {
                    if list.is_empty() {
                        continue;
                    }
                    for lo in (0..ne).step_by(512) {
                        let hi = (lo + 512).min(ne);
                        let list = list.clone();
                        jobs.push(Box::new(move || unsafe {
                            for r in lo..hi {
                                let mut sum = 0f32;
                                for &(dp, mi) in &list {
                                    let row = std::slice::from_raw_parts(dp.get().add(r * drb), drb);
                                    let mq = &*(midq_ptr.get() as *const quant::cpu_dot::Q8KRow).add(mi);
                                    sum += dot(dq, row, mq, nf);
                                }
                                *acc_ptr.get().add(t * ne + r) = sum;
                            }
                        }));
                    }
                }
                pool.wait(pool.submit(jobs));
                acc
            }
        }

        /// quants the lane can compute; extend together with dot()
        pub fn supported(quant: u32) -> bool {
            [
                kernels::QUANT_IQ2_XXS,
                kernels::QUANT_IQ2_XS,
                kernels::QUANT_IQ3_XXS,
                kernels::QUANT_Q2_K,
                kernels::QUANT_Q3_K,
                kernels::QUANT_Q4_K,
            ]
            .contains(&quant)
        }

        pub fn dot(quant: u32, row: &[u8], xq: &quant::cpu_dot::Q8KRow, n: usize) -> f32 {
            match quant {
                q if q == kernels::QUANT_IQ2_XS => {
                    quant::cpu_dot::vec_dot_iq2_xs_q8_k(row, xq, n)
                }
                q if q == kernels::QUANT_IQ3_XXS => {
                    quant::cpu_dot::vec_dot_iq3_xxs_q8_k(row, xq, n)
                }
                q if q == kernels::QUANT_Q2_K => quant::cpu_dot::vec_dot_q2_k_q8_k(row, xq, n),
                q if q == kernels::QUANT_Q3_K => quant::cpu_dot::vec_dot_q3_k_q8_k(row, xq, n),
                q if q == kernels::QUANT_Q4_K => quant::cpu_dot::vec_dot_q4_k_q8_k(row, xq, n),
                _ => quant::cpu_dot::vec_dot_iq2_xxs_q8_k(row, xq, n),
            }
        }

        /// mirrors pulsar_glu in pulsar_kernels.cu (0 = silu, 1 = gelu
        /// tanh, 2 = swiglu_oai, 3 = deepseek4 clamped silu)
        pub fn glu(g: f32, u: f32, op: u32) -> f32 {
            match op {
                1 => {
                    0.5 * g
                        * (1.0 + (0.797_884_560_802_865_4_f32 * (g + 0.044715 * g * g * g)).tanh())
                        * u
                }
                2 => {
                    let g = g.min(7.0);
                    let u = u.clamp(-7.0, 7.0);
                    g / (1.0 + (-1.702 * g).exp()) * (u + 1.0)
                }
                3 => {
                    let g = g.min(10.0);
                    let u = u.clamp(-10.0, 10.0);
                    g / (1.0 + (-g).exp()) * u
                }
                _ => g / (1.0 + (-g).exp()) * u,
            }
        }
    }

    fn warm_path(model: &Path) -> std::path::PathBuf {
        let mut p = model.as_os_str().to_owned();
        p.push(".warm");
        p.into()
    }

    /// How many header bytes to read before parsing; grows on Truncated.
    const HEAD_READ_START: usize = 32 << 20;

    fn parse_one_header(file: &File) -> Result<Gguf> {
        let mut n = HEAD_READ_START;
        loop {
            let mut head = vec![0u8; n];
            let got = file.read_at(&mut head, 0)?;
            head.truncate(got);
            match Gguf::parse(&head) {
                Ok(g) => return Ok(g),
                Err(gguf::Error::Truncated { .. }) if got == n => n *= 2,
                Err(e) => return Err(e.into()),
            }
        }
    }

    /// Open a model that may be a single gguf or a -00001-of-000NN split
    /// set. Returns the merged header over a virtual offset space plus the
    /// shard list ((virtual base, path); single file = one entry, base 0).
    pub fn parse_header(path: &Path) -> Result<(Vec<(u64, std::path::PathBuf)>, Gguf)> {
        let paths = gguf::split_shards(path)
            .unwrap_or_else(|| vec![path.to_path_buf()]);
        let mut shards = Vec::with_capacity(paths.len());
        let mut bases = Vec::with_capacity(paths.len());
        let mut ggufs = Vec::with_capacity(paths.len());
        let mut base = 0u64;
        for p in paths {
            let file = File::open(&p)?;
            ggufs.push(parse_one_header(&file)?);
            bases.push(base);
            shards.push((base, p.clone()));
            base += file.metadata()?.len();
        }
        if ggufs.len() > 1 {
            eprintln!("pulsar: split gguf: {} shards as one virtual file", ggufs.len());
        }
        Ok((shards, Gguf::merge_split(ggufs, &bases)))
    }

    /// Host requant: dense K-quant tensors -> q8_0 at load. Kimi K2 (and
    /// other community ggufs) put attention/embed/shexp weights in
    /// q2_K..q6_K, which the dense fast paths don't read; q8_0 is a
    /// superset precision-wise (the only loss is q8's own ~0.4% noise on
    /// top of values already coarsened to 2-6 bits), so one-time host
    /// conversion beats porting five dense matmul variants. Experts are
    /// untouched (they stream from disk and have native kernels).
    mod requant {
        pub fn f16_to_f32(h: u16) -> f32 {
            let s = ((h >> 15) & 1) as u32;
            let e = ((h >> 10) & 0x1f) as u32;
            let m = (h & 0x3ff) as u32;
            let bits = if e == 0 {
                if m == 0 { s << 31 } else {
                    // subnormal
                    let mut m = m;
                    let mut e = 127 - 15 + 1;
                    while m & 0x400 == 0 {
                        m <<= 1;
                        e -= 1;
                    }
                    (s << 31) | ((e as u32) << 23) | ((m & 0x3ff) << 13)
                }
            } else if e == 0x1f {
                (s << 31) | (0xff << 23) | (m << 13)
            } else {
                (s << 31) | ((e + 127 - 15) << 23) | (m << 13)
            };
            f32::from_bits(bits)
        }

        fn f32_to_f16(x: f32) -> u16 {
            let bits = x.to_bits();
            let s = ((bits >> 16) & 0x8000) as u16;
            let e = ((bits >> 23) & 0xff) as i32 - 127 + 15;
            let m = bits & 0x7f_ffff;
            if e <= 0 {
                s // flush to zero (scales here are never subnormal)
            } else if e >= 0x1f {
                s | 0x7c00
            } else {
                s | ((e as u16) << 10) | ((m >> 13) as u16)
            }
        }

        fn k4_scale_min(j: usize, q: &[u8], d: &mut u8, m: &mut u8) {
            if j < 4 {
                *d = q[j] & 63;
                *m = q[j + 4] & 63;
            } else {
                *d = (q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4);
                *m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
            }
        }

        /// Dequantize one 256-element block of `ty` at `src` into `out`.
        pub fn dequant_block(ty: gguf::TensorType, src: &[u8], out: &mut [f32; 256]) {
            use gguf::TensorType as T;
            match ty {
                T::Q2K => {
                    let (scales, qs) = (&src[0..16], &src[16..80]);
                    let d = f16_to_f32(u16::from_le_bytes([src[80], src[81]]));
                    let dmin = f16_to_f32(u16::from_le_bytes([src[82], src[83]]));
                    let mut i = 0;
                    for chunk in 0..2 {
                        for shift in [0u8, 2, 4, 6] {
                            let sub = i / 16;
                            let _ = sub;
                            for l in 0..32 {
                                let j = i / 16; // 16-value scale group
                                let sc = (scales[j] & 0x0f) as f32;
                                let mn = (scales[j] >> 4) as f32;
                                let q = ((qs[chunk * 32 + l] >> shift) & 3) as f32;
                                out[i] = d * sc * q - dmin * mn;
                                i += 1;
                            }
                        }
                    }
                }
                T::Q3K => {
                    let (hmask, qs, scales) = (&src[0..32], &src[32..96], &src[96..108]);
                    let d = f16_to_f32(u16::from_le_bytes([src[108], src[109]]));
                    let mut sc = [0i8; 16];
                    for j in 0..16 {
                        let s = if j < 8 {
                            (scales[j] & 0x0f) | (((scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)
                        } else {
                            (scales[j - 8] >> 4) | (((scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)
                        };
                        sc[j] = s as i8 - 32;
                    }
                    let mut i = 0;
                    let mut hbit = 1u8;
                    for chunk in 0..2 {
                        for shift in [0u8, 2, 4, 6] {
                            for l in 0..32 {
                                let mut q = ((qs[chunk * 32 + l] >> shift) & 3) as i32;
                                if hmask[l] & hbit == 0 {
                                    q -= 4;
                                }
                                out[i] = d * sc[i / 16] as f32 * q as f32;
                                i += 1;
                            }
                            hbit <<= 1;
                        }
                    }
                }
                T::Q4K | T::Q5K => {
                    let d = f16_to_f32(u16::from_le_bytes([src[0], src[1]]));
                    let dmin = f16_to_f32(u16::from_le_bytes([src[2], src[3]]));
                    let scales = &src[4..16];
                    let (qh, qs) = if ty == T::Q5K {
                        (&src[16..48], &src[48..176])
                    } else {
                        (&src[0..0], &src[16..144])
                    };
                    let mut i = 0;
                    for j in 0..4 {
                        let (mut s1, mut m1, mut s2, mut m2) = (0u8, 0u8, 0u8, 0u8);
                        k4_scale_min(2 * j, scales, &mut s1, &mut m1);
                        k4_scale_min(2 * j + 1, scales, &mut s2, &mut m2);
                        for l in 0..32 {
                            let mut q = (qs[j * 32 + l] & 0x0f) as f32;
                            if ty == T::Q5K && qh[l] & (1 << (2 * j)) != 0 {
                                q += 16.0;
                            }
                            out[i] = d * s1 as f32 * q - dmin * m1 as f32;
                            i += 1;
                        }
                        for l in 0..32 {
                            let mut q = (qs[j * 32 + l] >> 4) as f32;
                            if ty == T::Q5K && qh[l] & (1 << (2 * j + 1)) != 0 {
                                q += 16.0;
                            }
                            out[i] = d * s2 as f32 * q - dmin * m2 as f32;
                            i += 1;
                        }
                    }
                }
                T::Q6K => {
                    let (ql, qh, scales) = (&src[0..128], &src[128..192], &src[192..208]);
                    let d = f16_to_f32(u16::from_le_bytes([src[208], src[209]]));
                    let mut i = 0;
                    for chunk in 0..2 {
                        let (ql, qh) = (&ql[chunk * 64..], &qh[chunk * 32..]);
                        let sc = &scales[chunk * 8..];
                        for l in 0..32 {
                            let q0 = ((ql[l] & 0x0f) as i32 | (((qh[l] >> 0) & 3) as i32) << 4) - 32;
                            let q1 = ((ql[32 + l] & 0x0f) as i32 | (((qh[l] >> 2) & 3) as i32) << 4) - 32;
                            let q2 = ((ql[l] >> 4) as i32 | (((qh[l] >> 4) & 3) as i32) << 4) - 32;
                            let q3 = ((ql[32 + l] >> 4) as i32 | (((qh[l] >> 6) & 3) as i32) << 4) - 32;
                            out[i + l] = d * sc[l / 16] as i8 as f32 * q0 as f32;
                            out[i + 32 + l] = d * sc[2 + l / 16] as i8 as f32 * q1 as f32;
                            out[i + 64 + l] = d * sc[4 + l / 16] as i8 as f32 * q2 as f32;
                            out[i + 96 + l] = d * sc[6 + l / 16] as i8 as f32 * q3 as f32;
                        }
                        i += 128;
                    }
                }
                _ => unreachable!("requant: unsupported type"),
            }
        }

        /// f32 -> q8_0 (34-byte blocks of 32: f16 scale + int8 quants).
        pub fn quantize_q8_0(x: &[f32], out: &mut Vec<u8>) {
            for blk in x.chunks(32) {
                let amax = blk.iter().fold(0f32, |a, &v| a.max(v.abs()));
                let d = amax / 127.0;
                let id = if d > 0.0 { 1.0 / d } else { 0.0 };
                out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
                for &v in blk {
                    out.push((v * id).round().clamp(-128.0, 127.0) as i8 as u8);
                }
            }
        }
    }

    /// One logical model file that may span split-gguf shards: shard i
    /// covers [bases[i], bases[i]+size_i) of a virtual offset space (the
    /// same space the merged Gguf's tensor offsets live in).
    pub struct VFile {
        files: Vec<(u64, File)>,
    }

    impl VFile {
        fn open(shards: &[(u64, std::path::PathBuf)]) -> Result<VFile> {
            let mut files = Vec::with_capacity(shards.len());
            for (base, p) in shards {
                files.push((*base, File::open(p)?));
            }
            Ok(VFile { files })
        }

        fn read_exact_at(&self, buf: &mut [u8], offset: u64) -> std::io::Result<()> {
            let i = match self.files.binary_search_by(|(b, _)| b.cmp(&offset)) {
                Ok(i) => i,
                Err(i) => i.saturating_sub(1),
            };
            self.files[i].1.read_exact_at(buf, offset - self.files[i].0)
        }
    }

    fn read_tensor_bytes(file: &VFile, g: &Gguf, name: &str) -> Result<Vec<u8>> {
        let t = g.tensor(name).ok_or_else(|| meta_err(name))?;
        let bytes = t.byte_size().ok_or_else(|| meta_err(name))?;
        let mut buf = vec![0u8; bytes as usize];
        file.read_exact_at(&mut buf, g.data_offset + t.offset)?;

        // dense K-quant tensors -> q8_0 (see mod requant). output.weight
        // stays native: head_logits reads K-quants directly.
        let convert = matches!(
            t.ty,
            TensorType::Q2K | TensorType::Q3K | TensorType::Q4K | TensorType::Q5K | TensorType::Q6K
        ) && name != "output.weight";
        if convert {
            let n = t.n_elements() as usize;
            let (block_elems, block_bytes) = t.ty.block_layout().ok_or_else(|| meta_err(name))?;
            debug_assert_eq!(block_elems, 256);
            let mut out = Vec::with_capacity(n / 32 * 34);
            let mut f = [0f32; 256];
            for b in 0..n / 256 {
                requant::dequant_block(t.ty, &buf[b * block_bytes as usize..], &mut f);
                requant::quantize_q8_0(&f, &mut out);
            }
            return Ok(out);
        }
        Ok(buf)
    }

    /// Small f32 tensor -> host vec (scales, per-layer constants).
    fn read_tensor_f32(file: &VFile, g: &Gguf, name: &str) -> Result<Vec<f32>> {
        let t = g.tensor(name).ok_or_else(|| meta_err(name))?;
        if t.ty != TensorType::F32 {
            return Err(format!("{name}: expected f32, got {:?}", t.ty).into());
        }
        let mut buf = vec![0u8; t.n_elements() as usize * 4];
        file.read_exact_at(&mut buf, g.data_offset + t.offset)?;
        Ok(buf
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect())
    }

    fn upload(file: &VFile, g: &Gguf, name: &str) -> Result<DeviceBuf> {
        Ok(DeviceBuf::from_bytes(&read_tensor_bytes(file, g, name)?)?)
    }

    /// K-quant tensor -> resident device bytes + the matmul_kq metadata.
    /// Reads RAW file bytes: read_tensor_bytes would requant K-quants to
    /// q8_0 (1.9x the VRAM and the wrong layout for matmul_kq).
    fn upload_kq(file: &VFile, g: &Gguf, name: &str) -> Result<KqW> {
        let t = g.tensor(name).ok_or_else(|| meta_err(name))?;
        let quant = quant_code(t.ty)
            .ok_or_else(|| format!("{name}: unsupported K-quant type {:?}", t.ty))?;
        let bytes = t.byte_size().ok_or_else(|| meta_err(name))?;
        let mut buf = vec![0u8; bytes as usize];
        file.read_exact_at(&mut buf, g.data_offset + t.offset)?;
        Ok(KqW {
            w: DeviceBuf::from_bytes(&buf)?,
            row_bytes: t.ty.row_bytes(t.dims[0]).unwrap(),
            quant,
        })
    }

    /// f16 tensor -> host f32 (deepseek4 ships router/HC/compressor
    /// weights as f16; small ones convert to f32 for matmul_f32).
    fn read_f16_as_f32(file: &VFile, g: &Gguf, name: &str) -> Result<Vec<f32>> {
        let t = g.tensor(name).ok_or_else(|| meta_err(name))?;
        if t.ty != TensorType::F16 {
            return Err(format!("{name}: expected f16, got {:?}", t.ty).into());
        }
        let mut buf = vec![0u8; t.n_elements() as usize * 2];
        file.read_exact_at(&mut buf, g.data_offset + t.offset)?;
        Ok(buf
            .chunks_exact(2)
            .map(|c| requant::f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect())
    }

    fn upload_f16_as_f32(file: &VFile, g: &Gguf, name: &str) -> Result<DeviceBuf> {
        Ok(DeviceBuf::from_f32(&read_f16_as_f32(file, g, name)?)?)
    }

    /// Tensor -> device f32 regardless of source encoding. Small tensors
    /// whose consumers are matmul_f32 (qwen35 ssm_alpha/ssm_beta): dense
    /// 27B files K-quantize them where the 35B shipped f32.
    fn upload_as_f32(file: &VFile, g: &Gguf, name: &str) -> Result<DeviceBuf> {
        let t = g.tensor(name).ok_or_else(|| meta_err(name))?;
        match t.ty {
            TensorType::F32 => upload(file, g, name),
            TensorType::F16 => upload_f16_as_f32(file, g, name),
            TensorType::Q4K => {
                let n = t.n_elements() as usize;
                let mut buf = vec![0u8; n / 256 * 144];
                file.read_exact_at(&mut buf, g.data_offset + t.offset)?;
                Ok(DeviceBuf::from_f32(&quant::cpu_dot::dequant_q4_k(&buf, n))?)
            }
            other => Err(format!("{name}: no f32 path for {other:?}").into()),
        }
    }

    /// f16 tensor -> q8_0 bytes (deepseek4's bigger f16 matmul weights
    /// ride the q8_0 fast path; ~0.4% quantization noise).
    fn read_f16_as_q8(file: &VFile, g: &Gguf, name: &str) -> Result<Vec<u8>> {
        let t = g.tensor(name).ok_or_else(|| meta_err(name))?;
        if t.ty != TensorType::F16 {
            return Err(format!("{name}: expected f16, got {:?}", t.ty).into());
        }
        let n = t.n_elements() as usize;
        let mut buf = vec![0u8; n * 2];
        file.read_exact_at(&mut buf, g.data_offset + t.offset)?;
        let mut out = Vec::with_capacity(n / 32 * 34);
        let mut f = [0f32; 256];
        for blk in buf.chunks(512) {
            let m = blk.len() / 2;
            for (i, c) in blk.chunks_exact(2).enumerate() {
                f[i] = requant::f16_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
            requant::quantize_q8_0(&f[..m], &mut out);
        }
        Ok(out)
    }

    /// Big attention weights: VRAM while `vram_budget` lasts, then pinned
    /// host memory (zero-copy PCIe reads). Gqa attn always fits, so its
    /// budget is unlimited; Mla (GLM-class, ~12GB attn q8) spends a
    /// PULSAR_ATTN_VRAM_GB budget (default 5) on the tensors the caller
    /// routes here - zero-copy reads measure ~6GB/s vs VRAM's ~288GB/s, so
    /// every budgeted byte is ~50x cheaper to read each token.
    /// PULSAR_ATTN_HOST=1 forces everything pinned.
    fn upload_attn(
        file: &VFile,
        g: &Gguf,
        name: &str,
        vram_budget: &mut i64,
    ) -> Result<DeviceBuf> {
        let bytes = read_tensor_bytes(file, g, name)?;
        let force_host = std::env::var("PULSAR_ATTN_HOST").ok().as_deref() == Some("1");
        let use_vram = !force_host && *vram_budget >= bytes.len() as i64;
        let mut buf = if use_vram {
            *vram_budget -= bytes.len() as i64;
            DeviceBuf::alloc(bytes.len())?
        } else {
            DeviceBuf::alloc_pinned(bytes.len())?
        };
        buf.write(0, &bytes)?;
        Ok(buf)
    }

    impl Model {
        pub fn load(path: &Path) -> Result<Model> {
            let (shards, gguf) = parse_header(path)?;
            let file = VFile::open(&shards)?;
            let shape = Shape::from_gguf(&gguf)?;

            // the embedding table is read ~one row per token - pinned
            // host is free for it and returns ~1GB of VRAM to hot weights
            let token_embd = {
                // deepseek4 ships the table f16; embed_q8_0 wants q8_0
                let bytes = if shape.family == Family::Dsv4 {
                    read_f16_as_q8(&file, &gguf, "token_embd.weight")?
                } else {
                    read_tensor_bytes(&file, &gguf, "token_embd.weight")?
                };
                let mut buf = if matches!(shape.family, Family::Mla | Family::Dsv4) {
                    DeviceBuf::alloc_pinned(bytes.len())?
                } else {
                    DeviceBuf::alloc(bytes.len())?
                };
                buf.write(0, &bytes)?;
                buf
            };
            let output_norm = upload(&file, &gguf, "output_norm.weight")?;
            // tied embeddings (gemma4): no output.weight, the lm head IS
            // the (q8_0) embedding table
            let head_name = if gguf.tensor("output.weight").is_some() {
                "output.weight"
            } else {
                "token_embd.weight"
            };
            let output = upload(&file, &gguf, head_name)?;
            let output_kq = {
                let t = gguf.tensor(head_name).ok_or_else(|| meta_err(head_name))?;
                let quant = match t.ty {
                    TensorType::Q2K => Some(kernels::QUANT_Q2_K),
                    TensorType::IQ2XXS => Some(kernels::QUANT_IQ2_XXS),
                    TensorType::Q4K => Some(kernels::QUANT_Q4_K),
                    TensorType::Q5K => Some(kernels::QUANT_Q5_K),
                    TensorType::Q6K => Some(kernels::QUANT_Q6_K),
                    TensorType::Q3K => Some(kernels::QUANT_Q3_K),
                    _ => None,
                };
                quant.map(|q| (t.ty.row_bytes(t.dims[0]).unwrap(), q))
            };

            // Attn placement: park the whole stack on a second GPU when
            // one has room (Mla only; Gqa attn always fits beside the
            // experts). Roles by capability: expert streaming needs link
            // bandwidth (the primary, CUDA's fastest card), attn residency
            // only needs capacity - a bandwidth-crippled slot still serves
            // it at full speed, paying only once at load.
            // PULSAR_ATTN_GPU=<idx> forces, =off disables auto-detection.
            let primary = kernels::get_device();
            let attn_dev = match shape.family {
                Family::Mla => match std::env::var("PULSAR_ATTN_GPU").ok().as_deref() {
                    Some("off") | Some("-1") => None,
                    Some(v) => v.trim().parse::<i32>().ok().filter(|&d| {
                        let ok = d != primary && d >= 0 && d < kernels::device_count();
                        if !ok {
                            eprintln!("pulsar: ignoring PULSAR_ATTN_GPU={d} (primary is {primary}, {} devices)", kernels::device_count());
                        }
                        ok
                    }),
                    None => {
                        // auto: largest-free secondary that fits the stack
                        let mut need = 0u64;
                        for il in 0..shape.n_exec_layer {
                            for suf in [
                                "attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight",
                                "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight",
                                "attn_k_b.weight", "attn_v_b.weight", "attn_output.weight",
                            ] {
                                if let Some(ti) = gguf.tensor(&format!("blk.{il}.{suf}")) {
                                    need += ti.byte_size().unwrap_or(0);
                                }
                            }
                        }
                        // reserve: compact KV at ctx 4096 (both caches span
                        // n_kv_lora + qk_rope) + scratch/hop buffers + CUDA
                        // context overhead. Pinned overflow would be read
                        // over the attn card's own link, so it must FIT.
                        let kv = shape.n_exec_layer as u64
                            * 4096
                            * (shape.n_kv_lora + shape.qk_rope) as u64
                            * 4;
                        need += kv + (1 << 30);
                        let mut best: Option<(usize, i32)> = None;
                        for d in 0..kernels::device_count() {
                            if d == primary {
                                continue;
                            }
                            if let Ok((free, _)) = kernels::mem_info(d) {
                                if free as u64 >= need && best.is_none_or(|(bf, _)| free > bf) {
                                    best = Some((free, d));
                                } else if (free as u64) < need {
                                    eprintln!(
                                        "pulsar: CUDA device {d} skipped for attn ({:.1}GB free < {:.1}GB needed)",
                                        free as f64 / 1e9,
                                        need as f64 / 1e9
                                    );
                                }
                            }
                        }
                        best.map(|(free, d)| {
                            eprintln!(
                                "pulsar: auto-detected attn GPU: CUDA device {d} ({:.1}GB free, attn stack needs {:.1}GB)",
                                free as f64 / 1e9,
                                need as f64 / 1e9
                            );
                            d
                        })
                    }
                },
                // Gqa: opt-in only (PULSAR_ATTN_GPU=<idx>). Gqa attention is
                // already VRAM-resident on the primary, so offloading is a
                // capacity SHUFFLE: the attn stack's bytes migrate to the
                // second card, evicting that much expert tier from it. It
                // pays when the primary is squeezed (fat attn stacks, long
                // contexts); measured per model, not assumed.
                Family::Gqa => match std::env::var("PULSAR_ATTN_GPU").ok().as_deref() {
                    Some("off") | Some("-1") | None => None,
                    Some(v) => v.trim().parse::<i32>().ok().filter(|&d| {
                        let ok = d != primary && d >= 0 && d < kernels::device_count();
                        if !ok {
                            eprintln!("pulsar: ignoring PULSAR_ATTN_GPU={d} (primary is {primary}, {} devices)", kernels::device_count());
                        }
                        ok
                    }),
                },
                // ponytail: dsv4/qwen35 v1 run everything on the primary;
                // attn offload comes with the perf pass
                Family::Dsv4 | Family::Qwen35 => None,
            };
            if let Some(d) = attn_dev {
                eprintln!("pulsar: attn weights + KV resident on CUDA device {d}");
            }

            // Dense qwen35 on 2+ cards: whole-layer ownership. The model
            // fits in combined VRAM, so a layer's full stack (attn/GDN +
            // KV + FFN triple) is resident on ONE card and the residual
            // stream crosses once per boundary per chunk - the per-layer
            // tier round trips it replaces were ~55ms of a 103ms token.
            // Split point balances per-token weight reads; the lm head
            // (read every token) counts on the primary's side.
            // PULSAR_SPLIT=<n> forces n leading layers on the primary,
            // PULSAR_SPLIT=off keeps everything on one card.
            let qwen35_dense = shape.family == Family::Qwen35 && shape.n_expert == 1;
            let mut layer_dev = vec![primary; shape.n_exec_layer as usize];
            if qwen35_dense
                && kernels::device_count() > 1
                && std::env::var("PULSAR_SPLIT").ok().as_deref() != Some("off")
            {
                let second = (0..kernels::device_count())
                    .filter(|&d| d != primary)
                    .max_by_key(|&d| kernels::mem_info(d).map(|(f, _)| f).unwrap_or(0))
                    .unwrap();
                // VRAM bytes, not file bytes: MatW/DenseKq tensors upload
                // raw K-quant; the rest of the K-quants (and the embedding
                // table) requant to q8_0 (~1.9x for Q4_K)
                let vram = |t: &TensorInfo| -> u64 {
                    let raw = t.byte_size().unwrap_or(0);
                    let kq = matches!(
                        t.ty,
                        TensorType::Q2K | TensorType::Q3K | TensorType::Q4K
                            | TensorType::Q5K | TensorType::Q6K
                    );
                    let ffn_raw = t.name.ends_with("ffn_gate.weight")
                        || t.name.ends_with("ffn_up.weight")
                        || t.name.ends_with("ffn_down.weight");
                    if kq && !(ffn_raw || MatW::keep_native(t)) {
                        t.n_elements() / 32 * 34
                    } else {
                        raw
                    }
                };
                let lbytes: Vec<u64> = (0..shape.n_exec_layer)
                    .map(|il| {
                        let p = format!("blk.{il}.");
                        gguf.tensors
                            .iter()
                            .filter(|t| t.name.starts_with(&p))
                            .map(&vram)
                            .sum()
                    })
                    .collect();
                // resident on the primary regardless of the split: lm head
                // (native K-quant) + the q8_0-converted embedding table
                let fixed: u64 = gguf.tensor("output.weight").and_then(|t| t.byte_size()).unwrap_or(0)
                    + gguf.tensor("token_embd.weight").map(|t| t.n_elements() / 32 * 34).unwrap_or(0);
                // layers run SEQUENTIALLY within a token, so total time is
                // sum(bytes/bw) per card - minimized by filling the fast
                // primary to capacity, not by balancing
                let mut n0 = lbytes.len();
                if let Ok((free, _)) = kernels::mem_info(primary) {
                    let reserve = 2u64 << 30;
                    while n0 > 0 && fixed + lbytes[..n0].iter().sum::<u64>() + reserve > free as u64 {
                        n0 -= 1;
                    }
                }
                if let Some(n) = std::env::var("PULSAR_SPLIT").ok().and_then(|v| v.parse::<usize>().ok()) {
                    n0 = n.min(lbytes.len());
                }
                for d in layer_dev.iter_mut().skip(n0) {
                    *d = second;
                }
                let b1: u64 = lbytes[n0..].iter().sum();
                eprintln!(
                    "pulsar: dense split: layers 0..{n0} on device {primary}, {n0}..{} on device {second} ({:.1}GB)",
                    lbytes.len(),
                    b1 as f64 / 1e9
                );
            }

            // Mla: spend a VRAM budget on the two big per-layer attn
            // tensors (attn_output ~107MB, q_b ~36MB on GLM-5.2) - they are
            // 80%+ of the per-token pinned-host read traffic. Gqa attn is
            // small enough to always live in VRAM. With a dedicated attn
            // GPU the whole stack (~14GB q8) goes resident by default -
            // pinned overflow would be read over that card's own link.
            let gemma_arch = gguf.architecture() == Some("gemma4");
            let ink_arch = gguf.architecture() == Some("inkling");
            // per-layer attention geometry: gemma4 interleaves sliding-
            // window layers (own kv width, head_dim, theta) with full ones
            let geom: Vec<Geom> = if ink_arch {
                // inkling: 55/66 layers at window 512 with their own kv
                // width; no rope, so theta/factors are dead fields
                let kvh: Vec<u64> = match gguf.arch_meta("attention.head_count_kv") {
                    Some(Value::Array(a)) => a.iter().filter_map(Value::as_u64).collect(),
                    Some(v) => v.as_u64().map(|x| vec![x]).unwrap_or_default(),
                    None => Vec::new(),
                };
                let swa_pat: Vec<bool> = match gguf.arch_meta("attention.sliding_window_pattern") {
                    Some(Value::Array(a)) => a
                        .iter()
                        .map(|v| match v {
                            Value::Bool(b) => *b,
                            other => other.as_u64().unwrap_or(0) != 0,
                        })
                        .collect(),
                    _ => Vec::new(),
                };
                let window = gguf
                    .arch_meta("attention.sliding_window")
                    .and_then(Value::as_u64)
                    .unwrap_or(512) as u32;
                (0..shape.n_exec_layer as usize)
                    .map(|il| {
                        let swa = swa_pat.get(il).copied().unwrap_or(false);
                        Geom {
                            n_head_kv: kvh
                                .get(il)
                                .copied()
                                .unwrap_or(shape.n_head_kv as u64)
                                as u32,
                            head_dim: shape.head_dim,
                            theta: 0.0,
                            window: if swa { window } else { 0 },
                            factors: false,
                        }
                    })
                    .collect()
            } else if gemma_arch {
                let arr_u = |k: &str| -> Vec<u64> {
                    match gguf.arch_meta(k) {
                        Some(Value::Array(a)) => a.iter().filter_map(Value::as_u64).collect(),
                        Some(v) => v.as_u64().map(|x| vec![x]).unwrap_or_default(),
                        None => Vec::new(),
                    }
                };
                let kvh = arr_u("attention.head_count_kv");
                let swa_pat: Vec<bool> = match gguf.arch_meta("attention.sliding_window_pattern") {
                    Some(Value::Array(a)) => a
                        .iter()
                        .map(|v| matches!(v, Value::Bool(true)))
                        .collect(),
                    _ => Vec::new(),
                };
                let g_u = |k: &str, d: u32| -> u32 {
                    gguf.arch_meta(k).and_then(Value::as_u64).map(|v| v as u32).unwrap_or(d)
                };
                let g_f = |k: &str, d: f32| -> f32 {
                    gguf.arch_meta(k).and_then(Value::as_f32).unwrap_or(d)
                };
                let hd_full = g_u("attention.key_length", 512);
                let hd_swa = g_u("attention.key_length_swa", hd_full);
                let theta_full = g_f("rope.freq_base", 1_000_000.0);
                let theta_swa = g_f("rope.freq_base_swa", 10_000.0);
                let window = g_u("attention.sliding_window", 0);
                (0..shape.n_exec_layer as usize)
                    .map(|il| {
                        let swa = swa_pat.get(il).copied().unwrap_or(false);
                        Geom {
                            n_head_kv: kvh.get(il).copied().unwrap_or(1) as u32,
                            head_dim: if swa { hd_swa } else { hd_full },
                            theta: if swa { theta_swa } else { theta_full },
                            window: if swa { window } else { 0 },
                            factors: !swa,
                        }
                    })
                    .collect()
            } else {
                Vec::new()
            };
            let rope_factors = if gemma_arch && gguf.tensor("rope_freqs.weight").is_some() {
                // the rope kernel runs wherever q/k live - factors follow
                // the attn card under Gqa offload
                if let Some(d) = attn_dev {
                    kernels::set_device(d)?;
                }
                let f = upload(&file, &gguf, "rope_freqs.weight")?;
                if attn_dev.is_some() {
                    kernels::set_device(primary)?;
                }
                Some(f)
            } else {
                None
            };

            let env_budget = std::env::var("PULSAR_ATTN_VRAM_GB")
                .ok()
                .and_then(|v| v.parse::<i64>().ok())
                .map(|v| v << 30);
            let mut attn_vram_budget: i64 = match (shape.family, attn_dev) {
                (Family::Gqa, _) => i64::MAX,
                (Family::Mla, Some(_)) => env_budget.unwrap_or(i64::MAX),
                (Family::Mla, None) => env_budget.unwrap_or(6 << 30),
                // V4's attn+compressor+indexer q8 stack is ~6GB total;
                // resident on a 16GB card still leaves an expert cache
                (Family::Dsv4, _) => env_budget.unwrap_or(8 << 30),
                // qwen35: the whole non-expert stack is ~2GB - resident
                (Family::Qwen35, _) => env_budget.unwrap_or(i64::MAX),
            };
            // No-attn-GPU Mla: an oversized budget OOMs the load instead
            // of degrading (measured: 8GB+ on a 16GB primary fails at
            // cudaMalloc mid-upload; 10GB with 15.4 free still died in the
            // solver). Clamp to free minus a measured 9GB reserve (KV,
            // activations, staging, MLA scratch, CUDA context). The 6GB
            // default is already the feasible top on a 16GB card; per-
            // tensor placement has no headroom beyond this clamp because
            // every attn byte is read exactly once per token (flat value).
            if attn_dev.is_none() && shape.family == Family::Mla && attn_vram_budget < i64::MAX {
                if let Ok((free, _)) = kernels::mem_info(primary) {
                    let cap = (free as i64) - (9i64 << 30);
                    if cap > 0 && attn_vram_budget > cap {
                        eprintln!(
                            "pulsar: attn VRAM budget clamped {:.1} -> {:.1}GB (free {:.1}GB)",
                            attn_vram_budget as f64 / 1e9,
                            cap as f64 / 1e9,
                            free as f64 / 1e9
                        );
                        attn_vram_budget = cap;
                    }
                }
            }
            // small Mla attn tensors always go pinned (not worth budget) -
            // except on a dedicated attn GPU, where everything is resident
            let mut no_budget: i64 = if attn_dev.is_some() { i64::MAX } else { 0 };

            let dsv4_arch = shape.family == Family::Dsv4;
            let compress_ratios: Vec<u32> = if dsv4_arch {
                match gguf.arch_meta("attention.compress_ratios") {
                    Some(Value::Array(a)) => {
                        a.iter().filter_map(Value::as_u64).map(|v| v as u32).collect()
                    }
                    _ => return Err(meta_err("attention.compress_ratios")),
                }
            } else {
                Vec::new()
            };
            if dsv4_arch && compress_ratios.len() < shape.n_exec_layer as usize {
                return Err("compress_ratios shorter than the layer count".into());
            }

            let load_layer = |il: u32,
                              attn_vram_budget: &mut i64,
                              no_budget: &mut i64|
             -> Result<LayerW> {
                let t = |suffix: &str| format!("blk.{il}.{suffix}");
                let ffn = if il < shape.n_leading_dense {
                    Ffn::Dense {
                        gate: upload(&file, &gguf, &t("ffn_gate.weight"))?,
                        up: upload(&file, &gguf, &t("ffn_up.weight"))?,
                        down: upload(&file, &gguf, &t("ffn_down.weight"))?,
                    }
                } else if shape.family == Family::Qwen35
                    && gguf.tensor(&t("ffn_gate_exps.weight")).is_none()
                {
                    // dense qwen35 (27B): the FFN triple resident in
                    // native K-quant on whatever device is current (the
                    // layer's owner under the dense split)
                    Ffn::DenseKq {
                        gate: upload_kq(&file, &gguf, &t("ffn_gate.weight"))?,
                        up: upload_kq(&file, &gguf, &t("ffn_up.weight"))?,
                        down: upload_kq(&file, &gguf, &t("ffn_down.weight"))?,
                    }
                } else {
                    let exps = |suffix: &str| -> Result<ExpertTensor> {
                        let name = t(suffix);
                        let ti = gguf.tensor(&name).ok_or_else(|| meta_err(&name))?;
                        ExpertTensor::new(&gguf, ti, shape.n_expert)
                    };
                    // inkling shexp bank: same shape as routed experts but
                    // n_shexp_sink wide
                    let exps_sink = |suffix: &str| -> Result<ExpertTensor> {
                        let name = t(suffix);
                        let ti = gguf.tensor(&name).ok_or_else(|| meta_err(&name))?;
                        ExpertTensor::new(&gguf, ti, shape.n_shexp_sink)
                    };
                    // router bias name varies by converter: bare on the
                    // antirez Hy3/GLM files, ".bias" on others
                    let probs_b_name = if gguf.tensor(&t("exp_probs_b")).is_some() {
                        t("exp_probs_b")
                    } else {
                        t("exp_probs_b.bias")
                    };
                    // gemma4 fuses gate and up into one tensor: rows
                    // 0..n_ff are gate, n_ff..2n_ff are up. One slab per
                    // expert serves both (up = gate ptr + fused_up_off).
                    let fused = gguf.tensor(&t("ffn_gate_up_exps.weight")).is_some();
                    let (gate_exps, up_exps, fused_up_off) = if fused {
                        let g = exps("ffn_gate_up_exps.weight")?;
                        let off = g.row_bytes * shape.n_ff_exp as u64;
                        let u = g.clone();
                        (g, u, off)
                    } else {
                        (exps("ffn_gate_exps.weight")?, exps("ffn_up_exps.weight")?, 0)
                    };
                    Ffn::Moe {
                        // deepseek4 ships the router f16; matmul_f32
                        // wants f32 (router precision drives selection)
                        gate_inp: if dsv4_arch {
                            upload_f16_as_f32(&file, &gguf, &t("ffn_gate_inp.weight"))?
                        } else {
                            upload(&file, &gguf, &t("ffn_gate_inp.weight"))?
                        },
                        // no bias tensor (qwen3moe) -> zeros: score = prob
                        probs_b: if gguf.tensor(&probs_b_name).is_some() {
                            upload(&file, &gguf, &probs_b_name)?
                        } else {
                            let mut z = DeviceBuf::alloc(shape.n_expert as usize * 4)?;
                            kernels::zero(&mut z, shape.n_expert as usize * 4)?;
                            z
                        },
                        // inkling's ffn_*_shexp are 3D BANKS (the sink
                        // ExpertTensors below), not the 2D dense triple
                        shexp: if !ink_arch && gguf.tensor(&t("ffn_gate_shexp.weight")).is_some() {
                            Some((
                                upload(&file, &gguf, &t("ffn_gate_shexp.weight"))?,
                                upload(&file, &gguf, &t("ffn_up_shexp.weight"))?,
                                upload(&file, &gguf, &t("ffn_down_shexp.weight"))?,
                            ))
                        } else if gemma_arch {
                            // gemma's shared MLP: plain ffn tensors double
                            // as an always-on expert beside the routed set
                            Some((
                                upload(&file, &gguf, &t("ffn_gate.weight"))?,
                                upload(&file, &gguf, &t("ffn_up.weight"))?,
                                upload(&file, &gguf, &t("ffn_down.weight"))?,
                            ))
                        } else {
                            None
                        },
                        gate_exps,
                        up_exps,
                        down_exps: exps("ffn_down_exps.weight")?,
                        fused_up_off,
                        down_scale: if gguf.tensor(&t("ffn_down_exps.scale")).is_some() {
                            Some(upload(&file, &gguf, &t("ffn_down_exps.scale"))?)
                        } else {
                            None
                        },
                        sink: if ink_arch {
                            Some([
                                exps_sink("ffn_gate_shexp.weight")?,
                                exps_sink("ffn_up_shexp.weight")?,
                                exps_sink("ffn_down_shexp.weight")?,
                            ])
                        } else {
                            None
                        },
                    }
                };
                if let Some(d) = attn_dev {
                    kernels::set_device(d)?;
                }
                let attn = match shape.family {
                    Family::Gqa => Attn::Gqa {
                        attn_q: upload_attn(&file, &gguf, &t("attn_q.weight"), &mut *attn_vram_budget)?,
                        attn_k: upload_attn(&file, &gguf, &t("attn_k.weight"), &mut *attn_vram_budget)?,
                        attn_v: if gguf.tensor(&t("attn_v.weight")).is_some() {
                            Some(upload_attn(&file, &gguf, &t("attn_v.weight"), &mut *attn_vram_budget)?)
                        } else {
                            None // gemma attention_k_eq_v: k doubles as v
                        },
                        q_norm: upload(&file, &gguf, &t("attn_q_norm.weight"))?,
                        k_norm: upload(&file, &gguf, &t("attn_k_norm.weight"))?,
                    },
                    Family::Mla => Attn::Mla {
                        q_a: upload_attn(&file, &gguf, &t("attn_q_a.weight"), &mut *no_budget)?,
                        q_a_norm: upload(&file, &gguf, &t("attn_q_a_norm.weight"))?,
                        q_b: upload_attn(&file, &gguf, &t("attn_q_b.weight"), &mut *attn_vram_budget)?,
                        kv_a_mqa: upload_attn(&file, &gguf, &t("attn_kv_a_mqa.weight"), &mut *no_budget)?,
                        kv_a_norm: upload(&file, &gguf, &t("attn_kv_a_norm.weight"))?,
                        k_b: upload_attn(&file, &gguf, &t("attn_k_b.weight"), &mut *no_budget)?,
                        v_b: upload_attn(&file, &gguf, &t("attn_v_b.weight"), &mut *no_budget)?,
                        indexer: if shape.n_idx_topk > 0
                            && gguf.tensor(&t("indexer.attn_q_b.weight")).is_some()
                        {
                            Some(IdxW {
                                q_b: upload(&file, &gguf, &t("indexer.attn_q_b.weight"))?,
                                k: upload(&file, &gguf, &t("indexer.attn_k.weight"))?,
                                k_norm: upload(&file, &gguf, &t("indexer.k_norm.weight"))?,
                                k_norm_b: upload(&file, &gguf, &t("indexer.k_norm.bias"))?,
                                proj: upload(&file, &gguf, &t("indexer.proj.weight"))?,
                            })
                        } else {
                            None
                        },
                    },
                    Family::Dsv4 => {
                        let ratio = compress_ratios[il as usize];
                        // f16 attn-side matmul weights ride q8_0; budget
                        // placement like any other big attn tensor
                        let upload_f16_q8 = |name: &str, budget: &mut i64| -> Result<DeviceBuf> {
                            let bytes = read_f16_as_q8(&file, &gguf, name)?;
                            let use_vram = *budget >= bytes.len() as i64
                                && std::env::var("PULSAR_ATTN_HOST").ok().as_deref() != Some("1");
                            let mut buf = if use_vram {
                                *budget -= bytes.len() as i64;
                                DeviceBuf::alloc(bytes.len())?
                            } else {
                                DeviceBuf::alloc_pinned(bytes.len())?
                            };
                            buf.write(0, &bytes)?;
                            Ok(buf)
                        };
                        let comp_lane = |prefix: &str, budget: &mut i64| -> Result<Dsv4CompW> {
                            let kv_name = t(&format!("{prefix}_kv.weight"));
                            let ti = gguf.tensor(&kv_name).ok_or_else(|| meta_err(&kv_name))?;
                            let width = ti.dims[1] as u32;
                            Ok(Dsv4CompW {
                                kv_w: upload_f16_q8(&kv_name, budget)?,
                                gate_w: upload_f16_q8(&t(&format!("{prefix}_gate.weight")), budget)?,
                                ape: DeviceBuf::from_f32(&read_f16_as_f32(&file, &gguf, &t(&format!("{prefix}_ape.weight")))?)?,
                                norm: DeviceBuf::from_f32(&read_tensor_f32(&file, &gguf, &t(&format!("{prefix}_norm.weight")))?)?,
                                width,
                            })
                        };
                        let tid2eid = if gguf.tensor(&t("ffn_gate_tid2eid.weight")).is_some() {
                            let bytes = read_tensor_bytes(&file, &gguf, &t("ffn_gate_tid2eid.weight"))?;
                            Some(
                                bytes
                                    .chunks_exact(4)
                                    .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                                    .collect(),
                            )
                        } else {
                            None
                        };
                        Attn::Dsv4(Box::new(Dsv4W {
                            q_a: upload_attn(&file, &gguf, &t("attn_q_a.weight"), &mut *attn_vram_budget)?,
                            q_a_norm: upload(&file, &gguf, &t("attn_q_a_norm.weight"))?,
                            q_b: upload_attn(&file, &gguf, &t("attn_q_b.weight"), &mut *attn_vram_budget)?,
                            kv: upload_attn(&file, &gguf, &t("attn_kv.weight"), &mut *attn_vram_budget)?,
                            kv_a_norm: upload(&file, &gguf, &t("attn_kv_a_norm.weight"))?,
                            out_a: upload_attn(&file, &gguf, &t("attn_output_a.weight"), &mut *attn_vram_budget)?,
                            sinks: upload(&file, &gguf, &t("attn_sinks.weight"))?,
                            hc_attn_fn: upload_f16_as_f32(&file, &gguf, &t("hc_attn_fn.weight"))?,
                            hc_ffn_fn: upload_f16_as_f32(&file, &gguf, &t("hc_ffn_fn.weight"))?,
                            hc_attn_scale: DeviceBuf::from_f32(&read_tensor_f32(&file, &gguf, &t("hc_attn_scale.weight"))?)?,
                            hc_attn_base: DeviceBuf::from_f32(&read_tensor_f32(&file, &gguf, &t("hc_attn_base.weight"))?)?,
                            hc_ffn_scale: DeviceBuf::from_f32(&read_tensor_f32(&file, &gguf, &t("hc_ffn_scale.weight"))?)?,
                            hc_ffn_base: DeviceBuf::from_f32(&read_tensor_f32(&file, &gguf, &t("hc_ffn_base.weight"))?)?,
                            // absent on hash layers (selection is tid2eid
                            // there); zeros keep the top-k path harmless
                            probs_b: if gguf.tensor(&t("exp_probs_b.bias")).is_some() {
                                read_tensor_f32(&file, &gguf, &t("exp_probs_b.bias"))?
                            } else {
                                vec![0.0; shape.n_expert as usize]
                            },
                            tid2eid,
                            comp: if ratio != 0 {
                                Some(comp_lane("attn_compressor", &mut *attn_vram_budget)?)
                            } else {
                                None
                            },
                            idx: if ratio == 4 {
                                Some(Dsv4IdxW {
                                    q_b: upload_f16_q8(&t("indexer.attn_q_b.weight"), &mut *attn_vram_budget)?,
                                    proj: upload_f16_as_f32(&file, &gguf, &t("indexer.proj.weight"))?,
                                    comp: comp_lane("indexer_compressor", &mut *attn_vram_budget)?,
                                })
                            } else {
                                None
                            },
                            ratio,
                        }))
                    }
                    Family::Qwen35 => {
                        // probe, don't pattern-match: the nextn/MTP layer
                        // (blk.n_exec) is full attention regardless of the
                        // every-4th interval
                        let is_attn = gguf.tensor(&t("attn_q.weight")).is_some();
                        Attn::Qwen35(Box::new(Qwen35W {
                            attn: if is_attn {
                                Some(Qwen35Attn {
                                    wq: MatW::load(&file, &gguf, &t("attn_q.weight"))?,
                                    wk: MatW::load(&file, &gguf, &t("attn_k.weight"))?,
                                    wv: MatW::load(&file, &gguf, &t("attn_v.weight"))?,
                                    out: MatW::load(&file, &gguf, &t("attn_output.weight"))?,
                                    q_norm: upload(&file, &gguf, &t("attn_q_norm.weight"))?,
                                    k_norm: upload(&file, &gguf, &t("attn_k_norm.weight"))?,
                                })
                            } else {
                                None
                            },
                            gdn: if is_attn {
                                None
                            } else {
                                Some(Qwen35Gdn {
                                    wqkv: MatW::load(&file, &gguf, &t("attn_qkv.weight"))?,
                                    wz: MatW::load(&file, &gguf, &t("attn_gate.weight"))?,
                                    conv: upload(&file, &gguf, &t("ssm_conv1d.weight"))?,
                                    alpha_w: upload_as_f32(&file, &gguf, &t("ssm_alpha.weight"))?,
                                    beta_w: upload_as_f32(&file, &gguf, &t("ssm_beta.weight"))?,
                                    a: upload(&file, &gguf, &t("ssm_a"))?,
                                    dt_bias: upload(&file, &gguf, &t("ssm_dt.bias"))?,
                                    ssm_norm: upload(&file, &gguf, &t("ssm_norm.weight"))?,
                                    ssm_out: MatW::load(&file, &gguf, &t("ssm_out.weight"))?,
                                })
                            },
                            // dense qwen35 has no shared expert (and so
                            // no shexp gate); the ffn half never reads it
                            shexp_gate: if gguf.tensor(&t("ffn_gate_inp_shexp.weight")).is_some() {
                                upload(&file, &gguf, &t("ffn_gate_inp_shexp.weight"))?
                            } else {
                                DeviceBuf::alloc(4)?
                            },
                        }))
                    }
                };
                let attn_output = if dsv4_arch {
                    // V4's second-stage output projection
                    upload_attn(&file, &gguf, &t("attn_output_b.weight"), &mut *attn_vram_budget)?
                } else if shape.family == Family::Qwen35 {
                    // GDN layers project through ssm_out; attn layers
                    // through Qwen35Attn.out (MatW)
                    DeviceBuf::alloc(1)?
                } else {
                    upload_attn(&file, &gguf, &t("attn_output.weight"), &mut *attn_vram_budget)?
                };
                if attn_dev.is_some() {
                    kernels::set_device(primary)?;
                }
                let gemma = if gemma_arch {
                    // router input weight = gate_inp_s / sqrt(n_embd): the
                    // reference runs weightless rms, scales by 1/sqrt, then
                    // muls gate_inp_s - algebraically one weighted rms_norm
                    let raw = read_tensor_f32(&file, &gguf, &t("ffn_gate_inp.scale"))?;
                    let scaled: Vec<f32> = raw
                        .iter()
                        .map(|v| v / (shape.n_embd as f32).sqrt())
                        .collect();
                    let mut router_norm = DeviceBuf::alloc(scaled.len() * 4)?;
                    router_norm.write(0, kernels::as_bytes(&scaled))?;
                    let out_scale = read_tensor_f32(&file, &gguf, &t("layer_output_scale.weight"))
                        .map(|v| v[0])
                        .unwrap_or(1.0);
                    Some(GemmaW {
                        attn_post_norm: upload(&file, &gguf, &t("post_attention_norm.weight"))?,
                        router_norm,
                        pre_ffw_norm_2: upload(&file, &gguf, &t("pre_ffw_norm_2.weight"))?,
                        post_ffw_norm_1: upload(&file, &gguf, &t("post_ffw_norm_1.weight"))?,
                        post_ffw_norm_2: upload(&file, &gguf, &t("post_ffw_norm_2.weight"))?,
                        post_ffw_norm: upload(&file, &gguf, &t("post_ffw_norm.weight"))?,
                        out_scale,
                    })
                } else {
                    None
                };
                let ink = if ink_arch {
                    let gm = geom[il as usize];
                    // rel_proj gguf ne = [rel_extent, d_rel] (extent
                    // fastest): transpose to [extent][d_rel] rows so
                    // matmul_f32 contracts over d_rel
                    let raw = read_tensor_f32(&file, &gguf, &t("attn_rel_proj.weight"))?;
                    let ext = if gm.window != 0 { shape.rel_ext_swa } else { shape.rel_ext } as usize;
                    let dr = shape.d_rel as usize;
                    if raw.len() != ext * dr {
                        return Err(format!(
                            "blk.{il}.attn_rel_proj: {} elems, expected {ext}x{dr}",
                            raw.len()
                        )
                        .into());
                    }
                    let mut tr = vec![0f32; raw.len()];
                    for d in 0..dr {
                        for e in 0..ext {
                            tr[e * dr + d] = raw[d * ext + e];
                        }
                    }
                    let upload_f32 = |name: &str| -> Result<DeviceBuf> {
                        let v = read_tensor_f32(&file, &gguf, name)?;
                        let mut b = DeviceBuf::alloc(v.len() * 4)?;
                        b.write(0, kernels::as_bytes(&v))?;
                        Ok(b)
                    };
                    // attn-side weights (wr, rel_proj, k/v shortconvs) live
                    // where the attention segment computes; the attn/mlp
                    // stream shortconvs run on the primary after the hop
                    if let Some(d) = attn_dev {
                        kernels::set_device(d)?;
                    }
                    let wr = upload_attn(&file, &gguf, &t("attn_r.weight"), &mut *attn_vram_budget)?;
                    let mut rel_proj = DeviceBuf::alloc(tr.len() * 4)?;
                    rel_proj.write(0, kernels::as_bytes(&tr))?;
                    let sconv_k = upload_f32(&t("shortconv_k.weight"))?;
                    let sconv_v = upload_f32(&t("shortconv_v.weight"))?;
                    if attn_dev.is_some() {
                        kernels::set_device(primary)?;
                    }
                    Some(InkW {
                        wr,
                        rel_proj,
                        rel_extent: ext as u32,
                        sconv_k,
                        sconv_v,
                        sconv_attn: upload_f32(&t("shortconv_attn.weight"))?,
                        sconv_mlp: upload_f32(&t("shortconv_mlp.weight"))?,
                        gscale: read_tensor_f32(&file, &gguf, &t("ffn_gscale.weight"))?[0],
                    })
                } else {
                    None
                };
                Ok(LayerW {
                    attn_norm: upload(&file, &gguf, &t("attn_norm.weight"))?,
                    attn,
                    attn_output,
                    // qwen35 calls the pre-FFN norm post_attention_norm
                    ffn_norm: if gguf.tensor(&t("ffn_norm.weight")).is_some() {
                        upload(&file, &gguf, &t("ffn_norm.weight"))?
                    } else {
                        upload(&file, &gguf, &t("post_attention_norm.weight"))?
                    },
                    ffn,
                    gemma,
                    ink,
                })
            };

            let mut layers = Vec::with_capacity(shape.n_exec_layer as usize);
            for il in 0..shape.n_exec_layer {
                // dense split: the whole layer uploads to its owner
                kernels::set_device(layer_dev[il as usize])?;
                layers.push(load_layer(il, &mut attn_vram_budget, &mut no_budget)?);
            }
            kernels::set_device(primary)?;

            // MTP/nextn layer (PULSAR_MTP=1 opt-in): one extra transformer
            // block fed by eh_proj([enorm(embed(token)); hnorm(hidden)]),
            // sharing the base output head through shared_head_norm.
            let il = shape.n_exec_layer;
            let nextn = |suffix: &str| format!("blk.{il}.nextn.{suffix}.weight");
            let mtp = if std::env::var("PULSAR_MTP").ok().as_deref() == Some("1") {
                if gguf.tensor(&nextn("eh_proj")).is_none() {
                    eprintln!("pulsar: PULSAR_MTP=1 but the gguf has no nextn block - ignoring");
                    None
                } else {
                    let layer = load_layer(il, &mut attn_vram_budget, &mut no_budget)?;
                    let mut res_pool = DeviceBuf::alloc(1)?;
                    let mut res_map = std::collections::HashMap::new();
                    if let Ffn::Moe { gate_exps, up_exps, down_exps, .. } = &layer.ffn {
                        let total: usize = [gate_exps, up_exps, down_exps]
                            .iter()
                            .map(|t| t.expert_bytes as usize * shape.n_expert as usize)
                            .sum();
                        match DeviceBuf::alloc(total + SLAB_SLACK) {
                            Ok(mut pool) => {
                                let mut cursor = 0usize;
                                let mut slab = Vec::new();
                                for t in [gate_exps, up_exps, down_exps] {
                                    for e in 0..shape.n_expert as u64 {
                                        let off = t.abs_offset + e * t.expert_bytes;
                                        slab.resize(t.expert_bytes as usize, 0);
                                        file.read_exact_at(&mut slab, off)?;
                                        pool.write(cursor, &slab)?;
                                        res_map.insert(off, cursor);
                                        cursor += t.expert_bytes as usize;
                                    }
                                }
                                eprintln!(
                                    "pulsar: MTP draft experts resident ({:.1}GB, all {} triples)",
                                    total as f64 / 1e9,
                                    shape.n_expert
                                );
                                res_pool = pool;
                            }
                            Err(_) => eprintln!(
                                "pulsar: MTP expert residency didn't fit ({:.1}GB needed) - drafts will stream",
                                total as f64 / 1e9
                            ),
                        }
                    }
                    let m = MtpLayer {
                        layer,
                        eh_proj: upload(&file, &gguf, &nextn("eh_proj"))?,
                        enorm: upload(&file, &gguf, &nextn("enorm"))?,
                        hnorm: upload(&file, &gguf, &nextn("hnorm"))?,
                        head_norm: upload(&file, &gguf, &nextn("shared_head_norm"))?,
                        res_pool,
                        res_map,
                    };
                    eprintln!("pulsar: MTP draft layer loaded (speculative decode)");
                    Some(m)
                }
            } else {
                None
            };
            // depth default 1: the shipped nextn block is trained to
            // predict ONE step from a true hidden; self-fed chaining is
            // out-of-distribution and acceptance collapses with depth
            // (Hy3 measured 30% -> 23% -> 10% at depths 1/3/5)
            let mtp_depth = if mtp.is_some() {
                std::env::var("PULSAR_MTP_DEPTH")
                    .ok()
                    .and_then(|v| v.parse::<u32>().ok())
                    .unwrap_or(1)
                    .clamp(1, 8)
            } else {
                0
            };

            let logit_softcap = if gemma_arch {
                gguf.arch_meta("final_logit_softcapping")
                    .and_then(Value::as_f32)
                    .unwrap_or(30.0)
            } else {
                0.0
            };
            let tok_norm = if ink_arch {
                Some(upload(&file, &gguf, "token_embd_norm.weight")?)
            } else {
                None
            };
            let logit_scale = if ink_arch {
                let denom = gguf
                    .arch_meta("logit_scale_denom")
                    .and_then(Value::as_f32)
                    .ok_or_else(|| meta_err("logit_scale_denom"))?;
                1.0 / denom
            } else {
                1.0
            };
            let n_vocab_out = if ink_arch {
                gguf.arch_meta("unpadded_vocab_size")
                    .and_then(Value::as_u64)
                    .map(|v| v as u32)
                    .unwrap_or(shape.n_vocab)
            } else {
                shape.n_vocab
            };
            let (ones_hc, dsv4_out) = if dsv4_arch {
                let ones = vec![1.0f32; (shape.n_hc * shape.n_embd) as usize];
                (
                    Some(DeviceBuf::from_f32(&ones)?),
                    Some(Dsv4OutW {
                        fn_w: upload_f16_as_f32(&file, &gguf, "output_hc_fn.weight")?,
                        scale: read_tensor_f32(&file, &gguf, "output_hc_scale.weight")?[0],
                        base: read_tensor_f32(&file, &gguf, "output_hc_base.weight")?,
                    }),
                )
            } else {
                (None, None)
            };
            Ok(Model {
                path: path.to_path_buf(),
                shards,
                shape,
                gguf,
                token_embd,
                output_norm,
                output,
                layers,
                attn_dev,
                layer_dev,
                mtp,
                mtp_depth,
                output_kq,
                geom,
                rope_factors,
                embd_scale: if gemma_arch { (shape.n_embd as f32).sqrt() } else { 1.0 },
                logit_softcap,
                tok_norm,
                logit_scale,
                n_vocab_out,
                compress_ratios,
                ones_hc,
                dsv4_out,
            })
        }
    }

    /// Fill leftover GPUs (not primary, not the attn card) with the
    /// hottest expert triples from the warm census. First run has no
    /// census, so tiers activate from the second run on.
    fn build_tiers(m: &Model, mb: u32, primary: i32) -> Result<Vec<ExpertTier>> {
        let s = m.shape;
        if std::env::var("PULSAR_TIERS").ok().as_deref() == Some("off") {
            return Ok(Vec::new());
        }

        // dedicated cards first; the attn card joins LAST with whatever
        // VRAM the resident attn stack left over (the free-space check
        // below decides if that's worth a tier)
        let mut candidates: Vec<i32> = (0..kernels::device_count())
            .filter(|&d| d != primary && Some(d) != m.attn_dev)
            .collect();
        if let Some(ad) = m.attn_dev {
            if ad != primary {
                candidates.push(ad);
            }
        }
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let census: std::collections::HashMap<u64, u64> =
            read_census(&m.path).into_iter().map(|(off, _, count)| (off, count)).collect();
        if census.is_empty() {
            eprintln!("pulsar: no warm census yet - expert tiers idle until the next run");
            return Ok(Vec::new());
        }
        // rank whole triples by summed slab heat. Inkling's sink bank
        // ranks BELOW every routed triple despite its every-token heat:
        // the tier's marginal value is avoided DISK misses, and sinks
        // never disk-miss (the host LFU always keeps what every token
        // touches) - measured: sinks evicting routed triples cost 3%,
        // sinks filling spare tier capacity are free wins.
        let mut triples: Vec<(u64, [ (u64, u64); 3 ])> = Vec::new();
        let mut sink_triples: Vec<(u64, [ (u64, u64); 3 ])> = Vec::new();
        for l in &m.layers {
            let Ffn::Moe { gate_exps, up_exps, down_exps, sink, .. } = &l.ffn else {
                continue;
            };
            for e in 0..s.n_expert as u64 {
                let slabs = [gate_exps, up_exps, down_exps]
                    .map(|t| (t.abs_offset + e * t.expert_bytes, t.expert_bytes));
                let heat: u64 = slabs.iter().filter_map(|(off, _)| census.get(off)).sum();
                if heat > 0 {
                    triples.push((heat, slabs));
                }
            }
            if let Some(sk) = sink {
                for e in 0..s.n_shexp_sink as u64 {
                    let slabs = [&sk[0], &sk[1], &sk[2]]
                        .map(|t| (t.abs_offset + e * t.expert_bytes, t.expert_bytes));
                    let heat: u64 = slabs.iter().filter_map(|(off, _)| census.get(off)).sum();
                    if heat > 0 {
                        sink_triples.push((heat, slabs));
                    }
                }
            }
        }
        triples.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        sink_triples.sort_unstable_by(|a, b| b.0.cmp(&a.0));
        triples.extend(sink_triples);
        if triples.is_empty() {
            // fully-resident model (DenseKq): a tier would just grab the
            // free VRAM its own layers need
            return Ok(Vec::new());
        }

        let file = VFile::open(&m.shards)?;
        let mut tiers = Vec::new();
        let mut next = triples.into_iter();
        for d in candidates {
            let Ok((free, _)) = kernels::mem_info(d) else { continue };
            let reserve: usize = 1 << 30; // scratch + CUDA context
            if free <= reserve + (1 << 30) {
                continue; // not worth a tier
            }
            let t0 = std::time::Instant::now();
            kernels::set_device(d)?;
            let n_used = s.n_expert_used as usize;
            let mut tier = ExpertTier {
                dev: d,
                pool: DeviceBuf::alloc(free - reserve)?,
                map: std::collections::HashMap::new(),
                xin: DeviceBuf::alloc(mb as usize * s.n_embd as usize * 4)?,
                xq: DeviceBuf::alloc(
                    mb as usize * s.n_embd as usize / kernels::Q8_K_BLOCK_ELEMS
                        * kernels::Q8_K_BLOCK_BYTES,
                )?,
                mid: DeviceBuf::alloc(mb as usize * n_used * s.n_ff_exp as usize * 4)?,
                midq: DeviceBuf::alloc(
                    mb as usize * n_used * s.n_ff_exp as usize / kernels::Q8_K_BLOCK_ELEMS
                        * kernels::Q8_K_BLOCK_BYTES,
                )?,
                out: DeviceBuf::alloc(mb as usize * s.n_embd as usize * 4)?,
                ptrs: DeviceBuf::alloc(mb as usize * n_used * std::mem::size_of::<ExpertPtrs>())?,
                weights: DeviceBuf::alloc(mb as usize * n_used * 4)?,
                ptrs_sink: if s.n_shexp_sink > 0 {
                    DeviceBuf::alloc(mb as usize * n_used * std::mem::size_of::<ExpertPtrs>())?
                } else {
                    DeviceBuf::alloc(1)?
                },
                out_sink: if s.n_shexp_sink > 0 {
                    DeviceBuf::alloc(mb as usize * s.n_embd as usize * 4)?
                } else {
                    DeviceBuf::alloc(1)?
                },
                grp_ptrs: DeviceBuf::alloc(s.n_expert.max(1) as usize * std::mem::size_of::<ExpertPtrs>())?,
                grp_starts: DeviceBuf::alloc((s.n_expert as usize + 1) * 4)?,
                grp_pairs: DeviceBuf::alloc(mb as usize * n_used * 4)?,
                grp_partial: DeviceBuf::alloc(
                    // hybrid verify chunks cap at 16 tokens
                    16 * n_used * s.n_embd as usize * 4,
                )?,
                hits: 0,
            };
            let mut cursor = 0usize;
            let mut slab_buf = Vec::new();
            for (_, slabs) in next.by_ref() {
                let need: usize = slabs.iter().map(|&(_, len)| len as usize).sum();
                if cursor + need + SLAB_SLACK > tier.pool.bytes() {
                    break;
                }
                for (off, len) in slabs {
                    slab_buf.resize(len as usize, 0);
                    file.read_exact_at(&mut slab_buf, off)?;
                    tier.pool.write(cursor, &slab_buf)?;
                    tier.map.insert(off, tier.pool.ptr_at(cursor));
                    cursor += len as usize;
                }
            }
            kernels::set_device(primary)?;
            eprintln!(
                "pulsar: expert tier on CUDA device {d}: {} triples ({:.1}GB) resident in {:.1}s",
                tier.map.len() / 3,
                cursor as f64 / 1e9,
                t0.elapsed().as_secs_f32()
            );
            tiers.push(tier);
        }
        Ok(tiers)
    }

    /// Per-decode device state: activation buffers, KV caches, the routed
    /// expert staging arena, and reusable host staging.
    pub struct State {
        ctx: u32,
        max_batch: u32,
        tok: DeviceBuf,
        last_row: DeviceBuf,
        cur: DeviceBuf,
        normed: DeviceBuf,
        q: DeviceBuf,
        k: DeviceBuf,
        v: DeviceBuf,
        heads: DeviceBuf,
        attn_out: DeviceBuf,
        after_attn: DeviceBuf,
        gate_act: DeviceBuf,
        up_act: DeviceBuf,
        ffn_mid: DeviceBuf,
        ffn_out: DeviceBuf,
        shared_out: DeviceBuf,
        router_logits: DeviceBuf,
        router_selected: DeviceBuf,
        router_weights: DeviceBuf,
        moe_mid: DeviceBuf,
        moe_out: DeviceBuf,
        xq: DeviceBuf,
        midq: DeviceBuf,
        pub dev_cache: DeviceSlabCache,
        /// census count each touch entry was seeded with by load_warm, so
        /// save_warm can merge on this-run deltas (seeded counts otherwise
        /// ratchet: seed + delta > seed every run, a running sum in disguise)
        warm_seeds: std::collections::HashMap<u64, u64>,
        /// Primary staging arena for expert H2D (parity 0).
        staging: DeviceBuf,
        /// Alternate staging arena for cross-layer async H2D prefetch (parity 1).
        staging_alt: DeviceBuf,
        /// Side stream for expert H2D (overlaps disk / can pipeline with kernels).
        expert_h2d: kernels::CopyStream,
        /// Pending async H2D into staging_alt for the predicted next MoE layer.
        h2d_prefetch: Option<ExpertH2dPrefetch>,
        /// Disable async expert H2D (PULSAR_NO_ASYNC_H2D=1) — blocking path.
        async_expert_h2d: bool,
        expert_ptrs: DeviceBuf,
        kcache: Vec<DeviceBuf>,
        vcache: Vec<DeviceBuf>,
        /// Gqa KV storage: false = f32 (exact), true = fp8 e4m3 + per-row
        /// scale (PULSAR_KV=fp8, lossy, opt-in)
        kv_fp8: bool,
        logits: DeviceBuf,
        pub store: StreamingStore,
        prefetcher: Prefetcher,
        pred_logits: DeviceBuf,
        pred_selected: DeviceBuf,
        pred_weights: DeviceBuf,
        /// Cumulative per-stage wall time (PULSAR_PROFILE=1 to print).
        pub prof: Prof,
        stages: Option<[AttnStage; 2]>,
        // MLA scratch (dummies for Gqa); on the attn GPU when one is set
        q_rank: DeviceBuf,
        q_rank_norm: DeviceBuf,
        kv_raw: DeviceBuf,
        kv_norm: DeviceBuf,
        qk_low: DeviceBuf,
        mla_selected: DeviceBuf,
        // DSA indexer scratch (1-float dummies when absent): per-indexer-
        // layer K caches + q/weights/scores; selection count persists
        // across layers so non-indexer layers reuse the last list
        idx_kcache: Vec<DeviceBuf>,
        idx_kraw: DeviceBuf,
        /// f16 staging for the tensor-core batch scorer
        idx_q16: DeviceBuf,
        idx_q: DeviceBuf,
        idx_w: DeviceBuf,
        idx_scores: DeviceBuf,
        idx_last_sel: u32,
        // attn-GPU hop buffers (1-float dummies otherwise): normed input
        // copied primary->attn GPU, attn output copied back
        normed_a: DeviceBuf,
        attn_out_a: DeviceBuf,
        // resident expert tiers on leftover GPUs + the primary-side
        // buffer their partial outputs are gathered into
        pub tiers: Vec<ExpertTier>,
        tier_ret: DeviceBuf,
        /// CPU expert lane (PULSAR_CPU=1): worker pool + partial-return buf
        pub cpu_pool: Option<cpu_tier::Pool>,
        cpu_ret: DeviceBuf,
        pub cpu_hits: u64,
        // grouped batch-MoE scratch (grow-only; prefill chunks only)
        grp_ptrs: DeviceBuf,
        grp_starts: DeviceBuf,
        grp_pairs: DeviceBuf,
        grp_partial: DeviceBuf,
        // MTP scratch (1-float dummies without PULSAR_MTP=1): the draft
        // block's input pipeline + the last real token's hidden state
        mtp_e_raw: DeviceBuf,
        mtp_e: DeviceBuf,
        mtp_h: DeviceBuf,
        mtp_x: DeviceBuf,
        mtp_hidden: DeviceBuf,
        /// true-hidden anchor saved across a draft chain (the chain
        /// self-feeds approximate hiddens into mtp_hidden; the batched
        /// fill pass afterwards needs the pre-chain value back)
        mtp_hidden_save: DeviceBuf,
        pub mtp_drafted: u64,
        pub mtp_accepted: u64,
        /// q8_K activation scratch for a K-quant lm-head (1 f32 otherwise)
        head_xq: DeviceBuf,
        // Inkling scratch (empty/dummies elsewhere): per-layer packed
        // shortconv states [k | v | attn | mlp], the r projection, the
        // rel-bias logits, and sconv bounce buffers. The k/v streams
        // (states + tmp) and r/rel buffers live on the attn card under
        // Gqa offload; attn/mlp streams stay on the primary.
        sconv_state: Vec<[DeviceBuf; 4]>,
        sconv_tmp: DeviceBuf,
        sconv_tmp_kv: DeviceBuf,
        r_buf: DeviceBuf,
        rel_buf: DeviceBuf,
        /// Unified-memory box (GB10/Spark, Jetson): host-cache slabs are
        /// device-speed, so expert resolve hands their pinned pointers to
        /// the kernels directly - no VRAM cache, no staging copies. Safe
        /// because each layer's resolve runs after a full device sync, so
        /// an evicted slab can never have in-flight readers.
        unified: bool,
        /// deepseek4 runtime (HC streams, compressor state); None elsewhere
        dsv4: Option<dsv4::Dsv4Rt>,
        /// qwen35 runtime (GDN conv+delta states); None elsewhere
        qwen35: Option<qwen35::Qwen35Rt>,
    }

    /// Cross-layer expert H2D prefetch: slabs already copied into `staging_alt`
    /// (or primary staging when parity flips) for the predicted next MoE layer.
    struct ExpertH2dPrefetch {
        /// layer index the prefetch was built for
        layer: usize,
        /// offset -> device pointer inside the alt staging buffer
        map: std::collections::HashMap<u64, *const std::ffi::c_void>,
        /// true once `expert_h2d.record` was issued for this batch
        recorded: bool,
    }

    impl State {
        pub fn new(m: &Model, ctx: u32) -> Result<State> {
            // Mla keeps ~12GB of pinned attn weights in RAM; leave the
            // host expert cache smaller so the two fit in 30GB together.
            // With an attn GPU that RAM is free again - spend it on
            // experts, but derive the ceiling from MEASURED free RAM: a
            // fixed 22GB default memory-pressure-hung a 30GB box (twice,
            // power button both times). Pinned cache memory can't swap,
            // so the reserve must cover everything else on the machine.
            let gb = std::env::var("PULSAR_CACHE_GB")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or_else(|| {
                    let cap = if m.attn_dev.is_some() { 22 } else { 12 };
                    let avail_gb = std::fs::read_to_string("/proc/meminfo")
                        .ok()
                        .and_then(|s| {
                            s.lines().find(|l| l.starts_with("MemAvailable:"))?
                                .split_whitespace()
                                .nth(1)?
                                .parse::<usize>()
                                .ok()
                        })
                        .map(|kb| kb >> 20)
                        .unwrap_or(cap + 6);
                    cap.min(avail_gb.saturating_sub(6)).max(4)
                });
            Self::with_cache(m, ctx, gb << 30)
        }

        pub fn max_batch(&self) -> u32 {
            self.max_batch
        }

        pub fn ctx(&self) -> u32 {
            self.ctx
        }

        /// Persist the slab popularity census so the next run starts warm.
        pub fn save_warm(&self, m: &Model) -> Result {
            // Merge popularity across runs instead of overwriting. A save
            // REPLACES the file, so one short run (thin touch set) would
            // clobber a rich census and starve the next run's tier
            // placement - measured: a poisoned census halved Hy3's resident
            // tier hits, doubled h2d, and cut decode 8.2 -> 5.8 tok/s. Take
            // the per-slab max of PER-RUN heat: subtract the load_warm seed
            // first, because seeded counts increment from the old census
            // value, and max(old, seed + delta) is a running sum in
            // disguise. Cached slabs would ratchet cumulatively while
            // tier-resident slabs (never seeded) stayed per-run, so tier
            // ranking would drift toward whatever sat in the cache longest.
            // A thin run still can't lower a hot slab, and counts stay at
            // per-run scale (a running sum would ossify the cache).
            // ponytail: rm the .warm to reset a drifted hot set.
            let mut merged: std::collections::HashMap<u64, (u64, u64)> =
                read_census(&m.path)
                    .into_iter()
                    .map(|(off, len, count)| (off, (len, count)))
                    .collect();
            for (&off, &(count, len)) in self.dev_cache.touch.iter() {
                let seed = self.warm_seeds.get(&off).copied().unwrap_or(0);
                let this_run = count.saturating_sub(seed);
                let e = merged.entry(off).or_insert((len, 0));
                e.0 = len;
                e.1 = e.1.max(this_run);
            }
            let mut entries: Vec<(u64, u64, u64)> = merged
                .into_iter()
                .map(|(off, (len, count))| (count, off, len))
                .collect();
            entries.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            let mut bytes = Vec::with_capacity(entries.len() * 24);
            for (count, off, len) in &entries {
                bytes.extend_from_slice(&off.to_le_bytes());
                bytes.extend_from_slice(&len.to_le_bytes());
                bytes.extend_from_slice(&count.to_le_bytes());
            }
            std::fs::write(warm_path(&m.path), bytes)?;
            Ok(())
        }

        /// Load the popularity census: hottest **expert triples** into VRAM
        /// (gate+up+down colocated so a hit never leaves a sibling on disk),
        /// the next tier into the host cache, touch counts seeded for admission.
        fn load_warm(&mut self, m: &Model) -> Result<usize> {
            let Ok(bytes) = std::fs::read(warm_path(&m.path)) else {
                return Ok(0);
            };
            let mut heat: std::collections::HashMap<u64, (u64, u64)> =
                std::collections::HashMap::with_capacity(bytes.len() / 24);
            for c in bytes.chunks_exact(24) {
                let off = u64::from_le_bytes(c[0..8].try_into().unwrap());
                let len = u64::from_le_bytes(c[8..16].try_into().unwrap());
                let count = u64::from_le_bytes(c[16..24].try_into().unwrap());
                heat.insert(off, (count, len));
            }
            let in_tier =
                |off: u64| self.tiers.iter().any(|t| t.map.contains_key(&off));
            // Rank whole triples by summed slab heat. Fill VRAM with complete
            // triples only (slot count floored to a multiple of 3).
            let mut triples: Vec<(u64, [(u64, u64); 3])> = Vec::new();
            for l in &m.layers {
                let Ffn::Moe {
                    gate_exps,
                    up_exps,
                    down_exps,
                    sink,
                    ..
                } = &l.ffn
                else {
                    continue;
                };
                for e in 0..m.shape.n_expert as u64 {
                    let slabs = [gate_exps, up_exps, down_exps]
                        .map(|t| (t.abs_offset + e * t.expert_bytes, t.expert_bytes));
                    if slabs.iter().any(|(off, _)| in_tier(*off)) {
                        continue;
                    }
                    let h: u64 = slabs
                        .iter()
                        .map(|(off, _)| heat.get(off).map(|x| x.0).unwrap_or(0))
                        .sum();
                    if h > 0 {
                        triples.push((h, slabs));
                    }
                }
                if let Some(sk) = sink {
                    for e in 0..m.shape.n_shexp_sink as u64 {
                        let slabs = [&sk[0], &sk[1], &sk[2]]
                            .map(|t| (t.abs_offset + e * t.expert_bytes, t.expert_bytes));
                        if slabs.iter().any(|(off, _)| in_tier(*off)) {
                            continue;
                        }
                        let h: u64 = slabs
                            .iter()
                            .map(|(off, _)| heat.get(off).map(|x| x.0).unwrap_or(0))
                            .sum();
                        if h > 0 {
                            triples.push((h, slabs));
                        }
                    }
                }
            }
            triples.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            // seed touch for every census entry (including slabs not in a triple)
            for (&off, &(count, len)) in &heat {
                if in_tier(off) {
                    continue;
                }
                self.dev_cache.touch.insert(off, (count, len));
                self.warm_seeds.insert(off, count);
            }
            let dev_slots = self.dev_cache.meta.len();
            let n_dev_triples = dev_slots / 3;
            let mut dev_reads: Vec<stream::Read> = Vec::with_capacity(n_dev_triples * 3);
            let mut host_reads: Vec<stream::Read> = Vec::new();
            let host_budget = self.store.budget as u64;
            let mut host_bytes = 0u64;
            for (i, (_h, slabs)) in triples.iter().enumerate() {
                let reads = slabs.map(|(offset, len)| stream::Read { offset, len });
                if i < n_dev_triples {
                    dev_reads.extend_from_slice(&reads);
                } else {
                    let need: u64 = reads.iter().map(|r| r.len).sum();
                    if host_bytes + need > host_budget {
                        break;
                    }
                    host_bytes += need;
                    host_reads.extend_from_slice(&reads);
                }
            }
            // any remaining hot singleton census entries not covered by triples
            // still seed host if budget remains (fused odd tensors, etc.)
            let covered: std::collections::HashSet<u64> = dev_reads
                .iter()
                .chain(host_reads.iter())
                .map(|r| r.offset)
                .collect();
            let mut extras: Vec<(u64, u64, u64)> = heat
                .iter()
                .filter(|(&off, _)| !covered.contains(&off) && !in_tier(off))
                .map(|(&off, &(count, len))| (count, off, len))
                .collect();
            extras.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            for &(_c, offset, len) in &extras {
                if host_bytes + len > host_budget {
                    break;
                }
                host_bytes += len;
                host_reads.push(stream::Read { offset, len });
            }
            let n = dev_reads.len() + host_reads.len();
            // fetch VRAM triples one group at a time (avoid holding all
            // payloads in host RAM twice during warm load)
            for chunk in dev_reads.chunks_exact(3) {
                let mut pending: std::collections::HashMap<u64, Vec<u8>> =
                    std::collections::HashMap::with_capacity(3);
                self.store.fetch_direct(chunk, |off, payload| {
                    pending.insert(off, payload.to_vec());
                    Ok(())
                })?;
                let g = chunk[0].offset;
                let u = chunk[1].offset;
                let d = chunk[2].offset;
                let gp = pending.get(&g).map(|v| v.as_slice()).unwrap_or(&[]);
                let up = pending.get(&u).map(|v| v.as_slice()).unwrap_or(&[]);
                let dp = pending.get(&d).map(|v| v.as_slice()).unwrap_or(&[]);
                if gp.is_empty() || up.is_empty() || dp.is_empty() {
                    continue;
                }
                let _ = self.dev_cache.maybe_insert_triple(&[(g, gp), (u, up), (d, dp)], &[])?;
            }
            self.store.ensure_with(&host_reads, |_, _| Ok(()))?;
            self.store.reset_stats();
            self.dev_cache.hits = 0;
            self.dev_cache.misses = 0;
            Ok(n)
        }

        pub fn with_cache(m: &Model, ctx: u32, cache_bytes: usize) -> Result<State> {
            let s = m.shape;
            let f32s = |n: u32| DeviceBuf::alloc(n as usize * 4);
            let n_used = s.n_expert_used as usize;
            // uniform slab size across gate/up/down on this model; assert at fetch
            // include the MTP layer: its experts can use a DIFFERENT quant
            // (blk.80 is Q2_K on the Hy3 recipe, bigger slabs than IQ2_XXS)
            // - undersized slots make its slabs overflow into neighbors
            let max_slab = m
                .layers
                .iter()
                .chain(m.mtp.iter().map(|mt| &mt.layer))
                .filter_map(|l| match &l.ffn {
                    Ffn::Moe { gate_exps, up_exps, down_exps, sink, .. } => {
                        Some(
                            gate_exps
                                .expert_bytes
                                .max(up_exps.expert_bytes)
                                .max(down_exps.expert_bytes)
                                // sink bank slabs cache like any other
                                .max(sink.as_ref().map_or(0, |sk| {
                                    sk.iter().map(|t| t.expert_bytes).max().unwrap_or(0)
                                })),
                        )
                    }
                    _ => None,
                })
                .max()
                .unwrap_or(0) as usize;

            // Gqa: kcache/vcache are per-head K/V. Mla: kcache is the
            // compact latent cache (kv_lora wide), vcache the rope tail.
            // PULSAR_KV=fp8 stores Gqa rows as e4m3 + per-row f32 scale
            // (stride head_dim+4, ~3.9x smaller). Lossy, so opt-in: the
            // default f32 path keeps the bit-exact guarantees. MLA keeps
            // its compact latent cache as-is.
            let kv_fp8 = matches!(s.family, Family::Gqa)
                && std::env::var("PULSAR_KV").ok().as_deref() == Some("fp8");
            let kv_row = |hd: usize| if kv_fp8 { hd + 4 } else { hd * 4 };
            let (k_bytes, v_bytes) = match s.family {
                Family::Gqa => {
                    let b = s.n_head_kv as usize * ctx as usize * kv_row(s.head_dim as usize);
                    (b, b)
                }
                Family::Mla => (
                    ctx as usize * s.n_kv_lora as usize * 4,
                    ctx as usize * s.qk_rope as usize * 4,
                ),
                // raw SWA ring in kcache; the compressed-row cache rides
                // vcache, sized per layer in the loop below
                Family::Dsv4 => (s.n_swa as usize * s.head_dim as usize * 4, 4),
                Family::Qwen35 => {
                    let b = s.n_head_kv as usize * ctx as usize * kv_row(s.head_dim as usize);
                    (b, b)
                }
            };
            if kv_fp8 {
                let full = s.n_head_kv as usize * ctx as usize * s.head_dim as usize * 4;
                eprintln!(
                    "pulsar: fp8 KV cache on ({:.2} GB -> {:.2} GB over {} layers)",
                    (full * 2 * s.n_exec_layer as usize) as f64 / 1e9,
                    ((k_bytes + v_bytes) * s.n_exec_layer as usize) as f64 / 1e9,
                    s.n_exec_layer,
                );
            }
            // batch prefill: activations sized for max_batch tokens; the
            // logits/lm-head path stays single-row (last token only)
            // big default: each prefill chunk costs roughly one pass over
            // the expert corpus regardless of chunk size, so fewer chunks
            // win; activations at 512 cost only ~150MB
            let spec_rows = (m.mtp_depth + 1)
                .max(2)
                // qwen35 DFlash verify reads logits for a whole 16-row block
                .max(if s.family == Family::Qwen35 { 16 } else { 0 })
                .max(
                    std::env::var("PULSAR_NGRAM")
                        .ok()
                        .and_then(|v| v.parse::<u32>().ok())
                        .map(|d| d.clamp(1, 15) + 1)
                        .unwrap_or(0),
                );
            let mb = std::env::var("PULSAR_BATCH")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(256)
                .max(1);

            // everything the attn segment touches lives on the attn GPU
            // when one is set: KV, MLA scratch, q/heads, hop buffers
            let primary = kernels::get_device();
            if let Some(d) = m.attn_dev {
                kernels::set_device(d)?;
            }
            let mut kcache = Vec::new();
            let mut vcache = Vec::new();
            let n_kv_slots = s.n_exec_layer as usize + usize::from(m.mtp.is_some());
            let dense_split = m.layer_dev.iter().any(|&d| d != primary);
            for i in 0..n_kv_slots {
                if dense_split {
                    // dense split: KV lives with its layer (MTP slot ->
                    // primary, where the tail runs)
                    kernels::set_device(m.layer_dev.get(i).copied().unwrap_or(primary))?;
                }
                // per-layer geometry (gemma4): a SWA layer's cache is its
                // own kv width, not the Shape max
                let (kb, vb) = if s.family == Family::Qwen35 {
                    // only full-attention layers hold KV; the nextn/MTP
                    // draft slot is a full-attention layer too
                    if i == s.n_exec_layer as usize || (i as u32 + 1) % s.full_attn_interval == 0 {
                        (k_bytes, v_bytes)
                    } else {
                        (4, 4)
                    }
                } else if s.family == Family::Dsv4 {
                    let ratio = m.compress_ratios.get(i).copied().unwrap_or(0) as usize;
                    let comp = if ratio > 0 {
                        (ctx as usize / ratio + 2) * s.head_dim as usize * 4
                    } else {
                        4
                    };
                    (k_bytes, comp)
                } else {
                    match m.geom.get(i) {
                        Some(g) => {
                            let b = g.n_head_kv as usize * ctx as usize * kv_row(g.head_dim as usize);
                            (b, b)
                        }
                        None => (k_bytes, v_bytes),
                    }
                };
                let mut k = DeviceBuf::alloc(kb)?;
                let mut v = DeviceBuf::alloc(vb)?;
                if i == s.n_exec_layer as usize {
                    // MTP slot: position 0 is never written (no hidden
                    // before the first token) yet attention reads it -
                    // zero beats uninitialized VRAM
                    kernels::zero(&mut k, k_bytes)?;
                    kernels::zero(&mut v, v_bytes)?;
                }
                kcache.push(k);
                vcache.push(v);
            }
            if dense_split {
                kernels::set_device(primary)?;
            }
            let q = f32s(mb * s.n_head * s.head_dim.max(s.qk_dim()))?;
            let heads = f32s(mb * s.heads_dim().max(s.n_head * s.head_dim))?;
            let q_rank = f32s(mb * s.n_lora_q.max(1))?;
            let q_rank_norm = f32s(mb * s.n_lora_q.max(1))?;
            let kv_raw = f32s(mb * (s.n_kv_lora + s.qk_rope).max(1))?;
            let kv_norm = f32s(mb * s.n_kv_lora.max(1))?;
            let qk_low = f32s(mb * s.n_head * s.n_kv_lora.max(1))?;
            let mla_selected = DeviceBuf::alloc(mb as usize * ctx as usize * 4)?;
            // DSA indexer buffers live beside the attn stack (same device)
            let has_idx = s.n_idx_topk > 0 && s.family == Family::Mla;
            let mut idx_kcache = Vec::new();
            // n_kv_slots, not n_exec_layer: the MTP draft layer runs the
            // same MLA path as slot n_exec_layer and maintains its own
            // indexer keys
            for il in 0..n_kv_slots {
                idx_kcache.push(if has_idx && uses_full_indexer(il, s.n_leading_dense) {
                    DeviceBuf::alloc(ctx as usize * s.n_idx_dim as usize * 2)? // f16 keys
                } else {
                    f32s(1)?
                });
            }
            let idx_kraw = f32s(if has_idx { mb * s.n_idx_dim } else { 1 })?;
            let idx_q = f32s(if has_idx { mb * s.n_idx_head * s.n_idx_dim } else { 1 })?;
            let idx_q16 = DeviceBuf::alloc(if has_idx { (mb * s.n_idx_head * s.n_idx_dim) as usize * 2 } else { 1 })?;
            let idx_w = f32s(if has_idx { mb * s.n_idx_head } else { 1 })?;
            let idx_scores = f32s(if has_idx { mb * ctx } else { 1 })?;
            let (normed_a, attn_out_a) = if m.attn_dev.is_some() {
                (f32s(mb * s.n_embd)?, f32s(mb * s.n_embd)?)
            } else {
                (f32s(1)?, f32s(1)?)
            };
            // Gqa attention scratch beside the KV caches (attn card under
            // offload, primary otherwise): raw k/v projections, inkling's
            // rel-bias buffers and the k/v-stream shortconv state+tmp
            let kbuf = f32s(mb * s.n_head_kv * s.head_dim)?;
            let vbuf = f32s(mb * s.n_head_kv * s.head_dim)?;
            let r_buf = f32s(if s.d_rel > 0 { mb * s.n_head * s.d_rel } else { 1 })?;
            let rel_buf = f32s(if s.d_rel > 0 {
                mb * s.n_head * s.rel_ext.max(s.rel_ext_swa)
            } else {
                1
            })?;
            let sconv_tmp_kv = f32s(if s.sconv_k > 1 { mb * s.n_embd } else { 1 })?;
            let mut sconv_kv: Vec<(DeviceBuf, DeviceBuf)> = Vec::new();
            if s.sconv_k > 1 {
                let d = s.sconv_k - 1;
                for il in 0..s.n_exec_layer as usize {
                    let kvw = m
                        .geom
                        .get(il)
                        .map(|g| g.n_head_kv * g.head_dim)
                        .unwrap_or(s.n_head_kv * s.head_dim);
                    let mk = |w: u32| -> Result<DeviceBuf> {
                        let mut b = f32s(d * w)?;
                        let n = b.bytes();
                        kernels::zero(&mut b, n)?;
                        Ok(b)
                    };
                    sconv_kv.push((mk(kvw)?, mk(kvw)?));
                }
            }
            if m.attn_dev.is_some() {
                kernels::set_device(primary)?;
            }
            let tiers = build_tiers(m, mb, primary)?;
            let mut st = State {
                ctx,
                max_batch: mb,
                tok: DeviceBuf::alloc(mb as usize * 4)?,
                // spec verify reads depth+1 trailing rows (MTP or n-gram)
                last_row: f32s((spec_rows) * s.n_embd)?,
                cur: f32s(mb * s.n_embd)?,
                normed: f32s(mb * s.n_embd)?,
                q,
                k: kbuf,
                v: vbuf,
                heads,
                attn_out: f32s(mb * s.n_embd)?,
                after_attn: f32s(mb * s.n_embd)?,
                gate_act: f32s(mb * s.n_ff_dense.max(s.n_ff_exp))?,
                up_act: f32s(mb * s.n_ff_dense.max(s.n_ff_exp))?,
                ffn_mid: f32s(mb * s.n_ff_dense.max(s.n_ff_exp))?,
                ffn_out: f32s(mb * s.n_embd)?,
                shared_out: f32s(mb * s.n_embd)?,
                // +sink: the inkling gate matmul emits shared-expert
                // logits after the routed ones
                router_logits: f32s(mb * (s.n_expert + s.n_shexp_sink))?,
                router_selected: DeviceBuf::alloc(mb as usize * n_used * 4)?,
                router_weights: f32s(mb * s.n_expert_used)?,
                moe_mid: f32s(mb * s.n_expert_used * s.n_ff_exp)?,
                moe_out: f32s(mb * s.n_embd)?,
                xq: DeviceBuf::alloc(
                    mb as usize * s.n_embd as usize / kernels::Q8_K_BLOCK_ELEMS
                        * kernels::Q8_K_BLOCK_BYTES,
                )?,
                midq: DeviceBuf::alloc(
                    mb as usize * n_used * s.n_ff_exp as usize / kernels::Q8_K_BLOCK_ELEMS
                        * kernels::Q8_K_BLOCK_BYTES,
                )?,
                // placeholders: the capacity solver below sizes both from
                // MEASURED free VRAM once every fixed buffer has landed
                // (unified boxes keep the 1-byte cache: zero-copy resolve)
                dev_cache: DeviceSlabCache::new(1, max_slab)?,
                warm_seeds: std::collections::HashMap::new(),
                staging: DeviceBuf::alloc(1)?,
                staging_alt: DeviceBuf::alloc(1)?,
                expert_h2d: kernels::CopyStream::new()?,
                h2d_prefetch: None,
                async_expert_h2d: std::env::var_os("PULSAR_NO_ASYNC_H2D").is_none(),
                expert_ptrs: DeviceBuf::alloc(
                    mb as usize * n_used * std::mem::size_of::<ExpertPtrs>(),
                )?,
                kcache,
                vcache,
                kv_fp8,
                logits: f32s(spec_rows * s.n_vocab)?,
                store: StreamingStore::open(&m.shards, cache_bytes)?,
                prefetcher: Prefetcher::spawn(&m.shards)?,
                pred_logits: f32s(s.n_expert + s.n_shexp_sink)?,
                pred_selected: DeviceBuf::alloc(n_used * 4)?,
                pred_weights: f32s(s.n_expert_used)?,
                prof: Prof::default(),
                // ping-pong staging exists to hide PINNED attn reads; with
                // a dedicated attn GPU nothing is pinned, so no stages
                stages: match s.family {
                    Family::Mla if m.attn_dev.is_none() => Some([
                        AttnStage::new(&m.layers[0])?,
                        AttnStage::new(&m.layers[0])?,
                    ]),
                    _ => None,
                },
                q_rank,
                q_rank_norm,
                kv_raw,
                kv_norm,
                qk_low,
                mla_selected,
                idx_kcache,
                idx_kraw,
                idx_q16,
                idx_q,
                idx_w,
                idx_scores,
                idx_last_sel: 0,
                normed_a,
                attn_out_a,
                tier_ret: if tiers.is_empty() { f32s(1)? } else { f32s(mb * s.n_embd)? },
                cpu_pool: cpu_tier::Pool::from_env(),
                cpu_ret: f32s(1)?, // grows on first CPU-lane hit
                cpu_hits: 0,
                tiers,
                grp_ptrs: DeviceBuf::alloc(s.n_expert.max(1) as usize * std::mem::size_of::<ExpertPtrs>())?,
                grp_starts: DeviceBuf::alloc((s.n_expert as usize + 1) * 4)?,
                grp_pairs: DeviceBuf::alloc(mb as usize * n_used * 4)?,
                grp_partial: f32s(1)?, // grows on first grouped prefill
                mtp_e_raw: f32s(if m.mtp.is_some() { mb * s.n_embd } else { 1 })?,
                mtp_e: f32s(if m.mtp.is_some() { mb * s.n_embd } else { 1 })?,
                mtp_h: f32s(if m.mtp.is_some() { mb * s.n_embd } else { 1 })?,
                mtp_x: f32s(if m.mtp.is_some() { mb * 2 * s.n_embd } else { 1 })?,
                mtp_hidden: {
                    let mut b = f32s(if m.mtp.is_some() { s.n_embd } else { 1 })?;
                    // read before first write (position 0 has no prior
                    // hidden); zero beats uninitialized VRAM
                    let z = vec![0u8; b.bytes()];
                    b.write(0, &z)?;
                    b
                },
                mtp_hidden_save: f32s(if m.mtp.is_some() { s.n_embd } else { 1 })?,
                mtp_drafted: 0,
                mtp_accepted: 0,
                head_xq: if m.output_kq.is_some() {
                    DeviceBuf::alloc(
                        spec_rows as usize * s.n_embd as usize
                            / kernels::Q8_K_BLOCK_ELEMS
                            * kernels::Q8_K_BLOCK_BYTES,
                    )?
                } else {
                    f32s(1)?
                },
                // kv-stream states (attn card under offload) zip with the
                // attn/mlp-stream states (always primary: they run after
                // the hop back)
                sconv_state: if s.sconv_k > 1 {
                    let d = s.sconv_k - 1;
                    let mut v = Vec::with_capacity(s.n_exec_layer as usize);
                    for (kst, vst) in sconv_kv {
                        let mk = |w: u32| -> Result<DeviceBuf> {
                            let mut b = f32s(d * w)?;
                            let n = b.bytes();
                            kernels::zero(&mut b, n)?;
                            Ok(b)
                        };
                        v.push([kst, vst, mk(s.n_embd)?, mk(s.n_embd)?]);
                    }
                    v
                } else {
                    Vec::new()
                },
                sconv_tmp: f32s(if s.sconv_k > 1 { mb * s.n_embd } else { 1 })?,
                sconv_tmp_kv,
                r_buf,
                rel_buf,
                unified: {
                    let u = kernels::unified_memory();
                    if u {
                        eprintln!("pulsar: unified memory detected - zero-copy expert resolve");
                    }
                    u
                },
                dsv4: if s.family == Family::Dsv4 {
                    Some(dsv4::Dsv4Rt::new(m, ctx)?)
                } else {
                    None
                },
                qwen35: if s.family == Family::Qwen35 {
                    Some(qwen35::Qwen35Rt::new(m)?)
                } else {
                    None
                },
            };

            // ---- capacity solver: size the VRAM budget from MEASUREMENT.
            // Every fixed buffer has landed, so free VRAM on the primary IS
            // the pool; family-constant defaults OOM'd three models in one
            // week. Env knobs still win - the solver only fills what's
            // unset (PULSAR_DEV_CACHE_GB, PULSAR_BATCH).
            // max_slab == 0: no streamed experts anywhere (DenseKq
            // resident model) - skip the budget grab and the warm census
            if !st.unified && max_slab > 0 {
                let dev_env = std::env::var("PULSAR_DEV_CACHE_GB")
                    .ok()
                    .and_then(|v| v.parse::<usize>().ok())
                    .map(|g| g << 30);
                let batch_env = std::env::var("PULSAR_BATCH").is_ok();
                if let Ok((free, _)) = kernels::mem_info(primary) {
                    // CUDA context growth + allocator slack + kernel scratch
                    let reserve: usize = 768 << 20;
                    let pool = free.saturating_sub(reserve);
                    // prefill staging worst case for chunk c: every routed
                    // slot distinct until the expert count saturates, sink
                    // slabs always along (selected by every token). Fused
                    // gate_up shares one slab. Max over layers: quants vary.
                    let route_k = (s.n_expert_used - s.n_shexp_sink) as usize;
                    let stage_worst = |c: usize| -> usize {
                        let mut worst = 0usize;
                        for l in m.layers.iter().chain(m.mtp.iter().map(|mt| &mt.layer)) {
                            let Ffn::Moe { gate_exps, up_exps, down_exps, fused_up_off, sink, .. } = &l.ffn else {
                                continue;
                            };
                            let triple = gate_exps.expert_bytes as usize
                                + if *fused_up_off != 0 { 0 } else { up_exps.expert_bytes as usize }
                                + down_exps.expert_bytes as usize;
                            let distinct = (c * route_k.max(1)).min(s.n_expert as usize);
                            let mut b = distinct * triple;
                            if let Some(sk) = sink {
                                b += s.n_shexp_sink as usize
                                    * sk.iter().map(|t| t.expert_bytes as usize).sum::<usize>();
                            }
                            worst = worst.max(b);
                        }
                        worst
                    };
                    // chunk: biggest that keeps prefill staging within a
                    // third of the pool - decode wants the rest as cache
                    let chunk = if batch_env {
                        st.max_batch as usize
                    } else {
                        let share = pool / 3;
                        let mut c = st.max_batch as usize;
                        while c > 4 && stage_worst(c) > share {
                            c /= 2;
                        }
                        c.max(1)
                    };
                    // decode floor: one layer's slot resolve always fits.
                    // Only the primary staging arena is reserved from the
                    // budget; staging_alt (cross-layer async H2D) grows on
                    // demand so we do not steal ~2.5GB from the expert cache.
                    let staging_bytes = stage_worst(chunk).max(n_used * 3 * max_slab);
                    let dev_bytes = match dev_env {
                        Some(b) => b.max(1),
                        None => pool
                            .saturating_sub(staging_bytes)
                            .clamp(256 << 20, pool.max(256 << 20)),
                    };
                    st.dev_cache = DeviceSlabCache::new(dev_bytes, max_slab)?;
                    st.staging = DeviceBuf::alloc(staging_bytes + SLAB_SLACK)?;
                    // keep 1-byte placeholder; grown on first cross-layer prefetch
                    st.staging_alt = DeviceBuf::alloc(1)?;
                    st.max_batch = (chunk as u32).clamp(1, st.max_batch);
                    eprintln!(
                        "pulsar: auto budget: {:.1}GB VRAM free -> expert cache {:.1}GB, staging {:.1}GB, prefill chunk {}",
                        free as f64 / 1e9,
                        dev_bytes as f64 / 1e9,
                        staging_bytes as f64 / 1e9,
                        st.max_batch,
                    );
                }
            }

            let t0 = std::time::Instant::now();
            let warmed = if max_slab > 0 { st.load_warm(m)? } else { 0 };
            if warmed > 0 {
                eprintln!(
                    "pulsar: warm start: {warmed} slabs in {:.1}s",
                    t0.elapsed().as_secs_f32()
                );
            }
            Ok(st)
        }
    }

    impl Model {
        /// One full forward for one token at absolute position `pos`.
        /// Returns host logits when `want_logits`.
        pub fn forward_token(
            &self,
            st: &mut State,
            token: u32,
            pos: u32,
            want_logits: bool,
        ) -> Result<Option<Vec<f32>>> {
            self.forward_batch(st, &[token], pos, want_logits)
        }

        /// Forward `tokens` at absolute positions pos0..pos0+n. Union
        /// expert fetch per layer across the whole batch. Logits (when
        /// requested) are for the LAST token only.
        pub fn forward_batch(
            &self,
            st: &mut State,
            tokens: &[u32],
            pos0: u32,
            want_logits: bool,
        ) -> Result<Option<Vec<f32>>> {
            self.forward_rows(st, tokens, pos0, if want_logits { 1 } else { 0 })
        }

        /// Like forward_batch, but returns logits for the LAST `rows`
        /// positions (flattened rows x n_vocab); 0 rows = no logits.
        /// Speculative verification needs the draft row and its successor.
        pub fn forward_rows(
            &self,
            st: &mut State,
            tokens: &[u32],
            pos0: u32,
            rows: u32,
        ) -> Result<Option<Vec<f32>>> {
            let s = self.shape;
            if s.family == Family::Dsv4 {
                // V4 is a sequential state machine (SWA ring, streaming
                // compressor): prefill loops single-token forwards
                return self.forward_dsv4(st, tokens, pos0, rows);
            }
            if s.family == Family::Qwen35 {
                // GDN conv window + delta state are sequential too
                return self.forward_qwen35(st, tokens, pos0, rows);
            }
            // a batch must not straddle the indexer top_k boundary: rows
            // before it use causal range selection, rows after it need
            // scored top-k - split once here so every caller inherits it
            let topk = s.n_idx_topk;
            if topk > 0
                && pos0 < topk
                && pos0 + tokens.len() as u32 > topk
                && tokens.len() > 1
            {
                let split = (topk - pos0) as usize;
                self.forward_rows(st, &tokens[..split], pos0, 0)?;
                return self.forward_rows(st, &tokens[split..], topk, rows);
            }
            let n_tok = tokens.len() as u32;
            if n_tok == 0 || n_tok > st.max_batch {
                return Err(format!("batch {} outside 1..={}", n_tok, st.max_batch).into());
            }
            if pos0 + n_tok > st.ctx {
                return Err("position exceeds context".into());
            }
            let eps = s.rms_eps;
            let primary = kernels::get_device();
            let toks_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            st.tok.write(0, kernels::as_bytes(&toks_i32))?;
            kernels::embed_q8_0(&mut st.cur, &self.token_embd, &st.tok, s.n_embd, s.n_vocab, n_tok)?;
            if self.embd_scale != 1.0 {
                // gemma scales the residual stream by sqrt(n_embd)
                kernels::scale(&mut st.cur, n_tok * s.n_embd, self.embd_scale)?;
            }
            if let Some(tn) = &self.tok_norm {
                // inkling: rms-norm the embedding rows once, post-lookup
                kernels::rms_norm_inplace(&mut st.cur, tn, s.n_embd, n_tok, eps)?;
            }
            if pos0 == 0 {
                // fresh sequence: shortconv history restarts at zero
                // (k/v streams live on the attn card under offload)
                if let Some(d) = self.attn_dev {
                    kernels::set_device(d)?;
                }
                for states in st.sconv_state.iter_mut() {
                    for b in &mut states[..2] {
                        let n = b.bytes();
                        kernels::zero(b, n)?;
                    }
                }
                if self.attn_dev.is_some() {
                    kernels::set_device(primary)?;
                }
                for states in st.sconv_state.iter_mut() {
                    for b in &mut states[2..] {
                        let n = b.bytes();
                        kernels::zero(b, n)?;
                    }
                }
            }

            for (il, l) in self.layers.iter().enumerate() {
                // stage layer il+1's pinned attn tensors under this
                // layer's compute (decode only: prefill amortizes weights
                // over the whole batch already)
                if n_tok == 1 {
                    if let (Some(stages), Some(nl)) = (st.stages.as_mut(), self.layers.get(il + 1)) {
                        stages[(il + 1) % 2].kick(nl, il + 1)?;
                    }
                }
                self.eval_layer(st, il, l, n_tok, pos0, primary)?;
            }

            if rows == 0 {
                return Ok(None);
            }
            let k = rows.min(n_tok);
            let t_tail = std::time::Instant::now();
            let row = s.n_embd as usize * 4;
            kernels::copy_d2d(&mut st.last_row, 0, &st.cur, (n_tok - k) as usize * row, k as usize * row)?;
            kernels::rms_norm(&mut st.normed, &st.last_row, &self.output_norm, s.n_embd, k, eps)?;
            self.head_logits(st, k)?;
            kernels::sync()?;
            let out = st.logits.read_f32(k as usize * s.n_vocab as usize)?;
            st.prof.tail += t_tail.elapsed();
            Ok(Some(out))
        }

        /// Owner device of exec layer `il` (out of range - e.g. the MTP
        /// slot - falls back to layer 0's device, the primary).
        pub(crate) fn layer_dev(&self, il: usize) -> i32 {
            self.layer_dev.get(il).copied().unwrap_or(self.layer_dev[0])
        }

        /// True when the forward carries recurrent state beyond the KV
        /// cache (dsv4 compressor/HC lanes, qwen35 GDN, inkling
        /// shortconv). A prefix-cache may only APPEND to the forwarded
        /// stream for these; pure-KV families can rewind and overwrite.
        pub fn recurrent_state(&self) -> bool {
            matches!(self.shape.family, Family::Dsv4 | Family::Qwen35)
                || self.shape.sconv_k > 1
        }

        /// lm-head over the first `k` rows of st.normed into st.logits.
        fn head_logits(&self, st: &mut State, k: u32) -> Result {
            let s = self.shape;
            match self.output_kq {
                None => kernels::matmul_q8_0(&mut st.logits, &self.output, &st.normed, s.n_embd, s.n_vocab, k)?,
                Some((row_bytes, quant)) => {
                    kernels::quantize_q8_k(&mut st.head_xq, &st.normed, s.n_embd, k)?;
                    kernels::matmul_kq(&mut st.logits, &self.output, &st.head_xq, s.n_embd, s.n_vocab, k, row_bytes, quant)?;
                }
            }
            if self.logit_softcap > 0.0 {
                kernels::softcap(&mut st.logits, k * s.n_vocab, self.logit_softcap)?;
            }
            if self.logit_scale != 1.0 {
                // inkling muP head: logits / logit_scale_denom
                kernels::scale(&mut st.logits, k * s.n_vocab, self.logit_scale)?;
            }
            if self.n_vocab_out < s.n_vocab {
                // padded vocab rows hold garbage weights - poison them so
                // no sampler path can pick one
                kernels::fill_row_tail(&mut st.logits, k, s.n_vocab, self.n_vocab_out, f32::NEG_INFINITY)?;
            }
            Ok(())
        }

        /// One transformer layer over st.cur (residual stream in/out).
        /// `il` doubles as the KV-cache index; the MTP layer passes
        /// `self.layers.len()` (its own extra slot).
        fn eval_layer(
            &self,
            st: &mut State,
            il: usize,
            l: &LayerW,
            n_tok: u32,
            pos0: u32,
            primary: i32,
        ) -> Result {
            let s = self.shape;
            let eps = s.rms_eps;
            // per-layer attention geometry (gemma4 SWA/full interleave);
            // uniform families read straight from Shape
            let gm = self.geom.get(il).copied();
            let heads_dim = match (&l.attn, gm) {
                (Attn::Gqa { .. }, Some(g)) => s.n_head * g.head_dim,
                _ => s.heads_dim(),
            };
            {
                // attention
                kernels::rms_norm(&mut st.normed, &st.cur, &l.attn_norm, s.n_embd, n_tok, eps)?;
                let mut attn_output_w: &DeviceBuf = &l.attn_output;
                match &l.attn {
                    // dsv4/qwen35 have their own graphs
                    Attn::Dsv4(_) | Attn::Qwen35(_) => {
                        return Err("hybrid-family layer in the shared eval path".into())
                    }
                    Attn::Gqa { attn_q, attn_k, attn_v, q_norm, k_norm } => {
                        let (hkv, hd, theta, window) = match gm {
                            Some(g) => (g.n_head_kv, g.head_dim, g.theta, g.window),
                            None => (s.n_head_kv, s.head_dim, s.rope_freq_base, 0),
                        };
                        let rot = if gm.is_some() { hd } else { s.rot_dim };
                        let factors = gm
                            .filter(|g| g.factors)
                            .and_then(|_| self.rope_factors.as_ref());
                        // Gqa attn offload (opt-in): hop the normed input
                        // over and run the whole segment on the attn card,
                        // exactly like the Mla path below
                        if let Some(d) = self.attn_dev {
                            kernels::copy_across(&mut st.normed_a, &st.normed, (n_tok * s.n_embd) as usize * 4)?;
                            kernels::set_device(d)?;
                        }
                        let xin = if self.attn_dev.is_some() { &st.normed_a } else { &st.normed };
                        kernels::matmul_q8_0(&mut st.q, attn_q, xin, s.n_embd, s.n_head * hd, n_tok)?;
                        kernels::matmul_q8_0(&mut st.k, attn_k, xin, s.n_embd, hkv * hd, n_tok)?;
                        match attn_v {
                            Some(v_w) => kernels::matmul_q8_0(&mut st.v, v_w, xin, s.n_embd, hkv * hd, n_tok)?,
                            // attention_k_eq_v: v = the raw k projection
                            None => kernels::copy_across(&mut st.v, &st.k, (n_tok * hkv * hd) as usize * 4)?,
                        }
                        if let Some(ink) = &l.ink {
                            // inkling: k/v shortconvs on the flat
                            // projections, before head norm (reference
                            // order: matmul -> sconv -> reshape -> norm)
                            let kvb = (n_tok * hkv * hd) as usize * 4;
                            kernels::sconv(&mut st.sconv_tmp_kv, &st.k, &ink.sconv_k, &mut st.sconv_state[il][0], n_tok, hkv * hd, s.sconv_k)?;
                            kernels::copy_across(&mut st.k, &st.sconv_tmp_kv, kvb)?;
                            kernels::sconv(&mut st.sconv_tmp_kv, &st.v, &ink.sconv_v, &mut st.sconv_state[il][1], n_tok, hkv * hd, s.sconv_k)?;
                            kernels::copy_across(&mut st.v, &st.sconv_tmp_kv, kvb)?;
                        }
                        kernels::gqa_head_rms_norm(&mut st.q, Some(q_norm), n_tok * s.n_head, hd, eps)?;
                        kernels::gqa_head_rms_norm(&mut st.k, Some(k_norm), n_tok * hkv, hd, eps)?;
                        if gm.is_some() && l.ink.is_none() {
                            // gemma: v gets a weightless per-head rms norm
                            kernels::gqa_head_rms_norm(&mut st.v, None, n_tok * hkv, hd, eps)?;
                        }
                        if l.ink.is_none() {
                            // inkling has no rope: position rides the
                            // relative bias below (log-N tau is identity
                            // below 128k ctx, so it is skipped here)
                            kernels::gqa_rope(&mut st.q, n_tok, s.n_head, hd, rot, pos0, theta, factors)?;
                            kernels::gqa_rope(&mut st.k, n_tok, hkv, hd, rot, pos0, theta, factors)?;
                        }
                        let kvq = st.kv_fp8 as u32;
                        kernels::gqa_kv_append(&mut st.kcache[il], &st.k, n_tok, hkv, hd, st.ctx, pos0, kvq)?;
                        kernels::gqa_kv_append(&mut st.vcache[il], &st.v, n_tok, hkv, hd, st.ctx, pos0, kvq)?;
                        // gemma scores at scale 1.0 (q is per-head normed);
                        // inkling at muP 1/head_dim
                        let scale = if l.ink.is_some() {
                            1.0 / hd as f32
                        } else if gm.is_some() {
                            1.0
                        } else {
                            1.0 / (hd as f32).sqrt()
                        };
                        let rel_ext = if let Some(ink) = &l.ink {
                            // rel-pos bias: rel_proj^T . (x . wr), per
                            // (token, head) a rel_extent-long bias row
                            kernels::matmul_q8_0(&mut st.r_buf, &ink.wr, xin, s.n_embd, s.n_head * s.d_rel, n_tok)?;
                            kernels::matmul_f32(&mut st.rel_buf, &ink.rel_proj, &st.r_buf, s.d_rel, ink.rel_extent, n_tok * s.n_head)?;
                            ink.rel_extent
                        } else {
                            0
                        };
                        let rel = l.ink.as_ref().map(|_| &st.rel_buf);
                        kernels::gqa_attention_rel(&mut st.heads, &st.q, &st.kcache[il], &st.vcache[il], n_tok, s.n_head, hkv, hd, st.ctx, pos0, scale, window, rel, rel_ext, kvq)?;

                        // output projection on the attn card, hop back,
                        // restore the primary (mirrors the Mla path)
                        if self.attn_dev.is_some() {
                            kernels::matmul_q8_0(&mut st.attn_out_a, attn_output_w, &st.heads, s.n_head * hd, s.n_embd, n_tok)?;
                            kernels::copy_across(&mut st.attn_out, &st.attn_out_a, (n_tok * s.n_embd) as usize * 4)?;
                            kernels::set_device(primary)?;
                        }
                    }
                    Attn::Mla { q_a, q_a_norm, q_b, kv_a_mqa, kv_a_norm, k_b, v_b, indexer } => {
                        // ds4's GLM compact-KV decode path: q through the
                        // lora bottleneck, latent kv cached once for all
                        // heads, attention over all visible rows. Each
                        // pinned weight prefers its staged VRAM copy when
                        // the background copy already landed.
                        let stage = st
                            .stages
                            .as_ref()
                            .map(|sg| &sg[il % 2])
                            .filter(|sg| sg.ready_for(il));
                        let q_a_w = match stage { Some(sg) if q_a.is_pinned() => &sg.q_a, _ => q_a };
                        let q_b_w = match stage { Some(sg) if q_b.is_pinned() => &sg.q_b, _ => q_b };
                        let kv_a_w = match stage { Some(sg) if kv_a_mqa.is_pinned() => &sg.kv_a, _ => kv_a_mqa };
                        let k_b_w = match stage { Some(sg) if k_b.is_pinned() => &sg.k_b, _ => k_b };
                        let v_b_w = match stage { Some(sg) if v_b.is_pinned() => &sg.v_b, _ => v_b };
                        if let Some(sg) = stage {
                            if l.attn_output.is_pinned() {
                                attn_output_w = &sg.attn_output;
                            }
                        }

                        // attn-GPU offload: hop the normed input over,
                        // run the whole segment there. Blocking copies are
                        // legacy-stream ordered on the issuing device, so
                        // producer kernels have landed before they run.
                        if let Some(d) = self.attn_dev {
                            kernels::copy_across(&mut st.normed_a, &st.normed, (n_tok * s.n_embd) as usize * 4)?;
                            kernels::set_device(d)?;
                        }
                        let xin = if self.attn_dev.is_some() { &st.normed_a } else { &st.normed };

                        let rope = s.rope_cfg();
                        let kv_raw_dim = s.n_kv_lora + s.qk_rope;
                        kernels::matmul_q8_0(&mut st.q_rank, q_a_w, xin, s.n_embd, s.n_lora_q, n_tok)?;
                        kernels::rms_norm(&mut st.q_rank_norm, &st.q_rank, q_a_norm, s.n_lora_q, n_tok, eps)?;
                        kernels::matmul_q8_0(&mut st.q, q_b_w, &st.q_rank_norm, s.n_lora_q, s.n_head * s.qk_dim(), n_tok)?;
                        kernels::mla_rope_tail(&mut st.q, n_tok, s.n_head, s.qk_dim(), s.qk_rope, pos0, &rope)?;
                        kernels::matmul_q8_0(&mut st.kv_raw, kv_a_w, xin, s.n_embd, kv_raw_dim, n_tok)?;
                        kernels::mla_kv_lora_rms_norm(&mut st.kv_norm, &st.kv_raw, kv_a_norm, n_tok, kv_raw_dim, s.n_kv_lora, eps)?;
                        kernels::mla_store_compact_kv(&mut st.kcache[il], &mut st.vcache[il], &st.kv_norm, &st.kv_raw, pos0, n_tok, st.ctx, kv_raw_dim, s.n_kv_lora, s.qk_rope)?;
                        // DSA selection: within top_k every token sees the
                        // full range (bit-identical to the pre-indexer
                        // path). Beyond it, indexer layers score + top-k
                        // their own KV rows and the layers between reuse
                        // the last selection, exactly like ds4.
                        let visible = pos0 + n_tok;
                        let topk = s.n_idx_topk;
                        let is_idx_layer = uses_full_indexer(il, s.n_leading_dense);
                        if let (Some(idx), true) = (indexer, is_idx_layer) {
                            // maintain this layer's indexer K cache (xin =
                            // the attn-device copy of normed under offload)
                            kernels::matmul_q8_0(&mut st.idx_kraw, &idx.k, xin, s.n_embd, s.n_idx_dim, n_tok)?;
                            kernels::idx_store_k(&st.idx_kraw, &idx.k_norm, &idx.k_norm_b, &mut st.idx_kcache[il], pos0, n_tok, st.ctx, s.n_idx_dim, s.qk_rope, s.rms_eps, &s.rope_cfg(), 0.0, 1.0)?;
                        }
                        let n_sel = if topk == 0 || visible <= topk {
                            kernels::mla_fill_selected_range(&mut st.mla_selected, n_tok, pos0, visible, st.ctx)?;
                            st.idx_last_sel = visible;
                            visible
                        } else if is_idx_layer && indexer.is_some() {
                            let idx = indexer.as_ref().unwrap();
                            kernels::matmul_q8_0(&mut st.idx_q, &idx.q_b, &st.q_rank_norm, s.n_lora_q, s.n_idx_head * s.n_idx_dim, n_tok)?;
                            kernels::idx_rope0(&mut st.idx_q, n_tok, s.n_idx_head, s.n_idx_dim, s.qk_rope, pos0, &s.rope_cfg(), 0.0, 1.0)?;
                            // ds4 feeds proj the pre-norm residual (cur).
                            // Under attn offload cur is on the primary;
                            // borrow attn_out_a as the hop buffer - it is
                            // not written until the output projection.
                            if self.attn_dev.is_some() {
                                kernels::copy_across(&mut st.attn_out_a, &st.cur, (n_tok * s.n_embd) as usize * 4)?;
                                kernels::matmul_f32(&mut st.idx_w, &idx.proj, &st.attn_out_a, s.n_embd, s.n_idx_head, n_tok)?;
                            } else {
                                kernels::matmul_f32(&mut st.idx_w, &idx.proj, &st.cur, s.n_embd, s.n_idx_head, n_tok)?;
                            }
                            let scale = 1.0 / ((s.n_idx_dim * s.n_idx_head) as f32).sqrt();
                            if n_tok == 1 {
                                kernels::idx_score_one(&mut st.idx_scores, &st.idx_q, &st.idx_w, &st.idx_kcache[il], visible, s.n_idx_head, s.n_idx_dim, scale)?;
                                kernels::idx_topk(&mut st.mla_selected, &st.idx_scores, visible, topk)?;
                            } else {
                                // batch: every token in a post-boundary
                                // chunk has >= top_k visible rows (the
                                // forward_rows split guarantees it)
                                kernels::idx_scores_batch(&mut st.idx_scores, &st.idx_q, &st.idx_w, &st.idx_kcache[il], Some(&mut st.idx_q16), visible, n_tok, pos0, s.n_idx_head, s.n_idx_dim, scale)?;
                                kernels::idx_topk_batch(&mut st.mla_selected, &st.idx_scores, visible, n_tok, topk)?;
                            }
                            st.idx_last_sel = topk;
                            topk
                        } else {
                            // between indexer layers: reuse the last list
                            if st.idx_last_sel == 0 {
                                return Err("indexer selection missing (no indexer weights in gguf?)".into());
                            }
                            st.idx_last_sel
                        };
                        kernels::mla_qk_lowrank(&mut st.qk_low, &st.q, k_b_w, n_tok, s.n_head, s.n_kv_lora, s.qk_nope, s.qk_dim())?;
                        kernels::mla_attention(&mut st.heads, &st.q, &st.qk_low, &st.kcache[il], &st.vcache[il], v_b_w, &st.mla_selected, n_tok, n_sel, st.ctx, s.n_head, s.n_kv_lora, s.qk_nope, s.qk_rope, s.value_mla, &rope)?;

                        // output projection on the attn GPU, hop back,
                        // restore the primary for the ffn/expert half
                        if self.attn_dev.is_some() {
                            kernels::matmul_q8_0(&mut st.attn_out_a, attn_output_w, &st.heads, s.heads_dim(), s.n_embd, n_tok)?;
                            kernels::copy_across(&mut st.attn_out, &st.attn_out_a, (n_tok * s.n_embd) as usize * 4)?;
                            kernels::set_device(primary)?;
                        }
                    }
                }
                if self.attn_dev.is_none() {
                    kernels::matmul_q8_0(&mut st.attn_out, attn_output_w, &st.heads, heads_dim, s.n_embd, n_tok)?;
                }
                if let Some(gw) = &l.gemma {
                    // gemma post-attention norm sits INSIDE the residual
                    kernels::rms_norm_inplace(&mut st.attn_out, &gw.attn_post_norm, s.n_embd, n_tok, eps)?;
                }
                if let Some(ink) = &l.ink {
                    // inkling: the attention output stream gets its own
                    // shortconv before rejoining the residual
                    kernels::sconv(&mut st.sconv_tmp, &st.attn_out, &ink.sconv_attn, &mut st.sconv_state[il][2], n_tok, s.n_embd, s.sconv_k)?;
                    kernels::copy_across(&mut st.attn_out, &st.sconv_tmp, (n_tok * s.n_embd) as usize * 4)?;
                }
                kernels::add(&mut st.after_attn, &st.cur, &st.attn_out, n_tok * s.n_embd)?;

                // ffn
                kernels::rms_norm(&mut st.normed, &st.after_attn, &l.ffn_norm, s.n_embd, n_tok, eps)?;
                match &l.ffn {
                    // qwen35 (the only DenseKq family) never reaches the
                    // shared eval path
                    Ffn::DenseKq { .. } => {
                        return Err("DenseKq layer in the shared eval path".into())
                    }
                    Ffn::Dense { gate, up, down } => {
                        kernels::matmul_q8_0(&mut st.gate_act, gate, &st.normed, s.n_embd, s.n_ff_dense, n_tok)?;
                        kernels::matmul_q8_0(&mut st.up_act, up, &st.normed, s.n_embd, s.n_ff_dense, n_tok)?;
                        // leading-dense layers share the arch's gated-FFN op
                        // (M3: swiglu_oai on dense AND experts AND shexp)
                        kernels::swiglu(&mut st.ffn_mid, &st.gate_act, &st.up_act, n_tok * s.n_ff_dense, 0.0, 1.0, s.moe_act_op)?;
                        kernels::matmul_q8_0(&mut st.ffn_out, down, &st.ffn_mid, s.n_ff_dense, s.n_embd, n_tok)?;
                        if let Some(ink) = &l.ink {
                            // inkling: dense output rides gscale + its own
                            // shortconv stream before the residual
                            if ink.gscale != 1.0 {
                                kernels::scale(&mut st.ffn_out, n_tok * s.n_embd, ink.gscale)?;
                            }
                            kernels::sconv(&mut st.sconv_tmp, &st.ffn_out, &ink.sconv_mlp, &mut st.sconv_state[il][3], n_tok, s.n_embd, s.sconv_k)?;
                            kernels::copy_across(&mut st.ffn_out, &st.sconv_tmp, (n_tok * s.n_embd) as usize * 4)?;
                        }
                        kernels::add(&mut st.cur, &st.after_attn, &st.ffn_out, n_tok * s.n_embd)?;
                    }
                    Ffn::Moe { gate_inp, probs_b, shexp, gate_exps, up_exps, down_exps, fused_up_off, down_scale, sink } => {
                        let gw = l.gemma.as_ref();
                        // inkling: shared experts ride the router as
                        // always-on slots; per-layer gscale folds into the
                        // route-weight scale (every FFN output is linear
                        // in the weights)
                        let sink_n = if sink.is_some() { s.n_shexp_sink } else { 0 };
                        let route_k = s.n_expert_used - sink_n;
                        let route_scale = s.expert_weight_scale
                            * l.ink.as_ref().map_or(1.0, |i| i.gscale);
                        if let Some(gw) = gw {
                            // gemma routes on rms(attn_out) * gate_inp_s /
                            // sqrt(n_embd) - one weighted rms_norm; attn_out
                            // is dead here, reuse it as the scratch row
                            kernels::rms_norm(&mut st.attn_out, &st.after_attn, &gw.router_norm, s.n_embd, n_tok, eps)?;
                            kernels::matmul_f32(&mut st.router_logits, gate_inp, &st.attn_out, s.n_embd, s.n_expert, n_tok)?;
                        } else {
                            // inkling's gate matmul emits the sink logits
                            // after the n_expert routed ones
                            kernels::matmul_f32(&mut st.router_logits, gate_inp, &st.normed, s.n_embd, s.n_expert + sink_n, n_tok)?;
                        }
                        kernels::router_select(
                            &mut st.router_selected,
                            &mut st.router_weights,
                            &st.router_logits,
                            probs_b,
                            s.n_expert,
                            route_k,
                            route_scale,
                            n_tok,
                            if sink_n > 0 { 2 } else { s.router_softmax as u32 },
                            sink_n,
                        )?;
                        if let Some(ds) = down_scale {
                            // per-expert down scale folds into the route
                            // weight (the down projection is linear)
                            kernels::router_scale_selected(
                                &mut st.router_weights,
                                &st.router_selected,
                                ds,
                                n_tok * s.n_expert_used,
                                s.n_expert,
                            )?;
                        }

                        // Cross-layer prefetch (decode only): run the NEXT
                        // MoE layer's router on THIS layer's ffn input and
                        // ship the predicted slabs to the background
                        // fetcher. Rides the sync we need anyway.
                        let next_moe = if n_tok == 1
                            && std::env::var_os("PULSAR_NO_PREFETCH").is_none()
                        {
                            self.layers.get(il + 1).and_then(|nl| match &nl.ffn {
                                Ffn::Moe { gate_inp, probs_b, gate_exps, up_exps, down_exps, .. } => {
                                    Some((gate_inp, probs_b, [gate_exps, up_exps, down_exps]))
                                }
                                _ => None,
                            })
                        } else {
                            None
                        };
                        if let Some((n_gate_inp, n_probs_b, _)) = &next_moe {
                            kernels::matmul_f32(&mut st.pred_logits, n_gate_inp, &st.normed, s.n_embd, s.n_expert + sink_n, 1)?;
                            kernels::router_select(
                                &mut st.pred_selected,
                                &mut st.pred_weights,
                                &st.pred_logits,
                                n_probs_b,
                                s.n_expert,
                                route_k,
                                route_scale,
                                1,
                                if sink_n > 0 { 2 } else { s.router_softmax as u32 },
                                sink_n,
                            )?;
                        }

                        // shared expert: depends only on normed, so it is
                        // launched BEFORE the resolve - the GPU computes it
                        // under the disk/H2D wait. Gemma's "shared expert"
                        // is the full-width dense MLP (n_ff_dense, GELU)
                        // with its own post-norm.
                        if let Some((sg, su, sd)) = shexp {
                            let w = if gw.is_some() { s.n_ff_dense } else { s.n_ff_exp };
                            kernels::matmul_q8_0(&mut st.gate_act, sg, &st.normed, s.n_embd, w, n_tok)?;
                            kernels::matmul_q8_0(&mut st.up_act, su, &st.normed, s.n_embd, w, n_tok)?;
                            kernels::swiglu(&mut st.ffn_mid, &st.gate_act, &st.up_act, n_tok * w, 0.0, 1.0, s.moe_act_op)?;
                            kernels::matmul_q8_0(&mut st.shared_out, sd, &st.ffn_mid, w, s.n_embd, n_tok)?;
                            if let Some(gw) = gw {
                                kernels::rms_norm_inplace(&mut st.shared_out, &gw.post_ffw_norm_1, s.n_embd, n_tok, eps)?;
                            }
                        } else {
                            kernels::zero(&mut st.shared_out, (n_tok * s.n_embd) as usize * 4)?;
                        }
                        if let Some(gw) = gw {
                            // routed branch reads its own pre-norm of the
                            // residual, not the MLP norm
                            kernels::rms_norm(&mut st.normed, &st.after_attn, &gw.pre_ffw_norm_2, s.n_embd, n_tok, eps)?;
                        }
                        // also quantize the routed-expert activations now;
                        // only the expert weights are still in flight
                        kernels::quantize_q8_k(&mut st.xq, &st.normed, s.n_embd, n_tok)?;

                        // Expert resolve, batched: the union of distinct
                        // experts across all tokens fetches once. VRAM
                        // cache first, then host LFU + one io_uring batch.
                        let t_sync = std::time::Instant::now();
                        kernels::sync()?;
                        st.prof.sync += t_sync.elapsed();
                        let t_resolve = std::time::Instant::now();
                        let t_d2h = std::time::Instant::now();
                        let selected = st
                            .router_selected
                            .read_i32(n_tok as usize * s.n_expert_used as usize)?;
                        let pred_ids = if next_moe.is_some() {
                            Some(st.pred_selected.read_i32(s.n_expert_used as usize)?)
                        } else {
                            None
                        };
                        st.prof.resolve_d2h += t_d2h.elapsed();
                        if let (Some((_, _, next_exps)), Some(pred)) = (&next_moe, &pred_ids) {
                            let mut reads = Vec::with_capacity(3 * pred.len());
                            for &e in pred {
                                if e < 0 || e as u32 >= s.n_expert {
                                    continue;
                                }
                                for t in next_exps {
                                    let offset = t.abs_offset + e as u64 * t.expert_bytes;
                                    if !st.store.contains(offset)
                                        && !st.dev_cache.map.contains_key(&offset)
                                        && !st.tiers.iter().any(|tr| tr.map.contains_key(&offset))
                                    {
                                        reads.push(stream::Read { offset, len: t.expert_bytes });
                                    }
                                }
                            }
                            if !reads.is_empty() {
                                let _ = st.prefetcher.req_tx.send(reads);
                            }
                        }
                        // Prefill layer pipeline: a batch chunk touches
                        // ~every expert, so the next layer's want-list
                        // needs no prediction - it is all of them. Ship it
                        // to the background fetcher so the disk runs under
                        // this layer's GPU compute (ds4's ping-pong
                        // full-layer load, via the host-cache channel).
                        // real prefill chunks only: a 2-row spec-verify
                        // batch must not ship whole layers to the fetcher
                        if n_tok > 8 && std::env::var_os("PULSAR_NO_PREFETCH").is_none() {
                            if let Some(Ffn::Moe {
                                gate_exps: ng, up_exps: nu, down_exps: nd, ..
                            }) = self.layers.get(il + 1).map(|nl| &nl.ffn)
                            {
                                let mut reads = Vec::with_capacity(3 * s.n_expert as usize);
                                for e in 0..s.n_expert as u64 {
                                    for t in [ng, nu, nd] {
                                        let offset = t.abs_offset + e * t.expert_bytes;
                                        if !st.store.contains(offset)
                                            && !st.dev_cache.map.contains_key(&offset)
                                            && !st.tiers.iter().any(|tr| tr.map.contains_key(&offset))
                                        {
                                            reads.push(stream::Read {
                                                offset,
                                                len: t.expert_bytes,
                                            });
                                        }
                                    }
                                }
                                if !reads.is_empty() {
                                    let _ = st.prefetcher.req_tx.send(reads);
                                }
                            }
                        }
                        // Claim cross-layer async H2D BEFORE absorb: host
                        // DMA must finish before the host LFU can free the
                        // source pinned slabs.
                        let mut resolved = std::collections::HashMap::new();
                        if let Some(pf) = st.h2d_prefetch.take() {
                            if pf.layer == il {
                                if pf.recorded {
                                    let t = std::time::Instant::now();
                                    st.expert_h2d.synchronize()?;
                                    st.expert_h2d.wait_default()?;
                                    st.prof.h2d += t.elapsed();
                                }
                                for (off, p) in pf.map {
                                    resolved.insert(off, p);
                                }
                            } else if pf.recorded {
                                // stale prediction — must drain DMA before
                                // absorb can free host sources; count as h2d
                                // so it does not pollute disk/host
                                let t = std::time::Instant::now();
                                let _ = st.expert_h2d.synchronize();
                                st.prof.h2d += t.elapsed();
                            }
                        }
                        // absorb whatever the disk prefetcher finished
                        while let Ok((off, slab)) = st.prefetcher.done_rx.try_recv() {
                            st.store.absorb(off, slab);
                        }
                        let t_lists = std::time::Instant::now();
                        // gate/up/down may use different quants (K-quant
                        // recipes put ffn_down a tier higher); staging
                        // slots are strided by the largest of the three
                        let mut distinct: Vec<i32> = selected
                            .iter()
                            .copied()
                            .filter(|&e| e >= 0 && (e as u32) < s.n_expert + sink_n)
                            .collect();
                        distinct.sort_unstable();
                        distinct.dedup();
                        // id -> the three slabs it lives in: routed ids hit
                        // gate/up/down_exps, sink ids (>= n_expert) index
                        // the inkling shexp bank
                        let slabs_of = |e: u32| -> [(&ExpertTensor, u64); 3] {
                            if e < s.n_expert {
                                [(gate_exps, e as u64), (up_exps, e as u64), (down_exps, e as u64)]
                            } else {
                                let sk = sink.as_ref().unwrap();
                                let le = (e - s.n_expert) as u64;
                                [(&sk[0], le), (&sk[1], le), (&sk[2], le)]
                            }
                        };
                        let off_of = |t: &ExpertTensor, le: u64| t.abs_offset + le * t.expert_bytes;
                        // resolve tier placement once per distinct expert
                        // (was recomputed in cpu/offsets/ptrs loops)
                        let mut tier_place: std::collections::HashMap<
                            i32,
                            (usize, ExpertPtrs, bool),
                        > = std::collections::HashMap::with_capacity(distinct.len());
                        for &e in &distinct {
                            let is_sink = e as u32 >= s.n_expert;
                            let [g3, u3, d3] = slabs_of(e as u32);
                            let g = off_of(g3.0, g3.1);
                            if !is_sink
                                && self.mtp.as_ref().is_some_and(|mt| mt.res_map.contains_key(&g))
                            {
                                continue;
                            }
                            if let Some(place) = st.tiers.iter().enumerate().find_map(|(ti, t)| {
                                let gate = *t.map.get(&g)?;
                                Some((
                                    ti,
                                    ExpertPtrs {
                                        gate,
                                        up: byte_off(
                                            *t.map.get(&off_of(u3.0, u3.1))?,
                                            if is_sink { 0 } else { *fused_up_off },
                                        ),
                                        down: *t.map.get(&off_of(d3.0, d3.1))?,
                                    },
                                    is_sink,
                                ))
                            }) {
                                tier_place.insert(e, place);
                            }
                        }
                        let tier_of =
                            |e: i32| -> Option<(usize, ExpertPtrs, bool)> { tier_place.get(&e).copied() };
                        // CPU expert lane (PULSAR_CPU=1): host-cache-hit
                        // experts compute on CPU; decode-shaped batches only.
                        let cpu_on = st.cpu_pool.is_some()
                            && n_tok <= 8
                            && !st.unified
                            && s.n_embd % 256 == 0
                            && s.n_ff_exp % 256 == 0
                            && gate_exps.quant == up_exps.quant
                            && [gate_exps.quant, down_exps.quant]
                                .iter()
                                .all(|&q| cpu_tier::supported(q));
                        let n_used = s.n_expert_used as usize;
                        let (ne, nf) = (s.n_embd as usize, s.n_ff_exp as usize);
                        let mut lane = cpu_tier::Lane::new(
                            gate_exps.quant,
                            down_exps.quant,
                            gate_exps.row_bytes as usize,
                            down_exps.row_bytes as usize,
                            ne,
                            nf,
                            s.moe_act_op,
                        );
                        let mut cpu_guard: Option<cpu_tier::WaitGuard> = None;
                        // PULSAR_CPU_STEAL=0: leave dev-cache-resident
                        // experts to the GPU. Right call on boxes where
                        // warm VRAM coverage is high and the CPU is weak
                        // (a V100 user measured the lane net-negative
                        // there); default 1 = deterministic CPU ownership
                        // of host-cached experts, which is what stabilizes
                        // the cache ecology on high-miss boxes like mine.
                        let cpu_steal =
                            std::env::var("PULSAR_CPU_STEAL").ok().as_deref() != Some("0");
                        if cpu_on {
                            let mut pins = Vec::new();
                            for &e in &distinct {
                                if e < 0 || e as u32 >= s.n_expert || tier_of(e).is_some() {
                                    continue;
                                }
                                let [g3, u3, d3] = slabs_of(e as u32);
                                let (go, uo, dno) =
                                    (off_of(g3.0, g3.1), off_of(u3.0, u3.1), off_of(d3.0, d3.1));
                                // PULSAR_CPU_STEAL=0: leave VRAM-resident experts
                                // on the GPU (weak-CPU / high-coverage boxes).
                                if !cpu_steal
                                    && (st.dev_cache.map.contains_key(&go)
                                        || st.dev_cache.map.contains_key(&uo)
                                        || st.dev_cache.map.contains_key(&dno))
                                {
                                    continue;
                                }
                                // host-cached => CPU lane, even when a slab
                                // also sits in dev_cache: exclusion made
                                // ownership a first-touch race, bistable
                                // run to run (GLM oscillated 1.6-2.8).
                                if self
                                    .mtp
                                    .as_ref()
                                    .is_some_and(|mt| mt.res_map.contains_key(&go))
                                {
                                    continue;
                                }
                                let (Some(gp), Some(upp), Some(dp)) = (
                                    st.store.peek_ptr(go),
                                    st.store.peek_ptr(uo),
                                    st.store.peek_ptr(dno),
                                ) else {
                                    continue;
                                };
                                // PULSAR_CPU_CAP: bound lane experts per
                                // layer (bisection tool for the GLM loop)
                                if let Some(cap) = std::env::var("PULSAR_CPU_CAP")
                                    .ok()
                                    .and_then(|v| v.parse::<usize>().ok())
                                {
                                    if lane.idx.len() >= cap {
                                        continue;
                                    }
                                }
                                lane.add(e, gp.0, unsafe { upp.0.add(*fused_up_off as usize) }, dp.0);
                                pins.extend([go, uo, dno]);
                            }
                            st.store.pinned = pins;
                        }
                        if !lane.is_empty() {
                            let t_cpu_d2h = std::time::Instant::now();
                            let rw = st.router_weights.read_f32(n_tok as usize * n_used)?;
                            let normed_h = st.normed.read_f32(n_tok as usize * ne)?;
                            st.prof.resolve_d2h += t_cpu_d2h.elapsed();
                            let pool = st.cpu_pool.as_ref().unwrap();
                            cpu_guard = Some(cpu_tier::WaitGuard {
                                pool,
                                n: lane.submit_a(pool, &selected, n_used, &normed_h, &rw, n_tok as usize),
                            });
                        }
                        let mut offsets =
                            Vec::with_capacity(3 * distinct.len());
                        for &e in &distinct {
                            if tier_of(e).is_some() {
                                for (t, le) in slabs_of(e as u32) {
                                    let off = off_of(t, le);
                                    st.dev_cache.touch.entry(off).or_insert((0, t.expert_bytes)).0 += 1;
                                }
                                continue;
                            }
                            // PULSAR_CPU_VERIFY: fetch lane experts anyway
                            // so a full-pointer GPU pass can cross-check
                            // the lane partial (task #38 instrument)
                            let verify = std::env::var_os("PULSAR_CPU_VERIFY").is_some() && n_tok == 1;
                            if lane.idx.contains_key(&e) && !verify {
                                continue;
                            }
                            for (t, le) in slabs_of(e as u32) {
                                let r = stream::Read {
                                    offset: off_of(t, le),
                                    len: t.expert_bytes,
                                };
                                if offsets.last().map(|l: &stream::Read| l.offset) != Some(r.offset) {
                                    offsets.push(r);
                                }
                            }
                        }
                        let in_use: Vec<u64> = offsets.iter().map(|r| r.offset).collect();
                        let mut wants = Vec::new();
                        for r in &offsets {
                            if let Some(mt) = &self.mtp {
                                if let Some(&po) = mt.res_map.get(&r.offset) {
                                    resolved.insert(r.offset, mt.res_pool.ptr_at(po));
                                    continue;
                                }
                            }
                            if resolved.contains_key(&r.offset) {
                                st.dev_cache
                                    .touch
                                    .entry(r.offset)
                                    .or_insert((0, r.len))
                                    .0 += 1;
                                continue;
                            }
                            if st.unified {
                                wants.push(*r);
                                continue;
                            }
                            match st.dev_cache.get(r.offset, r.len) {
                                Some(p) => {
                                    resolved.insert(r.offset, p);
                                }
                                None => wants.push(*r),
                            }
                        }
                        st.prof.resolve_lists += t_lists.elapsed();
                        // Host LFU first; H2D overlaps remaining disk reads.
                        let mut stage_base = std::collections::HashMap::new();
                        let mut stage_total = 0usize;
                        for r in &wants {
                            stage_base.insert(r.offset, stage_total);
                            stage_total += r.len as usize;
                        }
                        if stage_total + SLAB_SLACK > st.staging.bytes() {
                            st.staging = DeviceBuf::alloc(stage_total + SLAB_SLACK)?;
                        }
                        let unified = st.unified;
                        let async_h2d = st.async_expert_h2d;
                        let mut h2d = std::time::Duration::ZERO;
                        let mut async_queued = false;
                        let t_host = std::time::Instant::now();
                        {
                            let dev_cache = &mut st.dev_cache;
                            let staging = &mut st.staging;
                            let expert_h2d = &st.expert_h2d;
                            st.store.ensure_with(&wants, |off, payload| {
                                if unified {
                                    resolved.insert(
                                        off,
                                        payload.as_ptr() as *const std::ffi::c_void,
                                    );
                                    return Ok(());
                                }
                                let t = std::time::Instant::now();
                                let p = match dev_cache.maybe_insert(off, payload, &in_use)? {
                                    Some(p) => p,
                                    None => {
                                        let base = stage_base[&off];
                                        if async_h2d {
                                            expert_h2d.copy_h2d_raw(
                                                staging,
                                                base,
                                                payload.as_ptr(),
                                                payload.len(),
                                            )?;
                                            async_queued = true;
                                        } else {
                                            staging.write(base, payload)?;
                                        }
                                        staging.ptr_at(base)
                                    }
                                };
                                h2d += t.elapsed();
                                resolved.insert(off, p);
                                Ok(())
                            })?;
                        }
                        let ensure_elapsed = t_host.elapsed();
                        // host bucket = ensure wall minus nested h2d copies
                        st.prof.resolve_host += ensure_elapsed.saturating_sub(h2d);
                        if async_queued {
                            let t = std::time::Instant::now();
                            st.expert_h2d.record()?;
                            st.expert_h2d.wait_default()?;
                            h2d += t.elapsed();
                        }
                        st.prof.h2d += h2d;
                        // sink slabs join the routed launch only when the
                        // bank shares quant AND row width; otherwise they
                        // run as a second NULL-masked launch below
                        let sink_same = sink.as_ref().is_none_or(|sk| {
                            sk[0].quant == gate_exps.quant && sk[0].row_bytes == gate_exps.row_bytes
                                && sk[1].quant == up_exps.quant && sk[1].row_bytes == up_exps.row_bytes
                                && sk[2].quant == down_exps.quant && sk[2].row_bytes == down_exps.row_bytes
                        });
                        let mut ptrs = Vec::with_capacity(selected.len());
                        let mut sink_ptrs: Vec<ExpertPtrs> = if sink_same {
                            Vec::new()
                        } else {
                            vec![ExpertPtrs::NULL; selected.len()]
                        };
                        let mut tptrs: Vec<Vec<ExpertPtrs>> = st
                            .tiers
                            .iter()
                            .map(|_| vec![ExpertPtrs::NULL; selected.len()])
                            .collect();
                        // sink slots on a differently-quantized bank get
                        // their own tier launch pair (mirrors the primary)
                        let mut tptrs_sink: Vec<Vec<ExpertPtrs>> = if sink_same {
                            Vec::new()
                        } else {
                            st.tiers.iter().map(|_| vec![ExpertPtrs::NULL; selected.len()]).collect()
                        };
                        let mut tier_slots = vec![0u64; st.tiers.len()];
                        let mut tier_slots_sink = vec![0u64; st.tiers.len()];
                        let verify = std::env::var_os("PULSAR_CPU_VERIFY").is_some() && n_tok == 1;
                        // verify: lane experts with REAL pointers, everything
                        // else NULL - isolates the lane set's GPU partial
                        let mut vptrs: Vec<ExpertPtrs> = Vec::new();
                        for (si, &e) in selected.iter().enumerate() {
                            if verify {
                                vptrs.push(if e >= 0 && lane.idx.contains_key(&e) && tier_of(e).is_none() {
                                    let [g3, u3, d3] = slabs_of(e as u32);
                                    ExpertPtrs {
                                        gate: resolved[&off_of(g3.0, g3.1)],
                                        up: byte_off(resolved[&off_of(u3.0, u3.1)], *fused_up_off),
                                        down: resolved[&off_of(d3.0, d3.1)],
                                    }
                                } else {
                                    ExpertPtrs::NULL
                                });
                            }
                            if e < 0 || e as u32 >= s.n_expert + sink_n {
                                ptrs.push(ExpertPtrs::NULL);
                                continue;
                            }
                            if let Some((ti, tp, is_sink)) = tier_of(e) {
                                ptrs.push(ExpertPtrs::NULL);
                                if is_sink && !sink_same {
                                    tptrs_sink[ti][si] = tp;
                                    tier_slots_sink[ti] += 1;
                                } else {
                                    tptrs[ti][si] = tp;
                                    tier_slots[ti] += 1;
                                }
                                continue;
                            }
                            if lane.idx.contains_key(&e) {
                                ptrs.push(ExpertPtrs::NULL);
                                continue;
                            }
                            let [g3, u3, d3] = slabs_of(e as u32);
                            let ep = ExpertPtrs {
                                gate: resolved[&off_of(g3.0, g3.1)],
                                // sink banks are never gate_up-fused (same
                                // rule as tier_of above)
                                up: byte_off(
                                    resolved[&off_of(u3.0, u3.1)],
                                    if e as u32 >= s.n_expert { 0 } else { *fused_up_off },
                                ),
                                down: resolved[&off_of(d3.0, d3.1)],
                            };
                            if !sink_same && e as u32 >= s.n_expert {
                                sink_ptrs[si] = ep;
                                ptrs.push(ExpertPtrs::NULL);
                            } else {
                                ptrs.push(ep);
                            }
                        }
                        st.expert_ptrs.write(0, kernels::as_bytes(&ptrs))?;

                        // grouped batch MoE (prefill): CSR of tokens per
                        // expert so each weight row is staged in shared
                        // memory once instead of re-read per token
                        let smem_ok = 2 * gate_exps.row_bytes.max(up_exps.row_bytes) * 4 <= 49152
                            && down_exps.row_bytes * 4 <= 49152;
                        let grouped = n_tok >= 16 && s.n_expert_used <= 16 && smem_ok
                            // grouped down stages rows in smem with no
                            // slack for the sub-block tail overread
                            && s.n_ff_exp % 256 == 0
                            && std::env::var_os("PULSAR_NO_GROUPED").is_none();
                        let mut n_group = 0u32;
                        if grouped {
                            let mut gid: std::collections::HashMap<*const std::ffi::c_void, u32> =
                                std::collections::HashMap::new();
                            let mut gptrs: Vec<ExpertPtrs> = Vec::new();
                            let mut members: Vec<Vec<u32>> = Vec::new();
                            for (si, p) in ptrs.iter().enumerate() {
                                if p.gate.is_null() {
                                    continue;
                                }
                                let g = *gid.entry(p.gate).or_insert_with(|| {
                                    gptrs.push(*p);
                                    members.push(Vec::new());
                                    (gptrs.len() - 1) as u32
                                });
                                let token = (si / s.n_expert_used as usize) as u32;
                                let slot = (si % s.n_expert_used as usize) as u32;
                                members[g as usize].push((token << 4) | slot);
                            }
                            n_group = gptrs.len() as u32;
                            if n_group > 0 {
                                let mut starts = Vec::with_capacity(n_group as usize + 1);
                                let mut pairs = Vec::with_capacity(n_tok as usize * s.n_expert_used as usize);
                                starts.push(0u32);
                                for m in &members {
                                    pairs.extend_from_slice(m);
                                    starts.push(pairs.len() as u32);
                                }
                                st.grp_ptrs.write(0, kernels::as_bytes(&gptrs))?;
                                st.grp_starts.write(0, kernels::as_bytes(&starts))?;
                                st.grp_pairs.write(0, kernels::as_bytes(&pairs))?;
                                let need = n_tok as usize * s.n_expert_used as usize * s.n_embd as usize * 4;
                                if st.grp_partial.bytes() < need {
                                    st.grp_partial = DeviceBuf::alloc(need)?;
                                }
                            }
                        }
                        st.prof.resolve += t_resolve.elapsed();
                        st.prof.calls += 1;

                        // tier partials first: their kernels run on other
                        // cards, overlapping the primary's MoE below
                        let mut active = Vec::new();
                        for ti in 0..st.tiers.len() {
                            let sink_hits = *tier_slots_sink.get(ti).unwrap_or(&0);
                            if tier_slots[ti] == 0 && sink_hits == 0 {
                                continue;
                            }
                            let tier = &mut st.tiers[ti];
                            tier.hits += tier_slots[ti] + sink_hits;
                            kernels::copy_across(&mut tier.xin, &st.normed, (n_tok * s.n_embd) as usize * 4)?;
                            kernels::copy_across(&mut tier.weights, &st.router_weights, (n_tok * s.n_expert_used) as usize * 4)?;
                            kernels::set_device(tier.dev)?;
                            // both ptr arrays land before any launch so the
                            // whole tier chain runs async under primary work
                            tier.ptrs.write(0, kernels::as_bytes(&tptrs[ti]))?;
                            if sink_hits > 0 {
                                tier.ptrs_sink.write(0, kernels::as_bytes(&tptrs_sink[ti]))?;
                            }
                            kernels::quantize_q8_k(&mut tier.xq, &tier.xin, s.n_embd, n_tok)?;
                            if tier_slots[ti] > 0 {
                                kernels::moe_pair_swiglu(
                                    &mut tier.mid, &tier.ptrs, &tier.weights, &tier.xq,
                                    s.n_embd, s.n_ff_exp, s.n_expert_used, n_tok, gate_exps.row_bytes, gate_exps.quant, s.moe_act_op,
                                )?;
                                kernels::quantize_q8_k(&mut tier.midq, &tier.mid, s.n_ff_exp, n_tok * s.n_expert_used)?;
                                kernels::moe_down(
                                    &mut tier.out, &tier.ptrs, &tier.midq,
                                    s.n_ff_exp, s.n_embd, s.n_expert_used, n_tok, down_exps.row_bytes, down_exps.quant,
                                )?;
                            }
                            if sink_hits > 0 {
                                // sink pass: same mid/midq scratch, stream-
                                // ordered after the routed pass consumed it
                                let sk = sink.as_ref().unwrap();
                                kernels::moe_pair_swiglu(
                                    &mut tier.mid, &tier.ptrs_sink, &tier.weights, &tier.xq,
                                    s.n_embd, s.n_ff_exp, s.n_expert_used, n_tok, sk[0].row_bytes, sk[0].quant, s.moe_act_op,
                                )?;
                                kernels::quantize_q8_k(&mut tier.midq, &tier.mid, s.n_ff_exp, n_tok * s.n_expert_used)?;
                                kernels::moe_down(
                                    &mut tier.out_sink, &tier.ptrs_sink, &tier.midq,
                                    s.n_ff_exp, s.n_embd, s.n_expert_used, n_tok, sk[2].row_bytes, sk[2].quant,
                                )?;
                            }
                            kernels::set_device(primary)?;
                            active.push((ti, tier_slots[ti] > 0, sink_hits > 0));
                        }

                        // PULSAR_CPU_VERIFY: GPU-compute the LANE experts
                        // alone (vptrs) and stash the partial; compared
                        // against the lane's CPU partial at the join
                        let mut verify_gpu: Option<Vec<f32>> = None;
                        if verify && !lane.is_empty() {
                            st.expert_ptrs.write(0, kernels::as_bytes(&vptrs))?;
                            kernels::moe_pair_swiglu(
                                &mut st.moe_mid, &st.expert_ptrs, &st.router_weights, &st.xq,
                                s.n_embd, s.n_ff_exp, s.n_expert_used, n_tok, gate_exps.row_bytes, gate_exps.quant, s.moe_act_op,
                            )?;
                            kernels::quantize_q8_k(&mut st.midq, &st.moe_mid, s.n_ff_exp, n_tok * s.n_expert_used)?;
                            kernels::moe_down(
                                &mut st.moe_out, &st.expert_ptrs, &st.midq,
                                s.n_ff_exp, s.n_embd, s.n_expert_used, n_tok, down_exps.row_bytes, down_exps.quant,
                            )?;
                            kernels::sync()?;
                            verify_gpu = Some(st.moe_out.read_f32((n_tok * s.n_embd) as usize)?);
                            st.expert_ptrs.write(0, kernels::as_bytes(&ptrs))?;
                        }

                        // routed experts: activations quantized to q8_K,
                        // integer dp4a dots (ds4's exact math)
                        if grouped && n_group > 0 {
                            kernels::moe_pair_swiglu_grouped(
                                &mut st.moe_mid, &st.grp_ptrs, &st.grp_starts, &st.grp_pairs,
                                &st.router_weights, &st.xq,
                                s.n_embd, s.n_ff_exp, s.n_expert_used, n_group, gate_exps.row_bytes, gate_exps.quant, s.moe_act_op,
                            )?;
                            kernels::quantize_q8_k(&mut st.midq, &st.moe_mid, s.n_ff_exp, n_tok * s.n_expert_used)?;
                            let pbytes = n_tok as usize * s.n_expert_used as usize * s.n_embd as usize * 4;
                            kernels::zero(&mut st.grp_partial, pbytes)?;
                            kernels::moe_down_grouped(
                                &mut st.grp_partial, &st.grp_ptrs, &st.grp_starts, &st.grp_pairs, &st.midq,
                                s.n_ff_exp, s.n_embd, s.n_expert_used, n_group, down_exps.row_bytes, down_exps.quant,
                            )?;
                            kernels::moe_slot_sum(&mut st.moe_out, &st.grp_partial, s.n_embd, s.n_expert_used, n_tok)?;
                        } else {
                            kernels::moe_pair_swiglu(
                                &mut st.moe_mid, &st.expert_ptrs, &st.router_weights, &st.xq,
                                s.n_embd, s.n_ff_exp, s.n_expert_used, n_tok, gate_exps.row_bytes, gate_exps.quant, s.moe_act_op,
                            )?;
                            kernels::quantize_q8_k(&mut st.midq, &st.moe_mid, s.n_ff_exp, n_tok * s.n_expert_used)?;
                            kernels::moe_down(
                                &mut st.moe_out, &st.expert_ptrs, &st.midq,
                                s.n_ff_exp, s.n_embd, s.n_expert_used, n_tok, down_exps.row_bytes, down_exps.quant,
                            )?;
                        }

                        // inkling sink bank on its own quant: second NULL-
                        // masked pass over the same slots (routed slots
                        // NULL here, so only the sink rows contribute);
                        // ffn_out is free until the final adds below
                        if !sink_same {
                            let sk = sink.as_ref().unwrap();
                            st.expert_ptrs.write(0, kernels::as_bytes(&sink_ptrs))?;
                            kernels::moe_pair_swiglu(
                                &mut st.moe_mid, &st.expert_ptrs, &st.router_weights, &st.xq,
                                s.n_embd, s.n_ff_exp, s.n_expert_used, n_tok, sk[0].row_bytes, sk[0].quant, s.moe_act_op,
                            )?;
                            kernels::quantize_q8_k(&mut st.midq, &st.moe_mid, s.n_ff_exp, n_tok * s.n_expert_used)?;
                            kernels::moe_down(
                                &mut st.ffn_out, &st.expert_ptrs, &st.midq,
                                s.n_ff_exp, s.n_embd, s.n_expert_used, n_tok, sk[2].row_bytes, sk[2].quant,
                            )?;
                            kernels::add_assign(&mut st.moe_out, &st.ffn_out, n_tok * s.n_embd)?;
                        }

                        // Cross-layer async H2D (OPT-IN: PULSAR_H2D_PREFETCH=1).
                        // Wrong predictions leave a recorded event that the next
                        // layer must host-synchronize before absorb — that wait
                        // lands in resolve "disk/host" and cost ~seconds on GLM.
                        // Same-layer async H2D (above) stays on by default.
                        if st.async_expert_h2d
                            && !st.unified
                            && n_tok == 1
                            && std::env::var_os("PULSAR_H2D_PREFETCH").is_some()
                            && std::env::var_os("PULSAR_NO_PREFETCH").is_none()
                        {
                            if let Some((_, _, next_exps)) = &next_moe {
                                if let Ok(pred) =
                                    st.pred_selected.read_i32(s.n_expert_used as usize)
                                {
                                    let mut pf_reads: Vec<stream::Read> = Vec::new();
                                    for &e in &pred {
                                        if e < 0 || e as u32 >= s.n_expert {
                                            continue;
                                        }
                                        for t in next_exps {
                                            let offset =
                                                t.abs_offset + e as u64 * t.expert_bytes;
                                            if st.tiers.iter().any(|tr| tr.map.contains_key(&offset))
                                                || st.dev_cache.peek(offset).is_some()
                                                || self.mtp.as_ref().is_some_and(|mt| {
                                                    mt.res_map.contains_key(&offset)
                                                })
                                            {
                                                continue;
                                            }
                                            if pf_reads.iter().any(|r| r.offset == offset) {
                                                continue;
                                            }
                                            // host must already hold the slab
                                            // (disk prefetcher / warm); skip if not
                                            if !st.store.contains(offset) {
                                                continue;
                                            }
                                            pf_reads.push(stream::Read {
                                                offset,
                                                len: t.expert_bytes,
                                            });
                                        }
                                    }
                                    if !pf_reads.is_empty() {
                                        let mut stage_total = 0usize;
                                        let mut bases = std::collections::HashMap::new();
                                        for r in &pf_reads {
                                            bases.insert(r.offset, stage_total);
                                            stage_total += r.len as usize;
                                        }
                                        if stage_total + SLAB_SLACK > st.staging_alt.bytes() {
                                            st.staging_alt =
                                                DeviceBuf::alloc(stage_total + SLAB_SLACK)?;
                                        }
                                        // default-stream MoE is already queued;
                                        // side-stream H2D runs concurrently.
                                        let mut map = std::collections::HashMap::new();
                                        let mut queued = false;
                                        for r in &pf_reads {
                                            if let Some(payload) = st.store.payload(r.offset) {
                                                let base = bases[&r.offset];
                                                st.expert_h2d.copy_h2d_raw(
                                                    &mut st.staging_alt,
                                                    base,
                                                    payload.as_ptr(),
                                                    payload.len(),
                                                )?;
                                                map.insert(
                                                    r.offset,
                                                    st.staging_alt.ptr_at(base),
                                                );
                                                queued = true;
                                            }
                                        }
                                        if queued {
                                            st.expert_h2d.record()?;
                                            st.h2d_prefetch = Some(ExpertH2dPrefetch {
                                                layer: il + 1,
                                                map,
                                                recorded: true,
                                            });
                                        }
                                    }
                                }
                            }
                        }

                        // gather tier partials (blocking copy issued on the
                        // tier's device = ordered after its kernels).
                        // NOTE: summing partials reorders float adds vs the
                        // single-kernel slot loop - same drift class as
                        // batch-vs-decode; PULSAR_TIERS=off restores exact.
                        for (ti, routed_out, sink_out) in active {
                            let tier = &st.tiers[ti];
                            if routed_out {
                                kernels::set_device(tier.dev)?;
                                kernels::copy_across(&mut st.tier_ret, &tier.out, (n_tok * s.n_embd) as usize * 4)?;
                                kernels::set_device(primary)?;
                                kernels::add_assign(&mut st.moe_out, &st.tier_ret, n_tok * s.n_embd)?;
                            }
                            if sink_out {
                                kernels::set_device(tier.dev)?;
                                kernels::copy_across(&mut st.tier_ret, &tier.out_sink, (n_tok * s.n_embd) as usize * 4)?;
                                kernels::set_device(primary)?;
                                kernels::add_assign(&mut st.moe_out, &st.tier_ret, n_tok * s.n_embd)?;
                            }
                        }

                        // CPU-lane join: stage A ran under the resolve
                        // and the GPU launches above; the down-proj fan-out
                        // runs here while those kernels are in flight, then
                        // one f32 upload joins moe_out on the primary.
                        if !lane.is_empty() {
                            drop(cpu_guard.take());
                            let t_cpu = std::time::Instant::now();
                            let pool = st.cpu_pool.as_ref().unwrap();
                            let acc = lane.finish(pool, n_tok as usize);
                            if let Some(gpu) = &verify_gpu {
                                let mut dmax = 0f32;
                                let mut gmax = 0f32;
                                let mut cmax = 0f32;
                                let mut at = 0usize;
                                for (i, (&g, &c)) in gpu.iter().zip(acc.iter()).enumerate() {
                                    let d = (g - c).abs();
                                    if d > dmax {
                                        dmax = d;
                                    }
                                    if g.abs() > gmax {
                                        gmax = g.abs();
                                        at = i;
                                    }
                                    cmax = cmax.max(c.abs());
                                }
                                eprintln!(
                                    "lane-verify L{il}: n={} max|gpu-cpu|={dmax:.5} max|gpu|={gmax:.5} max|cpu|={cmax:.5} at[{at}] gpu={:.5} cpu={:.5}",
                                    lane.idx.len(), gpu[at], acc[at]
                                );
                            }
                            st.store.pinned.clear();
                            st.cpu_hits += lane.idx.len() as u64;
                            st.prof.cpu += t_cpu.elapsed();
                            if st.cpu_ret.bytes() < acc.len() * 4 {
                                st.cpu_ret = DeviceBuf::alloc(acc.len() * 4)?;
                            }
                            st.cpu_ret.write(0, kernels::as_bytes(&acc))?;
                            kernels::add_assign(&mut st.moe_out, &st.cpu_ret, n_tok * s.n_embd)?;
                        }

                        // cur = after_attn + routed + shared (ds4's add3).
                        // gemma sandwiches norms around the sum and scales
                        // the whole stream by layer_output_scale.
                        if let Some(gw) = gw {
                            kernels::rms_norm_inplace(&mut st.moe_out, &gw.post_ffw_norm_2, s.n_embd, n_tok, eps)?;
                        }
                        kernels::add(&mut st.ffn_out, &st.moe_out, &st.shared_out, n_tok * s.n_embd)?;
                        if let Some(gw) = gw {
                            kernels::rms_norm_inplace(&mut st.ffn_out, &gw.post_ffw_norm, s.n_embd, n_tok, eps)?;
                        }
                        if let Some(ink) = &l.ink {
                            // inkling: the whole MoE output (routed + sink,
                            // gscale already in the route weights) gets the
                            // mlp shortconv before the residual
                            kernels::sconv(&mut st.sconv_tmp, &st.ffn_out, &ink.sconv_mlp, &mut st.sconv_state[il][3], n_tok, s.n_embd, s.sconv_k)?;
                            kernels::copy_across(&mut st.ffn_out, &st.sconv_tmp, (n_tok * s.n_embd) as usize * 4)?;
                        }
                        kernels::add(&mut st.cur, &st.after_attn, &st.ffn_out, n_tok * s.n_embd)?;
                        if let Some(gw) = gw {
                            if gw.out_scale != 1.0 {
                                kernels::scale(&mut st.cur, n_tok * s.n_embd, gw.out_scale)?;
                            }
                        }
                    }
                }
            }
            Ok(())
        }
    }

    /// Prefill `prompt` at pos0 (chunked), then sample until `stop`,
    /// ctx, or max_tokens; each sampled token goes to `on_token` and is
    /// forwarded into the KV cache (including the stop token, so the
    /// context stays template-shaped for a next turn). Returns the
    /// position after everything forwarded.
    impl Model {
        /// Build the MTP block's input rows for a prefill chunk and run it,
        /// so its KV covers the prompt (row for position p embeds token_p
        /// with hidden_{p-1}; st.mtp_hidden stitches chunk boundaries).
        /// Must run right after the chunk's forward while st.cur still
        /// holds its hidden states. Clobbers st.cur.
        fn mtp_prefill_fill(&self, st: &mut State, n_tok: u32, pos0: u32) -> Result {
            let Some(mtp) = &self.mtp else { return Ok(()) };
            let s = self.shape;
            let primary = kernels::get_device();
            let row = s.n_embd as usize * 4;
            // hidden inputs: [old mtp_hidden, cur rows 0..n-1]
            kernels::copy_d2d(&mut st.mtp_e_raw, 0, &st.mtp_hidden, 0, row)?;
            if n_tok > 1 {
                kernels::copy_d2d(&mut st.mtp_e_raw, row, &st.cur, 0, (n_tok as usize - 1) * row)?;
            }
            kernels::copy_d2d(&mut st.mtp_hidden, 0, &st.cur, (n_tok as usize - 1) * row, row)?;
            kernels::rms_norm(&mut st.mtp_h, &st.mtp_e_raw, &mtp.hnorm, s.n_embd, n_tok, s.rms_eps)?;
            // token embeddings (st.tok still holds the chunk)
            kernels::embed_q8_0(&mut st.mtp_e_raw, &self.token_embd, &st.tok, s.n_embd, s.n_vocab, n_tok)?;
            kernels::rms_norm(&mut st.mtp_e, &st.mtp_e_raw, &mtp.enorm, s.n_embd, n_tok, s.rms_eps)?;
            for i in 0..n_tok as usize {
                kernels::copy_d2d(&mut st.mtp_x, i * 2 * row, &st.mtp_e, i * row, row)?;
                kernels::copy_d2d(&mut st.mtp_x, i * 2 * row + row, &st.mtp_h, i * row, row)?;
            }
            kernels::matmul_q8_0(&mut st.cur, &mtp.eh_proj, &st.mtp_x, 2 * s.n_embd, s.n_embd, n_tok)?;
            self.mtp_eval_layer(st, n_tok, pos0, primary)
        }

        /// Eval the MTP draft layer over st.cur (family dispatch: the
        /// hybrid families have their own layer graphs).
        fn mtp_eval_layer(&self, st: &mut State, n_tok: u32, pos0: u32, primary: i32) -> Result {
            let mtp = self.mtp.as_ref().ok_or("mtp layer missing")?;
            match self.shape.family {
                Family::Qwen35 => {
                    let mut rt = st.qwen35.take().ok_or("qwen35 state missing")?;
                    let r = self.eval_qwen35_layer(st, &mut rt, self.layers.len(), &mtp.layer, pos0, n_tok);
                    st.qwen35 = Some(rt);
                    r
                }
                _ => self.eval_layer(st, self.layers.len(), &mtp.layer, n_tok, pos0, primary),
            }
        }

        /// One MTP pass: embed `token` at `pos` against st.mtp_hidden,
        /// append the block's KV, return the greedy draft for pos+1.
        /// Clobbers st.cur.
        fn mtp_draft(&self, st: &mut State, token: u32, pos: u32) -> Result<u32> {
            self.mtp_body(st, token, pos)?;
            let mtp = self.mtp.as_ref().ok_or("mtp_draft without an MTP layer")?;
            let s = self.shape;
            kernels::rms_norm(&mut st.normed, &st.cur, &mtp.head_norm, s.n_embd, 1, s.rms_eps)?;
            self.head_logits(st, 1)?;
            kernels::sync()?;
            let logits = st.logits.read_f32(s.n_vocab as usize)?;
            if std::env::var_os("PULSAR_MTP_DEBUG").is_some() {
                let bad = logits.iter().filter(|v| !v.is_finite()).count();
                eprintln!("mtp-draft pos={pos}: logits nan={bad}, draft={}", argmax(&logits));
            }
            Ok(argmax(&logits))
        }

        fn mtp_body(&self, st: &mut State, token: u32, pos: u32) -> Result {
            let mtp = self.mtp.as_ref().ok_or("mtp_draft without an MTP layer")?;
            let s = self.shape;
            let primary = kernels::get_device();
            let row = s.n_embd as usize * 4;
            st.tok.write(0, kernels::as_bytes(&[token as i32]))?;
            kernels::embed_q8_0(&mut st.mtp_e_raw, &self.token_embd, &st.tok, s.n_embd, s.n_vocab, 1)?;
            kernels::rms_norm(&mut st.mtp_e, &st.mtp_e_raw, &mtp.enorm, s.n_embd, 1, s.rms_eps)?;
            kernels::rms_norm(&mut st.mtp_h, &st.mtp_hidden, &mtp.hnorm, s.n_embd, 1, s.rms_eps)?;
            kernels::copy_d2d(&mut st.mtp_x, 0, &st.mtp_e, 0, row)?;
            kernels::copy_d2d(&mut st.mtp_x, row, &st.mtp_h, 0, row)?;
            kernels::matmul_q8_0(&mut st.cur, &mtp.eh_proj, &st.mtp_x, 2 * s.n_embd, s.n_embd, 1)?;
            self.mtp_eval_layer(st, 1, pos, primary)
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate(
        model: &Model,
        st: &mut State,
        prompt: &[u32],
        pos0: u32,
        sampler: &mut Sampler,
        max_tokens: usize,
        stop: impl Fn(u32) -> bool,
        mut on_token: impl FnMut(u32),
    ) -> Result<u32> {
        generate_cancellable(model, st, prompt, pos0, sampler, max_tokens, stop, on_token_shim(&mut on_token), || false)
    }

    fn on_token_shim(f: &mut impl FnMut(u32)) -> impl FnMut(u32) + '_ {
        move |t| f(t)
    }

    /// generate() with a cancel probe checked between prefill chunks and
    /// decode tokens: a server whose client disconnected mid-prefill can
    /// abandon the work instead of computing minutes of tokens for
    /// nobody. Returns the position reached; state/KV stay consistent
    /// with everything forwarded so far.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_cancellable(
        model: &Model,
        st: &mut State,
        prompt: &[u32],
        pos0: u32,
        sampler: &mut Sampler,
        max_tokens: usize,
        stop: impl Fn(u32) -> bool,
        mut on_token: impl FnMut(u32),
        cancel: impl Fn() -> bool,
    ) -> Result<u32> {
        // MTP speculative decode is greedy-only: acceptance compares the
        // draft against the verified argmax, which IS greedy sampling.
        let spec = model.mtp.is_some() && sampler.is_greedy();
        let mut pos = pos0;
        let mut logits = None;
        // qwen35 MTP prefill: the draft-layer scratch is 16-row and the
        // qwen35 forward leaves only its LAST 16-row chunk in st.cur, so
        // the fill pass needs outer chunks capped to match (the forward
        // is internally 16-chunked anyway - same work either way)
        let chunk_cap = if spec && model.shape.family == Family::Qwen35 {
            16
        } else {
            st.max_batch() as usize
        };
        let prof_chunks = std::env::var_os("PULSAR_PROFILE").is_some();
        for chunk in prompt.chunks(chunk_cap) {
            if cancel() {
                return Ok(pos);
            }
            let t0 = std::time::Instant::now();
            logits = model.forward_batch(st, chunk, pos, true)?;
            if spec {
                model.mtp_prefill_fill(st, chunk.len() as u32, pos)?;
            }
            if prof_chunks {
                eprintln!(
                    "pulsar: prefill chunk @{pos} len {} in {:.2}s",
                    chunk.len(),
                    t0.elapsed().as_secs_f64()
                );
            }
            pos += chunk.len() as u32;
        }

        // Draft-free n-gram speculation (PULSAR_NGRAM=depth, greedy only):
        // propose the tokens that followed the longest recent-suffix match
        // earlier in the context, verify the whole chain in ONE batch-union
        // forward (rows are cheap - the union fetch is shared), accept the
        // matching prefix. No draft model, no draft cost; pays exactly when
        // output repeats context (code, quotes, lists).
        let ngram_depth = std::env::var("PULSAR_NGRAM")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|_| sampler.is_greedy() && model.mtp.is_none());
        if let Some(depth) = ngram_depth {
            let v = model.shape.n_vocab as usize;
            let depth = depth.clamp(1, 15);
            let mut hist: Vec<u32> = prompt.to_vec();
            let mut emitted = 0usize;
            let mut next = argmax(logits.as_deref().ok_or("no logits")?);
            while emitted < max_tokens {
                if stop(next) || pos + 1 >= st.ctx() {
                    model.forward_batch(st, &[next], pos, false)?;
                    pos += 1;
                    break;
                }
                on_token(next);
                emitted += 1;
                hist.push(next);
                // longest suffix (4..=1) of hist that recurs earlier
                let mut draft: Vec<u32> = Vec::new();
                'outer: for m in (3..=4usize.min(hist.len().saturating_sub(1))).rev() {
                    let suf = &hist[hist.len() - m..];
                    let limit = hist.len() - m;
                    for i in (0..limit).rev() {
                        if &hist[i..i + m] == suf {
                            let mut j = i + m;
                            while draft.len() < depth && j < limit {
                                draft.push(hist[j]);
                                j += 1;
                            }
                            if !draft.is_empty() {
                                break 'outer;
                            }
                        }
                    }
                }
                if draft.is_empty() || pos + 2 + draft.len() as u32 >= st.ctx() {
                    let lg = model
                        .forward_batch(st, &[next], pos, true)?
                        .ok_or("no logits")?;
                    pos += 1;
                    next = argmax(&lg);
                    continue;
                }
                let mut chain = vec![next];
                chain.extend_from_slice(&draft);
                st.mtp_drafted += draft.len() as u64;
                let all = model
                    .forward_rows(st, &chain, pos, chain.len() as u32)?
                    .ok_or("no verify logits")?;
                let k = draft.len();
                let mut j = 0usize;
                while j < k && argmax(&all[j * v..(j + 1) * v]) == chain[j + 1] {
                    st.mtp_accepted += 1;
                    j += 1;
                }
                pos += (j + 1) as u32;
                next = argmax(&all[j * v..(j + 1) * v]);
                for &d in &chain[1..=j] {
                    if stop(d) || emitted >= max_tokens {
                        return Ok(pos);
                    }
                    on_token(d);
                    emitted += 1;
                    hist.push(d);
                }
            }
            return Ok(pos);
        }

        if spec {
            let v = model.shape.n_vocab as usize;
            let row = model.shape.n_embd as usize * 4;
            let depth_max = model.mtp_depth.max(1);
            let debug = std::env::var_os("PULSAR_MTP_DEBUG").is_some();
            let timing = std::env::var_os("PULSAR_MTP_TIMING").is_some();
            let (mut t_draft, mut t_verify, mut t_refwd, mut t_fill) =
                (std::time::Duration::ZERO, std::time::Duration::ZERO, std::time::Duration::ZERO, std::time::Duration::ZERO);
            let mut emitted = 0usize;
            let mut next = argmax(logits.as_deref().ok_or("no logits")?);
            'round: while emitted < max_tokens {
                if stop(next) || pos + 2 >= st.ctx() {
                    model.forward_batch(st, &[next], pos, false)?;
                    pos += 1;
                    break;
                }
                on_token(next);
                emitted += 1;

                // Draft a chain: each step self-feeds the MTP layer's own
                // output hidden (approximate but cheap - one layer/step).
                // Anchor the true pre-chain hidden for the fill pass.
                kernels::copy_d2d(&mut st.mtp_hidden_save, 0, &st.mtp_hidden, 0, row)?;
                let t0 = std::time::Instant::now();
                let depth = depth_max.min(st.ctx() - pos - 2);
                let mut chain = vec![next];
                for i in 0..depth {
                    let d = model.mtp_draft(st, chain[i as usize], pos + i)?;
                    st.mtp_drafted += 1;
                    kernels::copy_d2d(&mut st.mtp_hidden, 0, &st.cur, 0, row)?;
                    chain.push(d);
                    if stop(d) {
                        break; // no point speculating past a stop token
                    }
                }
                t_draft += t0.elapsed();
                let k = chain.len() - 1; // drafts in flight

                // Verify the whole chain in ONE forward: the per-layer
                // union expert fetch is what makes the extra rows cheap.
                // Greedy acceptance keeps the stream identical to plain
                // greedy decode.
                //
                // Recurrent families (qwen35 GDN): verify advances the
                // delta-rule/conv state over the WHOLE chain, and unlike
                // KV rows a recurrent state can't be overwritten next
                // round. Snapshot first; full acceptance means the state
                // is exactly right (free), partial acceptance restores
                // and re-forwards the accepted prefix.
                let recurrent = model.shape.family == Family::Qwen35;
                let t0 = std::time::Instant::now();
                if recurrent {
                    st.qwen35.as_mut().ok_or("qwen35 state missing")?.gdn_snapshot()?;
                }
                let all = model
                    .forward_rows(st, &chain, pos, (k + 1) as u32)?
                    .ok_or("no verify logits")?;
                t_verify += t0.elapsed();
                let mut j = 0usize;
                while j < k && argmax(&all[j * v..(j + 1) * v]) == chain[j + 1] {
                    st.mtp_accepted += 1;
                    j += 1;
                }
                if recurrent && j < k {
                    let t0 = std::time::Instant::now();
                    st.qwen35.as_mut().ok_or("qwen35 state missing")?.gdn_restore()?;
                    // no logits; leaves st.cur/st.tok holding exactly the
                    // accepted rows for the fill pass below
                    model.forward_batch(st, &chain[..=j], pos, false)?;
                    t_refwd += t0.elapsed();
                }
                if debug {
                    let nans = all.iter().filter(|x| !x.is_finite()).count();
                    eprintln!("mtp: pos={pos} chain={chain:?} accepted={j}/{k} nan={nans}");
                }

                // Re-anchor the MTP cache on TRUE hiddens for the accepted
                // prefix in one batched pass: st.tok still holds the chain,
                // st.cur its verified hiddens - exactly what a prefill
                // chunk looks like to mtp_prefill_fill.
                kernels::copy_d2d(&mut st.mtp_hidden, 0, &st.mtp_hidden_save, 0, row)?;
                let t0 = std::time::Instant::now();
                model.mtp_prefill_fill(st, (j + 1) as u32, pos)?;
                t_fill += t0.elapsed();
                pos += (j + 1) as u32;
                next = argmax(&all[j * v..(j + 1) * v]);

                for &d in &chain[1..=j] {
                    if stop(d) {
                        break 'round; // forwarded, not emitted - as non-spec
                    }
                    if emitted >= max_tokens {
                        break 'round;
                    }
                    on_token(d);
                    emitted += 1;
                }
            }
            if timing {
                eprintln!(
                    "mtp timing: draft {:.2}s verify {:.2}s refwd {:.2}s fill {:.2}s over {emitted} tokens",
                    t_draft.as_secs_f64(), t_verify.as_secs_f64(), t_refwd.as_secs_f64(), t_fill.as_secs_f64()
                );
            }
            return Ok(pos);
        }

        for _ in 0..max_tokens {
            if cancel() {
                return Ok(pos);
            }
            let next = sampler.sample(logits.as_ref().ok_or("no logits")?);
            if stop(next) || pos + 1 >= st.ctx() {
                model.forward_batch(st, &[next], pos, false)?;
                pos += 1;
                break;
            }
            on_token(next);
            logits = model.forward_batch(st, &[next], pos, true)?;
            pos += 1;
        }
        Ok(pos)
    }

    /// First-max argmax, matching ds4's sample_argmax.
    pub fn argmax(logits: &[f32]) -> u32 {
        let mut best = 0usize;
        for (i, &v) in logits.iter().enumerate() {
            if v > logits[best] {
                best = i;
            }
        }
        best as u32
    }

    /// Temperature + nucleus (top-p) + min-p sampling, seeded and
    /// reproducible. temp <= 0 is greedy.
    pub struct Sampler {
        pub temp: f32,
        pub top_p: f32,
        pub min_p: f32,
        state: u64,
    }

    impl Sampler {
        pub fn new(temp: f32, top_p: f32, min_p: f32, seed: u64) -> Sampler {
            Sampler { temp, top_p, min_p, state: seed | 1 }
        }

        pub fn is_greedy(&self) -> bool {
            self.temp <= 0.0
        }

        fn randf(&mut self) -> f32 {
            // xorshift64*
            let mut x = self.state;
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            self.state = x;
            ((x.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as f32) / (1u64 << 24) as f32
        }

        pub fn sample(&mut self, logits: &[f32]) -> u32 {
            if self.temp <= 0.0 {
                return argmax(logits);
            }
            let mut cand: Vec<(u32, f32)> =
                logits.iter().enumerate().map(|(i, &l)| (i as u32, l)).collect();
            cand.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            // softmax with temperature over the sorted candidates
            let maxl = cand[0].1;
            let mut sum = 0f32;
            for c in cand.iter_mut() {
                c.1 = ((c.1 - maxl) / self.temp).exp();
                sum += c.1;
            }
            let p0 = cand[0].1 / sum;
            let mut kept = 0usize;
            let mut cum = 0f32;
            for c in &cand {
                let p = c.1 / sum;
                if self.min_p > 0.0 && p < self.min_p * p0 && kept > 0 {
                    break;
                }
                cum += p;
                kept += 1;
                if self.top_p < 1.0 && cum >= self.top_p {
                    break;
                }
            }
            let kept_sum: f32 = cand[..kept].iter().map(|c| c.1).sum();
            let mut r = self.randf() * kept_sum;
            for c in &cand[..kept] {
                if r < c.1 {
                    return c.0;
                }
                r -= c.1;
            }
            cand[kept - 1].0
        }
    }
}

#[cfg(target_os = "linux")]
pub use real::*;
