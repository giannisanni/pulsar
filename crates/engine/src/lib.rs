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

/// Build identity for --version. The git sha is what a bug report needs;
/// the crate version alone never moves.
pub const VERSION: &str = concat!(
    env!("CARGO_PKG_VERSION"),
    " (",
    env!("PULSAR_GIT_SHA"),
    ")"
);

#[cfg(target_os = "linux")]
mod real {
    /// Bytes per GiB. Memory sizes are reported in GiB everywhere so a
    /// printed number can be compared directly against a card's capacity;
    /// mixing decimal GB with binary GiB under one "GB" label made a 69MiB
    /// board difference read as a 0.1GB gap. Throughput stays decimal GB/s,
    /// which is the convention for PCIe and NVMe.
    const GIB: f64 = (1u64 << 30) as f64;

    mod dsv4;
    mod k3;
    pub(crate) mod qwen35;
    pub use dsv4::generate_dspark;
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
        /// Kimi-K3 (kimi-k3): 3 KDA layers + 1 NoPE gated-MLA layer per
        /// block, Attention Residuals over depth, and a latent MoE whose
        /// 896 routed experts run in a half-width space. KDA is Qwen35's
        /// Gated DeltaNet with a per-key-channel forget gate. Decode-only
        /// graph; prefill loops tokens (the KDA recurrence is sequential).
        K3,
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
        /// YaRN applies to EVERY layer with the standard mscale, rather
        /// than laguna's scheme of yarn on full-window layers only with the
        /// kernel mscale cancelled. gpt-oss is the uniform kind.
        pub rope_yarn_uniform: bool,
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
        // kimi-k3 (zero elsewhere)
        /// KDA head width; d_inner = n_head * kda_head_dim.
        pub kda_head_dim: u32,
        /// g_min in `g = g_min * sigmoid(exp(A_log) * z)`. Negative.
        pub kda_gate_lb: f32,
        /// Width the routed experts live in (< n_embd), with
        /// ffn_routed_down/up projecting into and out of it.
        pub n_expert_latent: u32,
        /// Layers per AttnRes block; a checkpoint is banked every
        /// `attn_res_block` layers. Zero disables AttnRes.
        pub attn_res_block: u32,
        /// Shared-expert FFN width. K3 ships this already multiplied by
        /// the shared-expert count (2 x 3072 = 6144), matching the fused
        /// ffn_*_shexp tensors; elsewhere it falls back to n_ff_exp.
        pub n_ff_shexp: u32,
    }

    impl Shape {
        pub fn qk_dim(&self) -> u32 {
            self.qk_nope + self.qk_rope
        }

        /// Attention output width (input of attn_output).
        fn heads_dim(&self) -> u32 {
            match self.family {
                Family::Gqa | Family::Dsv4 | Family::Qwen35 => self.n_head * self.head_dim,
                // K3's two layer flavours agree here: the MLA half is
                // n_head * value_mla and the KDA half n_head *
                // kda_head_dim, both 12288, which is why one attn_output
                // shape serves both.
                Family::Mla | Family::K3 => self.n_head * self.value_mla,
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
        /// Resolve a parsed gguf header into a Shape. Public so config
        /// parsing for a new architecture can be checked against the real
        /// file (see examples/k3-shape.rs) before its weights exist.
        pub fn from_gguf(g: &Gguf) -> Result<Shape> {
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
                // poolside Laguna S/XS 2.1: GQA MoE with a sigmoid router
                // (exp_probs_b correction bias, topk-norm, shared expert),
                // interleaved full/sliding attention like gemma4, plus two
                // Laguna-only pieces: a per-head output gate (attn_gate,
                // softplus) and learnable sinks on the sliding layers.
                Some("laguna") => Family::Gqa,
                // OpenAI gpt-oss: GQA MoE with per-head attention sinks,
                // alternating sliding/full attention, q/k/v/output biases,
                // per-expert biases, and MXFP4 routed experts
                Some("gpt-oss") => Family::Gqa,
                // Qwen3.6-35B-A3B hybrid GDN (task #21)
                Some("qwen35moe") => Family::Qwen35,
                // Qwen3.6 dense (27B lineage, task #37): same GDN hybrid
                // stack; the dense FFN loads as a single always-on expert
                // so placement/caching/tiering machinery applies unchanged
                Some("qwen35") => Family::Qwen35,
                // Kimi-K3 2.8T: hybrid KDA/MLA + AttnRes + latent MoE
                Some("kimi-k3") => Family::K3,
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
                // laguna ships head_count as a per-layer array (48 on full
                // layers, 72 on sliding); the scalar Shape field takes the
                // MAX for buffer sizing, per-layer truth lives in
                // Model::geom.n_head_q - same split gemma4 uses for kv.
                n_head: u("attention.head_count").unwrap_or_else(|_| {
                    match g.arch_meta("attention.head_count") {
                        Some(Value::Array(a)) => {
                            a.iter().filter_map(Value::as_u64).max().unwrap_or(1) as u32
                        }
                        _ => 1,
                    }
                }),
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
                // softmax over the SELECTED top-k, not sigmoid per expert
                // and not softmax over all of them: llama.cpp calls this
                // SOFTMAX_WEIGHT, and gpt-oss uses it with no renorm after
                router_softmax: matches!(
                    g.architecture(),
                    Some("qwen3moe") | Some("gemma4") | Some("qwen35moe") | Some("gpt-oss")
                ),
                rope_yarn_uniform: g.architecture() == Some("gpt-oss"),
                // gated-FFN op: 1 = gelu (gemma4), 2 = swiglu_oai (MiniMax
                // M3: clamp 7, alpha 1.702, up+1 - llama.cpp PR 24523),
                // 0 = plain silu everywhere else (inkling included)
                moe_act_op: match g.architecture() {
                    Some("gemma4") => 1,
                    Some("minimax-m3") | Some("gpt-oss") => 2,
                    // kimi-k3 SiTU-GLU, on the routed AND shared experts
                    // and the leading dense FFN
                    Some("kimi-k3") => 4,
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
                kda_head_dim: 0,
                kda_gate_lb: 0.0,
                n_expert_latent: 0,
                attn_res_block: 0,
                n_ff_shexp: 0,
            };
            if g.architecture() == Some("laguna") {
                // laguna ships real YaRN (factor 32 over an 8192 native
                // window) on BOTH attention flavours; rope_cfg's
                // deepseek2 mscale path does not apply (log_mult 0).
                s.rope_scale_factor = f("rope.scaling.factor").unwrap_or(1.0);
                s.rope_orig_ctx = u("rope.scaling.original_context_length").unwrap_or(8192);
                s.rope_yarn_log_mult = 0.0;
            }
            if g.architecture() == Some("gpt-oss") {
                // yarn factor 32 over a 4096 native window, applied to every
                // layer; without this parse the factor defaulted to 1.0 and
                // rope ran unscaled, which is a different model
                s.rope_scale_factor = f("rope.scaling.factor").unwrap_or(1.0);
                s.rope_orig_ctx = u("rope.scaling.original_context_length").unwrap_or(4096);
                s.rope_yarn_log_mult = 0.0;
            }
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
            if family == Family::K3 {
                // MLA half: identical shape to the deepseek2/GLM lineage,
                // except nothing is ever rotated. The 64 rope-tail dims
                // still exist in the K/Q rows (attn_kv_a_mqa is
                // kv_lora + 64 wide) - K3 just never applies rope to
                // them, so qk_rope stays the slice width and no rope
                // kernel runs. rope.freq_base is present but unused
                // ("defaults_used: rope_theta" in the conversion keys).
                s.n_lora_q = u("attention.q_lora_rank").unwrap_or(1536);
                s.n_kv_lora = u("attention.kv_lora_rank").unwrap_or(512);
                s.qk_rope = u("rope.dimension_count").unwrap_or(64);
                let qk_mla = u("attention.key_length_mla").unwrap_or(192);
                s.qk_nope = qk_mla - s.qk_rope;
                s.value_mla = u("attention.value_length_mla").unwrap_or(128);
                // KDA half. head_count (96) is the KDA head count too, so
                // d_inner = 96 * 128 = 12288 = the ssm_* projection width.
                s.kda_head_dim = u("kda.head_dim").unwrap_or(128);
                s.ssm_conv_k = u("ssm.conv_kernel").unwrap_or(4);
                s.ssm_inner = s.n_head * s.kda_head_dim;
                s.ssm_k_heads = s.n_head;
                s.ssm_v_heads = s.n_head;
                s.ssm_state = s.kda_head_dim;
                s.kda_gate_lb = f("kda.gate_lower_bound").unwrap_or(-5.0);
                s.n_expert_latent = u("expert_latent_length").unwrap_or(0);
                s.attn_res_block = u("attn_res.block_size").unwrap_or(0);
                s.n_ff_shexp = u("expert_shared_feed_forward_length")
                    .unwrap_or(s.n_ff_exp * u("expert_shared_count").unwrap_or(1));
                s.rope_orig_ctx = u("rope.scaling.original_context_length")
                    .unwrap_or(1_048_576);
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
            TensorType::IQ4NL => kernels::QUANT_IQ4_NL,
            TensorType::IQ3S => kernels::QUANT_IQ3_S,
            TensorType::IQ2S => kernels::QUANT_IQ2_S,
            TensorType::IQ1S => kernels::QUANT_IQ1_S,
            TensorType::MXFP4 => kernels::QUANT_MXFP4,
            TensorType::NVFP4 => kernels::QUANT_NVFP4,
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
        /// True when this tensor should stay native: a quant matmul_kq
        /// dispatches, with a 256-divisible contraction dim. Wider than
        /// just the warp-cooperative pair (Q4K/Q6K) on purpose: unsloth
        /// UD dense ggufs ship Q5K/IQ4_XS attention and FFN tensors
        /// (Qwen3.8-27B), and requanting those to q8_0 costs 1.6-1.9x
        /// the VRAM and bytes-per-token AND puts a lazily-grown q8_0
        /// prequant scratch inside the GDN CUDA-graph capture, where
        /// cudaMalloc is illegal.
        fn keep_native(t: &TensorInfo) -> bool {
            matches!(
                t.ty,
                TensorType::Q4K | TensorType::Q5K | TensorType::Q6K | TensorType::IQ4XS | TensorType::NVFP4
            ) && t.dims[0].is_multiple_of(256)
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

    #[allow(clippy::large_enum_variant)] // one Ffn per layer; boxing would indirect the decode hot path
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
            /// per-expert f32 bias vectors, [n_expert][mid_dim] for gate/up
            /// and [n_expert][out_dim] for down. gpt-oss is the only arch
            /// here that ships them; everything else leaves this None and
            /// the kernels skip the add. Resident, never streamed: the
            /// whole set is n_expert * (2*mid + out) floats, a rounding
            /// error next to one expert's quantized weights.
            exp_bias: Option<[DeviceBuf; 3]>,
            /// gpt-oss router bias [n_expert], added to the gate logits
            /// before selection. Distinct from `probs_b`, which is the
            /// DeepSeek-style correction that steers selection WITHOUT
            /// entering the weights: this one is part of the linear layer,
            /// so it moves the softmax too.
            gate_inp_b: Option<DeviceBuf>,
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
        /// per-layer QUERY head count (laguna varies it: 48 full / 72
        /// sliding). 0 = uniform, use Shape::n_head.
        n_head_q: u32,
        n_head_kv: u32,
        head_dim: u32,
        theta: f32,
        window: u32,   /* 0 = full causal */
        factors: bool, /* proportional rope via rope_freqs */
        /// per-layer rotation width; 0 = rotate the whole head. laguna
        /// rotates 64 of 128 on full layers, 128 on sliding ones.
        rot: u32,
    }

    enum Attn {
        Gqa {
            attn_q: DeviceBuf,
            /// None = k reused as v (gemma E-series attention_k_eq_v)
            attn_v: Option<DeviceBuf>,
            attn_k: DeviceBuf,
            /// None = the arch has no qk-norm at all (gpt-oss uses q/k
            /// biases instead). Distinct from passing a null weight to the
            /// norm kernel, which still normalizes, just without a scale.
            q_norm: Option<DeviceBuf>,
            k_norm: Option<DeviceBuf>,
            /// gpt-oss per-head attention sink [n_head]: a learned logit
            /// that joins the softmax denominator and contributes no value,
            /// letting a head attend to nothing. None everywhere else.
            sinks: Option<DeviceBuf>,
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
        K3(Box<K3W>),
    }

    /// qwen35moe per-layer stack: exactly one of attn/gdn is Some.
    /// The MoE half reuses Ffn::Moe; LayerW.attn_norm doubles as the
    /// pre-attention norm and LayerW.ffn_norm as post_attention_norm.
    /// One K3 layer: exactly one of `kda`/`mla` is present (the gguf's
    /// attention.head_count_kv array marks KDA layers with 0), plus the
    /// two AttnRes score vectors every layer carries and the latent-MoE
    /// projections that wrap the routed experts.
    struct K3W {
        kda: Option<K3Kda>,
        mla: Option<K3Mla>,
        /// AttnRes score weights, f32 [n_embd]: one for the mix before
        /// attention, one for the mix before the FFN.
        attn_res_score: DeviceBuf,
        ffn_res_score: DeviceBuf,
        /// Latent MoE: routed experts run at n_expert_latent, so the
        /// input is projected down and the result normed and projected
        /// back up. Absent on the leading dense layer.
        routed: Option<K3Routed>,
        shexp: Option<K3Shexp>,
    }
    struct K3Kda {
        wq: MatW, // [n_embd -> d_inner]
        wk: MatW,
        wv: MatW,
        /// separate depthwise conv per stream, f32 [d_inner][conv_k]
        conv_q: DeviceBuf,
        conv_k: DeviceBuf,
        conv_v: DeviceBuf,
        /// decay logits factor through a rank-head_dim bottleneck
        f_a: MatW, // [n_embd -> head_dim]
        /// [head_dim -> d_inner], kept f32. It is small (6MB a layer) and
        /// it feeds an exponential, so the decay path is the last place to
        /// want requant error.
        f_b: DeviceBuf,
        beta_w: MatW, // [n_embd -> n_head]
        /// -exp(A_log), folded at conversion time, f32 [n_head]
        a: DeviceBuf,
        dt_bias: DeviceBuf, // f32 [d_inner]
        /// K3 uses one full-rank output gate where kimi-linear factors
        /// it as g_b(g_a(x))
        wg: MatW,           // [n_embd -> d_inner]
        ssm_norm: DeviceBuf, // f32 [head_dim]
        out: MatW,          // [d_inner -> n_embd]
    }
    struct K3Mla {
        q_a: DeviceBuf,
        q_a_norm: DeviceBuf,
        q_b: DeviceBuf,
        kv_a_mqa: DeviceBuf,
        kv_a_norm: DeviceBuf,
        k_b: DeviceBuf,
        v_b: DeviceBuf,
        /// sigmoid output gate applied before the output projection;
        /// reads the NORMED layer input, not the attention result
        gate: MatW, // [n_embd -> n_head*value_mla]
        out: MatW,
    }
    /// K3's shared experts stay in native K-quant. The generic Ffn::Moe
    /// slot requants to q8_0, which for K3's 2x6144 fused pair is 12.9GB
    /// across 92 layers - more than the primary card holds on its own.
    /// Native Q4_K halves that. Loaded on the primary, like Ffn::Moe's.
    struct K3Shexp {
        gate: MatW,
        up: MatW,
        down: MatW,
    }
    struct K3Routed {
        down: MatW,           // [n_embd -> latent]
        up: MatW,             // [latent -> n_embd]
        norm: Option<DeviceBuf>, // f32 [latent]
    }

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
        /// concatenated ssm_alpha ++ ssm_beta rows, f32
        /// [n_embd -> 2*ssm_v_heads]: at decode ONE matmul feeds both
        /// coefficient sets; the halves stay addressable by row offset
        ab_w: DeviceBuf,
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
        il < n_leading_dense as usize || (il >= 6 && (il - 6).is_multiple_of(4))
    }

    /// gpt-oss attention biases, all f32. Every other arch here projects
    /// without them and leaves this None.
    struct AttnBias {
        q: DeviceBuf,
        k: DeviceBuf,
        v: DeviceBuf,
        out: DeviceBuf,
    }

    struct LayerW {
        attn_norm: DeviceBuf,
        attn: Attn,
        attn_output: DeviceBuf,
        attn_bias: Option<AttnBias>,
        ffn_norm: DeviceBuf,
        ffn: Ffn,
        gemma: Option<GemmaW>,
        ink: Option<InkW>,
        /// laguna per-head output gate (g_proj): softplus(x @ w) scales
        /// each attention head row before attn_output. None elsewhere.
        attn_gate: Option<DeviceBuf>,
    }

    /// The nextn/MTP draft block: predicts token t+2 from (hidden of
    /// t, embedding of t+1) through one extra transformer layer.
    struct MtpLayer {
        layer: LayerW,
        /// [n_embd][2*n_embd] in whatever the file ships. NVFP4 ggufs
        /// (Qwen3.8 nextn, ggml type 40) kept this native, and reading
        /// those 36-byte blocks as q8_0's 34-byte ones made EVERY draft
        /// logit NaN -> argmax 0 -> 0% acceptance and a 3x slowdown.
        eh_proj: MatW,
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
        /// K3 AttnRes: the score vector for the final mix before the head
        output_res_score: Option<DeviceBuf>,
        output: DeviceBuf,
        layers: Vec<LayerW>,
        /// PULSAR_ATTN_GPU: second CUDA device holding ALL attn weights +
        /// KV resident (Mla only). Attention weights are read every layer
        /// every token, so residency is the one job a bandwidth-crippled
        /// PCIe link can still do: only activations cross per layer.
        /// Under the layer split this is the FIRST off-primary owner -
        /// coarse is-offload-on-at-all gates key off it; per-layer sites
        /// use attn_layer_dev.
        pub attn_dev: Option<i32>,
        /// Per-exec-layer (+ MTP slot) owner of the ATTENTION stack
        /// (weights, KV, idx cache, scratch) - Mla layer split. Contiguous
        /// ranges by construction so the DSA selection list hops only at
        /// range boundaries. All-attn_dev (or all-primary) when no split.
        pub attn_layer_dev: Vec<i32>,
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
        /// DSpark draft heads (fc / main_norm / markov / confidence):
        /// present only on a converted draft gguf, None on targets
        dspark: Option<dsv4::DsparkW>,
        /// FFN tensor-parallel shards on the second card (PULSAR_TP=1,
        /// dense qwen35); None everywhere else
        pub(crate) tp: Option<TpW>,
        pub(crate) tp_c: Option<TpCW>,
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
    /// A recurrent-state prefix checkpoint (position + family payload).
    /// KV rows are positional and rewritten on replay; only the rolling
    /// lane/GDN state needs copies.
    pub enum RecurrentCkpt {
        Dsv4(Vec<dsv4::Dsv4LayerCkpt>),
        Qwen35(Vec<Option<(DeviceBuf, DeviceBuf)>>),
    }

    #[derive(Default)]
    pub struct Prof {
        pub sync: std::time::Duration,
        pub resolve: std::time::Duration,
        /// D2H of router_selected / pred_selected inside resolve
        pub resolve_d2h: std::time::Duration,
        /// distinct/offsets/wants/tier placement list building
        pub resolve_lists: std::time::Duration,
        /// host-side lookup + LFU eviction (ensure_with wall minus h2d and disk)
        pub resolve_host: std::time::Duration,
        /// pure io_uring disk-fetch wait for cache misses (was hidden in host)
        pub resolve_fetch: std::time::Duration,
        pub h2d: std::time::Duration,
        /// draining the disk prefetcher into the host store (absorb), and
        /// how many slabs went through it - absorb evicts, and eviction
        /// is the one O(cache) operation on the per-layer path
        pub resolve_absorb: std::time::Duration,
        pub absorbed: u64,
        /// pointer/CSR build after the fetch, plus the blocking
        /// expert_ptrs upload. That upload is ordered behind the async
        /// expert H2D, so this bucket is dominated by PCIe drain, NOT by
        /// the 8-entry pointer loop: measured 12.9s with experts on a
        /// Gen4 x4 card (6.4 GB/s) vs 1.26s on Gen5 x8 (28.7 GB/s), same
        /// work either side. A big number here means "wrong card owns the
        /// experts", not "pointer building is slow".
        pub resolve_ptrs: std::time::Duration,
        /// CPU expert lane wall time after the stage-A overlap (mid
        /// quantize + down-proj fan-out + join)
        pub cpu: std::time::Duration,
        pub tail: std::time::Duration,
        pub calls: u64,
        /// Cross-layer prefetch accuracy: of the experts layer L+1 actually
        /// routed to, how many did the prediction made at layer L name?
        /// This gates the only overlap that is structurally possible - the
        /// layers are serial, so the GPU can ONLY be busy during a disk
        /// wait if the next layer's weights were already fetched.
        pub pred_hits: u64,
        pub pred_total: u64,
    }

    impl Prof {
        pub fn report(&self) -> String {
            let s = |d: std::time::Duration| d.as_secs_f64();
            let accounted = self.resolve_d2h + self.resolve_lists + self.resolve_host
                + self.resolve_fetch + self.h2d + self.resolve_absorb + self.resolve_ptrs;
            let other = self.resolve.saturating_sub(accounted);
            format!(
                "gpu-wait {:.2}s, resolve {:.2}s (d2h {:.2}s, lists {:.2}s, host {:.2}s, disk {:.2}s, h2d {:.2}s, absorb {:.2}s/{} slabs, ptrs {:.2}s, other {:.2}s), cpu-lane {:.2}s, logits-tail {:.2}s over {} layer steps",
                s(self.sync),
                s(self.resolve),
                s(self.resolve_d2h),
                s(self.resolve_lists),
                s(self.resolve_host),
                s(self.resolve_fetch),
                s(self.h2d),
                s(self.resolve_absorb),
                self.absorbed,
                s(self.resolve_ptrs),
                s(other),
                s(self.cpu),
                s(self.tail),
                self.calls
            ) + &if self.pred_total > 0 {
                format!(
                    "\npulsar: cross-layer prefetch predicted {}/{} routed experts ({:.1}%)",
                    self.pred_hits,
                    self.pred_total,
                    100.0 * self.pred_hits as f64 / self.pred_total as f64
                )
            } else {
                String::new()
            }
        }
    }

    /// One resident expert tier's placement + routing hits, for the /stats
    /// telemetry endpoint (the web UI's RAM/VRAM/Disk bars and per-tier heat).
    pub struct TierStat {
        pub dev: i32,
        pub bytes: usize,
        pub hits: u64,
    }

    /// Engine-side telemetry snapshot. Cumulative counters/timers; serve diffs
    /// consecutive snapshots to get per-turn deltas. Hardware (RAM/cores) and
    /// layer/expert counts are added serve-side from std + gguf metadata.
    /// One GPU's name and VRAM occupancy, for the /stats hardware panel.
    pub struct GpuStat {
        pub name: String,
        pub vram_free: usize,
        pub vram_total: usize,
    }

    pub struct Stats {
        pub gpu_count: i32,
        pub ctx: u32,
        pub tiers: Vec<TierStat>,
        pub cpu_hits: u64,
        pub cache_hits: u64,
        /// per-device name + VRAM free/total, all GPUs (not just expert tiers)
        pub gpus: Vec<GpuStat>,
        /// host RAM expert-cache used / capacity bytes (the RAM tier)
        pub host_used: usize,
        pub host_budget: usize,
        /// model bytes resident in VRAM: fixed expert tiers + the VRAM slab cache
        pub vram_resident: usize,
        /// total KV complex bytes at the CURRENT ctx (k/v caches + the DSA
        /// indexer key cache, across every layer and owner device). Divided
        /// by ctx this gives bytes-per-position, which is how a caller
        /// projects whether a different ctx would fit.
        pub kv_bytes: usize,
        /// VRAM a LARGER ctx could actually spend, on the cards that host
        /// KV: their free VRAM plus the expert tiers resident there (tiers
        /// are sized from what KV leaves over, so a resize rebuilds them
        /// smaller). Excludes cards that hold no KV - their free space is
        /// unreachable for this purpose.
        pub kv_headroom: usize,
        /// The format the KV actually resolved to, after PULSAR_KV and the
        /// size-aware auto-default were both applied. A caller showing
        /// "auto" needs this to say WHICH format auto landed on.
        pub kv_resolved: &'static str,
        /// Whether that KV is already in a compact format. False means a
        /// LARGER ctx would likely cost ~3.9x less per position than
        /// kv_bytes implies, because the auto-sizer switches to fp8 once
        /// the f32 projection gets big - a projection that ignores this
        /// rejects context sizes that would in fact load.
        pub kv_compact: bool,
        /// cumulative per-stage wall time, seconds (see Prof)
        pub prof_gpu_wait: f64,
        pub prof_resolve: f64,
        pub prof_h2d: f64,
        pub prof_fetch: f64,
        pub prof_cpu: f64,
        pub prof_tail: f64,
        pub prof_calls: u64,
    }

    /// One expert's residency + routing heat, for the Brain cortex viz.
    /// tier: 0 = disk-only, 2 = host RAM cache, 3 = VRAM-resident tier.
    pub struct ExpertCell {
        pub layer: u32,
        pub expert: u32,
        pub tier: u8,
        pub heat: u64,
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
                // Coalesce every pending request rather than keeping only
                // the newest. Each send is a DIFFERENT layer's predicted
                // experts, not a refresh of one list, so dropping a request
                // skips that layer's prefetch entirely and its slabs come
                // back as synchronous misses on the critical path.
                // Coalescing also deepens the batch, which is what actually
                // fills the drive: a ~7-read batch is latency-bound (~6GB/s
                // measured in decode) while the same NVMe sustains 11GB/s
                // once enough reads are in flight (fetch-bench, qd 8+).
                // Sorted because ascending offsets read faster than the
                // routing order, deduped because layers share slabs.
                // Bounded: coalesce up to a cap, then DISCARD whatever is
                // still queued. Prefetch is speculative, so a backlog is
                // strictly harmful - stale requests keep the drive busy on
                // layers the model has already passed while the current
                // layer waits. Unbounded coalescing let a prefill flood
                // queue ~177GiB of reads that drained through decode.
                const COALESCE_MAX: usize = 192;
                while let Ok(first) = req_rx.recv() {
                    let mut reads = first;
                    let mut dropped = false;
                    for more in req_rx.try_iter() {
                        if dropped || reads.len() >= COALESCE_MAX {
                            dropped = true; // drain and drop the rest
                            continue;
                        }
                        reads.extend(more);
                    }
                    // note: the FIRST request is never truncated - a real
                    // 256-token chunk legitimately asks for a whole layer
                    // (768 reads) and clipping that would gut the case this
                    // path exists for. The cap only bounds COALESCING.
                    reads.sort_unstable_by_key(|r| r.offset);
                    reads.dedup_by_key(|r| r.offset);
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
        /// Reserved capacity of the slab pool. Reclaimable by a larger KV:
        /// the auto budget sizes this from what the KV leaves over.
        pub fn pool_bytes(&self) -> usize {
            self.pool.bytes()
        }

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
            for (i, off, payload) in need.iter() {
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
        /// Returns the pure io_uring disk-fetch wait (excludes the caller's
        /// place/h2d copies) so the profiler can bucket disk separately from
        /// host-side lookup/eviction. Zero when everything hit the cache.
        fn ensure_with(
            &mut self,
            wants: &[stream::Read],
            mut place: impl FnMut(u64, &[u8]) -> Result,
        ) -> Result<std::time::Duration> {
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
                return Ok(std::time::Duration::ZERO);
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
            let t_fe = std::time::Instant::now();
            let mut fetch_place = std::time::Duration::ZERO;
            let place_err = {
                let Self { fetcher, cache, used, tick, .. } = self;
                let mut place_err = None;
                fetcher.fetch_each(&missing, |i, slab| {
                    if place_err.is_none() {
                        let tp = std::time::Instant::now();
                        if let Err(e) = place(missing[i].offset, slab.payload()) {
                            place_err = Some(e);
                        }
                        fetch_place += tp.elapsed();
                    }
                    *used += slab.bytes();
                    cache.insert(
                        missing[i].offset,
                        CacheEntry { slab, freq: 1, tick: *tick },
                    );
                    Ok(())
                })?;
                place_err
            };
            // pure disk wait = fetch_each wall minus the h2d copies nested in it
            let fetch = t_fe.elapsed().saturating_sub(fetch_place);
            match place_err {
                Some(e) => Err(e),
                None => Ok(fetch),
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
        /// tanh, 2 = swiglu_oai, 3 = deepseek4 clamped silu, 4 = kimi-k3
        /// SiTU-GLU)
        pub fn glu(g: f32, u: f32, op: u32) -> f32 {
            match op {
                1 => {
                    0.5 * g
                        * (1.0 + (0.797_884_6_f32 * (g + 0.044715 * g * g * g)).tanh())
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
                4 => {
                    // kimi-k3 SiTU-GLU; betas baked in as in pulsar_glu
                    let a = 4.0 * (g / 4.0).tanh() / (1.0 + (-g).exp());
                    a * (25.0 * (u / 25.0).tanh())
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

    /// Built-in {layer, expert} warm seed for a first run with no census
    /// yet (idea borrowed from ds4's ds4_streaming_hotlist_*.inc, MIT).
    /// Generated offline by `hotlist-gen` from a machine that has run the
    /// model; keyed by layer/expert index instead of byte offset, so it
    /// survives requantized ggufs and works on a fresh clone.
    fn builtin_hotlist(family: Family) -> Option<&'static str> {
        Some(match family {
            Family::Mla => include_str!("../hotlists/mla.hotlist"),
            Family::Gqa => include_str!("../hotlists/gqa.hotlist"),
            Family::Dsv4 => include_str!("../hotlists/dsv4.hotlist"),
            Family::Qwen35 => include_str!("../hotlists/qwen35.hotlist"),
            _ => return None,
        })
    }

    /// Parse a hotlist into the same offset-keyed heat map a census
    /// produces. Header `# n_layer=L n_expert=E` must match the model
    /// (a family can host several models; a mismatched seed is skipped
    /// rather than misapplied). Lines: `<layer> <expert> <count>` for
    /// routed experts, `<layer> s<idx> <count>` for sink/shexp banks.
    fn hotlist_heat(
        m: &Model,
        text: &str,
        heat: &mut std::collections::HashMap<u64, (u64, u64)>,
    ) {
        let mut lines = text.lines();
        let Some(header) = lines.next() else { return };
        let field = |k: &str| {
            header
                .split_whitespace()
                .find_map(|t| t.strip_prefix(k))
                .and_then(|v| v.parse::<u64>().ok())
        };
        // n_expert discriminates models within a family (GLM 256 vs Kimi
        // 384, qwen35 MoE 128 vs dense 1); per-line .get() bounds-checks
        // the layer index, so a stale n_layer only wastes lines.
        if field("n_expert=") != Some(m.shape.n_expert as u64) {
            return;
        }
        for line in lines {
            let mut it = line.split_whitespace();
            let (Some(l), Some(e), Some(c)) = (it.next(), it.next(), it.next()) else {
                continue;
            };
            let Ok(layer) = l.parse::<usize>() else { continue };
            let Ok(count) = c.parse::<u64>() else { continue };
            let Some(Ffn::Moe { gate_exps, up_exps, down_exps, sink, .. }) =
                m.layers.get(layer).map(|lw| &lw.ffn)
            else {
                continue;
            };
            let slabs: Option<[(u64, u64); 3]> = if let Some(se) = e.strip_prefix('s') {
                let Ok(idx) = se.parse::<u64>() else { continue };
                sink.as_ref().filter(|_| idx < m.shape.n_shexp_sink as u64).map(|sk| {
                    [&sk[0], &sk[1], &sk[2]]
                        .map(|t| (t.abs_offset + idx * t.expert_bytes, t.expert_bytes))
                })
            } else {
                let Ok(idx) = e.parse::<u64>() else { continue };
                (idx < m.shape.n_expert as u64).then(|| {
                    [gate_exps, up_exps, down_exps]
                        .map(|t| (t.abs_offset + idx * t.expert_bytes, t.expert_bytes))
                })
            };
            if let Some(slabs) = slabs {
                for (off, len) in slabs {
                    heat.insert(off, (count, len));
                }
            }
        }
    }

    /// Invert a model's census back to quantization-independent
    /// {layer, expert} heat and render it as hotlist text (the input
    /// format of `hotlist_heat`). Header-only: parses the gguf tensor
    /// table without touching the GPU, so it runs while a server owns
    /// the cards. Sink/shexp banks are not emitted (no census-bearing
    /// model uses them). Offline tool path: see `hotlist-gen`.
    pub fn hotlist_text(path: &Path) -> Result<String> {
        let (_shards, g) = parse_header(path)?;
        let meta_u = |k: &str| g.arch_meta(k).and_then(|v| v.as_u64());
        let n_expert = meta_u("expert_count").ok_or("hotlist: no expert_count")?;
        let n_layer = meta_u("block_count").ok_or("hotlist: no block_count")?;
        let census: std::collections::HashMap<u64, u64> =
            read_census(path).into_iter().map(|(off, _, count)| (off, count)).collect();
        if census.is_empty() {
            return Err("hotlist: no census (.warm) next to the model - run it once first".into());
        }
        let mut rows: Vec<(u64, String)> = Vec::new();
        for li in 0..n_layer {
            let slabs: Option<Vec<(u64, u64)>> = ["gate", "up", "down"]
                .iter()
                .map(|kind| {
                    let t = g.tensor(&format!("blk.{li}.ffn_{kind}_exps.weight"))?;
                    let eb = t.byte_size()? / n_expert;
                    Some((g.data_offset + t.offset, eb))
                })
                .collect();
            let Some(slabs) = slabs else { continue };
            for e in 0..n_expert {
                let h: u64 = slabs
                    .iter()
                    .filter_map(|(base, eb)| census.get(&(base + e * eb)))
                    .sum();
                if h > 0 {
                    rows.push((h, format!("{li} {e} {h}")));
                }
            }
        }
        rows.sort_unstable_by_key(|e| std::cmp::Reverse(e.0));
        rows.truncate(4096);
        let mut out = format!("# n_layer={n_layer} n_expert={n_expert}\n");
        for (_, line) in rows {
            out.push_str(&line);
            out.push('\n');
        }
        Ok(out)
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
                            let q0 = ((ql[l] & 0x0f) as i32 | ((qh[l] & 3) as i32) << 4) - 32;
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

        // F16 2D tensors -> q8_0: every dense-matmul consumer in the
        // engine reads q8_0 blocks, and raw f16 bytes fed to those
        // kernels decode as noise (poolside's Laguna ggufs ship ALL
        // attention projections in f16; same trap as the dspark drafts).
        // 1D f16 stays raw - callers that want it use read_f16_as_f32.
        // BF16 2D tensors get the SAME treatment: unsloth UD-* ships the
        // Q8_K_XL compressor/indexer projections as BF16, and raw BF16
        // bytes fed to q8_0 kernels are noise too (same 2-byte width,
        // different exponent bias - bf16 values read as f16 scales +
        // int8 quants decode to garbage and the model collapses at once).
        if (t.ty == TensorType::F16 || t.ty == TensorType::BF16) && t.dims.len() >= 2 {
            let n = t.n_elements() as usize;
            let mut out = Vec::with_capacity(n / 32 * 34);
            let mut f = vec![0f32; 32];
            for blk in buf.chunks_exact(64) {
                for (i, c) in blk.chunks_exact(2).enumerate() {
                    f[i] = if t.ty == TensorType::BF16 {
                        quant::bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))
                    } else {
                        requant::f16_to_f32(u16::from_le_bytes([c[0], c[1]]))
                    };
                }
                requant::quantize_q8_0(&f, &mut out);
            }
            if !n.is_multiple_of(32) {
                return Err(format!("{name}: f16 width not a multiple of 32").into());
            }
            return Ok(out);
        }
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

    /// Raw f16 tensor bytes -> device, no dequant (DFlash2 selector
    /// codebooks: gather tables read as __half by the kernel, kept f16
    /// to halve their 254MB VRAM cost). Bypasses read_tensor_bytes on
    /// purpose - its F16-2D branch requantizes to q8_0.
    fn upload_raw_f16(file: &VFile, g: &Gguf, name: &str) -> Result<DeviceBuf> {
        let t = g.tensor(name).ok_or_else(|| meta_err(name))?;
        if t.ty != TensorType::F16 {
            return Err(format!("{name}: expected f16, got {:?}", t.ty).into());
        }
        let mut buf = vec![0u8; t.n_elements() as usize * 2];
        file.read_exact_at(&mut buf, g.data_offset + t.offset)?;
        Ok(DeviceBuf::from_bytes(&buf)?)
    }

    /// FFN tensor-parallel shards (PULSAR_TP=1, dense qwen35): card B's
    /// halves. gate/up hold output rows [half..n_ff), down holds input
    /// blocks [half..) of every row repacked contiguously. Card A's
    /// Ffn::DenseKq holds the other halves; each card's down output is a
    /// partial sum and the eval adds them.
    struct TpFfnW {
        gate: KqW,
        up: KqW,
        down: KqW,
    }
    /// One layer's card-B shards: the FFN halves plus (phase 3) the GDN
    /// head halves, which reuse the A-side struct shape verbatim.
    struct TpLayerW {
        ffn: Option<TpFfnW>,
        gdn: Option<Qwen35Gdn>,
        attn: Option<Qwen35Attn>,
    }
    /// Third-card FFN peer (PULSAR_TP with 3+ devices). The GDN and
    /// attention halves stay on the A/B pair: only the FFN is split
    /// three ways, because it is the largest single weight block per
    /// layer and one code path, so the extra card's bandwidth joins the
    /// decode weight read without a third GDN state or KV shard.
    pub(crate) struct TpCW {
        pub(crate) dev: i32,
        pub(crate) width: u32,
        ffn: Vec<Option<TpFfnW>>,
    }

    pub(crate) struct TpW {
        pub(crate) dev: i32,
        /// per-card ffn width = n_ff_exp / 2
        pub(crate) half: u32,
        layers: Vec<Option<TpLayerW>>,
        /// card B's vocab rows [n_vocab/2..) of the lm head: the head is
        /// 2.06ms/token of serial card-A time (nsys), and the greedy /
        /// spec paths only need per-half argmaxes merged host-side
        head: Option<KqW>,
    }

    /// Raw K-quant tensor bytes + per-row size (shared by the whole and
    /// sliced uploads).
    fn read_kq_bytes(file: &VFile, g: &Gguf, name: &str) -> Result<(Vec<u8>, u64, u32)> {
        let t = g.tensor(name).ok_or_else(|| meta_err(name))?;
        let quant = quant_code(t.ty)
            .ok_or_else(|| format!("{name}: unsupported K-quant type {:?}", t.ty))?;
        let bytes = t.byte_size().ok_or_else(|| meta_err(name))?;
        let mut buf = vec![0u8; bytes as usize];
        file.read_exact_at(&mut buf, g.data_offset + t.offset)?;
        Ok((buf, t.ty.row_bytes(t.dims[0]).unwrap(), quant))
    }

    /// MatW built from ROW ranges of the tensor, on the CURRENT device
    /// (TP head/column slices). Follows keep_native so the slice rides
    /// the same kernel path the whole tensor would.
    fn matw_row_ranges(file: &VFile, g: &Gguf, name: &str, ranges: &[(u64, u64)]) -> Result<MatW> {
        let t = g.tensor(name).ok_or_else(|| meta_err(name))?;
        if MatW::keep_native(t) {
            let (bytes, rb, quant) = read_kq_bytes(file, g, name)?;
            let mut sl = Vec::new();
            for &(r0, r1) in ranges {
                sl.extend_from_slice(&bytes[(r0 * rb) as usize..(r1 * rb) as usize]);
            }
            Ok(MatW::Kq(KqW { w: DeviceBuf::from_bytes(&sl)?, row_bytes: rb, quant }))
        } else {
            // q8_0 rows (raw q8_0 or the F16/BF16 requant): 34 B / 32 elems
            let bytes = read_tensor_bytes(file, g, name)?;
            let rb = t.dims[0] as usize / 32 * 34;
            let mut sl = Vec::new();
            for &(r0, r1) in ranges {
                sl.extend_from_slice(&bytes[r0 as usize * rb..r1 as usize * rb]);
            }
            Ok(MatW::Q8(DeviceBuf::from_bytes(&sl)?))
        }
    }

    /// MatW with each output row's INPUT half repacked contiguously (TP
    /// row-parallel projections: ssm_out / attn_output / ffn_down). The
    /// halves are quant-block aligned by the TP gate's dim checks.
    fn matw_col_half(file: &VFile, g: &Gguf, name: &str, second: bool) -> Result<MatW> {
        let t = g.tensor(name).ok_or_else(|| meta_err(name))?;
        let native = MatW::keep_native(t);
        let (bytes, rb, quant) = if native {
            read_kq_bytes(file, g, name)?
        } else {
            let b = read_tensor_bytes(file, g, name)?;
            let rb = t.dims[0] as u64 / 32 * 34;
            (b, rb, 0)
        };
        let rb = rb as usize;
        let hb = rb / 2;
        let n = bytes.len() / rb;
        let mut sl = Vec::with_capacity(n * hb);
        for r in 0..n {
            let row = &bytes[r * rb..(r + 1) * rb];
            sl.extend_from_slice(if second { &row[hb..] } else { &row[..hb] });
        }
        if native {
            Ok(MatW::Kq(KqW { w: DeviceBuf::from_bytes(&sl)?, row_bytes: hb as u64, quant }))
        } else {
            Ok(MatW::Q8(DeviceBuf::from_bytes(&sl)?))
        }
    }

    /// MatW with an arbitrary set of INPUT column ranges repacked
    /// contiguously per output row (GDN TP: the v heads a card owns are
    /// interleaved, not a contiguous half). Ranges are in input values,
    /// quant-block aligned by construction (multiples of ssm_state).
    fn matw_col_ranges(file: &VFile, g: &Gguf, name: &str, cols: &[(u64, u64)]) -> Result<MatW> {
        let t = g.tensor(name).ok_or_else(|| meta_err(name))?;
        let in_dim = t.dims[0] as u64;
        let native = MatW::keep_native(t);
        let (bytes, rb, quant) = if native {
            read_kq_bytes(file, g, name)?
        } else {
            let b = read_tensor_bytes(file, g, name)?;
            (b, in_dim / 32 * 34, 0)
        };
        let rb = rb as usize;
        // value index -> byte offset inside a row; exact for block-aligned
        // ranges (256-value superblocks for K-quants, 32 for q8_0)
        let bo = |v: u64| -> usize { (v as u128 * rb as u128 / in_dim as u128) as usize };
        let hb: usize = cols.iter().map(|&(a, b)| bo(b) - bo(a)).sum();
        let n = bytes.len() / rb;
        let mut sl = Vec::with_capacity(n * hb);
        for r in 0..n {
            let row = &bytes[r * rb..(r + 1) * rb];
            for &(a, b) in cols {
                sl.extend_from_slice(&row[bo(a)..bo(b)]);
            }
        }
        if native {
            Ok(MatW::Kq(KqW { w: DeviceBuf::from_bytes(&sl)?, row_bytes: hb as u64, quant }))
        } else {
            Ok(MatW::Q8(DeviceBuf::from_bytes(&sl)?))
        }
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
    /// weights as f16; small ones convert to f32 for matmul_f32). F32
    /// tensors pass through (unsloth UD-* ships the same aux weights f32).
    /// BF16 is the same two bytes with the other exponent bias, so it gets
    /// its own decode rather than being read as f16 - unsloth UD-Q8_K_XL
    /// ships blk.N.indexer.proj.weight that way (issue #21).
    fn read_f16_as_f32(file: &VFile, g: &Gguf, name: &str) -> Result<Vec<f32>> {
        let t = g.tensor(name).ok_or_else(|| meta_err(name))?;
        let n = t.n_elements() as usize;
        match t.ty {
            TensorType::F16 | TensorType::BF16 => {
                let mut buf = vec![0u8; n * 2];
                file.read_exact_at(&mut buf, g.data_offset + t.offset)?;
                let bf16 = t.ty == TensorType::BF16;
                Ok(buf
                    .chunks_exact(2)
                    .map(|c| {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        if bf16 {
                            quant::bf16_to_f32(bits)
                        } else {
                            requant::f16_to_f32(bits)
                        }
                    })
                    .collect())
            }
            TensorType::F32 => {
                let mut buf = vec![0u8; n * 4];
                file.read_exact_at(&mut buf, g.data_offset + t.offset)?;
                Ok(buf
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect())
            }
            other => Err(format!("{name}: expected f16/bf16/f32, got {other:?}").into()),
        }
    }

    fn upload_f16_as_f32(file: &VFile, g: &Gguf, name: &str) -> Result<DeviceBuf> {
        Ok(DeviceBuf::from_f32(&read_f16_as_f32(file, g, name)?)?)
    }

    /// Tensor -> device f32 regardless of source encoding. Small tensors
    /// whose consumers are matmul_f32 (qwen35 ssm_alpha/ssm_beta): dense
    /// 27B files K-quantize them where the 35B shipped f32.
    /// Any small tensor -> host f32 (the decode half of upload_as_f32;
    /// the TP loader slices these row-wise before upload).
    fn read_any_f32(file: &VFile, g: &Gguf, name: &str) -> Result<Vec<f32>> {
        let t = g.tensor(name).ok_or_else(|| meta_err(name))?;
        match t.ty {
            TensorType::F32 | TensorType::F16 | TensorType::BF16 => read_f16_as_f32(file, g, name),
            TensorType::Q4K => {
                let n = t.n_elements() as usize;
                let mut buf = vec![0u8; n / 256 * 144];
                file.read_exact_at(&mut buf, g.data_offset + t.offset)?;
                Ok(quant::cpu_dot::dequant_q4_k(&buf, n))
            }
            TensorType::Q8_0 => {
                let n = t.n_elements() as usize;
                let bytes = t.byte_size().ok_or_else(|| meta_err(name))? as usize;
                if !n.is_multiple_of(32) {
                    return Err(format!("{name}: q8_0 width {n} not a multiple of 32").into());
                }
                let mut buf = vec![0u8; bytes];
                file.read_exact_at(&mut buf, g.data_offset + t.offset)?;
                Ok(quant::cpu_dot::dequant_q8_0(&buf, n))
            }
            // NVFP4 aux tensors (ssm_alpha/ssm_beta on the NVFP4
            // Qwen3.8 ggufs feed matmul_f32): mirror of ggml
            // dequantize_row_nvfp4 - 64-value blocks, four UE4M3-scaled
            // 16-value sub-blocks, kvalues_mxfp4 codebook (2x e2m1, the
            // ue4m3 decode returns value*0.5 to compensate)
            TensorType::NVFP4 => {
                let n = t.n_elements() as usize;
                let bytes = t.byte_size().ok_or_else(|| meta_err(name))? as usize;
                let mut buf = vec![0u8; bytes];
                file.read_exact_at(&mut buf, g.data_offset + t.offset)?;
                const KV: [f32; 16] = [
                    0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0,
                    0.0, -1.0, -2.0, -3.0, -4.0, -6.0, -8.0, -12.0,
                ];
                let ue4m3 = |x: u8| -> f32 {
                    if x == 0 || x == 0x7F {
                        return 0.0;
                    }
                    let exp = (x >> 3) & 0xF;
                    let man = f32::from(x & 7);
                    let raw = if exp == 0 {
                        man * 2f32.powi(-9)
                    } else {
                        (1.0 + man / 8.0) * 2f32.powi(i32::from(exp) - 7)
                    };
                    raw * 0.5
                };
                let mut out = Vec::with_capacity(n);
                for blk in buf.chunks_exact(36) {
                    for sub in 0..4 {
                        let d = ue4m3(blk[sub]);
                        let qs = &blk[4 + sub * 8..4 + sub * 8 + 8];
                        let mut vals = [0f32; 16];
                        for (j, &b) in qs.iter().enumerate() {
                            vals[j] = KV[(b & 0xF) as usize] * d;
                            vals[j + 8] = KV[(b >> 4) as usize] * d;
                        }
                        out.extend_from_slice(&vals);
                    }
                }
                out.truncate(n);
                Ok(out)
            }
            other => Err(format!("{name}: no f32 path for {other:?}").into()),
        }
    }

    fn upload_as_f32(file: &VFile, g: &Gguf, name: &str) -> Result<DeviceBuf> {
        let t = g.tensor(name).ok_or_else(|| meta_err(name))?;
        match t.ty {
            TensorType::F32 => upload(file, g, name),
            TensorType::F16 => upload_f16_as_f32(file, g, name),
            // unsloth UD-* ships the deepseek4 router (ffn_gate_inp) as
            // BF16; matmul_f32 needs f32 host data. quant::row_to_f32
            // already has the bf16->f32 decode.
            TensorType::BF16 => {
                let n = t.n_elements() as usize;
                let mut buf = vec![0u8; n * 2];
                file.read_exact_at(&mut buf, g.data_offset + t.offset)?;
                let mut f = Vec::with_capacity(n);
                quant::row_to_f32(TensorType::BF16, &buf, &mut f)?;
                Ok(DeviceBuf::from_f32(&f)?)
            }
            TensorType::Q4K => {
                let n = t.n_elements() as usize;
                let mut buf = vec![0u8; n / 256 * 144];
                file.read_exact_at(&mut buf, g.data_offset + t.offset)?;
                Ok(DeviceBuf::from_f32(&quant::cpu_dot::dequant_q4_k(&buf, n))?)
            }
            // pulsar-quant falls back to q8_0 for rows that are a multiple
            // of 32 but not 256, which is how ssm_alpha/ssm_beta arrive on
            // some GDN models. Dequantize rather than refuse the file.
            TensorType::Q8_0 => {
                let n = t.n_elements() as usize;
                // ask the type for its own size rather than hand-rolling
                // 34-bytes-per-32; a width that is not a multiple of 32
                // would otherwise short-read
                let bytes = t.byte_size().ok_or_else(|| meta_err(name))? as usize;
                if !n.is_multiple_of(32) {
                    return Err(format!("{name}: q8_0 width {n} not a multiple of 32").into());
                }
                let mut buf = vec![0u8; bytes];
                file.read_exact_at(&mut buf, g.data_offset + t.offset)?;
                Ok(DeviceBuf::from_f32(&quant::cpu_dot::dequant_q8_0(&buf, n))?)
            }
            other => Err(format!("{name}: no f32 path for {other:?}").into()),
        }
    }

    /// f16 tensor -> q8_0 bytes (deepseek4's bigger f16 matmul weights
    /// ride the q8_0 fast path; ~0.4% quantization noise). Q8_0 tensors
    /// pass through (unsloth UD-* ships the compressor/indexer projections
    /// already q8_0; re-quantizing would be lossy AND wrong-layout).
    /// BF16 decodes like f16 but with the other exponent bias: newer
    /// unsloth UD-Q8_K_XL deepseek4 shards carry attn_compressor_kv as
    /// BF16, and reading those bytes as f16 is silent garbage.
    fn read_f16_as_q8(file: &VFile, g: &Gguf, name: &str) -> Result<Vec<u8>> {
        let t = g.tensor(name).ok_or_else(|| meta_err(name))?;
        let n = t.n_elements() as usize;
        match t.ty {
            TensorType::F16 | TensorType::BF16 => {
                let mut buf = vec![0u8; n * 2];
                file.read_exact_at(&mut buf, g.data_offset + t.offset)?;
                let mut out = Vec::with_capacity(n / 32 * 34);
                let mut f = [0f32; 256];
                for blk in buf.chunks(512) {
                    let m = blk.len() / 2;
                    for (i, c) in blk.chunks_exact(2).enumerate() {
                        let bits = u16::from_le_bytes([c[0], c[1]]);
                        f[i] = if t.ty == TensorType::BF16 {
                            quant::bf16_to_f32(bits)
                        } else {
                            requant::f16_to_f32(bits)
                        };
                    }
                    requant::quantize_q8_0(&f[..m], &mut out);
                }
                Ok(out)
            }
            TensorType::Q8_0 => {
                let mut buf = vec![0u8; t.byte_size().ok_or_else(|| meta_err(name))? as usize];
                file.read_exact_at(&mut buf, g.data_offset + t.offset)?;
                Ok(buf)
            }
            other => Err(format!("{name}: expected f16/bf16/q8_0, got {other:?}").into()),
        }
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
        /// Card-B GDN head shard for exec layer `il` (PULSAR_TP); None
        /// when TP is off or the layer is full attention.
        pub(crate) fn tp_gdn(&self, il: usize) -> Option<&Qwen35Gdn> {
            self.tp.as_ref()?.layers.get(il)?.as_ref()?.gdn.as_ref()
        }
        /// Card-B attention head shard for exec layer `il` (PULSAR_TP).
        pub(crate) fn tp_attn(&self, il: usize) -> Option<&Qwen35Attn> {
            self.tp.as_ref()?.layers.get(il)?.as_ref()?.attn.as_ref()
        }
        /// Third-card FFN shard for exec layer `il` (PULSAR_TP, 3+ cards).
        pub(crate) fn tp_ffn3(&self, il: usize) -> Option<&TpFfnW> {
            self.tp_c.as_ref()?.ffn.get(il)?.as_ref()
        }
        pub fn load(path: &Path) -> Result<Model> {
            let (shards, gguf) = parse_header(path)?;
            let file = VFile::open(&shards)?;
            let shape = Shape::from_gguf(&gguf)?;

            // A model bigger than every card combined (or a single-card
            // run, where the dense-split planner never engages) must
            // degrade to pinned host RAM, not die at cudaMalloc: budget
            // each card's free VRAM minus a 3GiB reserve for
            // KV/activations/preq scratch/graph pools (state allocs come
            // AFTER weights and cannot pin). Spent only by weight
            // uploads; multi-GPU plans that already fit never touch the
            // ceiling, so placement there is unchanged.
            // Establish the measured primary FIRST: mem_info saves and
            // restores the current device, and before the first ensure
            // that is CUDA's default (0) - budgeting here would silently
            // re-pin every later allocation to device 0 even when the
            // probe picked a different primary.
            kernels::set_device(kernels::primary_device())?;
            for d in 0..kernels::device_count() {
                if let Ok((free, _)) = kernels::mem_info(d) {
                    kernels::set_weight_vram_budget(d, free as i64 - (3i64 << 30));
                }
            }

            // the embedding table is read ~one row per token - pinned
            // host is free for it and returns ~1GB of VRAM to hot weights
            let token_embd = {
                // deepseek4 embd arrives F16 (antirez ds4 recipe) or Q5_K
                // (unsloth UD-*); both convert to q8_0 for embed_q8_0.
                let bytes = read_tensor_bytes(&file, &gguf, "token_embd.weight")?;
                if matches!(shape.family, Family::Mla | Family::Dsv4) {
                    let mut buf = DeviceBuf::alloc_pinned(bytes.len())?;
                    buf.write(0, &bytes)?;
                    buf
                } else {
                    // budgeted weight path: VRAM while it lasts, pinned on
                    // overflow (a row-per-token read - pinned costs nothing)
                    DeviceBuf::from_bytes(&bytes)?
                }
            };
            let output_norm = upload(&file, &gguf, "output_norm.weight")?;
            // K3 mixes the banked AttnRes checkpoints one last time before
            // the head; absent on every other family.
            let output_res_score = if gguf.tensor("output_res_score.weight").is_some() {
                Some(upload(&file, &gguf, "output_res_score.weight")?)
            } else {
                None
            };
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
                // K3 joins Mla here for the same reason: its non-expert
                // stack is far too big for one card, so the per-layer
                // planner has to be allowed to spread it.
                Family::Mla | Family::K3 => match std::env::var("PULSAR_ATTN_GPU").ok().as_deref() {
                    Some("off") | Some("-1") => None,
                    Some(v) => v.trim().parse::<i32>().ok().filter(|&d| {
                        let ok = d != primary && d >= 0 && d < kernels::device_count();
                        if !ok {
                            eprintln!("pulsar: ignoring PULSAR_ATTN_GPU={d} (primary is {primary}, {} devices)", kernels::device_count());
                        }
                        ok
                    }),
                    // auto: the layer-split planner below assigns per-layer
                    // owners across every secondary; None here means "let
                    // the planner decide", not "no offload"
                    None => None,
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
            // ---- MLA layer-split planner: per-layer attention ownership.
            // Contiguous layer ranges spread across secondaries in
            // proportion to their post-reserve free VRAM, so each card's
            // KV/ctx headroom comes out even. Degenerates to one card (=
            // the old auto-detect) when one secondary holds everything
            // with headroom to spare, and to all-primary when there are no
            // secondaries. PULSAR_ATTN_GPU=<d> still forces one card,
            // =off forces primary.
            let n_attn_slots = shape.n_exec_layer as usize + 1; // + MTP draft slot
            let mut attn_layer_dev: Vec<i32> = vec![attn_dev.unwrap_or(primary); n_attn_slots];
            if matches!(shape.family, Family::Mla | Family::K3)
                && attn_dev.is_none()
                && std::env::var("PULSAR_ATTN_GPU").ok().as_deref().is_none_or(|v| v != "off" && v != "-1")
            {
                // K3 weighs exactly what the loader puts on the layer's
                // card: everything built inside the Attn::K3 arm. The
                // shared experts, the router and the dense FFN are
                // uploaded after the device is restored, so they live on
                // the primary and counting them here plans a split that
                // pushes layers onto an already-loaded primary until it
                // OOMs (measured: 10 layers overflowed, cudaMalloc died
                // 42s into the load).
                const MLA_SUF: &[&str] = &[
                    "attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight",
                    "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight",
                    "attn_k_b.weight", "attn_v_b.weight", "attn_output.weight",
                    "indexer.attn_q_b.weight", "indexer.attn_k.weight",
                    "indexer.k_norm.weight", "indexer.k_norm.bias",
                    "indexer.proj.weight",
                ];
                const K3_SUF: &[&str] = &[
                    "attn_q.weight", "attn_k.weight", "attn_v.weight",
                    "ssm_conv1d_q.weight", "ssm_conv1d_k.weight", "ssm_conv1d_v.weight",
                    "ssm_f_a.weight", "ssm_f_b.weight", "ssm_beta.weight",
                    "ssm_a", "ssm_dt.bias", "ssm_g.weight", "ssm_norm.weight",
                    "attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight",
                    "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight",
                    "attn_k_b.weight", "attn_v_b.weight", "attn_gate.weight",
                    "attn_output.weight",
                    // rides the layer's card (see the shexp load below)
                    "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
                ];
                let sufs = if shape.family == Family::K3 { K3_SUF } else { MLA_SUF };
                let mut lb = vec![0u64; shape.n_exec_layer as usize];
                for il in 0..shape.n_exec_layer {
                    for suf in sufs {
                        if let Some(ti) = gguf.tensor(&format!("blk.{il}.{suf}")) {
                            lb[il as usize] += ti.byte_size().unwrap_or(0);
                        }
                    }
                }
                let mut cards: Vec<(usize, i32)> = (0..kernels::device_count())
                    .filter(|&d| d != primary)
                    .filter_map(|d| kernels::mem_info(d).ok().map(|(f, _)| (f, d)))
                    .collect();
                cards.sort_unstable_by_key(|e| std::cmp::Reverse(e.0));
                // CUDA context + per-card scratch/hop buffers + slack; KV
                // and idx cache sizes are ctx-dependent and resolved at
                // State::new, which fails with a clear message when the
                // requested ctx outgrows the per-card leftover.
                let reserve: u64 = 5 << 28;
                let caps: Vec<u64> = cards
                    .iter()
                    .map(|&(f, _)| (f as u64).saturating_sub(reserve))
                    .collect();
                let total_w: u64 = lb.iter().sum();
                let total_cap: u64 = caps.iter().sum();
                if !cards.is_empty() && total_cap > 0 {
                    let spare = total_cap.saturating_sub(total_w);
                    let mut il = 0usize;
                    for (ci, &(_, d)) in cards.iter().enumerate() {
                        // weight share = cap minus this card's even slice
                        // of the spare headroom; last card sweeps rounding.
                        // u128: spare * cap overflows u64 at ~16GB * 16GB
                        let share = caps[ci]
                            .saturating_sub((spare as u128 * caps[ci] as u128 / total_cap as u128) as u64);
                        let mut used = 0u64;
                        while il < lb.len()
                            && (used + lb[il] <= share
                                || (ci == cards.len() - 1 && used + lb[il] <= caps[ci]))
                        {
                            used += lb[il];
                            attn_layer_dev[il] = d;
                            il += 1;
                        }
                        if il >= lb.len() {
                            break;
                        }
                    }
                    if il < lb.len() {
                        eprintln!(
                            "pulsar: attn split: {} layers overflow every secondary and stay on the primary",
                            lb.len() - il
                        );
                    }
                }
                // MTP draft slot runs right after the last layer
                attn_layer_dev[shape.n_exec_layer as usize] =
                    attn_layer_dev[(shape.n_exec_layer as usize).saturating_sub(1)];
            }
            let attn_dev = attn_layer_dev.iter().find(|&&d| d != primary).copied();
            if attn_dev.is_some() {
                // banner: one line per contiguous range
                let mut i = 0usize;
                while i < attn_layer_dev.len() - 1 {
                    let d = attn_layer_dev[i];
                    let mut j = i;
                    while j + 1 < attn_layer_dev.len() - 1 && attn_layer_dev[j + 1] == d {
                        j += 1;
                    }
                    eprintln!(
                        "pulsar: attn layers {i}..={j} resident on CUDA device {d}{}",
                        if d == primary { " (primary)" } else { "" }
                    );
                    i = j + 1;
                }
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
            // FFN tensor parallel (PULSAR_TP=1, dense qwen35, 2+ cards):
            // both cards read their FFN half of EVERY layer in parallel,
            // where the dense split reads each card's layers in sequence.
            // The FFN is ~57% of the per-token bytes on a dense 27B, so
            // this is most of TP's win at none of the KV/GDN sharding
            // cost; attention stays whole on the primary, and the dense
            // split is skipped (all layers owned by the primary).
            let mut tp3_cfg: Option<(i32, u32, u32)> = None;
            let tp_cfg: Option<(i32, u32)> = if qwen35_dense
                && kernels::device_count() > 1
                && std::env::var("PULSAR_TP").ok().as_deref() == Some("1")
                && shape.n_ff_exp.is_multiple_of(512)
                && shape.ssm_v_heads.is_multiple_of(2)
                && shape.ssm_k_heads.is_multiple_of(2)
                && (shape.ssm_k_heads * shape.ssm_state).is_multiple_of(512)
                && shape.n_head.is_multiple_of(2)
                && shape.n_head_kv.is_multiple_of(2)
                && (shape.n_head / 2 * shape.head_dim).is_multiple_of(256)
                // turbo KV rotates K/Q through a primary-resident Pi that
                // the B-side attention half has no copy of
                && !std::env::var("PULSAR_KV").ok().is_some_and(|v| v.starts_with("turbo"))
            {
                let dev_b = (0..kernels::device_count())
                    .filter(|&d| d != primary)
                    .max_by(|&a, &b| {
                        kernels::vram_bandwidth(a)
                            .partial_cmp(&kernels::vram_bandwidth(b))
                            .unwrap_or(std::cmp::Ordering::Equal)
                    })
                    .unwrap();
                // Split the FFN by MEASURED VRAM BANDWIDTH, not evenly.
                // Every sublayer is a barrier: both cards read their
                // shard's weights and the layer advances when the SLOWER
                // one finishes, so an even split on uneven cards runs the
                // whole model at the slow card's rate. Measured here with
                // nsys: the 4060 Ti moved 275 GB/s of its 288 GB/s peak
                // while the 5060 Ti idled at 61% of its own.
                //
                // A's share must stay a multiple of PULSAR_QK_K: ffn_down
                // splits along the CONTRACTION dim, so the cut has to land
                // on a quant block boundary. Clamped to [1/4, 3/4] so a
                // bogus bandwidth probe cannot starve a card.
                let qk = 256u32;
                let (ba, bb) = (kernels::vram_bandwidth(primary), kernels::vram_bandwidth(dev_b));
                let frac = if ba > 0.0 && bb > 0.0 { ba / (ba + bb) } else { 0.5 };
                let n_ff = shape.n_ff_exp;
                // round to the NEAREST block: truncating hands the
                // other card a whole extra block, which on a pair of
                // identical cards is pure straggler time (measured 47.8
                // -> 46.2 tok/s from one block of skew)
                let half = ((n_ff as f64 * frac / qk as f64).round() as u32 * qk)
                    .clamp(n_ff / 4 / qk * qk, n_ff * 3 / 4 / qk * qk);
                // Third card joins the FFN split when one exists and is
                // not being used for a PP tail. Shares stay proportional
                // to read bandwidth across all three.
                let third_ffn: Option<(i32, u32, u32)> = if std::env::var("PULSAR_TP3").ok().as_deref() == Some("off") {
                    None
                } else {
                    (0..kernels::device_count())
                        .filter(|&d| d != primary && d != dev_b)
                        .max_by(|&a, &b| {
                            kernels::vram_bandwidth(a)
                                .partial_cmp(&kernels::vram_bandwidth(b))
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .and_then(|d3| {
                            let b3 = kernels::vram_bandwidth(d3);
                            if b3 <= 0.0 || ba <= 0.0 || bb <= 0.0 {
                                return None;
                            }
                            let tot = ba + bb + b3;
                            let blocks = n_ff / qk;
                            let wa = ((blocks as f64 * ba / tot).round() as u32).max(1);
                            let wb = ((blocks as f64 * bb / tot).round() as u32).max(1);
                            if wa + wb >= blocks {
                                return None;
                            }
                            Some((d3, wa * qk, wb * qk))
                        })
                };
                if let Some((d3, wa, wb)) = third_ffn {
                    eprintln!(
                        "pulsar: FFN tensor parallel on devices {primary}+{dev_b}+{d3} ({wa}/{wb}/{} split, {:.0}/{:.0}/{:.0} GB/s)",
                        n_ff - wa - wb, ba, bb, kernels::vram_bandwidth(d3)
                    );
                } else {
                    eprintln!(
                        "pulsar: FFN tensor parallel on devices {primary}+{dev_b} ({half}/{} split, {:.0}/{:.0} GB/s)",
                        n_ff - half, ba, bb
                    );
                }
                tp3_cfg = third_ffn;
                Some((dev_b, third_ffn.map_or(half, |(_, wa, _)| wa)))
            } else {
                None
            };
            let tp_ffn_stash: std::cell::RefCell<Option<TpFfnW>> = std::cell::RefCell::new(None);
            let tp_ffn3_stash: std::cell::RefCell<Option<TpFfnW>> = std::cell::RefCell::new(None);
            let mut tp3_layers: Vec<Option<TpFfnW>> = Vec::new();
            let tp_gdn_stash: std::cell::RefCell<Option<Qwen35Gdn>> = std::cell::RefCell::new(None);
            let tp_attn_stash: std::cell::RefCell<Option<Qwen35Attn>> = std::cell::RefCell::new(None);
            let mut layer_dev = vec![primary; shape.n_exec_layer as usize];
            // TP + PP: TP splits each layer across the fast pair, so the
            // dense split used to be mutually exclusive with it and any
            // third card sat idle. A third device can still take a TAIL
            // of whole layers as a pipeline stage - those layers get no
            // TP weights (tp_attn/tp_gdn/tp_ffn are per-layer Options and
            // return None), run the plain path on their owner, and their
            // KV allocates there too via the dense_split branch in
            // State::new. That is the only way a third card of a
            // different class buys context: KV must live where its
            // attention runs, so whole layers move, not caches.
            // PULSAR_PP=<n> sets the tail length, =off disables.
            if let Some((dev_b, _)) = tp_cfg {
                let pp = std::env::var("PULSAR_PP").ok();
                if pp.as_deref() != Some("off") {
                    if let Some(third) = (0..kernels::device_count())
                        .filter(|&d| d != primary && d != dev_b)
                        .max_by(|&a, &b| {
                            kernels::vram_bandwidth(a)
                                .partial_cmp(&kernels::vram_bandwidth(b))
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                    {
                        // Per-layer bytes are ~uniform on this family, so
                        // size the tail from the third card's free VRAM
                        // rather than a fixed fraction: heterogeneous rigs
                        // are the point. Reserve covers its KV share plus
                        // prefill scratch.
                        let n_layers = shape.n_exec_layer as usize;
                        let mut n_tail = pp
                            .as_deref()
                            .and_then(|v| v.parse::<usize>().ok())
                            .unwrap_or(0);
                        // A tail costs decode: its layers read their
                        // weights from ONE card while the TP layers read
                        // theirs from two in parallel, and layers are
                        // sequential within a token. Measured on the 27B
                        // NVFP4 (2x 5060 Ti + 4060 Ti): 36.5 tok/s with a
                        // 32-layer tail vs 47.8 without, same prefill and
                        // same 196608 max ctx, because the KV auto-sizer
                        // absorbs the smaller budget by quantizing. So
                        // only spill to the third card when the TP pair
                        // genuinely cannot hold the weights; PULSAR_PP=<n>
                        // still forces a tail when capacity beats speed.
                        let fits_on_pair = {
                            let total: u64 = gguf
                                .tensors
                                .iter()
                                .filter_map(|t| t.byte_size())
                                .sum();
                            // TP halves the per-layer matrices, so each of
                            // the pair holds ~half; leave room for KV and
                            // the prefill hop/scratch buffers
                            let need = total / 2 + (4u64 << 30);
                            [primary, dev_b].iter().all(|&d| {
                                kernels::mem_info(d).map_or(false, |(free, _)| free as u64 >= need)
                            })
                        };
                        if n_tail == 0 && fits_on_pair {
                            eprintln!(
                                "pulsar: no pipeline tail: weights fit the TP pair, keeping every layer 2-way (PULSAR_PP=<n> to spill onto device {third} for more KV room)"
                            );
                        } else if n_tail == 0 {
                            // layer 0's own tensors ARE the per-layer cost
                            // on this family (uniform blocks); no averaging
                            // over a total that also carries embd/head
                            let per: u64 = gguf
                                .tensors
                                .iter()
                                .filter(|t| t.name.starts_with("blk.0."))
                                .filter_map(|t| t.byte_size())
                                .sum();
                            if let (Ok((free, _)), true) = (kernels::mem_info(third), per > 0) {
                                let reserve = 3u64 << 30;
                                let usable = (free as u64).saturating_sub(reserve);
                                n_tail = (usable / per) as usize;
                            }
                        }
                        // keep TP meaningful: never hand over more than
                        // half the stack
                        n_tail = n_tail.min(n_layers / 2);
                        if n_tail > 0 {
                            for d in layer_dev.iter_mut().skip(n_layers - n_tail) {
                                *d = third;
                            }
                            eprintln!(
                                "pulsar: pipeline tail: layers {}..{} on device {third} (TP {primary}+{dev_b} for 0..{})",
                                n_layers - n_tail,
                                n_layers,
                                n_layers - n_tail
                            );
                        }
                    }
                }
            }
            if qwen35_dense
                && tp_cfg.is_none()
                && kernels::device_count() > 1
                && std::env::var("PULSAR_SPLIT").ok().as_deref() != Some("off")
            {
                // Fastest VRAM wins, free bytes only break ties: the
                // second card serves resident weights from its own VRAM
                // every token, and picking by free space sent half of
                // Qwen3.8-27B to the 288GB/s 4060 Ti over a 69MiB free
                // difference while a 448GB/s 5060 Ti sat empty.
                let second = (0..kernels::device_count())
                    .filter(|&d| d != primary)
                    .max_by(|&a, &b| {
                        kernels::vram_bandwidth(a)
                            .partial_cmp(&kernels::vram_bandwidth(b))
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then_with(|| {
                                let free = |d| kernels::mem_info(d).map(|(f, _)| f).unwrap_or(0);
                                free(a).cmp(&free(b))
                            })
                    })
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
                // PULSAR_SPLIT=<n> pins the primary's layer count;
                // PULSAR_SPLIT=<n0>,<n1>,... pins EVERY card's count
                // (primary first, then secondaries fastest-first). The
                // list form exists because the auto plan sizes by WEIGHT
                // bytes only - it cannot see ctx, which is a State
                // parameter chosen after load - so it packs the fewest
                // cards and then KV OOMs at long ctx. Spreading layers
                // wider spreads their KV with them.
                let split_env = std::env::var("PULSAR_SPLIT").ok();
                let split_list: Vec<usize> = split_env
                    .as_deref()
                    .map(|v| v.split(',').filter_map(|x| x.trim().parse::<usize>().ok()).collect())
                    .unwrap_or_default();
                if let Some(&n) = split_list.first() {
                    n0 = n.min(lbytes.len());
                }
                // Spread the remainder over EVERY secondary, fastest
                // first, each filled to its own free VRAM. Two-card was
                // the old shape and it stranded a third card entirely;
                // rigs running pulsar are heterogeneous, and a slow card
                // still contributes capacity (its layers' KV allocates
                // there too, which is what raises the context ceiling).
                // Layers run sequentially within a token, so filling the
                // fast cards first minimizes sum(bytes/bw), same
                // reasoning as the primary fill above.
                let mut secs: Vec<i32> = (0..kernels::device_count())
                    .filter(|&d| d != primary)
                    .collect();
                secs.sort_by(|&a, &b| {
                    kernels::vram_bandwidth(b)
                        .partial_cmp(&kernels::vram_bandwidth(a))
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut i = n0;
                let n_sec = secs.len();
                for (si, d) in secs.iter().copied().enumerate() {
                    if i >= lbytes.len() {
                        break;
                    }
                    let free = kernels::mem_info(d).map(|(f, _)| f as u64).unwrap_or(0);
                    let reserve = 2u64 << 30;
                    let mut take = 0usize;
                    let mut used = 0u64;
                    while i + take < lbytes.len()
                        && used + lbytes[i + take] + reserve <= free
                    {
                        used += lbytes[i + take];
                        take += 1;
                    }
                    if let Some(&n) = split_list.get(si + 1) {
                        take = n.min(lbytes.len() - i);
                    } else if si + 1 == n_sec {
                        // last card swallows any remainder: leaving layers
                        // unassigned would silently keep them on an already
                        // full primary and die at cudaMalloc instead
                        take = lbytes.len() - i;
                    }
                    for k in 0..take {
                        layer_dev[i + k] = d;
                    }
                    if take > 0 {
                        let b: u64 = lbytes[i..i + take].iter().sum();
                        eprintln!(
                            "pulsar: dense split: layers {}..{} on device {d} ({:.1}GiB)",
                            i,
                            i + take,
                            b as f64 / GIB
                        );
                    }
                    i += take;
                }
                let b0: u64 = lbytes[..n0].iter().sum();
                eprintln!(
                    "pulsar: dense split: layers 0..{n0} on device {primary} ({:.1}GiB)",
                    b0 as f64 / GIB
                );
                let _ = second;
            }

            // Mla: spend a VRAM budget on the two big per-layer attn
            // tensors (attn_output ~107MB, q_b ~36MB on GLM-5.2) - they are
            // 80%+ of the per-token pinned-host read traffic. Gqa attn is
            // small enough to always live in VRAM. With a dedicated attn
            // GPU the whole stack (~14GB q8) goes resident by default -
            // pinned overflow would be read over that card's own link.
            let gemma_arch = gguf.architecture() == Some("gemma4");
            let ink_arch = gguf.architecture() == Some("inkling");
            let laguna_arch = gguf.architecture() == Some("laguna");
            let oss_arch = gguf.architecture() == Some("gpt-oss");
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
                            n_head_q: 0,
                            n_head_kv: kvh
                                .get(il)
                                .copied()
                                .unwrap_or(shape.n_head_kv as u64)
                                as u32,
                            head_dim: shape.head_dim,
                            theta: 0.0,
                            window: if swa { window } else { 0 },
                            factors: false,
                            rot: 0,
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
                            n_head_q: 0,
                            n_head_kv: kvh.get(il).copied().unwrap_or(1) as u32,
                            head_dim: if swa { hd_swa } else { hd_full },
                            theta: if swa { theta_swa } else { theta_full },
                            window: if swa { window } else { 0 },
                            factors: !swa,
                            rot: 0,
                        }
                    })
                    .collect()
            } else if laguna_arch {
                // Laguna interleaves one full-attention layer every 4th
                // with sliding ones, and varies the QUERY head count per
                // layer (48 full / 72 sliding) - unlike gemma, whose
                // per-layer array is head_count_kv. n_head_kv is constant
                // (8); the per-layer truth we need is the q head count,
                // carried in Geom::n_head_q.
                let arr_u = |k: &str| -> Vec<u64> {
                    match gguf.arch_meta(k) {
                        Some(Value::Array(a)) => a.iter().filter_map(Value::as_u64).collect(),
                        Some(v) => v.as_u64().map(|x| vec![x]).unwrap_or_default(),
                        None => Vec::new(),
                    }
                };
                let heads = arr_u("attention.head_count");
                let kvh = arr_u("attention.head_count_kv");
                let g_u = |k: &str, d: u32| -> u32 {
                    gguf.arch_meta(k).and_then(Value::as_u64).map(|v| v as u32).unwrap_or(d)
                };
                let g_f = |k: &str, d: f32| -> f32 {
                    gguf.arch_meta(k).and_then(Value::as_f32).unwrap_or(d)
                };
                let window = g_u("attention.sliding_window", 512);
                let theta_full = g_f("rope.freq_base", 10_000_000.0);
                let theta_swa = g_f("rope.freq_base_swa", theta_full);
                // partial rotary: full layers rotate rope.dimension_count
                // (64 of 128), sliding layers dimension_count_swa (128)
                let rot_full = g_u("rope.dimension_count", shape.head_dim);
                let rot_swa = g_u("rope.dimension_count_swa", shape.head_dim);
                // a layer is FULL when its q head count matches the
                // smaller of the two values in the array (48 vs 72); the
                // gguf carries no explicit sliding pattern for laguna
                let h_min = heads.iter().copied().min().unwrap_or(shape.n_head as u64);
                (0..shape.n_exec_layer as usize)
                    .map(|il| {
                        let nh = heads.get(il).copied().unwrap_or(shape.n_head as u64);
                        let full = nh == h_min;
                        Geom {
                            n_head_q: nh as u32,
                            n_head_kv: kvh.get(il).copied().unwrap_or(shape.n_head_kv as u64) as u32,
                            head_dim: shape.head_dim,
                            theta: if full { theta_full } else { theta_swa },
                            window: if full { 0 } else { window },
                            factors: false,
                            rot: if full { rot_full } else { rot_swa },
                        }
                    })
                    .collect()
            } else if oss_arch {
                // gpt-oss alternates sliding and full attention and ships no
                // sliding_window_pattern array to say so; the reference
                // pattern is sliding on even layers. Head geometry, rope and
                // head_dim are uniform, so window is the only thing that
                // varies per layer.
                let window = gguf
                    .arch_meta("attention.sliding_window")
                    .and_then(Value::as_u64)
                    .map(|v| v as u32)
                    .unwrap_or(128);
                (0..shape.n_exec_layer as usize)
                    .map(|il| Geom {
                        n_head_q: shape.n_head,
                        n_head_kv: shape.n_head_kv,
                        head_dim: shape.head_dim,
                        theta: shape.rope_freq_base,
                        window: if il % 2 == 0 { window } else { 0 },
                        factors: false,
                        // non-zero routes rope through the yarn path, which
                        // is what gpt-oss needs: uniform scaling with the
                        // kernel's mscale. Zero here silently drops YaRN.
                        rot: shape.head_dim,
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
                // K3's attention+KDA+shared-expert stack is ~29GB of Q4_K
                // over 93 layers: it cannot fit one card, so let the
                // layer-split planner place it and do not cap it here.
                (Family::K3, _) => env_budget.unwrap_or(i64::MAX),
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
                            "pulsar: attn VRAM budget clamped {:.1} -> {:.1}GiB (free {:.1}GiB)",
                            attn_vram_budget as f64 / GIB,
                            cap as f64 / GIB,
                            free as f64 / GIB
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
                    // A FFN triple matmul_kq cannot dispatch (q8_0 from a
                    // --default q8_0 quant, f16/f32 sources, or a
                    // non-256-multiple contraction dim) rides the plain
                    // q8_0 Dense path instead: matmul_q8_0 on the f32
                    // activations, preq scratch prewarmed before the GDN
                    // graph capture. ponytail: dense qwen35 + PULSAR_TP
                    // leaves this FFN un-split on the primary (the TP
                    // stash is K-quant only); split it when a q8_0 model
                    // needs TP.
                    let kq_dispatch = |ti: &TensorInfo| {
                        MatW::keep_native(ti)
                            || (matches!(
                                ti.ty,
                                TensorType::Q2K | TensorType::Q3K | TensorType::IQ2XXS
                            ) && ti.dims[0].is_multiple_of(256))
                    };
                    if gguf.tensor(&t("ffn_gate.weight")).is_some_and(|ti| !kq_dispatch(ti)) {
                        Ffn::Dense {
                            gate: upload(&file, &gguf, &t("ffn_gate.weight"))?,
                            up: upload(&file, &gguf, &t("ffn_up.weight"))?,
                            down: upload(&file, &gguf, &t("ffn_down.weight"))?,
                        }
                    } else
                    // dense qwen35 (27B): the FFN triple resident in
                    // native K-quant on whatever device is current (the
                    // layer's owner under the dense split). Under
                    // PULSAR_TP the triple splits across the pair -
                    // gate/up by output rows, down by input blocks - so
                    // both cards read their half of every layer's FFN in
                    // parallel and the partial down outputs add. The
                    // nextn/MTP layer stays whole (il >= n_exec_layer).
                    if let Some((dev_b, half)) = tp_cfg.filter(|_| il < shape.n_exec_layer
                            && layer_dev.get(il as usize).copied().unwrap_or(primary) == primary) {
                        let cur = kernels::get_device();
                        let (gb, grb, gq) = read_kq_bytes(&file, &gguf, &t("ffn_gate.weight"))?;
                        let (ub, urb, uq) = read_kq_bytes(&file, &gguf, &t("ffn_up.weight"))?;
                        let (db, drb, dq) = read_kq_bytes(&file, &gguf, &t("ffn_down.weight"))?;
                        let gsplit = half as usize * grb as usize;
                        let usplit = half as usize * urb as usize;
                        // B's share ends where C's begins (C absent -> B
                        // takes the remainder, the 2-card case)
                        let wb = tp3_cfg.map(|(_, _, wb)| wb as usize);
                        let gsplit2 = wb.map_or(gb.len(), |w| gsplit + w * grb as usize);
                        let usplit2 = wb.map_or(ub.len(), |w| usplit + w * urb as usize);
                        // down: the contraction dim halves, so every
                        // output row contributes its first half of
                        // quant blocks to A and the rest to B
                        let drb = drb as usize;
                        // same cut as gate/up, in bytes: exact because
                        // every share is a whole number of quant blocks
                        let n_ff_u = shape.n_ff_exp as usize;
                        let dh = half as usize * drb / n_ff_u;
                        let dh2 = wb.map_or(drb, |w| dh + w * drb / n_ff_u);
                        let n_out = db.len() / drb;
                        let mut da = Vec::with_capacity(n_out * dh);
                        let mut dbb = Vec::with_capacity(n_out * (dh2 - dh));
                        let mut dcc = Vec::with_capacity(n_out * (drb - dh2));
                        for r in 0..n_out {
                            let row = &db[r * drb..(r + 1) * drb];
                            da.extend_from_slice(&row[..dh]);
                            dbb.extend_from_slice(&row[dh..dh2]);
                            dcc.extend_from_slice(&row[dh2..]);
                        }
                        let kq = |bytes: &[u8], rb: u64, q: u32| -> Result<KqW> {
                            Ok(KqW { w: DeviceBuf::from_bytes(bytes)?, row_bytes: rb, quant: q })
                        };
                        let a = Ffn::DenseKq {
                            gate: kq(&gb[..gsplit], grb, gq)?,
                            up: kq(&ub[..usplit], urb, uq)?,
                            down: kq(&da, dh as u64, dq)?,
                        };
                        kernels::set_device(dev_b)?;
                        *tp_ffn_stash.borrow_mut() = Some(TpFfnW {
                            gate: kq(&gb[gsplit..gsplit2], grb, gq)?,
                            up: kq(&ub[usplit..usplit2], urb, uq)?,
                            // B's rows are its OWN span, which is only dh
                            // wide when the split is even
                            down: kq(&dbb, (dh2 - dh) as u64, dq)?,
                        });
                        if let Some((d3, _, _)) = tp3_cfg {
                            kernels::set_device(d3)?;
                            *tp_ffn3_stash.borrow_mut() = Some(TpFfnW {
                                gate: kq(&gb[gsplit2..], grb, gq)?,
                                up: kq(&ub[usplit2..], urb, uq)?,
                                down: kq(&dcc, (drb - dh2) as u64, dq)?,
                            });
                        }
                        kernels::set_device(cur)?;
                        a
                    } else {
                        Ffn::DenseKq {
                            gate: upload_kq(&file, &gguf, &t("ffn_gate.weight"))?,
                            up: upload_kq(&file, &gguf, &t("ffn_up.weight"))?,
                            down: upload_kq(&file, &gguf, &t("ffn_down.weight"))?,
                        }
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
                        // matmul_f32 wants f32 (router precision drives
                        // selection). upload_as_f32 covers every ship format:
                        // dsv4's f16, qwen35moe's q8_0, plain f32. The old
                        // upload() branch returned raw q8_0 bytes for qwen35moe,
                        // which matmul_f32 read past (8MB read on a 2MB buffer).
                        gate_inp: upload_as_f32(&file, &gguf, &t("ffn_gate_inp.weight"))?,
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
                        shexp: if !ink_arch
                            && shape.family != Family::K3
                            && gguf.tensor(&t("ffn_gate_shexp.weight")).is_some() {
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
                        // f32 and small enough to stay resident; presence is
                        // decided by the file, so an arch that grows biases
                        // later needs no code here
                        gate_inp_b: if gguf.tensor(&t("ffn_gate_inp.bias")).is_some() {
                            Some(upload(&file, &gguf, &t("ffn_gate_inp.bias"))?)
                        } else {
                            None
                        },
                        exp_bias: if gguf.tensor(&t("ffn_gate_exps.bias")).is_some() {
                            Some([
                                upload(&file, &gguf, &t("ffn_gate_exps.bias"))?,
                                upload(&file, &gguf, &t("ffn_up_exps.bias"))?,
                                upload(&file, &gguf, &t("ffn_down_exps.bias"))?,
                            ])
                        } else {
                            None
                        },
                    }
                };
                // attention stack lands on this layer's owner (the
                // layer split can spread ranges across secondaries)
                let a_dev = attn_layer_dev.get(il as usize).copied().unwrap_or(primary);
                if a_dev != primary {
                    kernels::set_device(a_dev)?;
                }
                let attn = match shape.family {
                    Family::K3 => {
                        // Probe rather than compute the 3-KDA-then-MLA
                        // pattern: the gguf marks KDA layers with
                        // head_count_kv 0, and the tensors are the same
                        // truth without trusting an interval constant.
                        let is_kda = gguf.tensor(&t("ssm_a")).is_some();
                        Attn::K3(Box::new(K3W {
                            kda: if is_kda {
                                Some(K3Kda {
                                    wq: MatW::load(&file, &gguf, &t("attn_q.weight"))?,
                                    wk: MatW::load(&file, &gguf, &t("attn_k.weight"))?,
                                    wv: MatW::load(&file, &gguf, &t("attn_v.weight"))?,
                                    conv_q: upload(&file, &gguf, &t("ssm_conv1d_q.weight"))?,
                                    conv_k: upload(&file, &gguf, &t("ssm_conv1d_k.weight"))?,
                                    conv_v: upload(&file, &gguf, &t("ssm_conv1d_v.weight"))?,
                                    f_a: MatW::load(&file, &gguf, &t("ssm_f_a.weight"))?,
                                    f_b: upload_as_f32(&file, &gguf, &t("ssm_f_b.weight"))?,
                                    beta_w: MatW::load(&file, &gguf, &t("ssm_beta.weight"))?,
                                    a: upload_as_f32(&file, &gguf, &t("ssm_a"))?,
                                    dt_bias: upload(&file, &gguf, &t("ssm_dt.bias"))?,
                                    wg: MatW::load(&file, &gguf, &t("ssm_g.weight"))?,
                                    ssm_norm: upload(&file, &gguf, &t("ssm_norm.weight"))?,
                                    out: MatW::load(&file, &gguf, &t("attn_output.weight"))?,
                                })
                            } else {
                                None
                            },
                            mla: if is_kda {
                                None
                            } else {
                                Some(K3Mla {
                                    q_a: upload_attn(&file, &gguf, &t("attn_q_a.weight"), &mut *attn_vram_budget)?,
                                    q_a_norm: upload(&file, &gguf, &t("attn_q_a_norm.weight"))?,
                                    q_b: upload_attn(&file, &gguf, &t("attn_q_b.weight"), &mut *attn_vram_budget)?,
                                    kv_a_mqa: upload_attn(&file, &gguf, &t("attn_kv_a_mqa.weight"), &mut *attn_vram_budget)?,
                                    kv_a_norm: upload(&file, &gguf, &t("attn_kv_a_norm.weight"))?,
                                    k_b: upload(&file, &gguf, &t("attn_k_b.weight"))?,
                                    v_b: upload(&file, &gguf, &t("attn_v_b.weight"))?,
                                    gate: MatW::load(&file, &gguf, &t("attn_gate.weight"))?,
                                    out: MatW::load(&file, &gguf, &t("attn_output.weight"))?,
                                })
                            },
                            // These four are consumed on the PRIMARY (the
                            // AttnRes mix and the FFN half both run against
                            // the primary's residual stream), so they must
                            // live there whatever card the attention half
                            // landed on. Reading a peer pointer from the
                            // wrong device is an illegal access, not a slow
                            // path.
                            attn_res_score: {
                                kernels::set_device(primary)?;
                                upload(&file, &gguf, &t("attn_res_score.weight"))?
                            },
                            ffn_res_score: upload(&file, &gguf, &t("ffn_res_score.weight"))?,
                            routed: if gguf.tensor(&t("ffn_routed_down.weight")).is_some() {
                                let r = K3Routed {
                                    down: MatW::load(&file, &gguf, &t("ffn_routed_down.weight"))?,
                                    up: MatW::load(&file, &gguf, &t("ffn_routed_up.weight"))?,
                                    norm: if gguf.tensor(&t("ffn_routed_norm.weight")).is_some() {
                                        Some(upload(&file, &gguf, &t("ffn_routed_norm.weight"))?)
                                    } else {
                                        None
                                    },
                                };
                                Some(r)
                            } else {
                                None // leading dense layer
                            },
                            shexp: if gguf.tensor(&t("ffn_gate_shexp.weight")).is_some() {
                                // 6.8GB across 92 layers: far too much to
                                // pile on the primary beside the router,
                                // latent projections and embeddings. It
                                // reads the layer's own input, so it rides
                                // the layer's card and the ffn-normed row
                                // hops over with it (28KB per layer).
                                if a_dev != primary {
                                    kernels::set_device(a_dev)?;
                                }
                                Some(K3Shexp {
                                    gate: MatW::load(&file, &gguf, &t("ffn_gate_shexp.weight"))?,
                                    up: MatW::load(&file, &gguf, &t("ffn_up_shexp.weight"))?,
                                    down: MatW::load(&file, &gguf, &t("ffn_down_shexp.weight"))?,
                                })
                            } else {
                                None
                            },
                        }))
                    }
                    Family::Gqa => Attn::Gqa {
                        attn_q: upload_attn(&file, &gguf, &t("attn_q.weight"), &mut *attn_vram_budget)?,
                        attn_k: upload_attn(&file, &gguf, &t("attn_k.weight"), &mut *attn_vram_budget)?,
                        attn_v: if gguf.tensor(&t("attn_v.weight")).is_some() {
                            Some(upload_attn(&file, &gguf, &t("attn_v.weight"), &mut *attn_vram_budget)?)
                        } else {
                            None // gemma attention_k_eq_v: k doubles as v
                        },
                        q_norm: if gguf.tensor(&t("attn_q_norm.weight")).is_some() {
                            Some(upload(&file, &gguf, &t("attn_q_norm.weight"))?)
                        } else {
                            None
                        },
                        k_norm: if gguf.tensor(&t("attn_k_norm.weight")).is_some() {
                            Some(upload(&file, &gguf, &t("attn_k_norm.weight"))?)
                        } else {
                            None
                        },
                        sinks: if gguf.tensor(&t("attn_sinks.weight")).is_some() {
                            Some(upload(&file, &gguf, &t("attn_sinks.weight"))?)
                        } else {
                            None
                        },
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
                            attn: if !is_attn {
                                None
                            } else if let Some((dev_b, _)) = tp_cfg.filter(|_| il < shape.n_exec_layer
                            && layer_dev.get(il as usize).copied().unwrap_or(primary) == primary) {
                                // TP head split: q|gate rows are per-head
                                // [q hd | gate hd], heads outer, so head
                                // ranges are contiguous row ranges; the
                                // output projection input follows A's
                                // head concat, so matw_col_half aligns.
                                let hd = shape.head_dim as u64;
                                let nh = shape.n_head as u64;
                                let nkv = shape.n_head_kv as u64;
                                let mk = |q0: u64, q1: u64, k0: u64, k1: u64, second: bool| -> Result<Qwen35Attn> {
                                    Ok(Qwen35Attn {
                                        wq: matw_row_ranges(&file, &gguf, &t("attn_q.weight"), &[(q0 * 2 * hd, q1 * 2 * hd)])?,
                                        wk: matw_row_ranges(&file, &gguf, &t("attn_k.weight"), &[(k0 * hd, k1 * hd)])?,
                                        wv: matw_row_ranges(&file, &gguf, &t("attn_v.weight"), &[(k0 * hd, k1 * hd)])?,
                                        out: matw_col_half(&file, &gguf, &t("attn_output.weight"), second)?,
                                        q_norm: upload(&file, &gguf, &t("attn_q_norm.weight"))?,
                                        k_norm: upload(&file, &gguf, &t("attn_k_norm.weight"))?,
                                    })
                                };
                                let a_half = mk(0, nh / 2, 0, nkv / 2, false)?;
                                let cur_dev = kernels::get_device();
                                kernels::set_device(dev_b)?;
                                *tp_attn_stash.borrow_mut() = Some(mk(nh / 2, nh, nkv / 2, nkv, true)?);
                                kernels::set_device(cur_dev)?;
                                Some(a_half)
                            } else {
                                Some(Qwen35Attn {
                                    wq: MatW::load(&file, &gguf, &t("attn_q.weight"))?,
                                    wk: MatW::load(&file, &gguf, &t("attn_k.weight"))?,
                                    wv: MatW::load(&file, &gguf, &t("attn_v.weight"))?,
                                    out: MatW::load(&file, &gguf, &t("attn_output.weight"))?,
                                    q_norm: upload(&file, &gguf, &t("attn_q_norm.weight"))?,
                                    k_norm: upload(&file, &gguf, &t("attn_k_norm.weight"))?,
                                })
                            },
                            gdn: if is_attn {
                                None
                            } else if let Some((dev_b, _)) = tp_cfg.filter(|_| il < shape.n_exec_layer
                            && layer_dev.get(il as usize).copied().unwrap_or(primary) == primary) {
                                // TP head split: A keeps the first half of
                                // the qk and v heads, B the second. The
                                // delta-rule state is head-local, so each
                                // card runs a complete half-GDN and only
                                // ssm_out's partial output crosses. Fused
                                // qkv row layout: [q key | k key | v value].
                                let kd = (shape.ssm_k_heads * shape.ssm_state) as u64;
                                let vd = (shape.ssm_v_heads * shape.ssm_state) as u64;
                                let kh = kd / 2;
                                let vheads = shape.ssm_v_heads as u64;
                                let hk = shape.ssm_k_heads as u64;
                                let sd = shape.ssm_state as u64;
                                let rep = vheads / hk;
                                let khh = hk / 2;
                                let vsel_of = |side: u64| -> Vec<(u64, u64)> {
                                    (0..rep)
                                        .map(|r| ((r * hk + side * khh) * sd, (r * hk + (side + 1) * khh) * sd))
                                        .collect()
                                };
                                let hsel_of = |side: u64| -> Vec<(u64, u64)> {
                                    (0..rep).map(|r| (r * hk + side * khh, r * hk + (side + 1) * khh)).collect()
                                };
                                let qkv_of = |side: u64, vsel: &[(u64, u64)]| -> Vec<(u64, u64)> {
                                    let mut v = vec![
                                        (side * kh, (side + 1) * kh),
                                        (kd + side * kh, kd + (side + 1) * kh),
                                    ];
                                    v.extend(vsel.iter().map(|&(a, b)| (2 * kd + a, 2 * kd + b)));
                                    v
                                };
                                let ck = shape.ssm_conv_k as usize;
                                let ne = shape.n_embd as usize;
                                let mk = |ranges: &[(u64, u64)], vsel: &[(u64, u64)], hsel: &[(u64, u64)]| -> Result<Qwen35Gdn> {
                                    let alpha = read_any_f32(&file, &gguf, &t("ssm_alpha.weight"))?;
                                    let beta = read_any_f32(&file, &gguf, &t("ssm_beta.weight"))?;
                                    let av = read_any_f32(&file, &gguf, &t("ssm_a"))?;
                                    let dtv = read_any_f32(&file, &gguf, &t("ssm_dt.bias"))?;
                                    let convv = read_any_f32(&file, &gguf, &t("ssm_conv1d.weight"))?;
                                    let mut conv_sl: Vec<f32> = Vec::new();
                                    for &(r0, r1) in ranges {
                                        conv_sl.extend_from_slice(&convv[r0 as usize * ck..r1 as usize * ck]);
                                    }
                                    Ok(Qwen35Gdn {
                                        wqkv: {
                                            let w = matw_row_ranges(&file, &gguf, &t("attn_qkv.weight"), ranges)?;
                                            if il == 0 && std::env::var_os("PULSAR_DEBUG_L2").is_some() {
                                                // does B's sliced weight actually hold the rows
                                                // the ranges name? compare its v-block's first
                                                // row against the full tensor's same row.
                                                if let MatW::Kq(k) = &w {
                                                    let vrow = ranges[2].0; // first v row, global
                                                    let off = (2 * (ranges[0].1 - ranges[0].0)) * k.row_bytes;
                                                    let mut got = vec![0u8; 16];
                                                    k.w.read(off as usize, &mut got)?;
                                                    let (full, frb, _q) = read_kq_bytes(&file, &gguf, &t("attn_qkv.weight"))?;
                                                    let fo = (vrow * frb) as usize;
                                                    eprintln!("  WSLICE il=0 rb={} vrow={} B[{:?}] full[{:?}] {}",
                                                        k.row_bytes, vrow, &got[..8], &full[fo..fo + 8],
                                                        if got[..] == full[fo..fo + 16] { "MATCH" } else { "DIFFER" });
                                                }
                                            }
                                            w
                                        },
                                        wz: matw_row_ranges(&file, &gguf, &t("attn_gate.weight"), vsel)?,
                                        conv: DeviceBuf::from_f32(&conv_sl)?,
                                        ab_w: {
                                            let mut ab: Vec<f32> = Vec::new();
                                            for &(h0, h1) in hsel {
                                                ab.extend_from_slice(&alpha[h0 as usize * ne..h1 as usize * ne]);
                                            }
                                            for &(h0, h1) in hsel {
                                                ab.extend_from_slice(&beta[h0 as usize * ne..h1 as usize * ne]);
                                            }
                                            DeviceBuf::from_f32(&ab)?
                                        },
                                        a: {
                                            let mut v: Vec<f32> = Vec::new();
                                            for &(h0, h1) in hsel {
                                                v.extend_from_slice(&av[h0 as usize..h1 as usize]);
                                            }
                                            DeviceBuf::from_f32(&v)?
                                        },
                                        dt_bias: {
                                            let mut v: Vec<f32> = Vec::new();
                                            for &(h0, h1) in hsel {
                                                v.extend_from_slice(&dtv[h0 as usize..h1 as usize]);
                                            }
                                            DeviceBuf::from_f32(&v)?
                                        },
                                        ssm_norm: upload(&file, &gguf, &t("ssm_norm.weight"))?,
                                        ssm_out: matw_col_ranges(&file, &gguf, &t("ssm_out.weight"), vsel)?,
                                    })
                                };
                                let va = vsel_of(0);
                                let a_half = mk(&qkv_of(0, &va), &va, &hsel_of(0))?;
                                let cur_dev = kernels::get_device();
                                kernels::set_device(dev_b)?;
                                let vb = vsel_of(1);
                                *tp_gdn_stash.borrow_mut() = Some(mk(&qkv_of(1, &vb), &vb, &hsel_of(1))?);
                                kernels::set_device(cur_dev)?;
                                Some(a_half)
                            } else {
                                Some(Qwen35Gdn {
                                    wqkv: MatW::load(&file, &gguf, &t("attn_qkv.weight"))?,
                                    wz: MatW::load(&file, &gguf, &t("attn_gate.weight"))?,
                                    // conv kernel reads this as f32 [conv_dim][ssm_conv_k];
                                    // upload() would quantize the F16 2D tensor to q8_0 bytes,
                                    // which the conv kernel reads past (4x size mismatch) -> OOB.
                                    conv: upload_as_f32(&file, &gguf, &t("ssm_conv1d.weight"))?,
                                    ab_w: {
                                        let mut ab = read_any_f32(&file, &gguf, &t("ssm_alpha.weight"))?;
                                        ab.extend(read_any_f32(&file, &gguf, &t("ssm_beta.weight"))?);
                                        DeviceBuf::from_f32(&ab)?
                                    },
                                    a: upload(&file, &gguf, &t("ssm_a"))?,
                                    dt_bias: upload(&file, &gguf, &t("ssm_dt.bias"))?,
                                    ssm_norm: upload(&file, &gguf, &t("ssm_norm.weight"))?,
                                    ssm_out: MatW::load(&file, &gguf, &t("ssm_out.weight"))?,
                                })
                            },
                            // dense qwen35 has no shared expert (and so
                            // no shexp gate); the ffn half never reads it
                            // matmul_f32 consumer (qwen35_row_sigmoid_scale
                            // path): q8_0 here would be read as f32 and OOB.
                            shexp_gate: if gguf.tensor(&t("ffn_gate_inp_shexp.weight")).is_some() {
                                upload_as_f32(&file, &gguf, &t("ffn_gate_inp_shexp.weight"))?
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
                } else if shape.family == Family::K3 {
                    // both K3 layer flavours keep their own out (MatW)
                    DeviceBuf::alloc(1)?
                } else {
                    upload_attn(&file, &gguf, &t("attn_output.weight"), &mut *attn_vram_budget)?
                };
                if a_dev != primary {
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
                    // presence decided by the file, so an arch that grows
                    // attention biases later needs no code here
                    attn_bias: if gguf.tensor(&t("attn_q.bias")).is_some() {
                        Some(AttnBias {
                            q: upload(&file, &gguf, &t("attn_q.bias"))?,
                            k: upload(&file, &gguf, &t("attn_k.bias"))?,
                            v: upload(&file, &gguf, &t("attn_v.bias"))?,
                            out: upload(&file, &gguf, &t("attn_output.bias"))?,
                        })
                    } else {
                        None
                    },
                    // qwen35 calls the pre-FFN norm post_attention_norm
                    ffn_norm: if gguf.tensor(&t("ffn_norm.weight")).is_some() {
                        upload(&file, &gguf, &t("ffn_norm.weight"))?
                    } else {
                        upload(&file, &gguf, &t("post_attention_norm.weight"))?
                    },
                    ffn,
                    gemma,
                    ink,
                    // laguna: per-head output gate (gating="per-head", so
                    // one column per QUERY head on this layer)
                    attn_gate: if gguf.tensor(&t("attn_gate.weight")).is_some() {
                        Some(upload(&file, &gguf, &t("attn_gate.weight"))?)
                    } else {
                        None
                    },
                })
            };

            let mut tp_head: Option<KqW> = None;
            if let Some((dev_b, _)) = tp_cfg {
                if output_kq.is_some()
                    && shape.n_vocab.is_multiple_of(2)
                    && gguf.tensor("output.weight").is_some()
                {
                    let (bytes, rb, quant) = read_kq_bytes(&file, &gguf, "output.weight")?;
                    let v2 = shape.n_vocab as u64 / 2;
                    kernels::set_device(dev_b)?;
                    tp_head = Some(KqW {
                        w: DeviceBuf::from_bytes(&bytes[(v2 * rb) as usize..])?,
                        row_bytes: rb,
                        quant,
                    });
                    kernels::set_device(primary)?;
                }
            }
            let mut layers = Vec::with_capacity(shape.n_exec_layer as usize);
            let mut tp_layers: Vec<Option<TpLayerW>> = Vec::new();
            for il in 0..shape.n_exec_layer {
                // dense split: the whole layer uploads to its owner
                kernels::set_device(layer_dev[il as usize])?;
                layers.push(load_layer(il, &mut attn_vram_budget, &mut no_budget)?);
                if tp_cfg.is_some() {
                    tp3_layers.push(tp_ffn3_stash.borrow_mut().take());
                    let f = tp_ffn_stash.borrow_mut().take();
                    let g2 = tp_gdn_stash.borrow_mut().take();
                    let at = tp_attn_stash.borrow_mut().take();
                    tp_layers.push(match (f, g2, at) {
                        (None, None, None) => None,
                        (f, g2, at) => Some(TpLayerW { ffn: f, gdn: g2, attn: at }),
                    });
                }
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
                                    "pulsar: MTP draft experts resident ({:.1}GiB, all {} triples)",
                                    total as f64 / GIB,
                                    shape.n_expert
                                );
                                res_pool = pool;
                            }
                            Err(_) => eprintln!(
                                "pulsar: MTP expert residency didn't fit ({:.1}GiB needed) - drafts will stream",
                                total as f64 / GIB
                            ),
                        }
                    }
                    let m = MtpLayer {
                        layer,
                        eh_proj: MatW::load(&file, &gguf, &nextn("eh_proj"))?,
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
                let raw = std::env::var("PULSAR_MTP_DEPTH").ok();
                let parsed = raw.as_deref().and_then(|v| v.parse::<u32>().ok());
                let d = parsed.unwrap_or(1).clamp(1, 8);
                if let Some(r) = raw.as_deref() {
                    if parsed.is_none() {
                        eprintln!(
                            "pulsar: PULSAR_MTP_DEPTH='{r}' not a u32 -> using default {d}"
                        );
                    } else {
                        eprintln!("pulsar: MTP depth={d}");
                    }
                } else {
                    eprintln!("pulsar: MTP depth={d} (default, PULSAR_MTP_DEPTH unset)");
                }
                d
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
            // DSpark draft heads: only a converted draft gguf carries
            // fc.weight, so every target model skips this in one probe
            let dspark = if dsv4_arch { dsv4::load_dspark(&file, &gguf)? } else { None };
            // the split/attn placement loops leave whatever device they
            // touched last current; restore the primary so post-load
            // allocations (draft models, probes) land on the fast card
            // instead of one the split already filled
            kernels::set_device(kernels::primary_device())?;
            // Weights the file ships that no load path asked for. Reading a
            // subset of an architecture's tensors is an error nowhere in the
            // loader, so a feature we do not model (gpt-oss ships 192 bias
            // tensors, for one) leaves a model that loads, runs, and answers
            // plausibly while ignoring part of itself. That is the worst way
            // to be wrong, and one pass over the table at load rules it out.
            let n_exec = shape.n_exec_layer as usize;
            let skipped: Vec<&str> = gguf
                .unconsumed()
                .into_iter()
                .filter(|n| {
                    // blocks past the executed depth are the MTP/nextn draft
                    // layer, which only loads when speculation is on (Hy3
                    // ships blk.80 over 80 exec layers). Leaving it unread is
                    // the design, not a miss - warning about it every load
                    // would train the eye to skip this line.
                    n.strip_prefix("blk.")
                        .and_then(|r| r.split('.').next())
                        .and_then(|d| d.parse::<usize>().ok())
                        .is_none_or(|il| il < n_exec)
                })
                .collect();
            if !skipped.is_empty() {
                const SHOW: usize = 6;
                let more = skipped.len().saturating_sub(SHOW);
                eprintln!(
                    "pulsar: WARNING {} tensor(s) in this gguf were never read, so output may be \
                     silently wrong: {}{}",
                    skipped.len(),
                    skipped[..skipped.len().min(SHOW)].join(", "),
                    if more > 0 { format!(", +{more} more") } else { String::new() },
                );
            }
            Ok(Model {
                path: path.to_path_buf(),
                shards,
                shape,
                gguf,
                token_embd,
                output_norm,
                output_res_score,
                output,
                layers,
                attn_dev,
                attn_layer_dev,
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
                dspark,
                tp: tp_cfg.map(|(dev, half)| TpW { dev, half, layers: tp_layers, head: tp_head }),
                tp_c: tp3_cfg.map(|(dev, _, _)| TpCW {
                    dev,
                    width: shape.n_ff_exp - tp_cfg.map_or(0, |(_, h)| h) - tp3_cfg.map_or(0, |(_, _, wb)| wb),
                    ffn: tp3_layers,
                }),
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
        // Expert biases live on the primary, and a tier kernel runs on
        // another device where that pointer is not dereferenceable, so a
        // tier would fault the moment it read one. Replicating the bias
        // buffers per tier device is the real fix (they are ~1MB); until
        // then, decline the tier rather than fault.
        if m.layers.iter().any(|l| matches!(&l.ffn, Ffn::Moe { exp_bias: Some(_), .. })) {
            eprintln!(
                "pulsar: expert tiers disabled - this model carries per-expert biases, \
                 which are resident on the primary only"
            );
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
        // PULSAR_DSPARK_RESIDENT=1: tier a DSpark draft's WHOLE expert
        // set (uniform heat, no census). Only worth it with ~10GB of
        // VRAM to spare: on substrate it evicted the target's tiers and
        // verify went 239ms -> 518ms, a measured net loss. Default is
        // the census ranking, same as any model.
        let all = m.dspark.is_some() && std::env::var_os("PULSAR_DSPARK_RESIDENT").is_some();
        let mut census: std::collections::HashMap<u64, u64> = if all {
            Default::default()
        } else {
            read_census(&m.path).into_iter().map(|(off, _, count)| (off, count)).collect()
        };
        if census.is_empty() && !all {
            // first run: rank tiers from the built-in hotlist seed too
            // (same fallback load_warm uses for the caches)
            if std::env::var_os("PULSAR_NO_HOTLIST").is_none() {
                if let Some(text) = builtin_hotlist(m.shape.family) {
                    let mut heat = std::collections::HashMap::new();
                    hotlist_heat(m, text, &mut heat);
                    census = heat.into_iter().map(|(off, (count, _len))| (off, count)).collect();
                }
            }
            if census.is_empty() {
                eprintln!("pulsar: no warm census yet - expert tiers idle until the next run");
                return Ok(Vec::new());
            }
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
                let heat: u64 = if all {
                    1
                } else {
                    slabs.iter().filter_map(|(off, _)| census.get(off)).sum()
                };
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
        triples.sort_unstable_by_key(|e| std::cmp::Reverse(e.0));
        sink_triples.sort_unstable_by_key(|e| std::cmp::Reverse(e.0));
        triples.extend(sink_triples);
        if triples.is_empty() {
            // fully-resident model (DenseKq): a tier would just grab the
            // free VRAM its own layers need
            if std::env::var_os("PULSAR_TIER_LOG").is_some() {
                eprintln!("pulsar: tier skip: no census-hot triples (primary {primary}, candidates {candidates:?})");
            }
            return Ok(Vec::new());
        }

        let file = VFile::open(&m.shards)?;
        let mut tiers = Vec::new();
        let mut next = triples.into_iter();
        for d in candidates {
            let Ok((free, _)) = kernels::mem_info(d) else { continue };
            let reserve: usize = 1 << 30; // scratch + CUDA context
            if free <= reserve + (1 << 30) {
                if std::env::var_os("PULSAR_TIER_LOG").is_some() {
                    eprintln!("pulsar: tier skip: device {d} only {:.1}GiB free", free as f64 / GIB);
                }
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
                    mb as usize
                        * (s.n_embd as usize).div_ceil(kernels::Q8_K_BLOCK_ELEMS)
                        * kernels::Q8_K_BLOCK_BYTES,
                )?,
                mid: DeviceBuf::alloc(mb as usize * n_used * s.n_ff_exp as usize * 4)?,
                midq: DeviceBuf::alloc(
                    mb as usize
                        * n_used
                        * (s.n_ff_exp as usize).div_ceil(kernels::Q8_K_BLOCK_ELEMS)
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
                "pulsar: expert tier on CUDA device {d}: {} triples ({:.1}GiB) resident in {:.1}s",
                tier.map.len() / 3,
                cursor as f64 / GIB,
                t0.elapsed().as_secs_f32()
            );
            tiers.push(tier);
        }
        Ok(tiers)
    }

    /// Per-decode device state: activation buffers, KV caches, the routed
    /// expert staging arena, and reusable host staging.
    /// See State.attn_sc.
    struct MlaScratch {
        dev: i32,
        normed_a: DeviceBuf,
        attn_out_a: DeviceBuf,
        q_rank: DeviceBuf,
        q_rank_norm: DeviceBuf,
        q: DeviceBuf,
        kv_raw: DeviceBuf,
        kv_norm: DeviceBuf,
        qk_low: DeviceBuf,
        heads: DeviceBuf,
        idx_kraw: DeviceBuf,
        idx_q: DeviceBuf,
        idx_q16: DeviceBuf,
        idx_w: DeviceBuf,
        idx_scores: DeviceBuf,
        mla_selected: DeviceBuf,
    }

    pub struct State {
        ctx: u32,
        max_batch: u32,
        /// Trailing logit rows last_row/logits were sized for. A forward
        /// that returns more rows than this writes past both buffers, and
        /// the corruption is silent, so every multi-row path checks it.
        spec_rows: u32,
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
        /// Per-device attention scratch for the Mla layer split: every
        /// buffer the attn segment touches, replicated once per distinct
        /// owner in attn_layer_dev. Single-card offload = one entry.
        /// Non-Mla families keep the flat fields below.
        attn_sc: Vec<MlaScratch>,
        /// Cards that host KV/idx caches; see Stats.kv_headroom.
        kv_devs: Vec<i32>,
        /// Device holding the live DSA selection list (mla_selected of
        /// that device's scratch); -1 = none written yet. Reuse layers on
        /// a different owner copy the list across before attending.
        sel_dev: i32,
        kcache: Vec<DeviceBuf>,
        vcache: Vec<DeviceBuf>,
        /// Card-B kv-head cache halves per attn layer (PULSAR_TP); empty
        /// without TP. Positional like kcache, so ckpt rewind needs no copy.
        tp_kcache: Vec<Option<(DeviceBuf, DeviceBuf)>>,
        /// Device argmax scratch + results for the spec loop's
        /// readback-free verify (skip_logit_read)
        amax_out: DeviceBuf,
        /// results / flag for the argmax-only head paths (spec verify,
        /// greedy decode); pub: the CLI's one-shot loop drives them
        pub last_argmax: Vec<u32>,
        pub skip_logit_read: bool,
        /// Gqa KV storage format (kvq). 0=f32 (exact, default), 1=fp8
        /// e4m3 + per-row scale, 2=fp16, 3=int8 + per-row scale, 4=q8_0,
        /// 5=q4_0, 6=turbo3, 7=turbo2, 8=turbo3_tcq, 9=turbo2_tcq,
        /// 10=turbo1_tcq. Codes 6-10 are dsv4-only (rotated centroids /
        /// trellis). All lossy formats opt-in via PULSAR_KV=<fmt>; the
        /// default f32 path keeps bit-exact guarantees. Dsv4's fused
        /// latent rows (raw ring + compressed) ride the same kvq field as
        /// a single flat head.
        kvq: u32,
        /// MLA latent KV storage format. 0=f32, 1=fp8 e4m3 + per-row
        /// scale, 2=fp16. Applies to both the latent rows and the rope
        /// tail; strides must match mla_lat_stride on the kernel side.
        kvq_lat: u32,
        /// Whether K/Q are rotated by `pi` (orthogonal Π) before block-quant.
        /// turbo4/turbo8 set this true. Drops back to false with a warning if
        /// qk_dim exceeds head_dim (q and qrot strides would disagree) or if Π
        /// fails its orthogonality check.
        kvq_rot: bool,
        /// Orthogonal rotation Π (head_dim×head_dim row-major, f32). Identity-
        /// sized (4 B) placeholder when kvq_rot is false. Applied as
        /// K_rot = K @ Πᵀ via matmul_f32 before KV append, and Q_rot = Q @ Πᵀ
        /// before attention. Decode-invariant: (QΠᵀ)·(KΠᵀ)ᵀ = QKᵀ.
        pi: DeviceBuf,
        /// Rotated-K scratch (n_tok * n_head_kv * head_dim, f32). Consumed
        /// once per layer; sized as a placeholder when rotation disabled.
        krot: DeviceBuf,
        /// Rotated-Q scratch (n_tok * n_head * head_dim, f32). Mirror of
        /// `q`'s attention-head layout, not the rope/`qk_dim`-padded layout.
        qrot: DeviceBuf,
        logits: DeviceBuf,
        pub store: StreamingStore,
        prefetcher: Prefetcher,
        /// Last cross-layer prediction and the layer it was made for, so the
        /// next layer can score it (PULSAR_PROFILE only).
        pred_prev: Vec<i32>,
        pred_prev_for: usize,
        pred_logits: DeviceBuf,
        pred_selected: DeviceBuf,
        pred_weights: DeviceBuf,
        /// Cumulative per-stage wall time (PULSAR_PROFILE=1 to print).
        pub prof: Prof,
        stages: Option<[AttnStage; 2]>,
        // MLA scratch (dummies for Gqa); on the attn GPU when one is set
        q_rank: DeviceBuf,
        q_rank_norm: DeviceBuf,
        // DSA indexer K caches (1-float dummies when absent); selection
        // count persists across layers so non-indexer layers reuse the
        // last list
        idx_kcache: Vec<DeviceBuf>,
        idx_last_sel: u32,
        // attn-GPU hop buffers (1-float dummies otherwise): normed input
        // copied primary->attn GPU, attn output copied back
        normed_a: DeviceBuf,
        attn_out_a: DeviceBuf,
        /// laguna per-head attention gate logits [max_batch][n_head]
        attn_gate_buf: DeviceBuf,
        // resident expert tiers on leftover GPUs + the primary-side
        // buffer their partial outputs are gathered into
        pub tiers: Vec<ExpertTier>,
        tier_ret: DeviceBuf,
        /// CPU expert lane (PULSAR_CPU=1): worker pool + partial-return buf
        pub cpu_pool: Option<cpu_tier::Pool>,
        cpu_ret: DeviceBuf,
        pub cpu_hits: u64,
        /// true per-(layer,expert) routing selections, tier-independent: counts
        /// every router pick, resident or streamed. Feeds /experts heat and the
        /// topic atlas without the host-cache blind spot. Index l*n_expert + e.
        route_counts: Vec<u64>,
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
        mtp_xq: DeviceBuf,
        mtp_hidden: DeviceBuf,
        /// true-hidden anchor saved across a draft chain (the chain
        /// self-feeds approximate hiddens into mtp_hidden; the batched
        /// fill pass afterwards needs the pre-chain value back)
        mtp_hidden_save: DeviceBuf,
        pub mtp_drafted: u64,
        pub mtp_accepted: u64,
        /// per-draft-position accept/draft counts (index 0 = first draft
        /// token). Acceptance DECAY across positions is the number that
        /// says whether a draft chain is healthy: a good MTP holds
        /// ~0.90/0.78/0.70, a self-feed that drifts collapses instead.
        pub mtp_pos_accepted: Vec<u64>,
        pub mtp_pos_drafted: Vec<u64>,
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
        /// k3 runtime (KDA conv+delta states, AttnRes bank); None elsewhere
        k3: Option<k3::K3Rt>,
        /// recurrent prefix checkpoints (pos ascending): a divergent
        /// request resumes from the nearest one instead of position 0
        ckpts: Vec<(u32, RecurrentCkpt)>,
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

        /// Arm DSpark hidden-state capture for a dsv4 target. Must be
        /// called before prefill; the ring fills as positions stream past.
        pub fn enable_dspark_capture(&mut self, m: &Model, layer_ids: Vec<usize>) -> Result {
            let rt = self.dsv4.as_mut().ok_or("dspark capture: not a dsv4 model")?;
            rt.enable_dflash(m, layer_ids)
        }

        /// One captured feature row: the target hidden states for `pos`,
        /// concatenated in `layer_ids` order. This is exactly the row the
        /// draft's `fc` consumes.
        pub fn dspark_feature_row(&self, m: &Model, pos: u32) -> Result<Vec<f32>> {
            let rt = self.dsv4.as_ref().ok_or("dspark capture: not a dsv4 model")?;
            let df = rt.dflash.as_ref().ok_or("dspark capture not enabled")?;
            let feat_w = df.layer_ids.len() * m.shape.n_embd as usize;
            let slot = pos as usize % qwen35::RING_CAP;
            Ok(df.ring.read_f32_at(slot * feat_w, feat_w)?)
        }

        /// Take a recurrent checkpoint at `pos` when the interval since
        /// the last one has passed (no-op for pure-KV families). GDN
        /// states are ~150MB per snapshot so qwen35 spaces them wide.
        pub fn maybe_checkpoint(&mut self, m: &Model, pos: u32) -> Result {
            if !m.recurrent_state() {
                return Ok(());
            }
            let qwen = m.shape.family == Family::Qwen35;
            let interval = if qwen { 4096 } else { 512 };
            let cap = if qwen { 2 } else { 8 };
            let last = self.ckpts.last().map(|(p, _)| *p).unwrap_or(0);
            if pos < last + interval {
                return Ok(());
            }
            let ck = match m.shape.family {
                Family::Dsv4 => RecurrentCkpt::Dsv4(
                    self.dsv4.as_ref().ok_or("dsv4 state missing")?.ckpt()?,
                ),
                Family::Qwen35 => RecurrentCkpt::Qwen35(
                    self.qwen35.as_ref().ok_or("qwen35 state missing")?.ckpt()?,
                ),
                _ => return Ok(()),
            };
            self.ckpts.push((pos, ck));
            if self.ckpts.len() > cap {
                // keep the earliest (early divergences hurt most) and
                // the recent tail
                self.ckpts.remove(1);
            }
            Ok(())
        }

        /// Restore the latest checkpoint at or before `upto`; drops the
        /// ones past it (their content no longer matches the stream).
        /// Returns the restored position.
        pub fn restore_nearest_ckpt(&mut self, _m: &Model, upto: u32) -> Result<Option<u32>> {
            let Some(i) = self
                .ckpts
                .iter()
                .rposition(|(p, _)| *p <= upto)
            else {
                self.ckpts.clear();
                return Ok(None);
            };
            let pos = self.ckpts[i].0;
            match &self.ckpts[i].1 {
                RecurrentCkpt::Dsv4(ck) => {
                    self.dsv4.as_mut().ok_or("dsv4 state missing")?.ckpt_restore(ck)?
                }
                RecurrentCkpt::Qwen35(ck) => {
                    self.qwen35.as_mut().ok_or("qwen35 state missing")?.ckpt_restore(ck)?
                }
            }
            self.ckpts.truncate(i + 1);
            Ok(Some(pos))
        }

        pub fn clear_ckpts(&mut self) {
            self.ckpts.clear();
        }

        /// Persist the current dsv4 prefix state (recurrent runtime, the
        /// append-only caches up to `hist.len()`, and the rollback
        /// checkpoints) so a fresh process can resume without re-prefilling.
        /// The big caches are append-only: rows past a later restore point
        /// are simply overwritten on replay, exactly like the in-process
        /// rollback. MTP-slot state is intentionally not saved - drafts
        /// self-heal through verify (brief acceptance dip, never wrong
        /// output). dsv4-only; other families return Err.
        pub fn save_prefix(&self, m: &Model, hist: &[u32], path: &Path) -> Result {
            let s = m.shape;
            if s.family != Family::Dsv4 {
                return Err("prefix persist: dsv4 only".into());
            }
            let rt = self.dsv4.as_ref().ok_or("dsv4 state missing")?;
            let mut out = Vec::with_capacity(64 << 20);
            out.extend_from_slice(b"PLSRPFX2");
            for v in [s.n_exec_layer, s.n_embd, s.n_vocab, self.ctx, s.n_swa, s.head_dim, s.n_idx_dim] {
                out.extend_from_slice(&v.to_le_bytes());
            }
            out.extend_from_slice(&(hist.len() as u64).to_le_bytes());
            out.extend_from_slice(kernels::as_bytes(hist));
            rt.save_layers(&mut out)?;
            let counts = rt.layer_counts();
            let put = |out: &mut Vec<u8>, b: &DeviceBuf, bytes: usize| -> Result {
                let bytes = bytes.min(b.bytes());
                let mut host = vec![0u8; bytes];
                b.read(0, &mut host)?;
                out.extend_from_slice(&(bytes as u64).to_le_bytes());
                out.extend_from_slice(&host);
                Ok(())
            };
            for (il, &(n_comp, _)) in counts.iter().enumerate().take(s.n_exec_layer as usize) {
                put(&mut out, &self.kcache[il], self.kcache[il].bytes())?;
                put(&mut out, &self.vcache[il], n_comp as usize * s.head_dim as usize * 4)?;
                let ik = &self.idx_kcache[il];
                let used = if ik.bytes() > 4 { hist.len() * s.n_idx_dim as usize * 2 } else { 0 };
                put(&mut out, ik, used)?;
            }
            out.extend_from_slice(&(self.ckpts.len() as u32).to_le_bytes());
            for (pos, ck) in &self.ckpts {
                out.extend_from_slice(&pos.to_le_bytes());
                match ck {
                    RecurrentCkpt::Dsv4(layers) => dsv4::ckpt_write(&mut out, layers)?,
                    _ => return Err("prefix persist: non-dsv4 checkpoint".into()),
                }
            }
            let tmp = path.with_extension("tmp");
            std::fs::write(&tmp, &out)?;
            std::fs::rename(&tmp, path)?;
            Ok(())
        }

        /// Load a prefix written by save_prefix into this (fresh) State.
        /// Returns the persisted history tokens; the caller installs them
        /// as its prompt-cache hist. Shape/ctx mismatches reject the file.
        pub fn load_prefix(&mut self, m: &Model, path: &Path) -> Result<Vec<u32>> {
            let s = m.shape;
            let data = std::fs::read(path)?;
            let mut inp: &[u8] = &data;
            if inp.len() < 8 || &inp[..8] != b"PLSRPFX2" {
                return Err("prefix file: bad magic".into());
            }
            inp = &inp[8..];
            let mut u32s = [0u32; 7];
            for v in &mut u32s {
                *v = u32::from_le_bytes(inp[..4].try_into().unwrap());
                inp = &inp[4..];
            }
            if u32s != [s.n_exec_layer, s.n_embd, s.n_vocab, self.ctx, s.n_swa, s.head_dim, s.n_idx_dim] {
                return Err("prefix file: shape/ctx mismatch".into());
            }
            let nh = u64::from_le_bytes(inp[..8].try_into().unwrap()) as usize;
            inp = &inp[8..];
            let hist: Vec<u32> = inp[..nh * 4]
                .chunks_exact(4)
                .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
                .collect();
            inp = &inp[nh * 4..];
            let rt = self.dsv4.as_mut().ok_or("dsv4 state missing")?;
            rt.load_layers(&mut inp)?;
            let take = |inp: &mut &[u8], b: &mut DeviceBuf| -> Result {
                let n = u64::from_le_bytes(inp[..8].try_into().unwrap()) as usize;
                *inp = &inp[8..];
                if n > b.bytes() {
                    return Err("prefix file: cache larger than allocation".into());
                }
                if n > 0 {
                    b.write(0, &inp[..n])?;
                }
                *inp = &inp[n..];
                Ok(())
            };
            for il in 0..s.n_exec_layer as usize {
                take(&mut inp, &mut self.kcache[il])?;
                take(&mut inp, &mut self.vcache[il])?;
                take(&mut inp, &mut self.idx_kcache[il])?;
            }
            let nck = u32::from_le_bytes(inp[..4].try_into().unwrap()) as usize;
            inp = &inp[4..];
            self.ckpts.clear();
            for _ in 0..nck {
                let pos = u32::from_le_bytes(inp[..4].try_into().unwrap());
                inp = &inp[4..];
                let layers = dsv4::ckpt_read(&mut inp)?;
                self.ckpts.push((pos, RecurrentCkpt::Dsv4(layers)));
            }
            Ok(hist)
        }

        pub fn ckpt_count(&self) -> usize {
            self.ckpts.len()
        }

        pub fn ctx(&self) -> u32 {
            self.ctx
        }

        /// Telemetry snapshot for the /stats endpoint. Cheap: reads counters
        /// and the resident tier list, no device work.
        pub fn stats(&self) -> Stats {
            let tiers: Vec<TierStat> = self
                .tiers
                .iter()
                .map(|t| TierStat {
                    dev: t.dev,
                    bytes: t.pool.bytes(),
                    hits: t.hits,
                })
                .collect();
            let n_gpu = kernels::device_count();
            let gpus = (0..n_gpu)
                .map(|d| {
                    let (free, total) = kernels::mem_info(d).unwrap_or((0, 0));
                    GpuStat { name: String::new(), vram_free: free, vram_total: total }
                })
                .collect();
            // model bytes actually in VRAM: fixed expert tiers + the *used* slabs
            // of the VRAM cache (pool.bytes() is reserved capacity, not occupancy)
            let vram_resident = tiers.iter().map(|t| t.bytes).sum::<usize>()
                + self.dev_cache.map.len() * self.dev_cache.slab_bytes;
            let s = |d: std::time::Duration| d.as_secs_f64();
            Stats {
                gpu_count: n_gpu,
                ctx: self.ctx,
                tiers,
                cpu_hits: self.cpu_hits,
                cache_hits: self.dev_cache.hits,
                gpus,
                host_used: self.store.used,
                host_budget: self.store.budget,
                vram_resident,
                kv_headroom: {
                    let primary = kernels::get_device();
                    self.kv_devs
                        .iter()
                        .map(|&d| {
                            let free = kernels::mem_info(d).map(|(f, _)| f).unwrap_or(0);
                            let tier: usize = self
                                .tiers
                                .iter()
                                .filter(|t| t.dev == d)
                                .map(|t| t.pool.bytes())
                                .sum();
                            // the primary's expert slab cache is reclaimable
                            // too: the auto budget sizes it from whatever the
                            // KV leaves over, so a bigger KV simply gets a
                            // smaller cache rather than failing
                            let slab = if d == primary { self.dev_cache.pool_bytes() } else { 0 };
                            free + tier + slab
                        })
                        .sum()
                },
                kv_resolved: match (self.kvq, self.kvq_lat) {
                    (1, _) | (_, 1) => "fp8",
                    (2, _) | (_, 2) => "fp16",
                    (3, _) => "int8",
                    (4, _) => if self.kvq_rot { "turbo8" } else { "q8_0" },
                    (5, _) => if self.kvq_rot { "turbo4" } else { "q4_0" },
                    (6, _) => "turbo3",
                    (7, _) => "turbo2",
                    (8, _) => "turbo3_tcq",
                    (9, _) => "turbo2_tcq",
                    (10, _) => "turbo1_tcq",
                    _ => "f32",
                },
                kv_compact: self.kvq != 0 || self.kvq_lat != 0,
                kv_bytes: self
                    .kcache
                    .iter()
                    .chain(self.vcache.iter())
                    .chain(self.idx_kcache.iter())
                    .map(|b| b.bytes())
                    .sum(),
                prof_gpu_wait: s(self.prof.sync),
                prof_resolve: s(self.prof.resolve),
                prof_h2d: s(self.prof.h2d),
                prof_fetch: s(self.prof.resolve_fetch),
                prof_cpu: s(self.prof.cpu),
                prof_tail: s(self.prof.tail),
                prof_calls: self.prof.calls,
            }
        }

        /// Per-expert residency + routing heat for the Brain cortex viz. One
        /// cell per (layer, expert) over MoE layers; heat is the live host-cache
        /// touch frequency, tier is where the expert's gate slab currently lives.
        pub fn expert_map(&self, m: &Model) -> Vec<ExpertCell> {
            let n_expert = m.shape.n_expert as usize;
            let mut cells = Vec::new();
            for (l, lw) in m.layers.iter().enumerate() {
                let Ffn::Moe { gate_exps, .. } = &lw.ffn else { continue };
                for e in 0..n_expert {
                    let off = gate_exps.abs_offset + e as u64 * gate_exps.expert_bytes;
                    let tier = if self.tiers.iter().any(|t| t.map.contains_key(&off)) {
                        3u8 // VRAM-resident tier
                    } else if self.dev_cache.map.contains_key(&off) {
                        2 // host RAM cache
                    } else {
                        0 // disk-only
                    };
                    let heat = self.route_counts.get(l * n_expert + e).copied().unwrap_or(0);
                    cells.push(ExpertCell { layer: l as u32, expert: e as u32, tier, heat });
                }
            }
            cells
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
            entries.sort_unstable_by_key(|e| std::cmp::Reverse(e.0));
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
            let mut heat: std::collections::HashMap<u64, (u64, u64)> =
                std::collections::HashMap::new();
            if let Ok(bytes) = std::fs::read(warm_path(&m.path)) {
                heat.reserve(bytes.len() / 24);
                for c in bytes.chunks_exact(24) {
                    let off = u64::from_le_bytes(c[0..8].try_into().unwrap());
                    let len = u64::from_le_bytes(c[8..16].try_into().unwrap());
                    let count = u64::from_le_bytes(c[16..24].try_into().unwrap());
                    heat.insert(off, (count, len));
                }
            } else if let Some(text) = builtin_hotlist(m.shape.family)
                .filter(|_| std::env::var_os("PULSAR_NO_HOTLIST").is_none())
            {
                // first run on a fresh machine: seed from the built-in
                // hotlist so tiers/cache start warm instead of idling
                // until the second run; the real census replaces it on
                // save_warm.
                hotlist_heat(m, text, &mut heat);
                if !heat.is_empty() {
                    eprintln!(
                        "pulsar: no census yet - warm set seeded from built-in hotlist ({} slabs)",
                        heat.len()
                    );
                }
            }
            if heat.is_empty() {
                return Ok(0);
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
            triples.sort_unstable_by_key(|e| std::cmp::Reverse(e.0));
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
            extras.sort_unstable_by_key(|e| std::cmp::Reverse(e.0));
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
            // PULSAR_KV selects the Gqa KV storage format, opt-in and
            // lossy (the default f32 path keeps bit-exact guarantees):
            //   fp8  -> e4m3 + per-row f32 scale (stride head_dim+4, ~3.9x)
            //   fp16 -> IEEE half           (stride head_dim*2,   ~2.0x)
            //   int8 -> int8 + per-row scale(stride head_dim+4,  ~4.0x)
            //   q8_0 -> 32-wide blocks f16 d + 32 i8 (stride head_dim/32*34)
            //   q4_0 -> 32-wide blocks f16 d + 16 nibbles (stride head_dim/32*18)
            //   turbo3/turbo2 / *_tcq -> dsv4-only sub-4-bit rotated centroids
            // MLA keeps its compact latent cache as-is.
            // turbo<4|8> / rotq<4|8> = q4_0/q8_0 with a fixed orthogonal
            // rotation folded into K (pre-append) and Q (pre-attention).
            // Rotation spreads per-32-block outliers across the block so no
            // single lane dominates blockmax `d` — see TurboQuant. Decode-
            // invariant: (Q@Πᵀ)·(K@Πᵀ)ᵀ = Q@Kᵀ since ΠᵀΠ=I. V is untouched.
            // turbo3/2 + tcq force rotation too: Π is the codec's own
            // block-diagonal FWHT, not the MGS matrix used by turbo4/8.
            //
            // qwen35-dense (n_expert==1) runs the dense-split path, which does
            // not support the quantized KV layout - applying it deadlocked the
            // forward (GPUs idle, no output). Keep dense on f32, and warn loudly
            // if PULSAR_KV was requested but the arch can't honor it (never
            // silently apply-and-hang; a stale env carries over on model switch).
            let kv_req = std::env::var("PULSAR_KV").ok();
            // ...but TENSOR PARALLEL replaces the dense split entirely
            // (no cross-card residual hop mid-forward, each card owns its
            // kv heads), so the deadlock cause is absent and quantized KV
            // is what makes long contexts fit: f32 costs 64KB/pos/card on
            // Qwen3.8-27B, fp8 ~16KB.
            // The deadlock this guards against was diagnosed before the
            // multi-bank rewrite. PULSAR_KV_DENSE=1 lifts it for testing:
            // fp8 KV is worth 4x the attention bandwidth (which prefill is
            // still bound by) and 4x the context ceiling.
            let kv_dense = s.family == Family::Qwen35
                && s.n_expert == 1
                && m.tp.is_none()
                && std::env::var("PULSAR_KV_DENSE").is_err();
            // Exhaustive: whether a family can honor PULSAR_KV is a property
            // of its cache layout, and the wrong answer here is not a slow
            // path but a hang (see the qwen35-dense note above). A new family
            // must say so rather than inherit whatever this line happened to
            // mean when it was written.
            let kv_ok = match s.family {
                // GQA keeps a plain [layer][kv_head][pos] cache the quant
                // kernels understand; qwen35's full-attention layers use the
                // same one, minus the dense split path. Dsv4's fused latent
                // row is a flat 512-wide vector divisible by 32, so every
                // codec (fp8/fp16/int8/q8_0/q4_0/turbo) applies to it as one
                // "head"; it rides the same kvq field as GQA.
                Family::Gqa | Family::Qwen35 | Family::Dsv4 => !kv_dense,
                // MLA carries its own compact latent cache (kvq_lat below);
                // K3's MLA quarter does too, and its KDA layers keep no KV
                // at all
                Family::Mla | Family::K3 => false,
            };
            // The MLA latent cache has its own quantized formats (fp8/fp16,
            // kvq_lat below) through mla_store_compact_kv/mla_attention.
            let kv_lat_ok = s.family == Family::Mla;
            // f32 KV projection across exec layers (+ MTP slot), mirroring
            // the per-layer sizing loop below. Only consulted for kv_ok /
            // kv_lat_ok families, so the qwen35-dense shape never reaches it.
            let kv_f32_total = || -> usize {
                let slots = s.n_exec_layer as usize + usize::from(m.mtp.is_some());
                (0..slots)
                    .map(|i| {
                        if s.family == Family::Mla {
                            (s.n_kv_lora + s.qk_rope) as usize * ctx as usize * 4
                        } else if s.family == Family::Qwen35 {
                            if i == s.n_exec_layer as usize
                                || (i as u32 + 1).is_multiple_of(s.full_attn_interval)
                            {
                                2 * s.n_head_kv as usize * ctx as usize * s.head_dim as usize * 4
                            } else {
                                8
                            }
                        } else if s.family == Family::Dsv4 {
                            // raw SWA ring + compressed rows (vcache) are
                            // both flat head_dim-wide latent rows in f32
                            let ratio = m.compress_ratios.get(i).copied().unwrap_or(0) as usize;
                            let comp = (ctx as usize).checked_div(ratio).map_or(0, |c| c + 2);
                            (s.n_swa as usize + comp) * s.head_dim as usize * 4
                        } else {
                            let (hkv, hd) = match m.geom.get(i) {
                                Some(g) => (g.n_head_kv as usize, g.head_dim as usize),
                                None => (s.n_head_kv as usize, s.head_dim as usize),
                            };
                            2 * hkv * ctx as usize * hd * 4
                        }
                    })
                    .sum()
            };
            // Shared adaptive-default rule (see the None arm below for the
            // full rationale): big f32 KV on a streaming model silently
            // starves the expert cache, so above 2GB absolute AND a third
            // of the KV card's free VRAM the default flips to a quantized
            // cache (int8 for GQA, fp8 for the MLA latent - the only
            // quantized latent format). The chosen codec logs itself.
            let kv_auto_quant = |total: usize| -> bool {
                let kv_dev = m.attn_dev.unwrap_or_else(kernels::get_device);
                let free = kernels::mem_info(kv_dev).map(|(f, _)| f).unwrap_or(usize::MAX);
                if total > (2usize << 30) && total > free / 3 {
                    eprintln!(
                        "pulsar: KV auto: f32 KV at ctx {} would be {:.1}GiB of {:.1}GiB free -> quantizing to ~{:.1}GiB; set PULSAR_KV=f32 to force exact f32 KV",
                        ctx,
                        total as f64 / GIB,
                        free as f64 / GIB,
                        total as f64 / 3.9e9,
                    );
                    true
                } else {
                    false
                }
            };
            let (kvq, kvq_rot) = if kv_ok {
                let (q, rot) = match kv_req.as_deref() {
                    Some("fp8") => (1, false),
                    Some("fp16") | Some("f16") => (2, false),
                    Some("int8") | Some("i8") => (3, false),
                    Some("q8_0") | Some("q8") => (4, false),
                    Some("q4_0") | Some("q4") => (5, false),
                    Some("turbo8") | Some("rotq8") | Some("turboq8") => (4, true),
                    Some("turbo4") | Some("rotq4") | Some("turboq4") => (5, true),
                    // buun turbo2/3 + trellis variants: intrinsically rotated
                    // (the encoder quantizes FWHT-rotated 128-groups), so all
                    // five force rotation on — Π is the codec's own R, see the
                    // pi build below.
                    Some("turbo3") => (6, true),
                    Some("turbo2") => (7, true),
                    Some("turbo3_tcq") => (8, true),
                    Some("turbo2_tcq") => (9, true),
                    Some("turbo1_tcq") => (10, true),
                    None
                        // A too-big f32 KV never OOMs on a streaming model -
                        // it silently eats the expert cache instead (measured:
                        // gpt-oss ctx 131072 one card, f32 left 0.4GB of
                        // expert cache and prefill chunk 4; fp8 left 9.7GB
                        // and chunk 256). So when nothing was requested and
                        // the projection is both large and a big share of
                        // the KV card's free VRAM, quantize. The 2GB
                        // absolute floor keeps small-ctx runs (bench.sh 512,
                        // check.sh 256) on the bit-exact f32 path.
                        //
                        // int8, not fp8: identical stride (head_dim+4) and
                        // scripts/kld-ab.sh scores it strictly better on
                        // Laguna - median KLD 0.0068 vs 0.0081, top-1 94.0%
                        // vs 92.9%. e4m3 spends 4 of its 8 bits on exponent
                        // range the per-row scale already provides.
                        if kv_auto_quant(kv_f32_total()) => { (3, false) }
                    _ => (0, false),
                };
                // Rotation writes head_dim-strided vectors into qrot, which is
                // sized n_head*head_dim per token. `q` itself is allocated at
                // head_dim.max(qk_dim()), so a head whose qk_dim exceeds
                // head_dim would leave the two strides disagreeing and the
                // rotation would read across head boundaries. qk_nope/qk_rope
                // are MLA-only and stay 0 for Gqa/Qwen35, so this holds today;
                // the guard keeps it from breaking silently if that changes.
                // Dsv4's q reads as [n_head][head_dim] (no qrot/qk_low split),
                // so it deliberately skips this guard and keeps rotation.
                // codes 6-10 (turbo2/3 + trellis) have encode arms only in
                // the dsv4 store/comp paths; the GQA append switch would
                // fall through to a plain f32 append and overflow the
                // smaller rows, so they are dsv4-only.
                if q >= 6 && s.family != Family::Dsv4 {
                    eprintln!(
                        "pulsar: PULSAR_KV={} ignored - turbo2/turbo3 and the tcq codecs are dsv4-only; using f32 KV",
                        kv_req.as_deref().unwrap_or(""),
                    );
                    (0, false)
                } else if rot && s.qk_dim() > s.head_dim && matches!(s.family, Family::Gqa | Family::Qwen35) {
                    eprintln!(
                        "pulsar: PULSAR_KV={} rotation disabled - qk_dim {} exceeds head_dim {} (split-rope q stride); falling back to plain block-KV",
                        kv_req.as_deref().unwrap_or(""),
                        s.qk_dim(),
                        s.head_dim,
                    );
                    (q, false)
                } else {
                    (q, rot)
                }
            } else {
                if !kv_lat_ok && kv_req.as_deref().is_some_and(|v| !v.is_empty() && v != "f32") {
                    eprintln!(
                        "pulsar: PULSAR_KV={} ignored - this arch keeps its f32 KV cache (quantized KV unsupported here)",
                        kv_req.as_deref().unwrap_or("")
                    );
                }
                (0, false)
            };
            // MLA latent storage format: 0=f32 (exact), 1=fp8 e4m3 +
            // per-row scale (~3.9x), 2=fp16 (2x). Block formats need
            // head-shaped rows, which the flat latent is not.
            let kvq_lat: u32 = if kv_lat_ok {
                match kv_req.as_deref() {
                    Some("fp8") => 1,
                    Some("fp16") | Some("f16") => 2,
                    None => {
                        if kv_auto_quant(kv_f32_total()) { 1 } else { 0 }
                    }
                    Some(v) if !v.is_empty() && v != "f32" => {
                        eprintln!(
                            "pulsar: PULSAR_KV={v} unsupported for the MLA latent cache (fp8|fp16|f32 only) - keeping f32"
                        );
                        0
                    }
                    _ => 0,
                }
            } else {
                0
            };
            let kv_row = |hd: usize| match kvq {
                0 => hd * 4,
                1 | 3 => hd + 4,      // per-row f32 scale
                2 => hd * 2,          // pure fp16
                4 => (hd / 32) * 34,  // q8_0: 34 B / 32 elems
                5 => (hd / 32) * 18,  // q4_0: 18 B / 32 elems
                // buun turbo blocks: f16 norm + packed codes, 4 sub-blocks
                // per 128-group sharing the corrected group norm
                6 => (hd / 32) * 14,  // turbo3: 2+8+4 B / 32 elems
                7 => (hd / 32) * 10,  // turbo2: 2+8 B / 32 elems
                // trellis: f16 norm + 6/6/7 init bits + 128 symbols of 3/2/1
                // bits, one packed qs per 128-group
                8 => (hd / 128) * 52, // turbo3_tcq: 3+49 B / 128 elems
                9 => (hd / 128) * 36, // turbo2_tcq: 2+33 B / 128 elems
                10 => (hd / 128) * 20, // turbo1_tcq: 2+17 B / 128 elems
                _ => hd * 4,
            };
            // fp8 latent rows carry one f32 scale at the tail, mirroring
            // the GQA fp8 layout (and mla_lat_stride on the kernel side)
            let lat_row = |d: usize| match kvq_lat {
                1 => d + 4,
                2 => d * 2,
                _ => d * 4,
            };
            let (k_bytes, v_bytes) = match s.family {
                Family::Gqa => {
                    let b = s.n_head_kv as usize * ctx as usize * kv_row(s.head_dim as usize);
                    (b, b)
                }
                Family::Mla => (
                    ctx as usize * lat_row(s.n_kv_lora as usize),
                    ctx as usize * lat_row(s.qk_rope as usize),
                ),
                // raw SWA ring in kcache; the compressed-row cache rides
                // vcache, sized per layer in the loop below. Both are flat
                // latent rows, quantized per kvq like a single GQA head.
                Family::Dsv4 => (
                    s.n_swa as usize * kv_row(s.head_dim as usize),
                    4,
                ),
                Family::Qwen35 => {
                    let b = s.n_head_kv as usize * ctx as usize * kv_row(s.head_dim as usize);
                    (b, b)
                }
                // K3 is per-layer mixed: these are the MLA-layer sizes
                // (same compact latent cache as Family::Mla). The KDA
                // layers take a placeholder in the loop below - their
                // history is the delta state in K3Rt, not a cache.
                Family::K3 => (
                    ctx as usize * lat_row(s.n_kv_lora as usize),
                    ctx as usize * lat_row(s.qk_rope as usize),
                ),
            };
            if kvq != 0 {
                // f32 baseline footprint: GQA/Qwen35 keep one row per head
                // per position; Dsv4 keeps one flat latent row per position
                // (n_swa raw ring + ctx/ratio compressed rows, per layer)
                let (full, packed) = if s.family == Family::Dsv4 {
                    (
                        (0..s.n_exec_layer as usize + usize::from(m.mtp.is_some()))
                            .map(|i| {
                                let ratio = m.compress_ratios.get(i).copied().unwrap_or(0) as usize;
                                let comp = (ctx as usize).checked_div(ratio).map_or(0, |c| c + 2);
                                (s.n_swa as usize + comp) * s.head_dim as usize * 4
                            })
                            .sum(),
                        (0..s.n_exec_layer as usize + usize::from(m.mtp.is_some()))
                            .map(|i| {
                                let ratio = m.compress_ratios.get(i).copied().unwrap_or(0) as usize;
                                let comp = (ctx as usize).checked_div(ratio).map_or(0, |c| c + 2);
                                (s.n_swa as usize + comp) * kv_row(s.head_dim as usize)
                            })
                            .sum(),
                    )
                } else {
                    (
                        s.n_head_kv as usize * ctx as usize * s.head_dim as usize * 4,
                        (k_bytes + v_bytes) * s.n_exec_layer as usize,
                    )
                };
                let name = match kvq {
                    1 => "fp8",
                    2 => "fp16",
                    3 => "int8",
                    4 => if kvq_rot { "turbo8" } else { "q8_0" },
                    5 => if kvq_rot { "turbo4" } else { "q4_0" },
                    6 => "turbo3",
                    7 => "turbo2",
                    8 => "turbo3_tcq",
                    9 => "turbo2_tcq",
                    _ => "turbo1_tcq",
                };
                // Dsv4's full is already ring+comp across all layers (its
                // kcache=raw ring, vcache=compressed rows are distinct data,
                // not K/V copies); GQA/Qwen35 double for the K and V copies.
                let full_gib = if s.family == Family::Dsv4 {
                    full as f64 / GIB
                } else {
                    (full * 2 * s.n_exec_layer as usize) as f64 / GIB
                };
                eprintln!(
                    "pulsar: {name} KV cache on ({:.2} GiB -> {:.2} GiB over {} layers)",
                    full_gib,
                    packed as f64 / GIB,
                    s.n_exec_layer,
                );
                // q8_0/q4_0/turbo2/turbo3 kernels require 32-wide sub-blocks
                // and the trellis codecs 128-wide groups; a non-multiple
                // head_dim makes the append guard return 0 silently (cache
                // stays uninitialized).
                if matches!(kvq, 4 | 5) {
                    eprintln!(
                        "pulsar: block-KV head_dim={} ({}divisible by 32)",
                        s.head_dim,
                        if s.head_dim.is_multiple_of(32) { "" } else { "NOT " }
                    );
                }
                if matches!(kvq, 6..=10) {
                    eprintln!(
                        "pulsar: block-KV head_dim={} ({}divisible by 128)",
                        s.head_dim,
                        if s.head_dim.is_multiple_of(128) { "" } else { "NOT " }
                    );
                }
                if kvq_rot {
                    eprintln!(
                        "pulsar: turbo rotation ON — K/Q rotated by orthogonal Π (head_dim={}) before block-quant",
                        s.head_dim,
                    );
                }
            }
            if kvq_lat != 0 {
                let full = (s.n_kv_lora + s.qk_rope) as usize * ctx as usize * 4;
                eprintln!(
                    "pulsar: {} MLA latent KV cache on ({:.2} GiB -> {:.2} GiB over {} layers)",
                    if kvq_lat == 1 { "fp8" } else { "fp16" },
                    (full * s.n_exec_layer as usize) as f64 / GIB,
                    ((k_bytes + v_bytes) * s.n_exec_layer as usize) as f64 / GIB,
                    s.n_exec_layer,
                );
            }
            // batch prefill: activations sized for max_batch tokens; the
            // logits/lm-head path stays single-row (last token only)
            // big default: each prefill chunk costs roughly one pass over
            // the expert corpus regardless of chunk size, so fewer chunks
            // win; activations at 512 cost only ~150MB.
            // Regime-dependent, measured 2026-07-24 on a 728-token prompt:
            // when the corpus streams (Hy3 iq2_xxs, 84GB) a "pass" is real
            // disk/PCIe traffic and chunk 128/256/768 = 38.2/30.4/24.3s, so
            // fewer chunks wins 1.6x. When the corpus is already resident in
            // tiers (Laguna Q2K, 39GB over 48GB of VRAM) there is nothing to
            // re-fetch and it goes flat: chunk 768/1536/2304 = 45.3/48.0/48.2s
            // on 4327 tokens. Cost tracks tokens, not chunks, in that regime.
            let spec_rows = (m.mtp_depth + 1)
                .max(2)
                // qwen35 DFlash verify reads logits for a whole 16-row
                // block; dsv4 verify is capped by its T_MAX sub-batch, so
                // 16 covers it too. Costs rows*n_vocab*4 of logit buffer
                // (8MB at deepseek4's 129k vocab), which is noise next to
                // the expert cache it sits beside.
                .max(if matches!(s.family, Family::Qwen35 | Family::Dsv4) { 16 } else { 0 })
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
                .max(1)
                // the qwen35 prefill chunk rides State's generic scratch
                // (cur/normed/xq/ffn set), so a wider PULSAR_PREFILL_CHUNK
                // lifts the batch floor with it
                .max(qwen35::chunk_max() as u32);

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
            let attn_split = m.attn_layer_dev.windows(2).any(|w| w[0] != w[1]);
            // Which cards actually host KV. A caller projecting a larger ctx
            // must count headroom on THESE devices only: KV cannot migrate to
            // a card that holds expert tiers. Summing every GPU's free VRAM
            // let a Gqa model (KV on the primary) borrow 28GiB of tier sitting
            // on two other cards, and the resize died at cudaMalloc.
            let mut kv_devs: Vec<i32> = Vec::new();
            for i in 0..n_kv_slots {
                let d = if dense_split {
                    m.layer_dev.get(i).copied().unwrap_or(primary)
                } else if attn_split {
                    m.attn_layer_dev.get(i).copied().unwrap_or(primary)
                } else {
                    m.attn_dev.unwrap_or(primary)
                };
                if !kv_devs.contains(&d) {
                    kv_devs.push(d);
                }
            }
            for i in 0..n_kv_slots {
                if dense_split {
                    // dense split: KV lives with its layer (MTP slot ->
                    // primary, where the tail runs)
                    kernels::set_device(m.layer_dev.get(i).copied().unwrap_or(primary))?;
                } else if attn_split {
                    // attn layer split: KV lives with its layer's owner
                    kernels::set_device(m.attn_layer_dev.get(i).copied().unwrap_or(primary))?;
                }
                // per-layer geometry (gemma4): a SWA layer's cache is its
                // own kv width, not the Shape max
                let (kb, vb) = if s.family == Family::Qwen35 {
                    // only full-attention layers hold KV; the nextn/MTP
                    // draft slot is a full-attention layer too (and stays
                    // whole on the primary, so it keeps full width).
                    // TP-split layers own HALF the kv heads per card -
                    // allocating full width there wasted 3.2GB a card at
                    // ctx 200k, which is the difference between fitting
                    // and not.
                    if i == s.n_exec_layer as usize {
                        (k_bytes, v_bytes)
                    } else if (i as u32 + 1).is_multiple_of(s.full_attn_interval) {
                        if m.tp_attn(i).is_some() {
                            (k_bytes / 2, v_bytes / 2)
                        } else {
                            (k_bytes, v_bytes)
                        }
                    } else {
                        (4, 4)
                    }
                } else if s.family == Family::K3 {
                    // only the MLA layers cache anything; ask the loaded
                    // weights rather than recomputing the 3-then-1 pattern
                    let is_mla = matches!(m.layers.get(i).map(|l| &l.attn),
                                          Some(Attn::K3(w)) if w.mla.is_some());
                    if is_mla { (k_bytes, v_bytes) } else { (4, 4) }
                } else if s.family == Family::Dsv4 {
                    let ratio = m.compress_ratios.get(i).copied().unwrap_or(0) as usize;
                    let comp = (ctx as usize)
                        .checked_div(ratio)
                        .map_or(4, |c| (c + 2) * kv_row(s.head_dim as usize));
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
            // TP attention halves: card B's kv-head caches, same codec
            // and stride math as A's
            let mut tp_kcache: Vec<Option<(DeviceBuf, DeviceBuf)>> = Vec::new();
            if let Some(tp) = &m.tp {
                kernels::set_device(tp.dev)?;
                for il in 0..s.n_exec_layer as usize {
                    tp_kcache.push(if m.tp_attn(il).is_some() {
                        let b = (s.n_head_kv as usize / 2) * ctx as usize * kv_row(s.head_dim as usize);
                        Some((DeviceBuf::alloc(b)?, DeviceBuf::alloc(b)?))
                    } else {
                        None
                    });
                }
            }
            if dense_split || m.tp.is_some() {
                kernels::set_device(primary)?;
            } else if attn_split {
                kernels::set_device(m.attn_dev.unwrap_or(primary))?;
            }
            // Mla keeps its attention scratch in attn_sc (one set per
            // owner device); the flat fields become stubs so the single-
            // card case pays no duplicate VRAM
            let mla = s.family == Family::Mla;
            let q = f32s(if mla { 1 } else { mb * s.n_head * s.head_dim.max(s.qk_dim()) })?;
            let heads = f32s(if mla { 1 } else { mb * s.heads_dim().max(s.n_head * s.head_dim) })?;
            let q_rank = f32s(if mla { 1 } else { mb * s.n_lora_q.max(1) })?;
            let q_rank_norm = f32s(if mla { 1 } else { mb * s.n_lora_q.max(1) })?;
            // DSA indexer buffers live beside the attn stack (same device)
            let has_idx = s.n_idx_topk > 0 && s.family == Family::Mla;
            let mut idx_kcache = Vec::new();
            // n_kv_slots, not n_exec_layer: the MTP draft layer runs the
            // same MLA path as slot n_exec_layer and maintains its own
            // indexer keys
            // f16 keys by default; fp8 (e4m3 + f32 row scale) rides the
            // same PULSAR_KV=fp8 that quantizes the latent cache
            let idx8 = (kvq_lat == 1) as u32;
            let idx_row = if idx8 != 0 { s.n_idx_dim as usize + 4 } else { s.n_idx_dim as usize * 2 };
            for il in 0..n_kv_slots {
                if attn_split {
                    kernels::set_device(m.attn_layer_dev.get(il).copied().unwrap_or(primary))?;
                }
                idx_kcache.push(if has_idx && uses_full_indexer(il, s.n_leading_dense) {
                    DeviceBuf::alloc(ctx as usize * idx_row)?
                } else {
                    f32s(1)?
                });
            }
            if attn_split {
                kernels::set_device(m.attn_dev.unwrap_or(primary))?;
            }
            // per-owner attention scratch (Mla only; see MlaScratch)
            let mut attn_sc: Vec<MlaScratch> = Vec::new();
            if mla {
                let mut devs: Vec<i32> = Vec::new();
                for &d in m.attn_layer_dev.iter() {
                    if !devs.contains(&d) {
                        devs.push(d);
                    }
                }
                for d in devs {
                    kernels::set_device(d)?;
                    let off = d != primary; // hop buffers only exist off-primary
                    attn_sc.push(MlaScratch {
                        dev: d,
                        normed_a: f32s(if off { mb * s.n_embd } else { 1 })?,
                        attn_out_a: f32s(if off { mb * s.n_embd } else { 1 })?,
                        q_rank: f32s(mb * s.n_lora_q.max(1))?,
                        q_rank_norm: f32s(mb * s.n_lora_q.max(1))?,
                        q: f32s(mb * s.n_head * s.head_dim.max(s.qk_dim()))?,
                        kv_raw: f32s(mb * (s.n_kv_lora + s.qk_rope).max(1))?,
                        kv_norm: f32s(mb * s.n_kv_lora.max(1))?,
                        qk_low: f32s(mb * s.n_head * s.n_kv_lora.max(1))?,
                        heads: f32s(mb * s.heads_dim().max(s.n_head * s.head_dim))?,
                        idx_kraw: f32s(if has_idx { mb * s.n_idx_dim } else { 1 })?,
                        idx_q: f32s(if has_idx { mb * s.n_idx_head * s.n_idx_dim } else { 1 })?,
                        idx_q16: DeviceBuf::alloc(if has_idx { (mb * s.n_idx_head * s.n_idx_dim) as usize * 2 } else { 1 })?,
                        idx_w: f32s(if has_idx { mb * s.n_idx_head } else { 1 })?,
                        idx_scores: f32s(if has_idx { mb * ctx } else { 1 })?,
                        mla_selected: DeviceBuf::alloc(mb as usize * ctx as usize * 4)?,
                    });
                }
                kernels::set_device(m.attn_dev.unwrap_or(primary))?;
            }
            let (normed_a, attn_out_a) = if m.attn_dev.is_some() && !mla {
                (f32s(mb * s.n_embd)?, f32s(mb * s.n_embd)?)
            } else {
                (f32s(1)?, f32s(1)?)
            };
            // laguna output gate: one logit per (token, query head)
            let has_gate = m.layers.iter().any(|l| l.attn_gate.is_some());
            let attn_gate_buf = f32s(if has_gate { mb * s.n_head } else { 1 })?;
            // Gqa attention scratch beside the KV caches (attn card under
            // offload, primary otherwise): raw k/v projections, inkling's
            // rel-bias buffers and the k/v-stream shortconv state+tmp
            let kbuf = f32s(mb * s.n_head_kv * s.head_dim)?;
            let vbuf = f32s(mb * s.n_head_kv * s.head_dim)?;
            // turbo rotation: orthogonal Π (head_dim×head_dim) built host-side
            // via modified Gram-Schmidt on a fixed xorshift seed — deterministic,
            // zero deps. Valid for any uniform-head_dim GQA layout: qk_nope/
            // qk_rope are MLA-only fields (0 for Gqa/Qwen35), so q's stride is
            // head_dim, which is what qrot mirrors (the PULSAR_KV parse guards
            // the qk_dim > head_dim case). Rope (rot_w lanes) is baked into
            // head_dim; rotating the full head preserves Q·Kᵀ regardless of the
            // internal nope/rope subdivision. Family is already gated by the
            // parse (Gqa|Qwen35), so kvq_rot flows through unchecked.
            //
            // The whole scheme rests on ΠᵀΠ = I, so Π is verified before use
            // rather than assumed: MGS loses orthogonality with the condition
            // number, and the degenerate-row fallback below patches a row
            // without re-orthogonalizing it. Either would silently break
            // decode-invariance, so a failed check disables rotation loudly.
            let mut kvq_rot = kvq_rot;
            let pi = if kvq_rot {
                let hd = s.head_dim as usize;
                let n = hd * hd;
                let mut m = vec![0.0f32; n];
                if kvq >= 6 {
                    // turbo3/turbo2 and the tcq trellis codecs are
                    // intrinsically rotated: their CUDA encoders quantize
                    // FWHT-rotated 128-groups and the decode codebooks live
                    // in rotated space, so Π must be the codec's own
                    // R = diag(s2)·FWHT₁₂₈·diag(s1)/√128 (buun seed-42 sign
                    // tables, transcribed verbatim), block-diagonal per
                    // 128-group. Built by pushing each unit vector through
                    // the same xor-butterfly the reference kernel runs, so
                    // there is no closed-form Hadamard indexing to get
                    // wrong. R is orthogonal by construction; the shared
                    // verify below still catches f32 drift.
                    const WHT_SIGNS1: [f32; 128] = [
                        -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0,
                        -1.0, -1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                        1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0,
                        1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0,
                        -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0,
                        -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0,
                        1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0, -1.0,
                        1.0, -1.0, 1.0, 1.0, 1.0, 1.0, -1.0, 1.0,
                        -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0,
                        -1.0, -1.0, 1.0, -1.0, -1.0, -1.0, 1.0, 1.0,
                        1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0,
                        1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0,
                        -1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
                        1.0, 1.0, 1.0, 1.0, -1.0, 1.0, -1.0, 1.0,
                        1.0, -1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0,
                        -1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0,
                    ];
                    const WHT_SIGNS2: [f32; 128] = [
                        1.0, 1.0, 1.0, 1.0, -1.0, 1.0, 1.0, -1.0,
                        1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, -1.0,
                        1.0, 1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0,
                        1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, 1.0,
                        1.0, 1.0, -1.0, -1.0, -1.0, 1.0, -1.0, -1.0,
                        -1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, -1.0,
                        1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0,
                        -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0,
                        1.0, -1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0,
                        -1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0,
                        -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0,
                        -1.0, -1.0, -1.0, 1.0, -1.0, -1.0, 1.0, -1.0,
                        1.0, -1.0, 1.0, 1.0, 1.0, -1.0, -1.0, 1.0,
                        -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0,
                        -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0,
                        1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 1.0, -1.0,
                    ];
                    for blk in 0..hd / 128 {
                        let o = blk * 128;
                        for col in 0..128usize {
                            let mut x = [0.0f32; 128];
                            x[col] = WHT_SIGNS1[col];
                            // xor-mask butterfly h=1,2,…,64 — the same
                            // pairing as the reference kernel's shuffle
                            // butterflies, each {j, j^h} pair once.
                            let mut h = 1usize;
                            while h < 128 {
                                for j in 0..128 {
                                    let k = j ^ h;
                                    if k > j {
                                        let (a, b) = (x[j], x[k]);
                                        x[j] = a + b;
                                        x[k] = a - b;
                                    }
                                }
                                h <<= 1;
                            }
                            for j in 0..128 {
                                m[(o + j) * hd + (o + col)] =
                                    WHT_SIGNS2[j] * x[j] * 0.08838834764831845f32;
                            }
                        }
                    }
                } else {
                    let mut g = 0x9E3779B97F4A7C15u64;
                    let mut rng = || {
                        // xorshift64 on the fixed seed — only need a spread of
                        // directions; orthogonality comes from MGS, not the RNG.
                        g ^= g << 13;
                        g ^= g >> 7;
                        g ^= g << 17;
                        ((g % 1_000_000) as f64 / 1_000_000.0 - 0.5) * 2.0
                    };
                    for i in 0..hd {
                        for j in 0..hd {
                            m[i * hd + j] = rng() as f32;
                        }
                    }
                    // Modified Gram-Schmidt: orthonormalize rows in place.
                    for i in 0..hd {
                        let (si, ei) = (i * hd, (i + 1) * hd);
                        for k in 0..i {
                            let (sk, ek) = (k * hd, (k + 1) * hd);
                            let dot = m[si..ei]
                                .iter()
                                .zip(&m[sk..ek])
                                .map(|(a, b)| a * b)
                                .sum::<f32>();
                            for j in 0..hd {
                                m[si + j] -= dot * m[sk + j];
                            }
                        }
                        let norm = m[si..ei].iter().map(|x| x * x).sum::<f32>().sqrt();
                        if norm < 1e-6 {
                            // Degenerate (vanishingly unlikely at f32 from a fixed
                            // seed); nudge row i to e_i to keep Π invertible.
                            for j in 0..hd {
                                m[si + j] = if j == i { 1.0 } else { 0.0 };
                            }
                        } else {
                            for j in 0..hd {
                                m[si + j] /= norm;
                            }
                        }
                    }
                }
                // Verify ΠΠᵀ = I. Square + orthonormal rows gives ΠᵀΠ = I
                // too, which is the identity (QΠᵀ)·(KΠᵀ)ᵀ = QKᵀ relies on.
                let mut worst = 0.0f32;
                for i in 0..hd {
                    for j in 0..hd {
                        let dot: f32 = (0..hd).map(|k| m[i * hd + k] * m[j * hd + k]).sum();
                        let want = if i == j { 1.0 } else { 0.0 };
                        worst = worst.max((dot - want).abs());
                    }
                }
                // MGS at f32 lands near 1e-5 for head_dim 128. 1e-3 sits well
                // above that drift floor and well below a deviation that would
                // move attention, so it separates "normal" from "broken".
                if worst > 1e-3 {
                    eprintln!(
                        "pulsar: turbo rotation disabled - Π failed orthogonality check (max |ΠΠᵀ-I| = {worst:.2e}, head_dim {hd}); falling back to plain block-KV",
                    );
                    kvq_rot = false;
                    f32s(1)?
                } else {
                    let mut pi = f32s(n as u32)?;
                    pi.write(0, kernels::as_bytes(&m))?;
                    pi
                }
            } else {
                f32s(1)?
            };
            // Rotated-K/Q scratch — sized only when rotation engages so the
            // q4_0/q8_0 kernels receive the rotated source buffer. Mirror
            // the layout the call sites expect: K is n_head_kv*head_dim per
            // token, Q is n_head*head_dim per token (attention head stride).
            let krot = if kvq_rot {
                f32s(mb * s.n_head_kv * s.head_dim)?
            } else {
                f32s(1)?
            };
            let qrot = if kvq_rot {
                f32s(mb * s.n_head * s.head_dim)?
            } else {
                f32s(1)?
            };
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
                spec_rows,
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
                    mb as usize
                        * (s.n_embd as usize).div_ceil(kernels::Q8_K_BLOCK_ELEMS)
                        * kernels::Q8_K_BLOCK_BYTES,
                )?,
                midq: DeviceBuf::alloc(
                    mb as usize
                        * n_used
                        * (s.n_ff_exp as usize).div_ceil(kernels::Q8_K_BLOCK_ELEMS)
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
                tp_kcache,
                amax_out: DeviceBuf::alloc(spec_rows.max(1) as usize * 8)?,
                last_argmax: Vec::new(),
                skip_logit_read: false,
                kvq,
                kv_devs,
                kvq_lat,
                kvq_rot,
                pi,
                krot,
                qrot,
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
                attn_sc,
                sel_dev: -1,
                idx_kcache,
                idx_last_sel: 0,
                normed_a,
                attn_out_a,
                attn_gate_buf,
                tier_ret: if tiers.is_empty() { f32s(1)? } else { f32s(mb * s.n_embd)? },
                // The CPU lane dots quantized weights directly and has no
                // bias term, so on an arch that carries expert biases it
                // would silently drop them for whatever it steals. Refuse
                // the lane rather than be quietly wrong.
                cpu_pool: {
                    let has_bias = m.layers.iter().any(|l| {
                        matches!(&l.ffn, Ffn::Moe { exp_bias: Some(_), .. })
                    });
                    match (cpu_tier::Pool::from_env(), has_bias) {
                        (Some(_), true) => {
                            eprintln!(
                                "pulsar: CPU expert lane disabled - this model carries per-expert \
                                 biases and the lane has no bias path"
                            );
                            None
                        }
                        (p, _) => p,
                    }
                },
                cpu_ret: f32s(1)?, // grows on first CPU-lane hit
                cpu_hits: 0,
                pred_prev: Vec::new(),
                pred_prev_for: usize::MAX,
                route_counts: vec![0u64; (m.shape.n_layer * m.shape.n_expert.max(1)) as usize],
                tiers,
                grp_ptrs: DeviceBuf::alloc(s.n_expert.max(1) as usize * std::mem::size_of::<ExpertPtrs>())?,
                grp_starts: DeviceBuf::alloc((s.n_expert as usize + 1) * 4)?,
                grp_pairs: DeviceBuf::alloc(mb as usize * n_used * 4)?,
                grp_partial: f32s(1)?, // grows on first grouped prefill
                mtp_e_raw: f32s(if m.mtp.is_some() { mb * s.n_embd } else { 1 })?,
                mtp_e: f32s(if m.mtp.is_some() { mb * s.n_embd } else { 1 })?,
                mtp_h: f32s(if m.mtp.is_some() { mb * s.n_embd } else { 1 })?,
                mtp_x: f32s(if m.mtp.is_some() { mb * 2 * s.n_embd } else { 1 })?,
                // q8_K activations for a native-quant eh_proj
                mtp_xq: DeviceBuf::alloc(if m.mtp.is_some() {
                    (mb * 2 * s.n_embd) as usize / kernels::Q8_K_BLOCK_ELEMS
                        * kernels::Q8_K_BLOCK_BYTES
                } else {
                    1
                })?,
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
                mtp_pos_accepted: vec![0; 8],
                mtp_pos_drafted: vec![0; 8],
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
                k3: if s.family == Family::K3 {
                    Some(k3::K3Rt::new(m, ctx)?)
                } else {
                    None
                },
                ckpts: Vec::new(),
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
                        "pulsar: auto budget: {:.1}GiB VRAM free -> expert cache {:.1}GiB, staging {:.1}GiB, prefill chunk {}",
                        free as f64 / GIB,
                        dev_bytes as f64 / GIB,
                        staging_bytes as f64 / GIB,
                        st.max_batch,
                    );
                    // A starved budget still "succeeds" - it just decodes at
                    // a tenth of the speed (prefill chunk 4, sub-GB cache).
                    // When the KV cache is the dominant eater on this card,
                    // say so and name the remedy instead of leaving a
                    // correct-looking line that hides the cause.
                    let kv_here: usize = st
                        .kcache
                        .iter()
                        .chain(st.vcache.iter())
                        .map(|b| b.bytes())
                        .sum();
                    let kv_on_primary = m.attn_dev.is_none_or(|d| d == primary) && !dense_split;
                    if kv_on_primary
                        && (st.max_batch <= 16 || dev_bytes < (1 << 30))
                        && kv_here > (free + kv_here) / 3
                    {
                        eprintln!(
                            "pulsar: WARNING: KV cache ({:.1}GiB at ctx {}) is starving the expert budget; {}",
                            kv_here as f64 / GIB,
                            st.ctx,
                            if (kv_ok && kvq == 0) || (kv_lat_ok && kvq_lat == 0) {
                                "set PULSAR_KV=fp8 (~4x smaller) or lower --ctx"
                            } else if kv_ok || kv_lat_ok {
                                "lower --ctx (KV is already quantized)"
                            } else {
                                "lower --ctx (this arch's KV cannot be quantized yet)"
                            },
                        );
                    }
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
            // last_row/logits are sized for spec_rows. Past that a forward
            // writes off the end of both; on the batched path that surfaced
            // as a raw assert inside copy_d2d, and on paths without a copy
            // it would just corrupt. One check here covers every family.
            if rows > st.spec_rows {
                return Err(format!(
                    "{rows} logit rows requested but State was sized for {} \
                     (raise with PULSAR_NGRAM, or a draft/MTP head)",
                    st.spec_rows
                )
                .into());
            }
            // Exhaustive on purpose. A family whose state advances token by
            // token cannot take the batched path below: the batch would be
            // computed against the wrong history and the logits would be
            // silently wrong, not obviously broken. Written as two `if`s a
            // new family inherits the batched arm for free, so keep this a
            // match and make the compiler ask.
            match s.family {
                // V4 is a sequential state machine (SWA ring, streaming
                // compressor): prefill loops single-token forwards
                Family::Dsv4 => return self.forward_dsv4(st, tokens, pos0, rows),
                // GDN conv window + delta state are sequential too
                Family::Qwen35 => return self.forward_qwen35(st, tokens, pos0, rows),
                // pure-KV attention: a row's whole history is the cache, so
                // batching rows is safe
                // K3's KDA conv window and delta state advance per token,
                // exactly like qwen35's GDN
                Family::K3 => return self.forward_k3(st, tokens, pos0, rows),
                Family::Gqa | Family::Mla => {}
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

            if std::env::var_os("PULSAR_L2_TRACE").is_some() {
                kernels::sync()?;
                let v = st.cur.read_f32((n_tok * s.n_embd) as usize)?;
                let last = &v[(n_tok as usize - 1) * s.n_embd as usize..];
                let l2: f32 = last.iter().map(|x| x * x).sum::<f32>().sqrt();
                eprintln!("l2 EMBD {l2:.4} first {:.5} {:.5} {:.5}", last[0], last[1], last[2]);
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
                if std::env::var_os("PULSAR_L2_TRACE").is_some() {
                    kernels::sync()?;
                    let v = st.cur.read_f32((n_tok * s.n_embd) as usize)?;
                    let last = &v[(n_tok as usize - 1) * s.n_embd as usize..];
                    let l2: f32 = last.iter().map(|x| x * x).sum::<f32>().sqrt();
                    eprintln!("l2 L{il} out {l2:.4} first {:.5} {:.5} {:.5}", last[0], last[1], last[2]);
                }
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
            // Exhaustive: answering `false` for a family that does carry
            // recurrent state lets the prefix cache rewind something that
            // cannot be rewound, which corrupts the stream rather than
            // failing. sconv_k > 1 catches inkling's shortconv on top.
            let by_family = match self.shape.family {
                // K3's KDA layers carry a conv window and a delta state
                Family::Dsv4 | Family::Qwen35 | Family::K3 => true,
                Family::Gqa | Family::Mla => false,
            };
            by_family || self.shape.sconv_k > 1
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
            // per-layer query head count: laguna varies it (48 full / 72
            // sliding); 0 or no geom = uniform s.n_head
            let nh_q = gm.map(|g| g.n_head_q).filter(|&v| v != 0).unwrap_or(s.n_head);
            let heads_dim = match (&l.attn, gm) {
                (Attn::Gqa { .. }, Some(g)) => nh_q * g.head_dim,
                _ => s.heads_dim(),
            };
            {
                // attention
                kernels::rms_norm(&mut st.normed, &st.cur, &l.attn_norm, s.n_embd, n_tok, eps)?;
                let mut attn_output_w: &DeviceBuf = &l.attn_output;
                // Mla runs its output projection inside the arm (per-layer
                // owner device); the common tail below must not re-run it
                let mut mla_attn_done = false;
                match &l.attn {
                    // dsv4/qwen35/k3 have their own graphs
                    Attn::Dsv4(_) | Attn::Qwen35(_) | Attn::K3(_) => {
                        return Err("hybrid-family layer in the shared eval path".into())
                    }
                    Attn::Gqa { attn_q, attn_k, attn_v, q_norm, k_norm, sinks } => {
                        let (hkv, hd, theta, window) = match gm {
                            Some(g) => (g.n_head_kv, g.head_dim, g.theta, g.window),
                            None => (s.n_head_kv, s.head_dim, s.rope_freq_base, 0),
                        };
                        let rot = if gm.is_some() { hd } else { s.rot_dim };
                        let factors = gm
                            .filter(|g| g.factors)
                            .and(self.rope_factors.as_ref());
                        // Gqa attn offload (opt-in): hop the normed input
                        // over and run the whole segment on the attn card,
                        // exactly like the Mla path below
                        if let Some(d) = self.attn_dev {
                            kernels::copy_across(&mut st.normed_a, &st.normed, (n_tok * s.n_embd) as usize * 4)?;
                            kernels::set_device(d)?;
                        }
                        let xin = if self.attn_dev.is_some() { &st.normed_a } else { &st.normed };
                        kernels::matmul_q8_0(&mut st.q, attn_q, xin, s.n_embd, nh_q * hd, n_tok)?;
                        kernels::matmul_q8_0(&mut st.k, attn_k, xin, s.n_embd, hkv * hd, n_tok)?;
                        if let Some(ab) = &l.attn_bias {
                            kernels::add_bias_rows(&mut st.q, &ab.q, nh_q * hd, n_tok)?;
                            kernels::add_bias_rows(&mut st.k, &ab.k, hkv * hd, n_tok)?;
                        }
                        match attn_v {
                            Some(v_w) => {
                                kernels::matmul_q8_0(&mut st.v, v_w, xin, s.n_embd, hkv * hd, n_tok)?;
                                if let Some(ab) = &l.attn_bias {
                                    kernels::add_bias_rows(&mut st.v, &ab.v, hkv * hd, n_tok)?;
                                }
                            }
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
                        // absent qk-norm means no normalization at all, not
                        // a weightless one - skip rather than pass None
                        if let Some(qn) = q_norm {
                            kernels::gqa_head_rms_norm(&mut st.q, Some(qn), n_tok * nh_q, hd, eps)?;
                        }
                        if let Some(kn) = k_norm {
                            kernels::gqa_head_rms_norm(&mut st.k, Some(kn), n_tok * hkv, hd, eps)?;
                        }
                        if gm.is_some() && l.ink.is_none() && l.attn_gate.is_none() && q_norm.is_some() {
                            // gemma: v gets a weightless per-head rms norm.
                            // laguna also has per-layer geom but does NOT
                            // do this (attn_gate marks it), and neither does
                            // gpt-oss (no qk-norm marks it - same
                            // discriminator as the attention scale above).
                            kernels::gqa_head_rms_norm(&mut st.v, None, n_tok * hkv, hd, eps)?;
                        }
                        if let Some(rot_w) = gm.map(|g| g.rot).filter(|&r| r != 0) {
                            // laguna per-layer-type rope (llama.cpp
                            // models/laguna.cpp is the validated spec):
                            // FULL layers run YaRN (factor 32 over 8192,
                            // theta 500k, rot 64) with attn_factor set to
                            // CANCEL the kernel-internal
                            // 1 + 0.1 ln(1/freq_scale) mscale - rotated
                            // dims stay unit-scaled. SLIDING layers run
                            // PLAIN rope (theta 10k, rot 128): freq_scale
                            // 1.0 and ext_factor 0, NOT the global yarn.
                            // gpt-oss scales every layer and keeps the
                            // kernel's mscale; laguna scales only the
                            // full-window ones and cancels it
                            let uni = s.rope_yarn_uniform;
                            let yarn = (uni || window == 0) && s.rope_scale_factor > 1.0;
                            let f = s.rope_scale_factor;
                            let rc = kernels::RopeCfg {
                                n_ctx_orig: s.rope_orig_ctx,
                                freq_base: theta,
                                freq_scale: if yarn { 1.0 / f } else { 1.0 },
                                ext_factor: if yarn { 1.0 } else { 0.0 },
                                attn_factor: if yarn && !uni { 1.0 / (1.0 + 0.1 * f.ln()) } else { 1.0 },
                                beta_fast: 32.0,
                                beta_slow: 1.0,
                                kq_mult: 1.0,
                            };
                            kernels::rope_yarn_partial(&mut st.q, n_tok, nh_q, hd, rot_w, pos0, &rc)?;
                            kernels::rope_yarn_partial(&mut st.k, n_tok, hkv, hd, rot_w, pos0, &rc)?;
                        } else if l.ink.is_none() {
                            // inkling has no rope: position rides the
                            // relative bias below (log-N tau is identity
                            // below 128k ctx, so it is skipped here)
                            kernels::gqa_rope(&mut st.q, n_tok, nh_q, hd, rot, pos0, theta, factors)?;
                            kernels::gqa_rope(&mut st.k, n_tok, hkv, hd, rot, pos0, theta, factors)?;
                        }
                        let kvq = st.kvq;
                        // turbo: rotate K by Πᵀ before block-quant append so
                        // outliers spread across the 32-wide block. V is NOT
                        // rotated — only K and Q preserve the dot-product.
                        let ksrc: &DeviceBuf = if st.kvq_rot {
                            kernels::matmul_f32(
                                &mut st.krot,
                                &st.pi,
                                &st.k,
                                hd,
                                hd,
                                n_tok * hkv,
                            )?;
                            &st.krot
                        } else {
                            &st.k
                        };
                        kernels::gqa_kv_append(&mut st.kcache[il], ksrc, n_tok, hkv, hd, st.ctx, pos0, kvq)?;
                        kernels::gqa_kv_append(&mut st.vcache[il], &st.v, n_tok, hkv, hd, st.ctx, pos0, kvq)?;
                        // gemma scores at scale 1.0 (q is per-head normed);
                        // inkling at muP 1/head_dim
                        let scale = if l.ink.is_some() {
                            1.0 / hd as f32
                        } else if gm.is_some() && l.attn_gate.is_none() && q_norm.is_some() {
                            // gemma only, and the reason is the qk-norm: q is
                            // already per-head normalized, so the scores need
                            // no 1/sqrt(head_dim). Keyed on that norm rather
                            // than on "has per-layer geometry", which gpt-oss
                            // also has while needing the ordinary scale.
                            1.0
                        } else {
                            // laguna: head_dim**-0.5 despite QK-norm
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
                        // turbo: rotate Q by the SAME Πᵀ so Q·Kᵀ is preserved
                        // ((QΠᵀ)·(KΠᵀ)ᵀ = QKᵀ) while the rotated K already sits
                        // in the cache. qrot mirrors q's attention-head layout
                        // (n_head*head_dim per token); q is allocated at
                        // head_dim.max(qk_dim()), and the PULSAR_KV parse
                        // disables rotation when qk_dim exceeds head_dim, so
                        // the two strides agree wherever this runs.
                        let qsrc: &DeviceBuf = if st.kvq_rot {
                            kernels::matmul_f32(
                                &mut st.qrot,
                                &st.pi,
                                &st.q,
                                hd,
                                hd,
                                n_tok * nh_q,
                            )?;
                            &st.qrot
                        } else {
                            &st.q
                        };
                        kernels::gqa_attention_rel(&mut st.heads, qsrc, &st.kcache[il], &st.vcache[il], n_tok, nh_q, hkv, hd, st.ctx, pos0, scale, window, rel, rel_ext, kvq, sinks.as_ref())?;

                        // laguna: per-head output gate. g_proj gives one
                        // logit per (token, head); softplus of it scales
                        // the whole head row before the output projection.
                        if let Some(gw) = &l.attn_gate {
                            kernels::matmul_q8_0(&mut st.attn_gate_buf, gw, xin, s.n_embd, nh_q, n_tok)?;
                            if il == 0 && std::env::var_os("PULSAR_L2_TRACE").is_some() {
                                kernels::sync()?;
                                let g = st.attn_gate_buf.read_f32((n_tok * nh_q) as usize)?;
                                let h = st.heads.read_f32((n_tok * nh_q * hd) as usize)?;
                                let hl2: f32 = h.iter().map(|x| x * x).sum::<f32>().sqrt();
                                let gmin = g.iter().cloned().fold(f32::INFINITY, f32::min);
                                let gmax = g.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                                eprintln!("l2 L0 gate logits [{gmin:.3}, {gmax:.3}] heads-pre-gate L2 {hl2:.4}");
                            }
                            kernels::laguna_head_gate(&mut st.heads, &st.attn_gate_buf, n_tok, nh_q, hd)?;
                            if il == 0 && std::env::var_os("PULSAR_L2_TRACE").is_some() {
                                kernels::sync()?;
                                let h = st.heads.read_f32((n_tok * nh_q * hd) as usize)?;
                                let hl2: f32 = h.iter().map(|x| x * x).sum::<f32>().sqrt();
                                eprintln!("l2 L0 heads-post-gate L2 {hl2:.4}");
                            }
                        }

                        // output projection on the attn card, hop back,
                        // restore the primary (mirrors the Mla path)
                        if self.attn_dev.is_some() {
                            kernels::matmul_q8_0(&mut st.attn_out_a, attn_output_w, &st.heads, nh_q * hd, s.n_embd, n_tok)?;
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

                        // layer split: this layer's owner runs the
                        // whole attention segment; hop the normed input
                        // over when off-primary. Blocking copies are
                        // legacy-stream ordered on the issuing device, so
                        // producer kernels have landed before they run.
                        let a_dev = self.attn_layer_dev.get(il).copied().unwrap_or(primary);
                        let sci = st
                            .attn_sc
                            .iter()
                            .position(|x| x.dev == a_dev)
                            .ok_or("mla attn scratch missing for owner device")?;
                        // DSA selection reuse crosses layers: when the live
                        // list was written on a different owner, copy it
                        // over before anything on this card reads it. The
                        // planner keeps ranges contiguous, so this fires
                        // once per boundary per chunk (~n_tok*topk*4 B).
                        if st.sel_dev >= 0 && st.sel_dev != a_dev && st.idx_last_sel > 0 {
                            if let Some(pi) = st.attn_sc.iter().position(|x| x.dev == st.sel_dev) {
                                let bytes = (n_tok * st.idx_last_sel) as usize * 4;
                                let (src_sc, dst_sc) = if pi < sci {
                                    let (l_h, r_h) = st.attn_sc.split_at_mut(sci);
                                    (&l_h[pi], &mut r_h[0])
                                } else {
                                    let (l_h, r_h) = st.attn_sc.split_at_mut(pi);
                                    (&r_h[0], &mut l_h[sci])
                                };
                                kernels::copy_across(&mut dst_sc.mla_selected, &src_sc.mla_selected, bytes)?;
                                st.sel_dev = a_dev;
                            }
                        }
                        let sc = &mut st.attn_sc[sci];
                        if a_dev != primary {
                            kernels::copy_across(&mut sc.normed_a, &st.normed, (n_tok * s.n_embd) as usize * 4)?;
                            kernels::set_device(a_dev)?;
                        }
                        let xin = if a_dev != primary { &sc.normed_a } else { &st.normed };

                        let rope = s.rope_cfg();
                        let kv_raw_dim = s.n_kv_lora + s.qk_rope;
                        kernels::matmul_q8_0(&mut sc.q_rank, q_a_w, xin, s.n_embd, s.n_lora_q, n_tok)?;
                        kernels::rms_norm(&mut sc.q_rank_norm, &sc.q_rank, q_a_norm, s.n_lora_q, n_tok, eps)?;
                        kernels::matmul_q8_0(&mut sc.q, q_b_w, &sc.q_rank_norm, s.n_lora_q, s.n_head * s.qk_dim(), n_tok)?;
                        kernels::mla_rope_tail(&mut sc.q, n_tok, s.n_head, s.qk_dim(), s.qk_rope, pos0, &rope)?;
                        kernels::matmul_q8_0(&mut sc.kv_raw, kv_a_w, xin, s.n_embd, kv_raw_dim, n_tok)?;
                        kernels::mla_kv_lora_rms_norm(&mut sc.kv_norm, &sc.kv_raw, kv_a_norm, n_tok, kv_raw_dim, s.n_kv_lora, eps)?;
                        kernels::mla_store_compact_kv(&mut st.kcache[il], &mut st.vcache[il], &sc.kv_norm, &sc.kv_raw, pos0, n_tok, st.ctx, kv_raw_dim, s.n_kv_lora, s.qk_rope, st.kvq_lat)?;
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
                            kernels::matmul_q8_0(&mut sc.idx_kraw, &idx.k, xin, s.n_embd, s.n_idx_dim, n_tok)?;
                            kernels::idx_store_k(&sc.idx_kraw, &idx.k_norm, &idx.k_norm_b, &mut st.idx_kcache[il], pos0, n_tok, st.ctx, s.n_idx_dim, s.qk_rope, s.rms_eps, &s.rope_cfg(), 0.0, 1.0, (st.kvq_lat == 1) as u32)?;
                        }
                        let n_sel = if topk == 0 || visible <= topk {
                            kernels::mla_fill_selected_range(&mut sc.mla_selected, n_tok, pos0, visible, st.ctx)?;
                            st.idx_last_sel = visible;
                            st.sel_dev = a_dev;
                            visible
                        } else if is_idx_layer && indexer.is_some() {
                            let idx = indexer.as_ref().unwrap();
                            kernels::matmul_q8_0(&mut sc.idx_q, &idx.q_b, &sc.q_rank_norm, s.n_lora_q, s.n_idx_head * s.n_idx_dim, n_tok)?;
                            kernels::idx_rope0(&mut sc.idx_q, n_tok, s.n_idx_head, s.n_idx_dim, s.qk_rope, pos0, &s.rope_cfg(), 0.0, 1.0)?;
                            // ds4 feeds proj the pre-norm residual (cur).
                            // Under attn offload cur is on the primary;
                            // borrow attn_out_a as the hop buffer - it is
                            // not written until the output projection.
                            if a_dev != primary {
                                kernels::copy_across(&mut sc.attn_out_a, &st.cur, (n_tok * s.n_embd) as usize * 4)?;
                                kernels::matmul_f32(&mut sc.idx_w, &idx.proj, &sc.attn_out_a, s.n_embd, s.n_idx_head, n_tok)?;
                            } else {
                                kernels::matmul_f32(&mut sc.idx_w, &idx.proj, &st.cur, s.n_embd, s.n_idx_head, n_tok)?;
                            }
                            let scale = 1.0 / ((s.n_idx_dim * s.n_idx_head) as f32).sqrt();
                            if n_tok == 1 {
                                kernels::idx_score_one(&mut sc.idx_scores, &sc.idx_q, &sc.idx_w, &st.idx_kcache[il], visible, s.n_idx_head, s.n_idx_dim, scale, (st.kvq_lat == 1) as u32)?;
                                kernels::idx_topk(&mut sc.mla_selected, &sc.idx_scores, visible, topk)?;
                            } else {
                                // batch: every token in a post-boundary
                                // chunk has >= top_k visible rows (the
                                // forward_rows split guarantees it)
                                kernels::idx_scores_batch(&mut sc.idx_scores, &sc.idx_q, &sc.idx_w, &st.idx_kcache[il], Some(&mut sc.idx_q16), visible, n_tok, pos0, s.n_idx_head, s.n_idx_dim, scale, (st.kvq_lat == 1) as u32)?;
                                kernels::idx_topk_batch(&mut sc.mla_selected, &sc.idx_scores, visible, n_tok, topk)?;
                            }
                            st.idx_last_sel = topk;
                            st.sel_dev = a_dev;
                            topk
                        } else {
                            // between indexer layers: reuse the last list
                            if st.idx_last_sel == 0 {
                                return Err("indexer selection missing (no indexer weights in gguf?)".into());
                            }
                            st.idx_last_sel
                        };
                        kernels::mla_qk_lowrank(&mut sc.qk_low, &sc.q, k_b_w, n_tok, s.n_head, s.n_kv_lora, s.qk_nope, s.qk_dim())?;
                        kernels::mla_attention(&mut sc.heads, &sc.q, &sc.qk_low, &st.kcache[il], &st.vcache[il], v_b_w, &sc.mla_selected, n_tok, n_sel, st.ctx, s.n_head, s.n_kv_lora, s.qk_nope, s.qk_rope, s.value_mla, &rope, st.kvq_lat)?;

                        // output projection on the owner, hop back
                        // when off-primary, restore the primary for the
                        // ffn/expert half
                        if a_dev != primary {
                            kernels::matmul_q8_0(&mut sc.attn_out_a, attn_output_w, &sc.heads, s.heads_dim(), s.n_embd, n_tok)?;
                            kernels::copy_across(&mut st.attn_out, &sc.attn_out_a, (n_tok * s.n_embd) as usize * 4)?;
                            kernels::set_device(primary)?;
                        } else {
                            kernels::matmul_q8_0(&mut st.attn_out, attn_output_w, &sc.heads, s.heads_dim(), s.n_embd, n_tok)?;
                        }
                        mla_attn_done = true;
                    }
                }
                if self.attn_dev.is_none() && !mla_attn_done {
                    kernels::matmul_q8_0(&mut st.attn_out, attn_output_w, &st.heads, heads_dim, s.n_embd, n_tok)?;
                    if let Some(ab) = &l.attn_bias {
                        kernels::add_bias_rows(&mut st.attn_out, &ab.out, s.n_embd, n_tok)?;
                    }
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
                if std::env::var_os("PULSAR_L2_TRACE").is_some() && il == 0 {
                    kernels::sync()?;
                    let a = st.attn_out.read_f32((n_tok * s.n_embd) as usize)?;
                    let l2: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                    eprintln!("l2 L0 attn_out(post o_proj) L2 {l2:.4} first {:.5} {:.5}", a[0], a[1]);
                }
                kernels::add(&mut st.after_attn, &st.cur, &st.attn_out, n_tok * s.n_embd)?;
                if std::env::var_os("PULSAR_L2_TRACE").is_some() && il == 0 {
                    kernels::sync()?;
                    let a = st.after_attn.read_f32((n_tok * s.n_embd) as usize)?;
                    let l2: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
                    eprintln!("l2 L0 after_attn(residual) L2 {l2:.4}");
                }

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
                    Ffn::Moe { gate_inp, probs_b, shexp, gate_exps, up_exps, down_exps, fused_up_off, down_scale, sink, exp_bias, gate_inp_b } => {
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
                        // part of the gate's linear layer, so it lands on
                        // the logits before both the top-k and the softmax
                        if let Some(gb) = gate_inp_b {
                            kernels::add_bias_rows(&mut st.router_logits, gb, s.n_expert, n_tok)?;
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
                        if std::env::var_os("PULSAR_ROUTER_HIST").is_some() && n_tok == 1 {
                            // Per-token gate weights by rank. The tail of
                            // this distribution is what a top-p router
                            // prune would drop, and its mass bounds the
                            // error that dropping costs - so measure it
                            // before believing any estimate.
                            kernels::sync()?;
                            let w = st.router_weights.read_f32(s.n_expert_used as usize)?;
                            let sum: f32 = w.iter().sum::<f32>().max(1e-9);
                            let mut line = String::new();
                            for v in &w {
                                line.push_str(&format!("{:.5} ", v / sum));
                            }
                            eprintln!("rw L{il} {line}");
                        }
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
                        // Score the prediction the PREVIOUS layer made for this
                        // one. Layers are serial, so prefetch is the only way
                        // the GPU can be busy during a disk wait; this ratio is
                        // the ceiling on how much of that wait can ever be
                        // hidden. Cheap set test - n_expert_used is 8.
                        if n_tok == 1 && std::env::var_os("PULSAR_PROFILE").is_some() {
                            if st.pred_prev_for == il && !st.pred_prev.is_empty() {
                                for &e in &selected {
                                    if e < 0 || e as u32 >= s.n_expert {
                                        continue;
                                    }
                                    st.prof.pred_total += 1;
                                    if st.pred_prev.contains(&e) {
                                        st.prof.pred_hits += 1;
                                    }
                                }
                            }
                            match &pred_ids {
                                Some(p) => {
                                    st.pred_prev.clear();
                                    st.pred_prev.extend_from_slice(p);
                                    st.pred_prev_for = il + 1;
                                }
                                None => st.pred_prev_for = usize::MAX,
                            }
                        }
                        // true routing count: every selection this layer, resident
                        // or streamed (topic atlas / Brain heat, no tier blind spot)
                        {
                            let base = il * s.n_expert as usize;
                            for &e in &selected {
                                if e >= 0 && (e as u32) < s.n_expert {
                                    st.route_counts[base + e as usize] += 1;
                                }
                            }
                        }
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
                        // Only when the chunk really does touch most of the
                        // layer. t tokens picking top-k of E experts reach
                        // E*(1-(1-k/E)^t) distinct ones: a 256-token chunk
                        // gets ~100% and this is the right call, but a
                        // 12-token prompt gets ~32% and it requests 2.36GiB
                        // per layer to use a third of it. Ungated (n_tok > 8)
                        // that was 177GiB of speculative reads across 75
                        // layers into a ~20GB cache - it saturated the drive,
                        // thrashed the cache, and the backlog drained into
                        // decode starving the reads actually being waited on.
                        // Measured on GLM-5.2: 1.43 tok/s with the flood vs
                        // 2.63 with prefetch off entirely.
                        let chunk_covers = {
                            let p_miss =
                                1.0 - (s.n_expert_used as f64 / s.n_expert.max(1) as f64);
                            1.0 - p_miss.powi(n_tok as i32) >= 0.75
                        };
                        if chunk_covers && std::env::var_os("PULSAR_NO_PREFETCH").is_none() {
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
                        let t_absorb = std::time::Instant::now();
                        while let Ok((off, slab)) = st.prefetcher.done_rx.try_recv() {
                            st.store.absorb(off, slab);
                            st.prof.absorbed += 1;
                        }
                        st.prof.resolve_absorb += t_absorb.elapsed();
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
                        // Per-expert bias pointers. Biases are resident f32
                        // and indexed by expert id, so unlike the weight
                        // slabs they need no cache/tier resolve; sink slots
                        // (id >= n_expert) have no bias tensor and stay null.
                        let bias_of = |e: i32| -> (
                            *const std::ffi::c_void,
                            *const std::ffi::c_void,
                            *const std::ffi::c_void,
                        ) {
                            match exp_bias {
                                Some([gb, ub, db]) if e >= 0 && (e as u32) < s.n_expert => {
                                    let mid = (e as u64) * s.n_ff_exp as u64 * 4;
                                    let out = (e as u64) * s.n_embd as u64 * 4;
                                    (
                                        byte_off(gb.ptr(), mid),
                                        byte_off(ub.ptr(), mid),
                                        byte_off(db.ptr(), out),
                                    )
                                }
                                _ => (std::ptr::null(), std::ptr::null(), std::ptr::null()),
                            }
                        };
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
                                        gate_b: bias_of(e).0,
                                        up_b: bias_of(e).1,
                                        down_b: bias_of(e).2,
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
                            && s.n_embd.is_multiple_of(256)
                            && s.n_ff_exp.is_multiple_of(256)
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
                        // ---- lane B: experts the DISK is about to deliver ----
                        // Lane A above can only claim experts already in RAM,
                        // so a disk-missed expert takes the most expensive
                        // route in the engine: disk -> RAM -> PCIe -> GPU.
                        // The bytes land in host memory anyway, so compute
                        // them where they land and skip the bus entirely.
                        // Unlike lane A this does NOT overlap the fetch (it
                        // runs after), so it only pays while the CPU finishes
                        // its share before the GPU finishes its own - hence
                        // the cap. DEFAULT OFF: the mechanism is confirmed
                        // (ptrs, the PCIe-drain bucket, falls 4.31 -> 3.05s on
                        // a Gen4 x4 card) but the throughput gain is inside
                        // the noise - slow card B=0 2.16/2.13/2.18 vs B=2
                        // 2.29/2.14, and dead neutral on Gen5 x8 (2.90 vs
                        // 2.91) where there is barely any PCIe cost to remove.
                        // Worth having on a box whose experts sit on a slow
                        // link; not worth changing the default path for.
                        // PULSAR_CPU_B=N enables, bounding experts per layer.
                        let lane_b_cap: usize = std::env::var("PULSAR_CPU_B")
                            .ok()
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(0);
                        let mut lane_b = cpu_tier::Lane::new(
                            gate_exps.quant, down_exps.quant,
                            gate_exps.row_bytes as usize, down_exps.row_bytes as usize,
                            ne, nf, s.moe_act_op,
                        );
                        // planned membership only - Lane::add needs the host
                        // pointers, which do not exist until the fetch lands
                        let mut lane_b_plan: Vec<i32> = Vec::new();
                        let mut lane_b_offs: std::collections::HashMap<u64, i32> =
                            std::collections::HashMap::new();
                        if cpu_on && lane_b_cap > 0 {
                            for &e in &distinct {
                                if e < 0 || e as u32 >= s.n_expert || tier_of(e).is_some() {
                                    continue;
                                }
                                if lane.idx.contains_key(&e) || lane_b_plan.len() >= lane_b_cap {
                                    continue;
                                }
                                let [g3, u3, d3] = slabs_of(e as u32);
                                let (go, uo, dno) =
                                    (off_of(g3.0, g3.1), off_of(u3.0, u3.1), off_of(d3.0, d3.1));
                                if self.mtp.as_ref().is_some_and(|mt| mt.res_map.contains_key(&go)) {
                                    continue;
                                }
                                // only when ALL THREE need the disk: a mixed
                                // expert would still pay a PCIe trip for its
                                // resident slabs and gain nothing
                                if st.store.contains(go) || st.store.contains(uo) || st.store.contains(dno)
                                    || st.dev_cache.map.contains_key(&go)
                                    || st.dev_cache.map.contains_key(&uo)
                                    || st.dev_cache.map.contains_key(&dno)
                                {
                                    continue;
                                }
                                lane_b_plan.push(e);
                                for o in [go, uo, dno] {
                                    lane_b_offs.insert(o, e);
                                }
                            }
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
                        let mut host_ptr: std::collections::HashMap<u64, *const u8> =
                            std::collections::HashMap::new();
                        let unified = st.unified;
                        let async_h2d = st.async_expert_h2d;
                        let mut h2d = std::time::Duration::ZERO;
                        let mut async_queued = false;
                        // TRIED AND REVERTED (2026-07-25): splitting this into
                        // two waves - launch the experts the host store already
                        // holds, THEN block on the disk for the rest - so the
                        // primary GPU computes through the disk wait. It is a
                        // regression: 2.52/2.53/2.51 tok/s against 2.85/2.54/
                        // 2.83 interleaved on GLM-5.2.
                        //
                        // The first explanation here (full-width kernels: NULL-
                        // masked launches plus an unconditional quantize_q8_k
                        // over n_ff_exp * n_expert_used) was WRONG. Measured by
                        // adding exactly that overhead to this path with none of
                        // the benefit - an extra expert_ptrs upload, then that
                        // plus the whole three-kernel chain over an all-NULL
                        // slot array:
                        //     baseline   2.55  2.56
                        //     +ptrs      2.63  2.77
                        //     +full      2.75  2.48
                        // Both variants land at or above baseline, so the extra
                        // launches and the extra upload cost nothing findable.
                        // Making the launches slot-sparse would optimise work
                        // that is not on the clock.
                        //
                        // What the probe did NOT replicate is the one real
                        // difference: the split calls ensure_with TWICE, and the
                        // disk batch goes SECOND. Single-wave issues every read,
                        // disk included, in one call, so io_uring starts filling
                        // immediately; the split makes the disk reads wait for
                        // the host-hit batch to finish staging first. Disk is
                        // ~86% of this model's decode wall, so delaying its
                        // start lands straight on the critical path - the split
                        // postponed the bottleneck to overlap something cheaper.
                        // A retry must issue the cold reads FIRST (the async
                        // prefetcher already does non-blocking fetch), compute
                        // the resident experts while they fly, then collect.
                        // Launching the TIER partials before this wait is also
                        // neutral (2.84/2.82) - they already overlapped the
                        // primary MoE, so it just moves them under a different
                        // shadow.
                        let t_host = std::time::Instant::now();
                        let fetch_wait = {
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
                                // lane B computes this on the CPU from the
                                // slab that just landed - keep the host
                                // pointer and skip the PCIe trip entirely.
                                // The whole point of the extension: these
                                // bytes never cross the bus.
                                if lane_b_offs.contains_key(&off) {
                                    host_ptr.insert(off, payload.as_ptr());
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
                            })?
                        };
                        let ensure_elapsed = t_host.elapsed();
                        // disk-fetch (pure io_uring wait) split into its own bucket;
                        // host = ensure wall minus h2d copies and the disk wait,
                        // leaving host-side lookup + LFU eviction
                        st.prof.resolve_fetch += fetch_wait;
                        st.prof.resolve_host +=
                            ensure_elapsed.saturating_sub(h2d).saturating_sub(fetch_wait);
                        if async_queued {
                            let t = std::time::Instant::now();
                            st.expert_h2d.record()?;
                            st.expert_h2d.wait_default()?;
                            h2d += t.elapsed();
                        }
                        // ---- submit lane B now its slabs are in RAM ----
                        let mut cpu_guard_b: Option<cpu_tier::WaitGuard> = None;
                        if !lane_b_plan.is_empty() {
                            let mut pins = std::mem::take(&mut st.store.pinned);
                            for &e in &lane_b_plan {
                                let [g3, u3, d3] = slabs_of(e as u32);
                                let (go, uo, dno) =
                                    (off_of(g3.0, g3.1), off_of(u3.0, u3.1), off_of(d3.0, d3.1));
                                let (Some(&gp), Some(&up), Some(&dp)) =
                                    (host_ptr.get(&go), host_ptr.get(&uo), host_ptr.get(&dno))
                                else {
                                    continue; // slab came from somewhere else; leave it to the GPU
                                };
                                lane_b.add(e, gp, unsafe { up.add(*fused_up_off as usize) }, dp);
                                // EXTEND, never replace: lane A's pins are
                                // still live and its workers are still reading
                                pins.extend([go, uo, dno]);
                            }
                            st.store.pinned = pins;
                            if !lane_b.is_empty() {
                                let rw = st.router_weights.read_f32(n_tok as usize * n_used)?;
                                let normed_h = st.normed.read_f32(n_tok as usize * ne)?;
                                let pool = st.cpu_pool.as_ref().unwrap();
                                cpu_guard_b = Some(cpu_tier::WaitGuard {
                                    pool,
                                    n: lane_b.submit_a(
                                        pool, &selected, n_used, &normed_h, &rw, n_tok as usize,
                                    ),
                                });
                            }
                        }
                        st.prof.h2d += h2d;
                        let t_ptrs = std::time::Instant::now();
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
                                        gate_b: bias_of(e).0,
                                        up_b: bias_of(e).1,
                                        down_b: bias_of(e).2,
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
                            if lane.idx.contains_key(&e) || lane_b.idx.contains_key(&e) {
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
                                gate_b: bias_of(e).0,
                                up_b: bias_of(e).1,
                                down_b: bias_of(e).2,
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
                            && s.n_ff_exp.is_multiple_of(256)
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
                        st.prof.resolve_ptrs += t_ptrs.elapsed();
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
                                if exp_bias.is_some() {
                                    kernels::moe_down_bias(
                                        &mut tier.out, &tier.ptrs, &tier.weights,
                                        s.n_embd, s.n_expert_used, n_tok,
                                    )?;
                                }
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
                        // The down bias is sum_s w_s * b_down_s, and the pair
                        // stage already folded w_s into mid, so it needs the
                        // weights again and cannot ride the down matmul. Both
                        // branches above land in moe_out through expert_ptrs,
                        // whose tier/lane/sink slots are NULL - those paths
                        // add their own bias against their own ptrs.
                        if exp_bias.is_some() {
                            kernels::moe_down_bias(
                                &mut st.moe_out, &st.expert_ptrs, &st.router_weights,
                                s.n_embd, s.n_expert_used, n_tok,
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
                        if !lane.is_empty() || !lane_b.is_empty() {
                            drop(cpu_guard.take());
                            drop(cpu_guard_b.take());
                            let t_cpu = std::time::Instant::now();
                            let pool = st.cpu_pool.as_ref().unwrap();
                            let mut acc = lane.finish(pool, n_tok as usize);
                            // fold lane B's partial into the same vector: both
                            // are sums over disjoint routed slots, so adding
                            // them is the same reduction the GPU would do
                            if !lane_b.is_empty() {
                                let b = lane_b.finish(pool, n_tok as usize);
                                if acc.is_empty() {
                                    acc = b;
                                } else {
                                    for (a, v) in acc.iter_mut().zip(b.iter()) {
                                        *a += *v;
                                    }
                                }
                            }
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
                            st.cpu_hits += (lane.idx.len() + lane_b.idx.len()) as u64;
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
            match &mtp.eh_proj {
                MatW::Q8(b) => kernels::matmul_q8_0(&mut st.cur, b, &st.mtp_x, 2 * s.n_embd, s.n_embd, n_tok)?,
                MatW::Kq(k) => {
                    kernels::quantize_q8_k(&mut st.mtp_xq, &st.mtp_x, 2 * s.n_embd, n_tok)?;
                    kernels::matmul_kq(&mut st.cur, &k.w, &st.mtp_xq, 2 * s.n_embd, s.n_embd, n_tok, k.row_bytes, k.quant)?
                }
            }
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
            // The draft needs one token id, not calibrated logits, so it
            // takes the vocab-split argmax head when TP is on: each card
            // reads half the 248k-row head instead of card A reading all
            // of it. The head is the single biggest read in a draft step.
            if self.shape.family == Family::Qwen35 {
                let mut rt = st.qwen35.take().ok_or("qwen35 state missing")?;
                let r = self.head_argmax_split(st, &mut rt, 1);
                st.qwen35 = Some(rt);
                if r? {
                    return Ok(*st.last_argmax.first().ok_or("no argmax row")?);
                }
            }
            self.head_logits(st, 1)?;
            if std::env::var_os("PULSAR_MTP_DEBUG").is_some() {
                kernels::sync()?;
                let logits = st.logits.read_f32(s.n_vocab as usize)?;
                let bad = logits.iter().filter(|v| !v.is_finite()).count();
                eprintln!("mtp-draft pos={pos}: logits nan={bad}, draft={}", argmax(&logits));
            }
            // argmax on device: the draft only needs the token id, not a
            // megabyte of logits over pageable PCIe
            let ids = kernels::argmax_rows(&mut st.amax_out, &st.logits, s.n_vocab, 1)?;
            Ok(ids[0])
        }

        /// Sampled MTP draft: like mtp_draft but reads the full nextn
        /// logits back and returns (sampled token, draft distribution q)
        /// so speculative SAMPLING can accept/reject against the target.
        /// No argmax-split shortcut - the residual math needs q's mass.
        fn mtp_draft_sampled(
            &self,
            st: &mut State,
            token: u32,
            pos: u32,
            sampler: &mut Sampler,
        ) -> Result<(u32, Vec<(u32, f32)>)> {
            self.mtp_body(st, token, pos)?;
            let mtp = self.mtp.as_ref().ok_or("mtp_draft without an MTP layer")?;
            let s = self.shape;
            kernels::rms_norm(&mut st.normed, &st.cur, &mtp.head_norm, s.n_embd, 1, s.rms_eps)?;
            self.head_logits(st, 1)?;
            kernels::sync()?;
            let logits = st.logits.read_f32(s.n_vocab as usize)?;
            let q = sampler.dist(&logits);
            let t = sampler.sample_from(&q);
            Ok((t, q))
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
            match &mtp.eh_proj {
                MatW::Q8(b) => kernels::matmul_q8_0(&mut st.cur, b, &st.mtp_x, 2 * s.n_embd, s.n_embd, 1)?,
                MatW::Kq(k) => {
                    kernels::quantize_q8_k(&mut st.mtp_xq, &st.mtp_x, 2 * s.n_embd, 1)?;
                    kernels::matmul_kq(&mut st.cur, &k.w, &st.mtp_xq, 2 * s.n_embd, s.n_embd, 1, k.row_bytes, k.quant)?
                }
            }
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
        // MTP greedy spec: acceptance compares the draft against the
        // verified argmax, which IS greedy sampling (readback-free).
        let spec = model.mtp.is_some() && sampler.is_greedy();
        // MTP sampled spec: rejection sampling against the target
        // distribution, distribution-preserving at any temperature. Reads
        // full logits (draft q + target p), so pricier than greedy verify
        // but lets temp>0 requests (pi) use the 87%-accept nextn head.
        let spec_sample = model.mtp.is_some() && !sampler.is_greedy()
            && std::env::var_os("PULSAR_NO_SPEC_SAMPLE").is_none();
        let mtp_prefill = spec || spec_sample;
        let mut pos = pos0;
        let mut logits = None;
        // qwen35 MTP prefill: the forward leaves only its LAST internal
        // chunk in st.cur and the fill pass reads every row of the chunk
        // back, so the outer chunks are capped to the forward's own width.
        //
        // This used to be hardcoded to 16 with the note that the forward
        // was internally 16-chunked anyway - same work either way.
        // That stopped being true when the internal width grew to 128:
        // 16-row outer chunks swept the whole 15.7GB of weights 8x more
        // often than needed, and the MTP prefill path was 5.4x slower
        // than the same prompt with MTP off (61s vs 11s on 7120 tokens).
        // The MTP scratch is max_batch-sized, so it was never the limit.
        let chunk_cap = if mtp_prefill && model.shape.family == Family::Qwen35 {
            (st.max_batch() as usize).min(qwen35::chunk_max())
        } else {
            st.max_batch() as usize
        };
        let prof_chunks = std::env::var_os("PULSAR_PROFILE").is_some();
        if pos0 == 0 {
            st.clear_ckpts();
        }
        for chunk in prompt.chunks(chunk_cap) {
            if cancel() {
                return Ok(pos);
            }
            let t0 = std::time::Instant::now();
            logits = model.forward_batch(st, chunk, pos, true)?;
            if mtp_prefill {
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
            st.maybe_checkpoint(model, pos)?;
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

        // MTP speculative SAMPLING (temp>0): same draft/verify/GDN-state
        // flow as the greedy block below, but the draft samples from the
        // nextn head (recording q), the verify reads full target logits
        // (p), and acceptance is Leviathan/Chen rejection sampling so the
        // emitted stream is distributed EXACTLY as non-spec sampling at
        // the same temp/top_p/min_p. Reuses the sampler RNG for accept
        // draws, residual, and bonus - one deterministic stream.
        if spec_sample {
            let v = model.shape.n_vocab as usize;
            let row = model.shape.n_embd as usize * 4;
            // sampled spec: verify reads full logits per row, so each
            // extra draft row is pricey - depth-1 measured optimal
            // (37.2 vs 32.8 tok/s at depth-2), unlike GREEDY where the
            // cheap argmax verify makes depth-2 win. Cap here, keep the
            // model-loaded depth for greedy; PULSAR_MTP_SAMPLE_DEPTH
            // overrides, never exceeding the loaded depth.
            let depth_max = std::env::var("PULSAR_MTP_SAMPLE_DEPTH")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(1)
                .max(1)
                .min(model.mtp_depth.max(1));
            let debug = std::env::var_os("PULSAR_MTP_DEBUG").is_some();
            let mut emitted = 0usize;
            let mut next = sampler.sample(logits.as_deref().ok_or("no logits")?);
            'sround: while emitted < max_tokens {
                if stop(next) || pos + 2 >= st.ctx() {
                    model.forward_batch(st, &[next], pos, false)?;
                    pos += 1;
                    break;
                }
                on_token(next);
                emitted += 1;

                // draft a chain, recording each step's draft distribution q
                kernels::copy_d2d(&mut st.mtp_hidden_save, 0, &st.mtp_hidden, 0, row)?;
                let depth = depth_max.min(st.ctx() - pos - 2);
                let mut chain = vec![next];
                let mut qs: Vec<Vec<(u32, f32)>> = Vec::with_capacity(depth as usize);
                for i in 0..depth {
                    let (d, q) = model.mtp_draft_sampled(st, chain[i as usize], pos + i, sampler)?;
                    st.mtp_drafted += 1;
                    kernels::copy_d2d(&mut st.mtp_hidden, 0, &st.cur, 0, row)?;
                    chain.push(d);
                    qs.push(q);
                    if stop(d) {
                        break;
                    }
                }
                let k = chain.len() - 1;

                // verify: full target logits per row (no argmax shortcut)
                let recurrent = model.shape.family == Family::Qwen35;
                if recurrent {
                    st.qwen35.as_mut().ok_or("qwen35 state missing")?.gdn_snapshot()?;
                }
                let all = model
                    .forward_rows(st, &chain, pos, (k + 1) as u32)?
                    .ok_or("no verify logits")?;

                // rejection sampling over the drafted prefix
                let mut j = 0usize;
                let mut resid_next: Option<u32> = None;
                while j < k {
                    let p = sampler.dist(&all[j * v..(j + 1) * v]);
                    match sampler.spec_step(chain[j + 1], &qs[j], &p) {
                        Ok(_) => {
                            st.mtp_accepted += 1;
                            j += 1;
                        }
                        Err(resid) => {
                            resid_next = Some(resid);
                            break;
                        }
                    }
                }
                // next token: residual on a reject, else the bonus sampled
                // from the target's row-k distribution (whole chain accepted)
                next = match resid_next {
                    Some(r) => r,
                    None => {
                        // whole chain accepted: bonus from the target's
                        // row-k distribution, via the same dist() the
                        // accept step used (consistency)
                        let bonus = sampler.dist(&all[k * v..(k + 1) * v]);
                        sampler.sample_from(&bonus)
                    }
                };
                if debug {
                    eprintln!("mtp-sample: pos={pos} chain={chain:?} accepted={j}/{k} next={next}");
                }

                if recurrent && j < k {
                    st.qwen35.as_mut().ok_or("qwen35 state missing")?.gdn_restore()?;
                    model.forward_batch(st, &chain[..=j], pos, false)?;
                }
                kernels::copy_d2d(&mut st.mtp_hidden, 0, &st.mtp_hidden_save, 0, row)?;
                model.mtp_prefill_fill(st, (j + 1) as u32, pos)?;
                pos += (j + 1) as u32;

                for &d in &chain[1..=j] {
                    if stop(d) {
                        break 'sround;
                    }
                    if emitted >= max_tokens {
                        break 'sround;
                    }
                    on_token(d);
                    emitted += 1;
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
            // Lookup-augmented block: extend the MTP draft chain up to this
            // many total rows with tokens pulled straight from the context
            // (PULSAR_SPEC_LOOKUP=<block>, e.g. 16). MTP drafts the first
            // few (its strength: generative continuation); context lookup
            // fills the rest (its strength: verbatim copy). One verify step
            // then accepts many tokens on copy-heavy work. Capped by
            // spec_rows (16 on qwen35). 0/unset = plain MTP, unchanged.
            let lookup_block = std::env::var("PULSAR_SPEC_LOOKUP")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(0)
                .min(st.spec_rows as usize);
            let mut hist: Vec<u32> = if lookup_block > 0 { prompt.to_vec() } else { Vec::new() };
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
                if lookup_block > 0 {
                    hist.push(next);
                }

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
                // Extend the block from context (lookup drafts need no MTP
                // hidden - the verify forward computes real hiddens and the
                // fill re-anchors on the accepted prefix). Cap the whole
                // chain at both lookup_block and the ctx headroom.
                if lookup_block > chain.len() && !stop(chain[chain.len() - 1]) {
                    let room = (st.ctx() - pos - 2) as usize;
                    let want = lookup_block.min(room + 1).saturating_sub(chain.len());
                    let mut lseq = hist.clone();
                    lseq.extend_from_slice(&chain[1..]);
                    let ext = ngram_continuation(&lseq, want);
                    st.mtp_drafted += ext.len() as u64;
                    chain.extend(ext);
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
                let mut t_snap = std::time::Duration::ZERO;
                if recurrent {
                    let ts = std::time::Instant::now();
                    st.qwen35.as_mut().ok_or("qwen35 state missing")?.gdn_snapshot()?;
                    t_snap = ts.elapsed();
                }
                // qwen35: readback-free verify (device argmax, 4 bytes a
                // row); other spec families keep the logits readback
                let mut t_amax = std::time::Duration::ZERO;
                let row_amax: Vec<u32> = if recurrent {
                    st.skip_logit_read = true;
                    let tf = std::time::Instant::now();
                    let r = model.forward_rows(st, &chain, pos, (k + 1) as u32);
                    st.skip_logit_read = false;
                    t_amax = tf.elapsed();
                    r?.ok_or("no verify logits")?;
                    std::mem::take(&mut st.last_argmax)
                } else {
                    let all = model
                        .forward_rows(st, &chain, pos, (k + 1) as u32)?
                        .ok_or("no verify logits")?;
                    (0..=k).map(|i| argmax(&all[i * v..(i + 1) * v])).collect()
                };
                t_verify += t0.elapsed();
                if timing {
                    eprintln!("mtp step: draft {:.2}ms snap {:.2}ms fwd+amax {:.2}ms verify-total {:.2}ms",
                        t_draft.as_secs_f64() * 1e3 / (emitted + 1).max(1) as f64,
                        t_snap.as_secs_f64() * 1e3,
                        t_amax.as_secs_f64() * 1e3,
                        t0.elapsed().as_secs_f64() * 1e3);
                }
                // Per-position draft quality, measured INDEPENDENTLY of
                // the prefix rule: position i counts as a hit when the
                // verified argmax equals the drafted token there, even if
                // an earlier position already broke the chain. Prefix
                // acceptance hides drift (everything after a miss reads
                // as 0); this exposes it.
                for i in 0..k {
                    if i < st.mtp_pos_drafted.len() {
                        st.mtp_pos_drafted[i] += 1;
                        if row_amax[i] == chain[i + 1] {
                            st.mtp_pos_accepted[i] += 1;
                        }
                    }
                }
                let mut j = 0usize;
                while j < k && row_amax[j] == chain[j + 1] {
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
                    eprintln!("mtp: pos={pos} chain={chain:?} accepted={j}/{k}");
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
                next = row_amax[j];

                for &d in &chain[1..=j] {
                    if stop(d) {
                        break 'round; // forwarded, not emitted - as non-spec
                    }
                    if emitted >= max_tokens {
                        break 'round;
                    }
                    on_token(d);
                    emitted += 1;
                    if lookup_block > 0 {
                        hist.push(d);
                    }
                }
            }
            if timing {
                eprintln!(
                    "mtp timing: draft {:.2}s verify {:.2}s refwd {:.2}s fill {:.2}s over {emitted} tokens",
                    t_draft.as_secs_f64(), t_verify.as_secs_f64(), t_refwd.as_secs_f64(), t_fill.as_secs_f64()
                );
                let rates: Vec<String> = st
                    .mtp_pos_drafted
                    .iter()
                    .zip(&st.mtp_pos_accepted)
                    .take_while(|(d, _)| **d > 0)
                    .map(|(d, a)| format!("{:.3}", *a as f64 / *d as f64))
                    .collect();
                if !rates.is_empty() {
                    eprintln!("mtp per-position acceptance: {}", rates.join(", "));
                }
            }
            return Ok(pos);
        }

        // plain greedy on qwen35: argmax-only rows ride the device
        // argmax (and the vocab-split head under TP) - no megabyte
        // logits readback per token. Sampling paths keep full logits.
        if sampler.is_greedy() && model.shape.family == Family::Qwen35 {
            let mut next = argmax(logits.as_deref().ok_or("no logits")?);
            for _ in 0..max_tokens {
                if cancel() {
                    return Ok(pos);
                }
                if stop(next) || pos + 1 >= st.ctx() {
                    model.forward_batch(st, &[next], pos, false)?;
                    pos += 1;
                    break;
                }
                on_token(next);
                st.skip_logit_read = true;
                let r = model.forward_batch(st, &[next], pos, true);
                st.skip_logit_read = false;
                r?;
                next = *st.last_argmax.first().ok_or("no argmax row")?;
                pos += 1;
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

    /// Lookup-augmented draft continuation (prompt-lookup / n-gram): find
    /// the most recent earlier occurrence of `seq`'s 2..=4-token suffix and
    /// return up to `want` of the tokens that FOLLOWED it. On copy-heavy
    /// work (RAG, doc reproduction, applying an edit) this fills the draft
    /// block straight from the context, so one verify step accepts many
    /// tokens - the trick behind the 15/16-accept "reproduce a document"
    /// throughput. Returns empty when nothing matches (generative text),
    /// so the caller falls back to the model's own drafts.
    pub fn ngram_continuation(seq: &[u32], want: usize) -> Vec<u32> {
        if want == 0 || seq.len() < 4 {
            return Vec::new();
        }
        // min 3-token match: 2-token suffixes match spuriously on
        // generative text (drafts get rejected, wasting verify rows);
        // real context runs (docs, code) match 3+ exactly.
        for m in (3..=4usize.min(seq.len() - 1)).rev() {
            let suf = &seq[seq.len() - m..];
            let limit = seq.len() - m;
            for i in (0..limit).rev() {
                if &seq[i..i + m] == suf {
                    let mut out = Vec::new();
                    let mut j = i + m;
                    while out.len() < want && j < seq.len() {
                        out.push(seq[j]);
                        j += 1;
                    }
                    if !out.is_empty() {
                        return out;
                    }
                }
            }
        }
        Vec::new()
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

        /// The truncated, normalized sampling distribution as (token,
        /// prob) pairs over the kept support - EXACTLY the set `sample`
        /// would draw from (same temp/top_p/min_p filter), so speculative
        /// draft (q) and target (p) distributions match token-for-token
        /// where their supports overlap. Sorted descending by prob.
        /// Empty when greedy (temp<=0): the caller uses the argmax path.
        pub fn dist(&self, logits: &[f32]) -> Vec<(u32, f32)> {
            if self.temp <= 0.0 {
                return Vec::new();
            }
            // full-softmax denominator + max (the min_p/top_p thresholds
            // are against the FULL distribution, so this reduction is
            // needed - but it is O(n), not a sort)
            let maxl = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut full_sum = 0f32;
            for &l in logits {
                full_sum += ((l - maxl) / self.temp).exp();
            }
            // only the highest logits can survive top_p/min_p, so
            // partial-select a top slab and sort THAT instead of the whole
            // 248k vocab (the full sort was ~40% of spec-sampling time).
            // 4096 covers top_p at any sane temp; the fallback keeps it
            // exact if a pathologically flat row needs more.
            let n = logits.len();
            let build = |idx: &[u32]| -> (Vec<(u32, f32)>, f32) {
                let p0 = ((logits[idx[0] as usize] - maxl) / self.temp).exp() / full_sum;
                let mut kept: Vec<(u32, f32)> = Vec::new();
                let (mut cum, mut kept_sum) = (0f32, 0f32);
                for &i in idx {
                    let e = ((logits[i as usize] - maxl) / self.temp).exp();
                    let p = e / full_sum;
                    if self.min_p > 0.0 && p < self.min_p * p0 && !kept.is_empty() {
                        break;
                    }
                    cum += p;
                    kept.push((i, e));
                    kept_sum += e;
                    if self.top_p < 1.0 && cum >= self.top_p {
                        break;
                    }
                }
                for c in kept.iter_mut() {
                    c.1 /= kept_sum;
                }
                (kept, cum)
            };
            let slab = 4096.min(n);
            let mut idx: Vec<u32> = (0..n as u32).collect();
            let cmp = |a: &u32, b: &u32| logits[*b as usize].total_cmp(&logits[*a as usize]);
            if slab < n {
                idx.select_nth_unstable_by(slab - 1, cmp);
                idx.truncate(slab);
            }
            idx.sort_unstable_by(cmp);
            let (kept, cum) = build(&idx);
            // slab exhausted before reaching top_p (only a near-uniform
            // row): redo exactly over the full sorted vocab.
            if self.top_p < 1.0 && cum < self.top_p && slab < n {
                let mut full: Vec<u32> = (0..n as u32).collect();
                full.sort_unstable_by(cmp);
                return build(&full).0;
            }
            kept
        }

        /// A uniform draw in [0,1) from the sampler's own RNG, for the
        /// speculative accept/reject test.
        pub fn accept_uniform(&mut self) -> f32 {
            self.randf()
        }

        /// Prob of `tok` in a (token,prob) distribution, 0 if outside its
        /// support.
        fn prob_of(dist: &[(u32, f32)], tok: u32) -> f32 {
            dist.iter().find(|(t, _)| *t == tok).map_or(0.0, |(_, p)| *p)
        }

        /// Draw one token from an explicit (token,prob) distribution
        /// (used for the bonus token when the whole chain is accepted).
        pub fn sample_from(&mut self, dist: &[(u32, f32)]) -> u32 {
            let sum: f32 = dist.iter().map(|(_, p)| p).sum();
            let mut r = self.randf() * sum;
            for &(t, p) in dist {
                if r < p {
                    return t;
                }
                r -= p;
            }
            dist.last().map_or(0, |&(t, _)| t)
        }

        /// One speculative accept/reject step (Leviathan/Chen). `drafted`
        /// was sampled from `q`; `p` is the target distribution at the
        /// same position. Returns Some(accepted token) when the draft is
        /// accepted, or None with the residual token to emit instead
        /// (sampled from norm(relu(p-q)) over the union support), which
        /// also ends the chain. Distribution-preserving: over the RNG the
        /// emitted token is distributed exactly as p.
        pub fn spec_step(&mut self, drafted: u32, q: &[(u32, f32)], p: &[(u32, f32)]) -> std::result::Result<u32, u32> {
            let qd = Self::prob_of(q, drafted).max(1e-30);
            let pd = Self::prob_of(p, drafted);
            if self.randf() <= (pd / qd).min(1.0) {
                return Ok(drafted);
            }
            // residual: (p - q)+ over the union of supports, normalized
            let mut resid: Vec<(u32, f32)> = Vec::with_capacity(p.len());
            for &(t, pt) in p {
                let r = pt - Self::prob_of(q, t);
                if r > 0.0 {
                    resid.push((t, r));
                }
            }
            let s: f32 = resid.iter().map(|(_, r)| r).sum();
            if s <= 0.0 {
                // p is a subset of q at this token's mass; fall back to a
                // fresh draw from p (rare; keeps output within p's support)
                return Err(self.sample_from(p));
            }
            for c in resid.iter_mut() {
                c.1 /= s;
            }
            Err(self.sample_from(&resid))
        }
    }
}

#[cfg(target_os = "linux")]
pub use real::*;
