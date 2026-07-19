//! Qwen3.5/3.6 MoE hybrid (qwen35moe) forward path + DFlash speculative
//! decoding, tasks #21/#23.
//!
//! References: llama.cpp qwen35moe.cpp + delta-net-base.cpp; lucebox
//! draft_graph.cpp + dflash_spec_decode.cpp (docs/qwen36-port-notes.md,
//! docs/dflash-port-notes.md). 3 of 4 layers run Gated DeltaNet linear
//! attention (conv window + delta-rule state, O(1) memory); every 4th
//! layer is sigmoid-gated full attention with partial neox rope. MoE on
//! every layer: softmax top-8 of 256 + a shared expert behind a scalar
//! sigmoid gate.
//!
//! The forward is BATCHED in chunks of up to 16 tokens: projections,
//! attention, and the MoE union run batched while the GDN recurrences
//! loop tokens inside single kernel launches. That chunk width is what
//! makes DFlash verify (16 candidate rows for the cost of a few
//! sequential tokens) and chunked prefill work.

use super::{Attn, Ffn, LayerW, Model, Result, State};
use kernels::DeviceBuf;

/// Verify/prefill chunk width (DFlash block size; also the register
/// budget the batched GDN kernel was written for).
const T_MAX: usize = 16;
/// DFlash feature-ring capacity = the draft context window (lucebox
/// defaults to 2048; v1 keeps the fc cost down with 256).
const RING_CAP: usize = 256;

fn argmax(row: &[f32]) -> u32 {
    let mut best = 0usize;
    for (i, &v) in row.iter().enumerate() {
        if v > row[best] {
            best = i;
        }
    }
    best as u32
}

/// Per-GDN-layer device state.
struct GdnState {
    /// delta-rule state [ssm_v_heads][ssm_state][ssm_state]
    s: DeviceBuf,
    /// conv window [ssm_conv_k - 1][conv_dim]
    conv: DeviceBuf,
}

/// DFlash runtime state riding on Qwen35Rt (allocated on first use).
struct DflashRt {
    /// captured target features [RING_CAP][n_capture * n_embd] f32,
    /// slot = position % RING_CAP
    ring: DeviceBuf,
    /// pre-verify snapshots (S + conv per GDN layer)
    snap_s: Vec<Option<DeviceBuf>>,
    snap_conv: Vec<Option<DeviceBuf>>,
    /// fast-rollback stashes, per GDN layer: the verify chunk's raw
    /// qkv projections [16][conv_dim] and final g/beta [16][v_heads] -
    /// enough to replay ONLY the conv+delta recurrences after a
    /// snapshot restore (no matmuls, no MoE, no attention)
    stash_qkv: Vec<Option<DeviceBuf>>,
    stash_g: Vec<Option<DeviceBuf>>,
    stash_beta: Vec<Option<DeviceBuf>>,
    /// stash during the CURRENT forward (set for verify passes)
    capture_gdn: bool,
    /// capture layer ids from the draft gguf
    layer_ids: Vec<usize>,
}

/// qwen35 runtime: GDN states + scratch sized for T_MAX-token chunks.
pub(super) struct Qwen35Rt {
    states: Vec<Option<GdnState>>,
    qkv: DeviceBuf,      // [T][conv_dim] raw projection
    conv_out: DeviceBuf, // [T][conv_dim] conv+silu, layout [q|k|v] per row
    z: DeviceBuf,        // [T][value_dim]
    gq: DeviceBuf,       // [T][key_dim] delta-rule inputs
    gk: DeviceBuf,       // [T][key_dim]
    gv: DeviceBuf,       // [T][value_dim]
    small: DeviceBuf,    // [T][ssm_v_heads] alpha/beta matvec scratch
    g: DeviceBuf,        // [T][ssm_v_heads] log-decay upload
    beta: DeviceBuf,     // [T][ssm_v_heads]
    gdn_o: DeviceBuf,    // [T][value_dim]
    gdn_tmp: DeviceBuf,  // [T][value_dim]
    qfull: DeviceBuf,    // [T][2*n_head*head_dim] fused q+gate
    gate: DeviceBuf,     // [T][n_head*head_dim]
    shg: DeviceBuf,      // [T] shared-expert gate logits
    dflash: Option<DflashRt>,
}

impl Qwen35Rt {
    pub fn new(m: &Model) -> Result<Qwen35Rt> {
        let s = m.shape;
        let key_dim = (s.ssm_k_heads * s.ssm_state) as usize;
        let value_dim = (s.ssm_v_heads * s.ssm_state) as usize;
        let conv_dim = 2 * key_dim + value_dim;
        let mut states = Vec::with_capacity(s.n_exec_layer as usize);
        for il in 0..s.n_exec_layer {
            if (il + 1) % s.full_attn_interval == 0 {
                states.push(None);
            } else {
                let sbytes = s.ssm_v_heads as usize
                    * s.ssm_state as usize
                    * s.ssm_state as usize
                    * 4;
                let cbytes = (s.ssm_conv_k as usize - 1) * conv_dim * 4;
                let mut st = GdnState {
                    s: DeviceBuf::alloc(sbytes)?,
                    conv: DeviceBuf::alloc(cbytes)?,
                };
                kernels::zero(&mut st.s, sbytes)?;
                kernels::zero(&mut st.conv, cbytes)?;
                states.push(Some(st));
            }
        }
        let f32s = |n: usize| DeviceBuf::alloc(n * 4);
        Ok(Qwen35Rt {
            states,
            qkv: f32s(T_MAX * conv_dim)?,
            conv_out: f32s(T_MAX * conv_dim)?,
            z: f32s(T_MAX * value_dim)?,
            gq: f32s(T_MAX * key_dim)?,
            gk: f32s(T_MAX * key_dim)?,
            gv: f32s(T_MAX * value_dim)?,
            small: f32s(T_MAX * s.ssm_v_heads as usize)?,
            g: f32s(T_MAX * s.ssm_v_heads as usize)?,
            beta: f32s(T_MAX * s.ssm_v_heads as usize)?,
            gdn_o: f32s(T_MAX * value_dim)?,
            gdn_tmp: f32s(T_MAX * value_dim)?,
            qfull: f32s(T_MAX * 2 * (s.n_head * s.head_dim) as usize)?,
            gate: f32s(T_MAX * (s.n_head * s.head_dim) as usize)?,
            shg: f32s(T_MAX)?,
            dflash: None,
        })
    }

    fn reset(&mut self) -> Result {
        for st in self.states.iter_mut().flatten() {
            let (sb, cb) = (st.s.bytes(), st.conv.bytes());
            kernels::zero(&mut st.s, sb)?;
            kernels::zero(&mut st.conv, cb)?;
        }
        Ok(())
    }

    fn enable_dflash(&mut self, m: &Model, layer_ids: Vec<usize>) -> Result {
        if self.dflash.is_some() {
            return Ok(());
        }
        let s = m.shape;
        let feat_w = layer_ids.len() * s.n_embd as usize;
        let key_dim = (s.ssm_k_heads * s.ssm_state) as usize;
        let value_dim = (s.ssm_v_heads * s.ssm_state) as usize;
        let conv_dim = 2 * key_dim + value_dim;
        let mut snap_s = Vec::new();
        let mut snap_conv = Vec::new();
        let mut stash_qkv = Vec::new();
        let mut stash_g = Vec::new();
        let mut stash_beta = Vec::new();
        for gs in &self.states {
            match gs {
                Some(g) => {
                    snap_s.push(Some(DeviceBuf::alloc(g.s.bytes())?));
                    snap_conv.push(Some(DeviceBuf::alloc(g.conv.bytes())?));
                    stash_qkv.push(Some(DeviceBuf::alloc(T_MAX * conv_dim * 4)?));
                    stash_g.push(Some(DeviceBuf::alloc(T_MAX * s.ssm_v_heads as usize * 4)?));
                    stash_beta.push(Some(DeviceBuf::alloc(T_MAX * s.ssm_v_heads as usize * 4)?));
                }
                None => {
                    snap_s.push(None);
                    snap_conv.push(None);
                    stash_qkv.push(None);
                    stash_g.push(None);
                    stash_beta.push(None);
                }
            }
        }
        self.dflash = Some(DflashRt {
            ring: DeviceBuf::alloc(RING_CAP * feat_w * 4)?,
            snap_s,
            snap_conv,
            stash_qkv,
            stash_g,
            stash_beta,
            capture_gdn: false,
            layer_ids,
        });
        Ok(())
    }

    fn snapshot(&mut self) -> Result {
        let Some(df) = &mut self.dflash else {
            return Err("dflash not enabled".into());
        };
        for (il, gs) in self.states.iter().enumerate() {
            if let Some(g) = gs {
                let ss = df.snap_s[il].as_mut().unwrap();
                let sc = df.snap_conv[il].as_mut().unwrap();
                kernels::copy_d2d(ss, 0, &g.s, 0, g.s.bytes())?;
                kernels::copy_d2d(sc, 0, &g.conv, 0, g.conv.bytes())?;
            }
        }
        Ok(())
    }

    /// Fast rollback: restore the pre-verify snapshots and replay ONLY
    /// the conv + delta recurrences for the accepted prefix from the
    /// stashed inputs - no matmuls, no MoE, no attention, no lm head.
    /// KV caches and the feature ring already hold the correct rows
    /// (deterministic kernels wrote identical values during verify).
    fn rollback_to(&mut self, m: &Model, accept_n: u32) -> Result {
        let s = m.shape;
        let key_dim = s.ssm_k_heads * s.ssm_state;
        let value_dim = s.ssm_v_heads * s.ssm_state;
        let conv_dim = 2 * key_dim + value_dim;
        let Some(df) = &self.dflash else {
            return Err("dflash not enabled".into());
        };
        for (il, gs) in self.states.iter_mut().enumerate() {
            let Some(g) = gs else { continue };
            let ss = df.snap_s[il].as_ref().unwrap();
            let sc = df.snap_conv[il].as_ref().unwrap();
            kernels::copy_d2d(&mut g.s, 0, ss, 0, ss.bytes())?;
            kernels::copy_d2d(&mut g.conv, 0, sc, 0, sc.bytes())?;
            if accept_n == 0 {
                continue;
            }
            let sq = df.stash_qkv[il].as_ref().unwrap();
            let Attn::Qwen35(w) = &m.layers[il].attn else {
                return Err("qwen35 layer expected".into());
            };
            let gdn = w.gdn.as_ref().ok_or("gdn weights missing")?;
            kernels::qwen35_conv_batch(&mut self.conv_out, sq, &gdn.conv, &mut g.conv, conv_dim, s.ssm_conv_k, accept_n)?;
            kernels::qwen35_split_qkv(&mut self.gq, &mut self.gk, &mut self.gv, &self.conv_out, accept_n, key_dim, value_dim)?;
            kernels::qwen35_l2_norm(&mut self.gq, accept_n * s.ssm_k_heads, s.ssm_state, s.rms_eps)?;
            kernels::qwen35_l2_norm(&mut self.gk, accept_n * s.ssm_k_heads, s.ssm_state, s.rms_eps)?;
            kernels::qwen35_gdn_batch(
                &mut self.gdn_o, &mut g.s, &self.gq, &self.gk, &self.gv,
                df.stash_g[il].as_ref().unwrap(),
                df.stash_beta[il].as_ref().unwrap(),
                s.ssm_v_heads, s.ssm_k_heads, s.ssm_state, accept_n,
            )?;
        }
        Ok(())
    }

    /// Full-snapshot restore (legacy path; rollback_to supersedes it).
    #[allow(dead_code)]
    fn restore(&mut self) -> Result {
        let Some(df) = &self.dflash else {
            return Err("dflash not enabled".into());
        };
        for (il, gs) in self.states.iter_mut().enumerate() {
            if let Some(g) = gs {
                let ss = df.snap_s[il].as_ref().unwrap();
                let sc = df.snap_conv[il].as_ref().unwrap();
                kernels::copy_d2d(&mut g.s, 0, ss, 0, ss.bytes())?;
                kernels::copy_d2d(&mut g.conv, 0, sc, 0, sc.bytes())?;
            }
        }
        Ok(())
    }
}

/* ---- DFlash draft model -------------------------------------------------- */

struct DraftLayer {
    attn_norm: DeviceBuf,
    wq: DeviceBuf,
    wk: DeviceBuf,
    wv: DeviceBuf,
    q_norm: DeviceBuf,
    k_norm: DeviceBuf,
    wo: DeviceBuf,
    ffn_norm: DeviceBuf, // post_attention_norm
    gate: DeviceBuf,
    up: DeviceBuf,
    down: DeviceBuf,
}

/// The DFlash block-diffusion draft (lucebox draft_graph semantics).
/// Shares the TARGET's token embedding and lm head.
pub struct DraftModel {
    layers: Vec<DraftLayer>,
    fc: DeviceBuf,          // q8_0 [n_capture*n_embd -> n_embd]
    hidden_norm: DeviceBuf, // f32 [n_embd]
    out_norm: DeviceBuf,
    pub block_size: usize,
    pub mask_id: u32,
    pub layer_ids: Vec<usize>,
    n_head: u32,
    n_kv: u32,
    head_dim: u32,
    rope: kernels::RopeCfg,
    n_embd: u32,
    ff: u32,
    // scratch
    feat_in: DeviceBuf, // [RING_CAP][n_capture*n_embd] window gather
    feat: DeviceBuf,    // [RING_CAP][n_embd] fused features
    h: DeviceBuf,    // [16][n_embd] block hidden
    hn: DeviceBuf,
    q: DeviceBuf,    // [16][n_head*dim]
    kcat: DeviceBuf, // [RING_CAP+16][n_kv*dim]
    vcat: DeviceBuf,
    attn: DeviceBuf, // [16][n_head*dim]
    ffa: DeviceBuf,  // [16][ff]
    ffb: DeviceBuf,
    ffm: DeviceBuf,
    tmp: DeviceBuf, // [16][n_embd]
}

impl DraftModel {
    pub fn load(path: &std::path::Path) -> Result<DraftModel> {
        let (shards, g) = super::parse_header(path)?;
        if g.architecture() != Some("dflash-draft") {
            return Err(format!("{path:?}: not a dflash-draft gguf").into());
        }
        let file = super::VFile::open(&shards)?;
        let u = |k: &str| -> Result<u32> {
            Ok(g.arch_meta(k)
                .and_then(gguf::Value::as_u64)
                .ok_or_else(|| format!("draft gguf missing {k}"))? as u32)
        };
        let n_layer = u("block_count")?;
        let n_embd = u("embedding_length")?;
        let ff = u("feed_forward_length")?;
        let n_head = u("attention.head_count")?;
        let n_kv = u("attention.head_count_kv")?;
        let head_dim = u("attention.key_length")?;
        let block_size = u("dflash.block_size")? as usize;
        let mask_id = u("dflash.mask_token_id")?;
        let rope_base = g
            .arch_meta("rope.freq_base")
            .and_then(gguf::Value::as_f32)
            .unwrap_or(10_000_000.0);
        // the z-lab draft is TRAINED with yarn (factor 64 / orig 4096);
        // ggml semantics: attn_factor 1.0, the kernel-internal
        // 1 + 0.1 ln(1/freq_scale) supplies the HF mscale
        let yarn_factor = g
            .arch_meta("rope.scaling.factor")
            .and_then(gguf::Value::as_f32)
            .unwrap_or(1.0);
        let rope = if yarn_factor > 1.0 {
            kernels::RopeCfg {
                n_ctx_orig: g
                    .arch_meta("rope.scaling.original_context_length")
                    .and_then(gguf::Value::as_u64)
                    .unwrap_or(4096) as u32,
                freq_base: rope_base,
                freq_scale: 1.0 / yarn_factor,
                ext_factor: 1.0,
                attn_factor: 1.0,
                beta_fast: 32.0,
                beta_slow: 1.0,
                kq_mult: 1.0,
            }
        } else {
            kernels::RopeCfg {
                n_ctx_orig: 0,
                freq_base: rope_base,
                freq_scale: 1.0,
                ext_factor: 0.0,
                attn_factor: 1.0,
                beta_fast: 0.0,
                beta_slow: 0.0,
                kq_mult: 1.0,
            }
        };
        let layer_ids: Vec<usize> = match g.arch_meta("dflash.target_layer_ids") {
            Some(gguf::Value::Array(a)) => {
                a.iter().filter_map(gguf::Value::as_u64).map(|v| v as usize).collect()
            }
            _ => return Err("draft gguf missing dflash.target_layer_ids".into()),
        };
        if block_size > T_MAX {
            return Err("draft block_size exceeds T_MAX".into());
        }
        let up = |name: &str| super::upload(&file, &g, name);
        let mut layers = Vec::with_capacity(n_layer as usize);
        for il in 0..n_layer {
            let t = |suf: &str| format!("blk.{il}.{suf}");
            layers.push(DraftLayer {
                attn_norm: up(&t("attn_norm.weight"))?,
                wq: up(&t("attn_q.weight"))?,
                wk: up(&t("attn_k.weight"))?,
                wv: up(&t("attn_v.weight"))?,
                q_norm: up(&t("attn_q_norm.weight"))?,
                k_norm: up(&t("attn_k_norm.weight"))?,
                wo: up(&t("attn_output.weight"))?,
                ffn_norm: up(&t("post_attention_norm.weight"))?,
                gate: up(&t("ffn_gate.weight"))?,
                up: up(&t("ffn_up.weight"))?,
                down: up(&t("ffn_down.weight"))?,
            });
        }
        let f32s = |n: usize| DeviceBuf::alloc(n * 4);
        let bs = block_size;
        let kv_rows = RING_CAP + bs;
        let n_cap = layer_ids.len();
        Ok(DraftModel {
            fc: up("dflash_fc.weight")?,
            hidden_norm: up("dflash_hidden_norm.weight")?,
            out_norm: up("output_norm.weight")?,
            layers,
            block_size: bs,
            mask_id,
            layer_ids,
            n_head,
            n_kv,
            head_dim,
            rope,
            n_embd,
            ff,
            feat_in: f32s(RING_CAP * n_cap * n_embd as usize)?,
            feat: f32s(RING_CAP * n_embd as usize)?,
            h: f32s(bs * n_embd as usize)?,
            hn: f32s(bs * n_embd as usize)?,
            q: f32s(bs * (n_head * head_dim) as usize)?,
            kcat: f32s(kv_rows * (n_kv * head_dim) as usize)?,
            vcat: f32s(kv_rows * (n_kv * head_dim) as usize)?,
            attn: f32s(bs * (n_head * head_dim) as usize)?,
            ffa: f32s(bs * ff as usize)?,
            ffb: f32s(bs * ff as usize)?,
            ffm: f32s(bs * ff as usize)?,
            tmp: f32s(bs * n_embd as usize)?,
        })
    }
}

/* ---- forward ------------------------------------------------------------- */

impl Model {
    pub(super) fn forward_qwen35(&self, st: &mut State, tokens: &[u32], pos0: u32, rows: u32) -> Result<Option<Vec<f32>>> {
        if tokens.is_empty() {
            return Err("empty batch".into());
        }
        if rows as usize > T_MAX {
            return Err("qwen35: rows exceeds the verify chunk".into());
        }
        if pos0 + tokens.len() as u32 > st.ctx {
            return Err("position exceeds context".into());
        }
        let mut rt = st.qwen35.take().ok_or("qwen35 state missing")?;
        let r = self.forward_qwen35_inner(st, &mut rt, tokens, pos0, rows);
        st.qwen35 = Some(rt);
        r
    }

    fn forward_qwen35_inner(&self, st: &mut State, rt: &mut Qwen35Rt, tokens: &[u32], pos0: u32, rows: u32) -> Result<Option<Vec<f32>>> {
        let s = self.shape;
        if pos0 == 0 {
            rt.reset()?;
        }
        // chunked batched forward; `rows` logits must come from ONE
        // final chunk (callers keep verify blocks <= T_MAX)
        let mut pos = pos0;
        let mut last_t = 0u32;
        for chunk in tokens.chunks(T_MAX) {
            let t = chunk.len() as u32;
            let ids: Vec<i32> = chunk.iter().map(|&x| x as i32).collect();
            st.tok.write(0, kernels::as_bytes(&ids))?;
            kernels::embed_q8_0(&mut st.cur, &self.token_embd, &st.tok, s.n_embd, s.n_vocab, t)?;
            for (il, l) in self.layers.iter().enumerate() {
                self.eval_qwen35_layer(st, rt, il, l, pos, t)?;
                if std::env::var_os("PULSAR_DEBUG_L2").is_some() {
                    let a = st.after_attn.read_f32(4)?;
                    let v = st.cur.read_f32(4)?;
                    eprintln!("L{il}: attn {:?} cur {:?}", &a[..2], &v[..2]);
                }
            }
            pos += t;
            last_t = t;
        }
        if rows == 0 {
            return Ok(None);
        }
        if rows > last_t {
            return Err("qwen35: rows exceeds the final chunk".into());
        }
        let k = rows;
        let row = s.n_embd as usize * 4;
        kernels::copy_d2d(&mut st.last_row, 0, &st.cur, (last_t - k) as usize * row, k as usize * row)?;
        kernels::rms_norm(&mut st.normed, &st.last_row, &self.output_norm, s.n_embd, k, s.rms_eps)?;
        self.head_logits(st, k)?;
        kernels::sync()?;
        Ok(Some(st.logits.read_f32(k as usize * s.n_vocab as usize)?))
    }

    fn eval_qwen35_layer(&self, st: &mut State, rt: &mut Qwen35Rt, il: usize, l: &LayerW, pos: u32, t: u32) -> Result {
        let s = self.shape;
        let eps = s.rms_eps;
        let Attn::Qwen35(w) = &l.attn else {
            return Err("qwen35 layer without Qwen35 attn weights".into());
        };
        let key_dim = s.ssm_k_heads * s.ssm_state;
        let value_dim = s.ssm_v_heads * s.ssm_state;
        let conv_dim = 2 * key_dim + value_dim;

        // ---- DFlash feature capture: HF hidden_states[il] convention -
        // the residual stream ENTERING layer il (= output of layer il-1)
        if let Some(df) = &mut rt.dflash {
            if let Some(idx) = df.layer_ids.iter().position(|&x| x == il) {
                let stride = (df.layer_ids.len() as u32) * s.n_embd;
                kernels::qwen35_ring_scatter(
                    &mut df.ring,
                    &st.cur,
                    pos,
                    RING_CAP as u32,
                    t,
                    s.n_embd,
                    stride,
                    idx as u32 * s.n_embd,
                )?;
            }
        }

        kernels::rms_norm(&mut st.normed, &st.cur, &l.attn_norm, s.n_embd, t, eps)?;

        if let Some(gdn) = &w.gdn {
            // ---- Gated DeltaNet (recurrences loop inside the launches)
            kernels::matmul_q8_0(&mut rt.qkv, &gdn.wqkv, &st.normed, s.n_embd, conv_dim, t)?;
            kernels::matmul_q8_0(&mut rt.z, &gdn.wz, &st.normed, s.n_embd, value_dim, t)?;
            // g/beta coefficients fully on-device (no host readbacks)
            kernels::matmul_f32(&mut rt.g, &gdn.alpha_w, &st.normed, s.n_embd, s.ssm_v_heads, t)?;
            kernels::matmul_f32(&mut rt.beta, &gdn.beta_w, &st.normed, s.n_embd, s.ssm_v_heads, t)?;
            kernels::qwen35_gdn_coeffs(&mut rt.g, &mut rt.beta, &gdn.a, &gdn.dt_bias, t, s.ssm_v_heads)?;

            // fast-rollback stash: the raw qkv rows + final coeffs are
            // all a state-only replay needs
            if let Some(df) = &mut rt.dflash {
                if df.capture_gdn {
                    let sq = df.stash_qkv[il].as_mut().ok_or("stash missing")?;
                    kernels::copy_d2d(sq, 0, &rt.qkv, 0, (t * conv_dim) as usize * 4)?;
                    let sg = df.stash_g[il].as_mut().unwrap();
                    kernels::copy_d2d(sg, 0, &rt.g, 0, (t * s.ssm_v_heads) as usize * 4)?;
                    let sb = df.stash_beta[il].as_mut().unwrap();
                    kernels::copy_d2d(sb, 0, &rt.beta, 0, (t * s.ssm_v_heads) as usize * 4)?;
                }
            }
            let gs = rt.states[il].as_mut().ok_or("gdn state missing")?;
            kernels::qwen35_conv_batch(&mut rt.conv_out, &rt.qkv, &gdn.conv, &mut gs.conv, conv_dim, s.ssm_conv_k, t)?;
            // split [q|k|v] rows into contiguous batch buffers, one launch
            kernels::qwen35_split_qkv(&mut rt.gq, &mut rt.gk, &mut rt.gv, &rt.conv_out, t, key_dim, value_dim)?;
            kernels::qwen35_l2_norm(&mut rt.gq, t * s.ssm_k_heads, s.ssm_state, eps)?;
            kernels::qwen35_l2_norm(&mut rt.gk, t * s.ssm_k_heads, s.ssm_state, eps)?;
            kernels::qwen35_gdn_batch(
                &mut rt.gdn_o, &mut gs.s, &rt.gq, &rt.gk, &rt.gv, &rt.g, &rt.beta,
                s.ssm_v_heads, s.ssm_k_heads, s.ssm_state, t,
            )?;
            kernels::gqa_head_rms_norm(&mut rt.gdn_o, Some(&gdn.ssm_norm), t * s.ssm_v_heads, s.ssm_state, eps)?;
            kernels::swiglu(&mut rt.gdn_tmp, &rt.z, &rt.gdn_o, t * value_dim, 0.0, 1.0, 0)?;
            kernels::matmul_q8_0(&mut st.attn_out, &gdn.ssm_out, &rt.gdn_tmp, value_dim, s.n_embd, t)?;
        } else if let Some(attn) = &w.attn {
            // ---- sigmoid-gated full attention (partial neox rope)
            let hd = s.head_dim;
            kernels::matmul_q8_0(&mut rt.qfull, &attn.wq, &st.normed, s.n_embd, 2 * s.n_head * hd, t)?;
            // per-token rows are contiguous: treat (token, head) as one
            // flat head axis for the strided split
            kernels::qwen35_split_gate(&mut st.q, &mut rt.gate, &rt.qfull, t * s.n_head, hd)?;
            kernels::matmul_q8_0(&mut st.k, &attn.wk, &st.normed, s.n_embd, s.n_head_kv * hd, t)?;
            kernels::matmul_q8_0(&mut st.v, &attn.wv, &st.normed, s.n_embd, s.n_head_kv * hd, t)?;
            kernels::gqa_head_rms_norm(&mut st.q, Some(&attn.q_norm), t * s.n_head, hd, eps)?;
            kernels::gqa_head_rms_norm(&mut st.k, Some(&attn.k_norm), t * s.n_head_kv, hd, eps)?;
            kernels::gqa_rope(&mut st.q, t, s.n_head, hd, s.rot_dim, pos, s.rope_freq_base, None)?;
            kernels::gqa_rope(&mut st.k, t, s.n_head_kv, hd, s.rot_dim, pos, s.rope_freq_base, None)?;
            kernels::gqa_kv_append(&mut st.kcache[il], &st.k, t, s.n_head_kv, hd, st.ctx, pos, 0)?;
            kernels::gqa_kv_append(&mut st.vcache[il], &st.v, t, s.n_head_kv, hd, st.ctx, pos, 0)?;
            kernels::gqa_attention_rel(
                &mut st.heads, &st.q, &st.kcache[il], &st.vcache[il],
                t, s.n_head, s.n_head_kv, hd, st.ctx, pos,
                1.0 / (hd as f32).sqrt(), 0, None, 0, 0,
            )?;
            kernels::qwen35_sigmoid_gate(&mut st.heads, &rt.gate, t * s.n_head * hd)?;
            kernels::matmul_q8_0(&mut st.attn_out, &l.attn_output, &st.heads, s.n_head * hd, s.n_embd, t)?;
        } else {
            return Err("qwen35 layer with neither attn nor gdn".into());
        }
        kernels::add(&mut st.after_attn, &st.cur, &st.attn_out, t * s.n_embd)?;

        // ---- MoE (pre-norm residual)
        kernels::rms_norm(&mut st.normed, &st.after_attn, &l.ffn_norm, s.n_embd, t, eps)?;
        let Ffn::Moe { gate_inp, probs_b, shexp, gate_exps, up_exps, down_exps, .. } = &l.ffn else {
            return Err("qwen35 layer without MoE ffn".into());
        };
        if s.n_expert > 1 {
            kernels::matmul_f32(&mut st.router_logits, gate_inp, &st.normed, s.n_embd, s.n_expert, t)?;
            kernels::router_select(
                &mut st.router_selected,
                &mut st.router_weights,
                &st.router_logits,
                probs_b,
                s.n_expert,
                s.n_expert_used,
                s.expert_weight_scale,
                t,
                1, // softmax mode
                0,
            )?;
        } else {
            // dense-as-one-expert (qwen35 27B): expert 0, weight 1.0
            let ones = vec![1.0f32; t as usize];
            st.router_weights.write(0, kernels::as_bytes(&ones))?;
        }
        if let Some((sg, su, sd)) = shexp {
            kernels::matmul_q8_0(&mut st.gate_act, sg, &st.normed, s.n_embd, s.n_ff_exp, t)?;
            kernels::matmul_q8_0(&mut st.up_act, su, &st.normed, s.n_embd, s.n_ff_exp, t)?;
            kernels::swiglu(&mut st.ffn_mid, &st.gate_act, &st.up_act, t * s.n_ff_exp, 0.0, 1.0, 0)?;
            kernels::matmul_q8_0(&mut st.shared_out, sd, &st.ffn_mid, s.n_ff_exp, s.n_embd, t)?;
            kernels::matmul_f32(&mut rt.shg, &w.shexp_gate, &st.normed, s.n_embd, 1, t)?;
            kernels::qwen35_row_sigmoid_scale(&mut st.shared_out, &rt.shg, t, s.n_embd)?;
        } else {
            kernels::zero(&mut st.shared_out, (t * s.n_embd) as usize * 4)?;
        }
        kernels::quantize_q8_k(&mut st.xq, &st.normed, s.n_embd, t)?;
        kernels::sync()?;
        let selected = if s.n_expert > 1 {
            st.router_selected.read_i32((t * s.n_expert_used) as usize)?
        } else {
            vec![0i32; t as usize]
        };
        self.dsv4_moe(st, &selected, gate_exps, up_exps, down_exps, 0, t)?;
        kernels::add(&mut st.ffn_out, &st.moe_out, &st.shared_out, t * s.n_embd)?;
        kernels::add(&mut st.cur, &st.after_attn, &st.ffn_out, t * s.n_embd)?;
        Ok(())
    }
}

/* ---- DFlash spec decode --------------------------------------------------- */

impl Model {
    /// One draft forward: [last_tok, MASK x bs-1] + the feature window
    /// -> bs candidate ids (row 0 overwritten with last_tok).
    fn dflash_draft(&self, st: &mut State, d: &mut DraftModel, committed: u32, last_tok: u32) -> Result<Vec<u32>> {
        let s = self.shape;
        let bs = d.block_size;
        let w_eff = (committed as usize).min(RING_CAP);
        let start = committed as usize - w_eff;
        let feat_w = d.layer_ids.len() * s.n_embd as usize * 4;
        // gather the window in position order (one modulo-gather launch)
        {
            let rt = st.qwen35.as_ref().ok_or("qwen35 state missing")?;
            let df = rt.dflash.as_ref().ok_or("dflash not enabled")?;
            kernels::qwen35_ring_gather(
                &mut d.feat_in,
                &df.ring,
                (start % RING_CAP) as u32,
                RING_CAP as u32,
                w_eff as u32,
                (feat_w / 4) as u32,
            )?;
        }
        // noise block: [last_tok, MASK x bs-1] embedded with the target table
        let mut ids: Vec<i32> = vec![d.mask_id as i32; bs];
        ids[0] = last_tok as i32;
        st.tok.write(0, kernels::as_bytes(&ids))?;
        kernels::embed_q8_0(&mut d.h, &self.token_embd, &st.tok, s.n_embd, s.n_vocab, bs as u32)?;

        let eps = s.rms_eps;
        let n_cap = d.layer_ids.len() as u32;
        // fuse: fc @ features -> rms(hidden_norm)
        kernels::matmul_q8_0(&mut d.feat, &d.fc, &d.feat_in, n_cap * s.n_embd, s.n_embd, w_eff as u32)?;
        kernels::rms_norm_inplace(&mut d.feat, &d.hidden_norm, s.n_embd, w_eff as u32, eps)?;

        let kv_dim = d.n_kv * d.head_dim;
        let q_dim = d.n_head * d.head_dim;
        let total_k = (w_eff + bs) as u32;
        for l in &d.layers {
            kernels::rms_norm(&mut d.hn, &d.h, &l.attn_norm, s.n_embd, bs as u32, eps)?;
            // K/V: context rows from features, block rows from hn
            kernels::matmul_q8_0(&mut d.kcat, &l.wk, &d.feat, s.n_embd, kv_dim, w_eff as u32)?;
            kernels::matmul_q8_0_off(&mut d.kcat, w_eff * kv_dim as usize * 4, &l.wk, 0, &d.hn, 0, s.n_embd, kv_dim, bs as u32)?;
            kernels::matmul_q8_0(&mut d.vcat, &l.wv, &d.feat, s.n_embd, kv_dim, w_eff as u32)?;
            kernels::matmul_q8_0_off(&mut d.vcat, w_eff * kv_dim as usize * 4, &l.wv, 0, &d.hn, 0, s.n_embd, kv_dim, bs as u32)?;
            kernels::gqa_head_rms_norm(&mut d.kcat, Some(&l.k_norm), total_k * d.n_kv, d.head_dim, eps)?;
            // Q from the block only
            kernels::matmul_q8_0(&mut d.q, &l.wq, &d.hn, s.n_embd, q_dim, bs as u32)?;
            kernels::gqa_head_rms_norm(&mut d.q, Some(&l.q_norm), bs as u32 * d.n_head, d.head_dim, eps)?;
            // plain neox rope, full head, rebased positions
            kernels::qwen35_rope_yarn(&mut d.kcat, total_k, d.n_kv, d.head_dim, 0, &d.rope)?;
            kernels::qwen35_rope_yarn(&mut d.q, bs as u32, d.n_head, d.head_dim, w_eff as u32, &d.rope)?;
            // non-causal attention over all context + block rows
            kernels::qwen35_draft_attn(
                &mut d.attn, &d.q, &d.kcat, &d.vcat,
                bs as u32, total_k, d.n_head, d.n_kv, d.head_dim,
                1.0 / (d.head_dim as f32).sqrt(),
            )?;
            kernels::matmul_q8_0(&mut d.tmp, &l.wo, &d.attn, q_dim, s.n_embd, bs as u32)?;
            kernels::add_assign(&mut d.h, &d.tmp, bs as u32 * s.n_embd)?;
            // FFN
            kernels::rms_norm(&mut d.hn, &d.h, &l.ffn_norm, s.n_embd, bs as u32, eps)?;
            kernels::matmul_q8_0(&mut d.ffa, &l.gate, &d.hn, s.n_embd, d.ff, bs as u32)?;
            kernels::matmul_q8_0(&mut d.ffb, &l.up, &d.hn, s.n_embd, d.ff, bs as u32)?;
            kernels::swiglu(&mut d.ffm, &d.ffa, &d.ffb, bs as u32 * d.ff, 0.0, 1.0, 0)?;
            kernels::matmul_q8_0(&mut d.tmp, &l.down, &d.ffm, d.ff, s.n_embd, bs as u32)?;
            kernels::add_assign(&mut d.h, &d.tmp, bs as u32 * s.n_embd)?;
        }
        // final norm -> target lm head (head_logits reads st.normed)
        kernels::rms_norm(&mut st.normed, &d.h, &d.out_norm, s.n_embd, bs as u32, eps)?;
        self.head_logits(st, bs as u32)?;
        kernels::sync()?;
        let logits = st.logits.read_f32(bs * s.n_vocab as usize)?;
        let v = s.n_vocab as usize;
        let mut out: Vec<u32> = (0..bs).map(|i| argmax(&logits[i * v..(i + 1) * v])).collect();
        out[0] = last_tok;
        Ok(out)
    }
}

/// DFlash speculative generation (greedy): draft a 16-block, verify in
/// one batched target forward, accept the matching prefix, restore the
/// pre-verify recurrent state and replay the accepted tokens.
#[allow(clippy::too_many_arguments)]
pub fn generate_dflash(
    model: &Model,
    draft: &mut DraftModel,
    st: &mut State,
    prompt: &[u32],
    pos0: u32,
    max_tokens: usize,
    stop: impl Fn(u32) -> bool,
    mut on_token: impl FnMut(u32),
) -> Result<u32> {
    let s = model.shape;
    let v = s.n_vocab as usize;
    // arm the capture ring BEFORE prefill
    {
        let rt = st.qwen35.as_mut().ok_or("qwen35 state missing")?;
        rt.enable_dflash(model, draft.layer_ids.clone())?;
    }
    let logits = model
        .forward_rows(st, prompt, pos0, 1)?
        .ok_or("no prefill logits")?;
    let mut last_tok = argmax(&logits);
    let mut committed = pos0 + prompt.len() as u32;
    let mut emitted = 0usize;
    let bs = draft.block_size;

    while emitted < max_tokens {
        if committed + bs as u32 + 1 >= st.ctx {
            break;
        }
        // 1. draft
        let t0 = std::time::Instant::now();
        let draft_tok = model.dflash_draft(st, draft, committed, last_tok)?;
        st.mtp_drafted += (bs - 1) as u64;
        let t_draft = t0.elapsed();
        // 2. snapshot + batched verify (gdn inputs stashed for rollback)
        let t0 = std::time::Instant::now();
        {
            let rt = st.qwen35.as_mut().unwrap();
            rt.snapshot()?;
            rt.dflash.as_mut().unwrap().capture_gdn = true;
        }
        let all = model
            .forward_rows(st, &draft_tok, committed, bs as u32)?
            .ok_or("no verify logits")?;
        st.qwen35.as_mut().unwrap().dflash.as_mut().unwrap().capture_gdn = false;
        let t_verify = t0.elapsed();
        let target_tok: Vec<u32> =
            (0..bs).map(|i| argmax(&all[i * v..(i + 1) * v])).collect();
        if std::env::var_os("PULSAR_DFLASH_DEBUG").is_some() {
            eprintln!("dflash round @{committed}:\n  draft  {draft_tok:?}\n  target {target_tok:?}");
        }
        // 3. accept the matching prefix (row i predicts the token after
        //    draft_tok[i]; draft_tok[0] = last_tok is always accepted)
        let mut accept_n = 1usize;
        while accept_n < bs && draft_tok[accept_n] == target_tok[accept_n - 1] {
            accept_n += 1;
        }
        st.mtp_accepted += (accept_n - 1) as u64;
        // 4. restore + replay the accepted prefix (deterministic kernels:
        //    identical values land in the caches and the feature ring)
        let t0 = std::time::Instant::now();
        {
            let mut rt = st.qwen35.take().ok_or("qwen35 state missing")?;
            let r = rt.rollback_to(model, accept_n as u32);
            st.qwen35 = Some(rt);
            r?;
        }
        let t_replay = t0.elapsed();
        if std::env::var_os("PULSAR_DFLASH_DEBUG").is_some() {
            eprintln!(
                "dflash timing: draft {:.0}ms verify {:.0}ms replay({accept_n}) {:.0}ms",
                t_draft.as_secs_f64() * 1e3,
                t_verify.as_secs_f64() * 1e3,
                t_replay.as_secs_f64() * 1e3
            );
        }
        // 5. emit (stop tokens are forwarded into state but not
        //    emitted, matching engine::generate)
        let mut hit_stop = false;
        for &tokv in &draft_tok[..accept_n] {
            if stop(tokv) {
                hit_stop = true;
                break;
            }
            on_token(tokv);
            emitted += 1;
            if emitted >= max_tokens {
                hit_stop = true;
                break;
            }
        }
        committed += accept_n as u32;
        last_tok = target_tok[accept_n - 1];
        if hit_stop {
            break;
        }
    }
    Ok(committed)
}
