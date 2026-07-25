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

use super::{Attn, Ffn, LayerW, MatW, Model, Result, State};
use kernels::DeviceBuf;

/// Matmul over either weight encoding. `x` is the f32 input; `xq` must
/// hold its q8_K quantization when the weight is K-quant (callers
/// quantize once per distinct input).
fn matw(out: &mut DeviceBuf, w: &MatW, x: &DeviceBuf, xq: &DeviceBuf, in_dim: u32, out_dim: u32, t: u32) -> Result {
    match w {
        MatW::Q8(b) => kernels::matmul_q8_0(out, b, x, in_dim, out_dim, t)?,
        MatW::Kq(k) => kernels::matmul_kq(out, &k.w, xq, in_dim, out_dim, t, k.row_bytes, k.quant)?,
    }
    Ok(())
}

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

/// Per-GDN-layer device state (on the layer's owner card).
struct GdnState {
    /// delta-rule state [ssm_v_heads][ssm_state][ssm_state]
    s: DeviceBuf,
    /// conv window [ssm_conv_k - 1][conv_dim]
    conv: DeviceBuf,
    /// owner device (dense split; primary elsewhere)
    dev: i32,
    /// pre-verify snapshots for nextn-MTP rounds (lazy, on `dev`)
    snap_s: Option<DeviceBuf>,
    snap_conv: Option<DeviceBuf>,
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
    /// dense-split second-span captures (empty without a bank)
    stage2: Vec<Stage2>,
    /// bank-side replay scratch: rollback kernels mix scratch with the
    /// layer's GDN state in one launch, so split layers need scratch on
    /// their own card (None without a bank)
    bank_rb: Option<BankRb>,
}

/// Rollback replay scratch on the bank device (mirrors the Qwen35Rt
/// conv_out/gq/gk/gv/gdn_o set for layers the second card owns).
struct BankRb {
    dev: i32,
    first: usize,
    conv_out: DeviceBuf,
    gq: DeviceBuf,
    gk: DeviceBuf,
    gv: DeviceBuf,
    gdn_o: DeviceBuf,
}

/// Deferred capture for a layer owned by the second card: rows stage on
/// that card during its span and flush through a primary-side bounce
/// into the ring after the hop back.
struct Stage2 {
    /// capture layer (>= bank.first)
    il: usize,
    /// slot index within layer_ids (ring column offset)
    slot: usize,
    /// [T_MAX][n_embd] f32 on the bank device
    stage: DeviceBuf,
    /// [T_MAX][n_embd] f32 on the primary
    bounce: DeviceBuf,
}

/// Second-card copies of every activation/scratch buffer the layer
/// eval touches, for the dense split (Model.layer_dev): swapped into
/// State/Qwen35Rt at the ownership boundary so eval_qwen35_layer runs
/// unchanged, then swapped back for the tail. The residual stream
/// crosses cards exactly twice per chunk.
struct DenseBank {
    dev: i32,
    /// first layer owned by dev (layers split contiguously)
    first: usize,
    // State halves
    cur: DeviceBuf,
    normed: DeviceBuf,
    attn_out: DeviceBuf,
    after_attn: DeviceBuf,
    q: DeviceBuf,
    k: DeviceBuf,
    v: DeviceBuf,
    heads: DeviceBuf,
    xq: DeviceBuf,
    gate_act: DeviceBuf,
    up_act: DeviceBuf,
    ffn_mid: DeviceBuf,
    midq: DeviceBuf,
    ffn_out: DeviceBuf,
    // Qwen35Rt scratch halves
    qkv: DeviceBuf,
    conv_out: DeviceBuf,
    z: DeviceBuf,
    gq: DeviceBuf,
    gk: DeviceBuf,
    gv: DeviceBuf,
    small: DeviceBuf,
    g: DeviceBuf,
    beta: DeviceBuf,
    gdn_o: DeviceBuf,
    gdn_tmp: DeviceBuf,
    qfull: DeviceBuf,
    gate: DeviceBuf,
}

impl DenseBank {
    fn new(m: &Model) -> Result<Option<DenseBank>> {
        let s = m.shape;
        let primary = kernels::get_device();
        let Some(first) = m.layers.iter().enumerate().position(|(il, _)| m.layer_dev(il) != primary)
        else {
            return Ok(None);
        };
        let dev = m.layer_dev(first);
        let key_dim = (s.ssm_k_heads * s.ssm_state) as usize;
        let value_dim = (s.ssm_v_heads * s.ssm_state) as usize;
        let conv_dim = 2 * key_dim + value_dim;
        let n_ff = s.n_ff_exp as usize;
        let n_embd = s.n_embd as usize;
        kernels::set_device(dev)?;
        let f32s = |n: usize| DeviceBuf::alloc(n * 4);
        let q8k = |n: usize| {
            DeviceBuf::alloc(n / kernels::Q8_K_BLOCK_ELEMS * kernels::Q8_K_BLOCK_BYTES)
        };
        let b = DenseBank {
            dev,
            first,
            cur: f32s(T_MAX * n_embd)?,
            normed: f32s(T_MAX * n_embd)?,
            attn_out: f32s(T_MAX * n_embd)?,
            after_attn: f32s(T_MAX * n_embd)?,
            q: f32s(T_MAX * (s.n_head * s.head_dim) as usize)?,
            k: f32s(T_MAX * (s.n_head_kv * s.head_dim) as usize)?,
            v: f32s(T_MAX * (s.n_head_kv * s.head_dim) as usize)?,
            heads: f32s(T_MAX * (s.n_head * s.head_dim) as usize)?,
            xq: q8k(T_MAX * n_embd)?,
            gate_act: f32s(T_MAX * n_ff)?,
            up_act: f32s(T_MAX * n_ff)?,
            ffn_mid: f32s(T_MAX * n_ff)?,
            midq: q8k(T_MAX * n_ff)?,
            ffn_out: f32s(T_MAX * n_embd)?,
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
        };
        kernels::set_device(primary)?;
        Ok(Some(b))
    }

    /// Exchange the eval buffers with State/Qwen35Rt (call with the bank
    /// taken OUT of rt). Symmetric: calling it twice restores the wiring.
    fn swap(&mut self, st: &mut State, rt_scratch: &mut Qwen35Rt) {
        use std::mem::swap;
        swap(&mut self.cur, &mut st.cur);
        swap(&mut self.normed, &mut st.normed);
        swap(&mut self.attn_out, &mut st.attn_out);
        swap(&mut self.after_attn, &mut st.after_attn);
        swap(&mut self.q, &mut st.q);
        swap(&mut self.k, &mut st.k);
        swap(&mut self.v, &mut st.v);
        swap(&mut self.heads, &mut st.heads);
        swap(&mut self.xq, &mut st.xq);
        swap(&mut self.gate_act, &mut st.gate_act);
        swap(&mut self.up_act, &mut st.up_act);
        swap(&mut self.ffn_mid, &mut st.ffn_mid);
        swap(&mut self.midq, &mut st.midq);
        swap(&mut self.ffn_out, &mut st.ffn_out);
        swap(&mut self.qkv, &mut rt_scratch.qkv);
        swap(&mut self.conv_out, &mut rt_scratch.conv_out);
        swap(&mut self.z, &mut rt_scratch.z);
        swap(&mut self.gq, &mut rt_scratch.gq);
        swap(&mut self.gk, &mut rt_scratch.gk);
        swap(&mut self.gv, &mut rt_scratch.gv);
        swap(&mut self.small, &mut rt_scratch.small);
        swap(&mut self.g, &mut rt_scratch.g);
        swap(&mut self.beta, &mut rt_scratch.beta);
        swap(&mut self.gdn_o, &mut rt_scratch.gdn_o);
        swap(&mut self.gdn_tmp, &mut rt_scratch.gdn_tmp);
        swap(&mut self.qfull, &mut rt_scratch.qfull);
        swap(&mut self.gate, &mut rt_scratch.gate);
    }
}

/// qwen35 runtime: GDN states + scratch sized for T_MAX-token chunks.
pub(super) struct Qwen35Rt {
    states: Vec<Option<GdnState>>,
    /// dense-split second-card buffers (None single-card)
    bank: Option<DenseBank>,
    /// CUDA graphs for GDN layer runs, keyed by (rows, first layer).
    /// GDN layers have no position inputs (recurrent state, no rope/KV)
    /// and the dense path is pure kernel launches, so a run's chain is
    /// static across tokens; attention layers stay outside (their
    /// split-K grid is derived from pos on the host). PULSAR_GRAPHS=0
    /// disables.
    graphs: std::collections::HashMap<(u32, usize), kernels::Graph>,
    graphs_on: bool,
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
        let primary = kernels::get_device();
        let key_dim = (s.ssm_k_heads * s.ssm_state) as usize;
        let value_dim = (s.ssm_v_heads * s.ssm_state) as usize;
        let conv_dim = 2 * key_dim + value_dim;
        let mut states = Vec::with_capacity(s.n_exec_layer as usize);
        for il in 0..s.n_exec_layer {
            if (il + 1) % s.full_attn_interval == 0 {
                states.push(None);
            } else {
                let dev = m.layer_dev(il as usize);
                kernels::set_device(dev)?;
                let sbytes = s.ssm_v_heads as usize
                    * s.ssm_state as usize
                    * s.ssm_state as usize
                    * 4;
                let cbytes = (s.ssm_conv_k as usize - 1) * conv_dim * 4;
                let mut st = GdnState {
                    s: DeviceBuf::alloc(sbytes)?,
                    conv: DeviceBuf::alloc(cbytes)?,
                    dev,
                    snap_s: None,
                    snap_conv: None,
                };
                kernels::zero(&mut st.s, sbytes)?;
                kernels::zero(&mut st.conv, cbytes)?;
                states.push(Some(st));
            }
        }
        kernels::set_device(primary)?;
        let bank = DenseBank::new(m)?;
        let f32s = |n: usize| DeviceBuf::alloc(n * 4);
        let graphs_on = std::env::var("PULSAR_GRAPHS").ok().as_deref() != Some("0")
            && m.layers.iter().any(|l| matches!(l.ffn, super::Ffn::DenseKq { .. }));
        Ok(Qwen35Rt {
            bank,
            graphs: std::collections::HashMap::new(),
            graphs_on,
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

    /// Snapshot every GDN state (nextn-MTP verify rounds; buffers live
    /// beside their state, so the copies never cross cards).
    pub(super) fn gdn_snapshot(&mut self) -> Result {
        let primary = kernels::get_device();
        for gs in self.states.iter_mut().flatten() {
            kernels::set_device(gs.dev)?;
            if gs.snap_s.is_none() {
                gs.snap_s = Some(DeviceBuf::alloc(gs.s.bytes())?);
                gs.snap_conv = Some(DeviceBuf::alloc(gs.conv.bytes())?);
            }
            let ss = gs.snap_s.as_mut().unwrap();
            kernels::copy_d2d(ss, 0, &gs.s, 0, gs.s.bytes())?;
            let sc = gs.snap_conv.as_mut().unwrap();
            kernels::copy_d2d(sc, 0, &gs.conv, 0, gs.conv.bytes())?;
        }
        kernels::set_device(primary)?;
        Ok(())
    }

    /// Restore the last gdn_snapshot (partial MTP acceptance).
    pub(super) fn gdn_restore(&mut self) -> Result {
        let primary = kernels::get_device();
        for gs in self.states.iter_mut().flatten() {
            let (Some(ss), Some(sc)) = (&gs.snap_s, &gs.snap_conv) else {
                return Err("gdn_restore without a snapshot".into());
            };
            kernels::set_device(gs.dev)?;
            let n = gs.s.bytes();
            kernels::copy_d2d(&mut gs.s, 0, ss, 0, n)?;
            let n = gs.conv.bytes();
            kernels::copy_d2d(&mut gs.conv, 0, sc, 0, n)?;
        }
        kernels::set_device(primary)?;
        Ok(())
    }

    /// Snapshot every GDN state for a prefix checkpoint (device-local
    /// copies on each state's owner card).
    pub(super) fn ckpt(&self) -> Result<Vec<Option<(DeviceBuf, DeviceBuf)>>> {
        let primary = kernels::get_device();
        let mut out = Vec::with_capacity(self.states.len());
        for gs in &self.states {
            out.push(match gs {
                Some(g) => {
                    kernels::set_device(g.dev)?;
                    let mut s2 = DeviceBuf::alloc(g.s.bytes())?;
                    kernels::copy_d2d(&mut s2, 0, &g.s, 0, g.s.bytes())?;
                    let mut c2 = DeviceBuf::alloc(g.conv.bytes())?;
                    kernels::copy_d2d(&mut c2, 0, &g.conv, 0, g.conv.bytes())?;
                    Some((s2, c2))
                }
                None => None,
            });
        }
        kernels::set_device(primary)?;
        Ok(out)
    }

    pub(super) fn ckpt_restore(&mut self, ck: &[Option<(DeviceBuf, DeviceBuf)>]) -> Result {
        let primary = kernels::get_device();
        for (gs, c) in self.states.iter_mut().zip(ck) {
            if let (Some(g), Some((s2, c2))) = (gs, c) {
                kernels::set_device(g.dev)?;
                kernels::copy_d2d(&mut g.s, 0, s2, 0, s2.bytes())?;
                kernels::copy_d2d(&mut g.conv, 0, c2, 0, c2.bytes())?;
            }
        }
        kernels::set_device(primary)?;
        Ok(())
    }

    fn reset(&mut self) -> Result {
        let primary = kernels::get_device();
        for st in self.states.iter_mut().flatten() {
            kernels::set_device(st.dev)?;
            let (sb, cb) = (st.s.bytes(), st.conv.bytes());
            kernels::zero(&mut st.s, sb)?;
            kernels::zero(&mut st.conv, cb)?;
        }
        kernels::set_device(primary)?;
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
        // dense split: captures on second-card layers cannot scatter into
        // the primary-side ring mid-span. Stage them on their own card and
        // flush through a primary bounce after the hop back (same
        // producer-current ordering trick as the residual hop).
        let mut stage2 = Vec::new();
        let mut bank_rb = None;
        if let Some(b) = &self.bank {
            let prev = kernels::get_device();
            for (slot, &il) in layer_ids.iter().enumerate() {
                if il >= b.first {
                    kernels::set_device(b.dev)?;
                    let stage = DeviceBuf::alloc(T_MAX * s.n_embd as usize * 4)?;
                    kernels::set_device(prev)?;
                    let bounce = DeviceBuf::alloc(T_MAX * s.n_embd as usize * 4)?;
                    stage2.push(Stage2 { il, slot, stage, bounce });
                }
            }
            kernels::set_device(b.dev)?;
            let f32s = |n: usize| DeviceBuf::alloc(n * 4);
            bank_rb = Some(BankRb {
                dev: b.dev,
                first: b.first,
                conv_out: f32s(T_MAX * conv_dim)?,
                gq: f32s(T_MAX * key_dim)?,
                gk: f32s(T_MAX * key_dim)?,
                gv: f32s(T_MAX * value_dim)?,
                gdn_o: f32s(T_MAX * value_dim)?,
            });
            kernels::set_device(prev)?;
        }
        let mut snap_s = Vec::new();
        let mut snap_conv = Vec::new();
        let mut stash_qkv = Vec::new();
        let mut stash_g = Vec::new();
        let mut stash_beta = Vec::new();
        let prev_dev = kernels::get_device();
        for (il, gs) in self.states.iter().enumerate() {
            match gs {
                Some(g) => {
                    // snapshots/stashes copy_d2d against the layer's GDN
                    // state, so they must live on the layer's OWNER card
                    // (plain cudaMemcpy cannot cross devices without P2P)
                    let bank_dev = self.bank.as_ref().filter(|b| il >= b.first).map(|b| b.dev);
                    if let Some(d) = bank_dev {
                        kernels::set_device(d)?;
                    }
                    snap_s.push(Some(DeviceBuf::alloc(g.s.bytes())?));
                    snap_conv.push(Some(DeviceBuf::alloc(g.conv.bytes())?));
                    stash_qkv.push(Some(DeviceBuf::alloc(T_MAX * conv_dim * 4)?));
                    stash_g.push(Some(DeviceBuf::alloc(T_MAX * s.ssm_v_heads as usize * 4)?));
                    stash_beta.push(Some(DeviceBuf::alloc(T_MAX * s.ssm_v_heads as usize * 4)?));
                    if bank_dev.is_some() {
                        kernels::set_device(prev_dev)?;
                    }
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
            stage2,
            bank_rb,
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
        let mut df = self.dflash.take().ok_or("dflash not enabled")?;
        let primary = kernels::primary_device();
        let r = (|| -> Result {
            let DflashRt {
                snap_s, snap_conv, stash_qkv, stash_g, stash_beta, bank_rb, ..
            } = &mut df;
            for (il, gs) in self.states.iter_mut().enumerate() {
                let Some(g) = gs else { continue };
                // split layers replay entirely on their own card: the
                // kernels mix scratch with the GDN state in one launch,
                // and cross-device args fault asynchronously
                let on_bank = bank_rb.as_ref().is_some_and(|rb| il >= rb.first);
                if on_bank {
                    kernels::set_device(bank_rb.as_ref().unwrap().dev)?;
                }
                let ss = snap_s[il].as_ref().unwrap();
                let sc = snap_conv[il].as_ref().unwrap();
                kernels::copy_d2d(&mut g.s, 0, ss, 0, ss.bytes())?;
                kernels::copy_d2d(&mut g.conv, 0, sc, 0, sc.bytes())?;
                if accept_n == 0 {
                    if on_bank {
                        kernels::set_device(primary)?;
                    }
                    continue;
                }
                let sq = stash_qkv[il].as_ref().unwrap();
                let Attn::Qwen35(w) = &m.layers[il].attn else {
                    return Err("qwen35 layer expected".into());
                };
                let gdn = w.gdn.as_ref().ok_or("gdn weights missing")?;
                let (conv_out, gq, gk, gv, gdn_o) = if on_bank {
                    let rb = bank_rb.as_mut().unwrap();
                    (&mut rb.conv_out, &mut rb.gq, &mut rb.gk, &mut rb.gv, &mut rb.gdn_o)
                } else {
                    (&mut self.conv_out, &mut self.gq, &mut self.gk, &mut self.gv, &mut self.gdn_o)
                };
                kernels::qwen35_conv_batch(conv_out, sq, &gdn.conv, &mut g.conv, conv_dim, s.ssm_conv_k, accept_n)?;
                kernels::qwen35_split_qkv(gq, gk, gv, conv_out, accept_n, key_dim, value_dim)?;
                kernels::qwen35_l2_norm(gq, accept_n * s.ssm_k_heads, s.ssm_state, s.rms_eps)?;
                kernels::qwen35_l2_norm(gk, accept_n * s.ssm_k_heads, s.ssm_state, s.rms_eps)?;
                kernels::qwen35_gdn_batch(
                    gdn_o, &mut g.s, gq, gk, gv,
                    stash_g[il].as_ref().unwrap(),
                    stash_beta[il].as_ref().unwrap(),
                    s.ssm_v_heads, s.ssm_k_heads, s.ssm_state, accept_n,
                )?;
                if on_bank {
                    kernels::set_device(primary)?;
                }
            }
            Ok(())
        })();
        kernels::set_device(primary)?;
        self.dflash = Some(df);
        r
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

/// DSpark heads riding on the DFlash trunk (DeepSpec: DSpark = DFlash +
/// markov head + confidence head). The markov head biases each draft
/// step's logits with w2 @ w1[prev_token]; the confidence head predicts
/// per-slot acceptance so low-confidence draft tails are cut before the
/// (expensive) batched verify.
struct DsparkHeads {
    /// q8_0 [n_vocab x rank] prev-token embedding (row-gathered like an
    /// embedding table)
    w1: DeviceBuf,
    /// q8_0 [n_vocab x rank] rank -> vocab bias projection
    w2: DeviceBuf,
    rank: u32,
    /// f32 [rank] dequantized w1[prev] for the current step
    state: DeviceBuf,
    /// per-block argmax winners (128 * 8B) for the fused kernel
    scratch: DeviceBuf,
    /// 4B argmax result
    out_id: DeviceBuf,
    /// host f32 [n_embd + rank] confidence projection (None = no head)
    conf_w: Option<Vec<f32>>,
    conf_bias: f32,
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
    /// DSpark markov + confidence heads (None on plain DFlash drafts)
    dspark: Option<DsparkHeads>,
    /// DeepSpec-trained drafts emit NEXT-token rows (row j predicts the
    /// token after slot j); z-lab drafts fill the mask at row j. Detected
    /// from the dspark metadata the converter writes.
    next_rows: bool,
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
        let next_rows = g.arch_meta("dspark.confidence_head").is_some()
            || std::env::var_os("PULSAR_DFLASH_DEEPSPEC").is_some();
        // DeepSpec extract_context_feature reads hidden_states[l + 1]
        // (the residual ENTERING layer l+1); our ring captures at layer
        // entry, so DeepSpec-trained drafts shift the capture points up
        let layer_ids: Vec<usize> = if next_rows {
            layer_ids.iter().map(|&l| l + 1).collect()
        } else {
            layer_ids
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
        // DSpark heads (markov bias + confidence prefix cut); plain
        // DFlash drafts carry no dspark metadata and skip all of this.
        let dspark = match g.arch_meta("dspark.markov_rank").and_then(gguf::Value::as_u64) {
            Some(rank) if rank > 0 && std::env::var_os("PULSAR_NO_DSPARK").is_none() => {
                let rank = rank as u32;
                let has_conf = matches!(
                    g.arch_meta("dspark.confidence_head"),
                    Some(gguf::Value::Bool(true))
                );
                let conf_w = if has_conf {
                    Some(super::read_f16_as_f32(&file, &g, "confidence_proj.weight")?)
                } else {
                    None
                };
                let conf_bias = g
                    .arch_meta("dspark.confidence_bias")
                    .and_then(gguf::Value::as_f32)
                    .unwrap_or(0.0);
                eprintln!(
                    "pulsar: dspark heads active (markov rank {rank}{})",
                    if conf_w.is_some() { " + confidence" } else { "" }
                );
                Some(DsparkHeads {
                    w1: up("markov_w1.weight")?,
                    w2: up("markov_w2.weight")?,
                    rank,
                    state: f32s(rank as usize)?,
                    scratch: DeviceBuf::alloc(128 * 8)?,
                    out_id: DeviceBuf::alloc(4)?,
                    conf_w,
                    conf_bias,
                })
            }
            _ => None,
        };
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
            dspark,
            next_rows,
        })
    }
}

/// Confidence cut threshold in logit space. PULSAR_DSPARK_CONF is the
/// sigmoid probability (default 0.5: cut only slots the head thinks are
/// more likely rejected than accepted); "off" disables the cut.
fn dspark_conf_threshold() -> f32 {
    match std::env::var("PULSAR_DSPARK_CONF").ok().as_deref() {
        Some("off") => f32::NEG_INFINITY,
        Some(v) => v
            .parse::<f32>()
            .ok()
            .filter(|p| *p > 0.0 && *p < 1.0)
            .map(|p| (p / (1.0 - p)).ln())
            .unwrap_or(0.0),
        None => 0.0,
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
        let primary = kernels::get_device();
        let mut bank = rt.bank.take();
        let n0 = bank.as_ref().map_or(self.layers.len(), |b| b.first);
        let mut pos = pos0;
        let mut last_t = 0u32;
        let mut run = |st: &mut State, rt: &mut Qwen35Rt, chunk: &[u32], pos: u32| -> Result {
            let t = chunk.len() as u32;
            let ids: Vec<i32> = chunk.iter().map(|&x| x as i32).collect();
            st.tok.write(0, kernels::as_bytes(&ids))?;
            kernels::embed_q8_0(&mut st.cur, &self.token_embd, &st.tok, s.n_embd, s.n_vocab, t)?;
            self.eval_qwen35_span(st, rt, 0, n0, pos, t)?;
            if let Some(b) = &mut bank {
                // hop 1: residual over to the second card (issued with
                // the producer current, so the consumer's launches order
                // after it), then run its layers on its own buffers
                let bytes = (t * s.n_embd) as usize * 4;
                kernels::copy_across(&mut b.cur, &st.cur, bytes)?;
                b.swap(st, rt);
                kernels::set_device(b.dev)?;
                self.eval_qwen35_span(st, rt, n0, self.layers.len(), pos, t)?;
                // hop 2 back: after the swap b.cur is the card-1 buffer
                // holding the final residual
                b.swap(st, rt);
                kernels::copy_across(&mut st.cur, &b.cur, bytes)?;
                // dflash second-span captures: bounce to the primary while
                // the producer card is still current (same ordering trick
                // as the residual hop), scatter once back on the primary
                if let Some(df) = &mut rt.dflash {
                    for e in &mut df.stage2 {
                        kernels::copy_across(&mut e.bounce, &e.stage, bytes)?;
                    }
                }
                kernels::set_device(primary)?;
                if let Some(df) = &mut rt.dflash {
                    let DflashRt { ring, stage2, layer_ids, .. } = df;
                    let stride = (layer_ids.len() as u32) * s.n_embd;
                    for e in stage2.iter() {
                        kernels::qwen35_ring_scatter(
                            ring,
                            &e.bounce,
                            pos,
                            RING_CAP as u32,
                            t,
                            s.n_embd,
                            stride,
                            e.slot as u32 * s.n_embd,
                        )?;
                    }
                }
            }
            Ok(())
        };
        for chunk in tokens.chunks(T_MAX) {
            run(st, rt, chunk, pos)?;
            pos += chunk.len() as u32;
            last_t = chunk.len() as u32;
        }
        rt.bank = bank;
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

    /// Eval layers [lo, hi) on the current device. Runs of GDN+DenseKq
    /// layers replay as CUDA graphs: no position inputs, no host work -
    /// one launch instead of ~22 per layer. Attention layers (and any
    /// debug/prof mode) launch plain.
    fn eval_qwen35_span(&self, st: &mut State, rt: &mut Qwen35Rt, lo: usize, hi: usize, pos: u32, t: u32) -> Result {
        let dbg = std::env::var_os("PULSAR_DEBUG_L2").is_some()
            || std::env::var_os("PULSAR_DENSE_PROF").is_some();
        // no graphs while dflash is active: capture layers scatter with
        // the runtime position (a replayed graph would bake it stale),
        // and the capture_gdn stash is a runtime branch with cudaMemcpy -
        // illegal under stream capture, silently skipped on replay
        let dflash_on = rt.dflash.is_some();
        let graphable = |il: usize| {
            let l = &self.layers[il];
            !dflash_on
                && matches!(&l.attn, Attn::Qwen35(w) if w.gdn.is_some())
                && matches!(l.ffn, Ffn::DenseKq { .. })
        };
        let mut graphs = std::mem::take(&mut rt.graphs);
        let mut il = lo;
        let r = (|| -> Result {
            while il < hi {
                if rt.graphs_on && !dbg && graphable(il) {
                    let mut end = il + 1;
                    while end < hi && graphable(end) {
                        end += 1;
                    }
                    let key = (t, il);
                    if !graphs.contains_key(&key) {
                        let (lo2, hi2) = (il, end);
                        // capture takes a kernels::Result closure; stash
                        // the real engine error across the boundary
                        let mut inner: Option<Box<dyn std::error::Error>> = None;
                        let g = kernels::Graph::capture(|| {
                            for j in lo2..hi2 {
                                if let Err(e) = self.eval_qwen35_layer(st, rt, j, &self.layers[j], pos, t) {
                                    inner = Some(e);
                                    return Err(kernels::Error("graph capture body failed"));
                                }
                            }
                            Ok(())
                        });
                        if let Some(e) = inner {
                            return Err(e);
                        }
                        graphs.insert(key, g?);
                    }
                    graphs[&key].launch()?;
                    il = end;
                } else {
                    self.eval_qwen35_layer(st, rt, il, &self.layers[il], pos, t)?;
                    if std::env::var_os("PULSAR_DEBUG_L2").is_some() {
                        let a = st.after_attn.read_f32(4)?;
                        let v = st.cur.read_f32(4)?;
                        eprintln!("L{il}: attn {:?} cur {:?}", &a[..2], &v[..2]);
                    }
                    il += 1;
                }
            }
            Ok(())
        })();
        rt.graphs = graphs;
        r
    }

    pub(super) fn eval_qwen35_layer(&self, st: &mut State, rt: &mut Qwen35Rt, il: usize, l: &LayerW, pos: u32, t: u32) -> Result {
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
            if let Some(e) = df.stage2.iter_mut().find(|e| e.il == il) {
                // second-span layer: the ring lives on the primary, so
                // stage locally; the run loop flushes after the hop back
                kernels::copy_d2d(&mut e.stage, 0, &st.cur, 0, (t * s.n_embd) as usize * 4)
                    .map_err(|e| format!("dflash stage2 capture at layer {il}: {e}"))?;
            } else if let Some(idx) = df.layer_ids.iter().position(|&x| x == il) {
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
        let dbg = il == 0 && std::env::var_os("PULSAR_DEBUG_L2").is_some();
        if dbg {
            eprintln!("  embd {:?} normed {:?}", &st.cur.read_f32(2)?, &st.normed.read_f32(2)?);
        }

        if let Some(gdn) = &w.gdn {
            // ---- Gated DeltaNet (recurrences loop inside the launches)
            if matches!(gdn.wqkv, MatW::Kq(_)) {
                kernels::quantize_q8_k(&mut st.xq, &st.normed, s.n_embd, t)?;
            }
            matw(&mut rt.qkv, &gdn.wqkv, &st.normed, &st.xq, s.n_embd, conv_dim, t)?;
            matw(&mut rt.z, &gdn.wz, &st.normed, &st.xq, s.n_embd, value_dim, t)?;
            if dbg {
                eprintln!("  qkv {:?} z {:?}", &rt.qkv.read_f32(2)?, &rt.z.read_f32(2)?);
            }
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
            if dbg {
                eprintln!(
                    "  conv {:?} gq {:?} g {:?} beta {:?} gdn_o {:?}",
                    &rt.conv_out.read_f32(2)?, &rt.gq.read_f32(2)?,
                    &rt.g.read_f32(2)?, &rt.beta.read_f32(2)?, &rt.gdn_o.read_f32(2)?
                );
            }
            kernels::gqa_head_rms_norm(&mut rt.gdn_o, Some(&gdn.ssm_norm), t * s.ssm_v_heads, s.ssm_state, eps)?;
            kernels::swiglu(&mut rt.gdn_tmp, &rt.z, &rt.gdn_o, t * value_dim, 0.0, 1.0, 0)?;
            if matches!(gdn.ssm_out, MatW::Kq(_)) {
                kernels::quantize_q8_k(&mut st.midq, &rt.gdn_tmp, value_dim, t)?;
            }
            matw(&mut st.attn_out, &gdn.ssm_out, &rt.gdn_tmp, &st.midq, value_dim, s.n_embd, t)?;
        } else if let Some(attn) = &w.attn {
            // ---- sigmoid-gated full attention (partial neox rope)
            let hd = s.head_dim;
            if matches!(attn.wq, MatW::Kq(_)) {
                kernels::quantize_q8_k(&mut st.xq, &st.normed, s.n_embd, t)?;
            }
            matw(&mut rt.qfull, &attn.wq, &st.normed, &st.xq, s.n_embd, 2 * s.n_head * hd, t)?;
            // per-token rows are contiguous: treat (token, head) as one
            // flat head axis for the strided split
            kernels::qwen35_split_gate(&mut st.q, &mut rt.gate, &rt.qfull, t * s.n_head, hd)?;
            matw(&mut st.k, &attn.wk, &st.normed, &st.xq, s.n_embd, s.n_head_kv * hd, t)?;
            matw(&mut st.v, &attn.wv, &st.normed, &st.xq, s.n_embd, s.n_head_kv * hd, t)?;
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
                None, // qwen35 has no attention sinks
            )?;
            kernels::qwen35_sigmoid_gate(&mut st.heads, &rt.gate, t * s.n_head * hd)?;
            if matches!(attn.out, MatW::Kq(_)) {
                kernels::quantize_q8_k(&mut st.midq, &st.heads, s.n_head * hd, t)?;
            }
            matw(&mut st.attn_out, &attn.out, &st.heads, &st.midq, s.n_head * hd, s.n_embd, t)?;
        } else {
            return Err("qwen35 layer with neither attn nor gdn".into());
        }
        kernels::add(&mut st.after_attn, &st.cur, &st.attn_out, t * s.n_embd)?;

        // ---- FFN (pre-norm residual)
        // PULSAR_DENSE_PROF=1: sync-bracketed phase totals (attn+GDN into
        // the gpu-wait bucket, dense ffn into resolve). Syncs distort the
        // absolute rate - read the SPLIT, not the total.
        let prof = std::env::var_os("PULSAR_DENSE_PROF").is_some();
        let mut mark = std::time::Instant::now();
        if prof {
            kernels::sync()?;
            st.prof.sync += mark.elapsed();
            mark = std::time::Instant::now();
        }
        kernels::rms_norm(&mut st.normed, &st.after_attn, &l.ffn_norm, s.n_embd, t, eps)?;
        if let Ffn::DenseKq { gate, up, down } = &l.ffn {
            // dense 27B: resident K-quant triple, no experts, no syncs
            kernels::quantize_q8_k(&mut st.xq, &st.normed, s.n_embd, t)?;
            kernels::matmul_kq(&mut st.gate_act, &gate.w, &st.xq, s.n_embd, s.n_ff_exp, t, gate.row_bytes, gate.quant)?;
            kernels::matmul_kq(&mut st.up_act, &up.w, &st.xq, s.n_embd, s.n_ff_exp, t, up.row_bytes, up.quant)?;
            kernels::swiglu(&mut st.ffn_mid, &st.gate_act, &st.up_act, t * s.n_ff_exp, 0.0, 1.0, 0)?;
            kernels::quantize_q8_k(&mut st.midq, &st.ffn_mid, s.n_ff_exp, t)?;
            kernels::matmul_kq(&mut st.ffn_out, &down.w, &st.midq, s.n_ff_exp, s.n_embd, t, down.row_bytes, down.quant)?;
            kernels::add(&mut st.cur, &st.after_attn, &st.ffn_out, t * s.n_embd)?;
            if prof {
                kernels::sync()?;
                st.prof.resolve += mark.elapsed();
            }
            return Ok(());
        }
        let Ffn::Moe { gate_inp, probs_b, shexp, gate_exps, up_exps, down_exps, .. } = &l.ffn else {
            return Err("qwen35 layer without MoE ffn".into());
        };
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
        let selected = st.router_selected.read_i32((t * s.n_expert_used) as usize)?;
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
        let v = s.n_vocab as usize;
        let mut out: Vec<u32>;
        // DeepSpec next-token rows: draft slot j comes from logits row
        // j-1 (row 0 is keyed on the anchor token); z-lab mask-fill rows
        // read slot j from row j.
        let row_of = |j: usize| if d.next_rows { j - 1 } else { j };
        if let Some(dk) = &mut d.dspark {
            // markov-biased greedy: each slot's argmax includes the
            // w2 @ w1[prev] bigram bias, sequenced on the previous pick
            // (fused kernel; the vocab-sized bias never materializes)
            out = vec![last_tok; bs];
            let mut states_host: Vec<Vec<f32>> = Vec::new();
            for i in 1..bs {
                let prev = [out[i - 1] as i32];
                st.tok.write(0, kernels::as_bytes(&prev))?;
                kernels::embed_q8_0(&mut dk.state, &dk.w1, &st.tok, dk.rank, s.n_vocab, 1)?;
                out[i] = kernels::dspark_markov_argmax(
                    &st.logits,
                    row_of(i) * v,
                    &dk.w2,
                    &dk.state,
                    s.n_vocab,
                    dk.rank,
                    &mut dk.scratch,
                    &mut dk.out_id,
                )?;
                if dk.conf_w.is_some() {
                    states_host.push(dk.state.read_f32(dk.rank as usize)?);
                }
            }
            // confidence prefix cut: slot i survives while the predicted
            // acceptance logit stays above the threshold; the verify then
            // runs only the surviving rows
            if let Some(w) = &dk.conf_w {
                let thr = dspark_conf_threshold();
                let h_host = d.h.read_f32(bs * s.n_embd as usize)?;
                let ne = s.n_embd as usize;
                let mut keep = 1usize;
                for i in 1..bs {
                    let hrow = &h_host[row_of(i) * ne..(row_of(i) + 1) * ne];
                    let stt = &states_host[i - 1];
                    let mut acc = dk.conf_bias;
                    for (a, b) in w[..ne].iter().zip(hrow) {
                        acc += a * b;
                    }
                    for (a, b) in w[ne..].iter().zip(stt) {
                        acc += a * b;
                    }
                    if acc < thr {
                        break;
                    }
                    keep += 1;
                }
                out.truncate(keep);
            }
        } else {
            kernels::sync()?;
            let logits = st.logits.read_f32(bs * s.n_vocab as usize)?;
            out = (0..bs)
                .map(|i| {
                    let r = if i == 0 { 0 } else { row_of(i) };
                    argmax(&logits[r * v..(r + 1) * v])
                })
                .collect();
            out[0] = last_tok;
        }
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
        // 1. draft (DSpark confidence may cut the block short: nrows <= bs)
        let t0 = std::time::Instant::now();
        let draft_tok = model.dflash_draft(st, draft, committed, last_tok)?;
        let nrows = draft_tok.len();
        st.mtp_drafted += (nrows - 1) as u64;
        let t_draft = t0.elapsed();
        // 2. snapshot + batched verify (gdn inputs stashed for rollback)
        let t0 = std::time::Instant::now();
        {
            let rt = st.qwen35.as_mut().unwrap();
            rt.snapshot()?;
            rt.dflash.as_mut().unwrap().capture_gdn = true;
        }
        let all = model
            .forward_rows(st, &draft_tok, committed, nrows as u32)?
            .ok_or("no verify logits")?;
        st.qwen35.as_mut().unwrap().dflash.as_mut().unwrap().capture_gdn = false;
        let t_verify = t0.elapsed();
        let target_tok: Vec<u32> =
            (0..nrows).map(|i| argmax(&all[i * v..(i + 1) * v])).collect();
        if std::env::var_os("PULSAR_DFLASH_DEBUG").is_some() {
            eprintln!("dflash round @{committed}:\n  draft  {draft_tok:?}\n  target {target_tok:?}");
        }
        // 3. accept the matching prefix (row i predicts the token after
        //    draft_tok[i]; draft_tok[0] = last_tok is always accepted)
        let mut accept_n = 1usize;
        while accept_n < nrows && draft_tok[accept_n] == target_tok[accept_n - 1] {
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
