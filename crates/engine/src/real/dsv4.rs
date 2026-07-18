//! DeepSeek-V4-Flash (deepseek4) forward path, task #22.
//!
//! Reference: antirez ds4.c @ 80ebbc3 (see docs/deepseek4-port-notes.md).
//! Split of labor: big matvecs and attention run on the GPU through the
//! existing kernel set plus the dsv4 kernels; the control-heavy small
//! math (Sinkhorn hyper-connection gates, sqrt-softplus router, the
//! streaming KV compressor, indexer QAT + top-k selection) runs on the
//! host in exact reference f32 form. Decode-only graph: prefill loops
//! tokens because the SWA ring and compressor are sequential state
//! machines. ponytail: no tiers/prefetch/batched prefill yet - fold
//! dsv4 into the shared MoE resolve when the perf pass starts.

use super::{Attn, Ffn, LayerW, Model, Result, Shape, State, SLAB_SLACK};
use kernels::DeviceBuf;

const NEG_INF: f32 = -1.0e30; // DS4_NEG_INF (finite on purpose)
/// Batched prefill chunk width (matches the qwen35 blueprint; the
/// per-token interleave keeps the SWA ring correct within a chunk).
const T_MAX: usize = 16;
const ROUTER_FLOOR: f32 = 6.103515625e-5;

/* ---- host float helpers (reference math) ------------------------------- */

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn softplus_stable(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        x.exp().ln_1p()
    }
}

/// f32 -> f16 bits, round-to-nearest-even (matches __float2half; the
/// requant module's converter truncates, which is fine for scales but
/// not for cache-value parity).
#[allow(dead_code)] // host reference, exercised by the unit tests
fn f32_to_f16_rte(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let man = bits & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | (((man != 0) as u16) << 9);
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m24 = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let halfway = 1u32 << (shift - 1);
        let rem = m24 & ((1u32 << shift) - 1);
        let mut m = m24 >> shift;
        if rem > halfway || (rem == halfway && (m & 1) == 1) {
            m += 1;
        }
        return sign | m as u16;
    }
    let rem = man & 0x1fff;
    let mut m = man >> 13;
    if rem > 0x1000 || (rem == 0x1000 && (m & 1) == 1) {
        m += 1;
    }
    let mut e = e as u32;
    if m == 0x400 {
        m = 0;
        e += 1;
        if e >= 0x1f {
            return sign | 0x7c00;
        }
    }
    sign | ((e << 10) as u16) | m as u16
}

#[allow(dead_code)] // host reference, exercised by the unit tests
fn f16_round(x: f32) -> f32 {
    super::requant::f16_to_f32(f32_to_f16_rte(x))
}

/// e4m3fn decode (mirrors e4m3_dec_common in the CUDA side).
#[allow(dead_code)] // host reference, exercised by the unit tests
fn e4m3_dec(b: u8) -> f32 {
    let e = (b >> 3) & 0xf;
    let m = (b & 7) as f32;
    let v = if e == 0 {
        m * 0.001953125 // 2^-9 subnormal step
    } else {
        f32::from_bits((((e as u32) + 120) << 23) | ((b as u32 & 7) << 20))
    };
    if b & 0x80 != 0 {
        -v
    } else {
        v
    }
}

/// Round-half-to-even for non-negative x (avoids the round_ties_even
/// MSRV dependency; matches __float2int_rn on this domain).
#[allow(dead_code)] // host reference, exercised by the unit tests
fn round_half_even(x: f32) -> i32 {
    let f = x.floor();
    let diff = x - f;
    if diff > 0.5 {
        f as i32 + 1
    } else if diff < 0.5 {
        f as i32
    } else {
        let i = f as i32;
        if i % 2 == 0 {
            i
        } else {
            i + 1
        }
    }
}

/// e4m3fn encode, nearest-even (mirrors e4m3_enc in the CUDA side).
#[allow(dead_code)] // host reference, exercised by the unit tests
fn e4m3_enc(x: f32) -> u8 {
    let (s, x) = if x < 0.0 { (0x80u8, -x) } else { (0u8, x) };
    if !(x > 0.0) {
        return s; // zero and NaN
    }
    if x >= 448.0 {
        return s | 0x7e;
    }
    if x < 0.0009765625 {
        let m = round_half_even(x * 512.0);
        return if m >= 8 { s | 0x08 } else { s | m as u8 };
    }
    let e = ((x.to_bits() >> 23) as i32) - 127;
    let mut q = round_half_even(x * (2.0f32).powi(3 - e));
    let mut e = e;
    if q == 16 {
        q = 8;
        e += 1;
    }
    if e < -6 {
        let m = round_half_even(x * 512.0);
        return if m >= 8 { s | 0x08 } else { s | m as u8 };
    }
    if e > 8 {
        return s | 0x7e;
    }
    s | ((((e + 7) << 3) | (q - 8)) as u8)
}

/// ds4's fp8 sim: e4m3 round-trip on the first head_dim-n_rot dims,
/// 64-wide blocks, power-of-2 scale, clamp +-448 (host lane).
#[allow(dead_code)] // host reference, exercised by the unit tests
fn fp8_sim_row(x: &mut [f32], n_rot: usize) {
    let n_nope = x.len() - n_rot;
    for blk in x[..n_nope].chunks_mut(64) {
        let mut amax = blk.iter().fold(0f32, |a, &v| a.max(v.abs()));
        if amax < 1.0e-4 {
            amax = 1.0e-4;
        }
        let scale = (amax / 448.0).log2().ceil().exp2();
        for v in blk {
            let c = (*v / scale).clamp(-448.0, 448.0);
            *v = e4m3_dec(e4m3_enc(c)) * scale;
        }
    }
}

/// 128-wide Hadamard transform, orthonormal (scale 1/sqrt(128)).
fn hadamard128(x: &mut [f32; 128]) {
    let mut stride = 1;
    while stride < 128 {
        let mut base = 0;
        while base < 128 {
            for i in 0..stride {
                let a = x[base + i];
                let b = x[base + stride + i];
                x[base + i] = a + b;
                x[base + stride + i] = a - b;
            }
            base += 2 * stride;
        }
        stride <<= 1;
    }
    for v in x.iter_mut() {
        *v *= 0.088_388_35;
    }
}

const E2M1_VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// e2m1 round-trip: nearest value, ties toward the even table index.
fn e2m1_roundtrip(x: f32) -> f32 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let ax = x.abs().min(6.0);
    let mut best = 0usize;
    let mut best_diff = (ax - E2M1_VALUES[0]).abs();
    for (i, &v) in E2M1_VALUES.iter().enumerate().skip(1) {
        let diff = (ax - v).abs();
        if diff < best_diff || (diff == best_diff && i % 2 == 0 && best % 2 == 1) {
            best = i;
            best_diff = diff;
        }
    }
    sign * E2M1_VALUES[best]
}

/// fp4 activation sim: 32-wide blocks, power-of-2 scale, clamp +-6.
fn fp4_sim_row(x: &mut [f32]) {
    debug_assert_eq!(x.len() % 32, 0);
    for blk in x.chunks_mut(32) {
        let mut amax = blk.iter().fold(0f32, |a, &v| a.max(v.abs()));
        if amax < 7.052_966e-38 {
            amax = 7.052_966e-38;
        }
        let scale = (amax / 6.0).log2().ceil().exp2();
        for v in blk {
            let c = (*v / scale).clamp(-6.0, 6.0);
            *v = e2m1_roundtrip(c) * scale;
        }
    }
}

/// Indexer QAT: hadamard128 + fp4 round-trip per 128-wide row (applies
/// to indexer Q and indexer compressor rows - selection parity needs it).
fn qat_row(x: &mut [f32]) {
    debug_assert_eq!(x.len(), 128);
    let mut buf = [0f32; 128];
    buf.copy_from_slice(x);
    hadamard128(&mut buf);
    fp4_sim_row(&mut buf);
    x.copy_from_slice(&buf);
}

/* ---- host rope (ds4's rope_tail_ext_inplace) --------------------------- */

fn rope_ramp(low: f32, high: f32, i0: u32) -> f32 {
    let y = ((i0 / 2) as f32 - low) / (high - low).max(0.001);
    1.0 - y.clamp(0.0, 1.0)
}

fn rope_corr_dims(n_dims: u32, n_ctx_orig: u32, base: f32, beta_fast: f32, beta_slow: f32) -> (f32, f32) {
    let dim = |beta: f32| {
        n_dims as f32 * (n_ctx_orig as f32 / (beta * 2.0 * std::f32::consts::PI)).ln()
            / (2.0 * base.ln())
    };
    (dim(beta_fast).floor().max(0.0), dim(beta_slow).ceil().min(n_dims as f32 - 1.0))
}

/// Rotate the last n_rot dims of each head (compressed-layer YaRN when
/// r.ext_factor != 0). Matches the device rope tail.
fn rope_tail_host(x: &mut [f32], n_head: usize, head_dim: usize, n_rot: usize, pos: u32, r: &kernels::RopeCfg) {
    let n_nope = head_dim - n_rot;
    let theta_scale = r.freq_base.powf(-2.0 / n_rot as f32);
    let (c0, c1) = if r.ext_factor != 0.0 {
        rope_corr_dims(n_rot as u32, r.n_ctx_orig, r.freq_base, r.beta_fast, r.beta_slow)
    } else {
        (0.0, 0.0)
    };
    for h in 0..n_head {
        let tail = &mut x[h * head_dim + n_nope..h * head_dim + head_dim];
        let mut theta_extrap = pos as f32;
        let mut i = 0;
        while i < n_rot {
            let theta_interp = r.freq_scale * theta_extrap;
            let mut theta = theta_interp;
            let mut mscale = r.attn_factor;
            if r.ext_factor != 0.0 {
                let ramp = rope_ramp(c0, c1, i as u32) * r.ext_factor;
                theta = theta_interp * (1.0 - ramp) + theta_extrap * ramp;
                mscale *= 1.0 + 0.1 * (1.0 / r.freq_scale).ln();
            }
            let c = theta.cos() * mscale;
            let s = theta.sin() * mscale;
            let x0 = tail[i];
            let x1 = tail[i + 1];
            tail[i] = x0 * c - x1 * s;
            tail[i + 1] = x0 * s + x1 * c;
            theta_extrap *= theta_scale;
            i += 2;
        }
    }
}

/// Per-layer rope config: dense (ratio 0) layers run plain base-10000
/// rope; compressed layers run YaRN on the long-context base with the
/// magnitude factor cancelled (V4 interpolates without mscale).
pub(super) fn rope_cfg(s: &Shape, ratio: u32) -> kernels::RopeCfg {
    if ratio == 0 {
        kernels::RopeCfg {
            n_ctx_orig: 0,
            freq_base: s.rope_freq_base,
            freq_scale: 1.0,
            ext_factor: 0.0,
            attn_factor: 1.0,
            beta_fast: 0.0,
            beta_slow: 0.0,
            kq_mult: 1.0,
        }
    } else {
        let f = s.rope_scale_factor;
        kernels::RopeCfg {
            n_ctx_orig: s.rope_orig_ctx,
            freq_base: s.compress_rope_base,
            freq_scale: 1.0 / f,
            ext_factor: 1.0,
            attn_factor: 1.0 / (1.0 + 0.1 * f.ln()),
            beta_fast: 32.0,
            beta_slow: 1.0,
            kq_mult: 1.0,
        }
    }
}

/* ---- Sinkhorn hyper-connection gates (hc_split_sinkhorn_one) ----------- */

/// mix[6*n_hc] + scale[3] + base[6*n_hc] -> (pre[n_hc], post[n_hc],
/// comb_k[n_hc*n_hc]). comb_k is already in the kernel's [dst*n_hc+src]
/// layout: the reference's post step reads its row-softmax matrix
/// TRANSPOSED (comb[dst + src*n_hc]), so the transpose happens here.
#[allow(dead_code)] // host reference, exercised by the unit tests
fn sinkhorn_split(mix: &[f32], scale: &[f32], base: &[f32], n_hc: usize, iters: u32, eps: f32) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut pre = vec![0f32; n_hc];
    let mut post = vec![0f32; n_hc];
    for i in 0..n_hc {
        pre[i] = sigmoid(mix[i] * scale[0] + base[i]) + eps;
        post[i] = 2.0 * sigmoid(mix[n_hc + i] * scale[1] + base[n_hc + i]);
    }
    // c[src + dst*n_hc], row (dst) softmax + eps
    let mut c = vec![0f32; n_hc * n_hc];
    for dst in 0..n_hc {
        let mut row_max = f32::NEG_INFINITY;
        for src in 0..n_hc {
            let idx = src + dst * n_hc;
            let off = 2 * n_hc + idx;
            c[idx] = mix[off] * scale[2] + base[off];
            row_max = row_max.max(c[idx]);
        }
        let mut sum = 0f32;
        for src in 0..n_hc {
            let idx = src + dst * n_hc;
            c[idx] = (c[idx] - row_max).exp();
            sum += c[idx];
        }
        let inv = 1.0 / sum;
        for src in 0..n_hc {
            c[src + dst * n_hc] = c[src + dst * n_hc] * inv + eps;
        }
    }
    // first column pass, then (iters-1) row+column rounds - the exact
    // reference order (the initial row softmax counts as the first row pass)
    let col_pass = |c: &mut [f32]| {
        for src in 0..n_hc {
            let sum: f32 = (0..n_hc).map(|dst| c[src + dst * n_hc]).sum();
            let inv = 1.0 / (sum + eps);
            for dst in 0..n_hc {
                c[src + dst * n_hc] *= inv;
            }
        }
    };
    let row_pass = |c: &mut [f32]| {
        for dst in 0..n_hc {
            let sum: f32 = (0..n_hc).map(|src| c[src + dst * n_hc]).sum();
            let inv = 1.0 / (sum + eps);
            for src in 0..n_hc {
                c[src + dst * n_hc] *= inv;
            }
        }
    };
    col_pass(&mut c);
    for _ in 1..iters {
        row_pass(&mut c);
        col_pass(&mut c);
    }
    // kernel layout: coefficient for (out dst, in src) = c[dst + src*n_hc]
    // (the reference post step's transposed read)
    let mut comb_k = vec![0f32; n_hc * n_hc];
    for dst in 0..n_hc {
        for src in 0..n_hc {
            comb_k[dst * n_hc + src] = c[dst + src * n_hc];
        }
    }
    (pre, post, comb_k)
}

/* ---- router (sqrt-softplus, biased top-k / tid2eid hash) --------------- */

fn topk_desc(score: &[f32], k: usize) -> Vec<i32> {
    let mut idx = vec![-1i32; k];
    for (i, &s) in score.iter().enumerate() {
        for j in 0..k {
            if idx[j] < 0 || s > score[idx[j] as usize] {
                for m in (j + 1..k).rev() {
                    idx[m] = idx[m - 1];
                }
                idx[j] = i as i32;
                break;
            }
        }
    }
    idx
}

/// probs = sqrt(softplus(logits)); selection = biased top-k (or the
/// tid2eid row on hash layers); weights = unbiased probs of the winners,
/// sum-normalized (floored) and scaled.
fn route(logits: &[f32], bias: &[f32], tid2eid: Option<&Vec<i32>>, token: u32, k: usize, weight_scale: f32) -> Result<(Vec<i32>, Vec<f32>)> {
    let n = logits.len();
    let probs: Vec<f32> = logits.iter().map(|&l| softplus_stable(l).sqrt()).collect();
    let selected = match tid2eid {
        Some(table) => {
            let row = &table[token as usize * k..token as usize * k + k];
            for &e in row {
                if e < 0 || e as usize >= n {
                    return Err("hash-selected expert outside router range".into());
                }
            }
            row.to_vec()
        }
        None => {
            let sel_score: Vec<f32> = probs.iter().zip(bias).map(|(p, b)| p + b).collect();
            topk_desc(&sel_score, k)
        }
    };
    let mut weights: Vec<f32> = selected.iter().map(|&e| probs[e as usize]).collect();
    let sum: f32 = weights.iter().sum::<f32>().max(ROUTER_FLOOR);
    for w in &mut weights {
        *w = *w / sum * weight_scale;
    }
    Ok((selected, weights))
}

/* ---- streaming compressor (compressor_decode_one) ---------------------- */

/// One compressor lane's rolling state.
#[allow(dead_code)] // host reference, exercised by the unit tests
pub(super) struct CompLane {
    st_kv: Vec<f32>,
    st_sc: Vec<f32>,
    width: usize,
    head_dim: usize,
    ratio: usize,
}

#[allow(dead_code)]
impl CompLane {
    fn new(ratio: u32, head_dim: u32) -> CompLane {
        let coff = if ratio == 4 { 2 } else { 1 };
        let width = coff * head_dim as usize;
        let rows = coff * ratio as usize;
        CompLane {
            st_kv: vec![0.0; rows * width],
            st_sc: vec![NEG_INF; rows * width],
            width,
            head_dim: head_dim as usize,
            ratio: ratio as usize,
        }
    }

    fn reset(&mut self) {
        self.st_kv.iter_mut().for_each(|v| *v = 0.0);
        self.st_sc.iter_mut().for_each(|v| *v = NEG_INF);
    }

    /// Per-dimension softmax pool over the window (two lanes on ratio 4:
    /// previous-window rows read column j, current rows column hd + j).
    fn pool(&self) -> Vec<f32> {
        let (w, hd, r) = (self.width, self.head_dim, self.ratio);
        let mut out = vec![0f32; hd];
        for j in 0..hd {
            let mut max_score = NEG_INF;
            if r == 4 {
                for row in 0..r {
                    max_score = max_score.max(self.st_sc[row * w + j]);
                    max_score = max_score.max(self.st_sc[(r + row) * w + hd + j]);
                }
            } else {
                for row in 0..r {
                    max_score = max_score.max(self.st_sc[row * w + j]);
                }
            }
            if max_score <= NEG_INF * 0.5 {
                continue;
            }
            let mut denom = 0f32;
            let mut sum = 0f32;
            if r == 4 {
                for row in 0..r {
                    let wp = (self.st_sc[row * w + j] - max_score).exp();
                    let wc = (self.st_sc[(r + row) * w + hd + j] - max_score).exp();
                    denom += wp + wc;
                    sum += wp * self.st_kv[row * w + j];
                    sum += wc * self.st_kv[(r + row) * w + hd + j];
                }
            } else {
                for row in 0..r {
                    let ws = (self.st_sc[row * w + j] - max_score).exp();
                    denom += ws;
                    sum += ws * self.st_kv[row * w + j];
                }
            }
            out[j] = if denom > 0.0 { sum / denom } else { 0.0 };
        }
        out
    }

    /// Feed one token's kv/score projections; on a ratio boundary emit
    /// the pooled + normed + roped + quant-sim'd + f16-rounded row.
    fn step(&mut self, kv_cur: &[f32], sc_cur: &[f32], ape: &[f32], norm: &[f32], pos: u32, rms_eps: f32, rope: &kernels::RopeCfg, n_rot: usize) -> Option<Vec<f32>> {
        let (w, hd, r) = (self.width, self.head_dim, self.ratio);
        let pos_mod = pos as usize % r;
        let row = if r == 4 { r + pos_mod } else { pos_mod };
        for j in 0..w {
            self.st_kv[row * w + j] = kv_cur[j];
            self.st_sc[row * w + j] = sc_cur[j] + ape[pos_mod * w + j];
        }
        if (pos as usize + 1) % r != 0 {
            return None;
        }
        let pooled = self.pool();
        let ss: f64 = pooled.iter().map(|&v| (v as f64) * (v as f64)).sum();
        let rms = 1.0 / ((ss as f32 / hd as f32) + rms_eps).sqrt();
        let mut out: Vec<f32> =
            pooled.iter().zip(norm).map(|(&v, &n)| v * rms * n).collect();
        let comp_pos = pos + 1 - r as u32;
        rope_tail_host(&mut out, 1, hd, n_rot, comp_pos, rope);
        if hd == 512 {
            fp8_sim_row(&mut out, n_rot);
        } else if hd == 128 {
            qat_row(&mut out);
        }
        if r == 4 {
            // shift: previous-window rows take the finished window, then
            // the current rows mirror them (the reference's double copy)
            for row in 0..r {
                for j in 0..w {
                    self.st_kv[row * w + j] = self.st_kv[(r + row) * w + j];
                    self.st_sc[row * w + j] = self.st_sc[(r + row) * w + j];
                }
            }
        }
        // cache rows are stored f16 (kv_cache_push_comp)
        for v in &mut out {
            *v = f16_round(*v);
        }
        Some(out)
    }
}

/// Per-layer host state.
/// Device-resident compressor lane state (the host CompLane stays as
/// the unit-test reference).
struct DevLane {
    st_kv: DeviceBuf, // [rows][width]
    st_sc: DeviceBuf,
    width: u32,
    ratio: u32,
}

impl DevLane {
    fn new(ratio: u32, head_dim: u32) -> Result<DevLane> {
        let coff = if ratio == 4 { 2u32 } else { 1 };
        let width = coff * head_dim;
        let rows = coff * ratio;
        let mut l = DevLane {
            st_kv: DeviceBuf::alloc((rows * width) as usize * 4)?,
            st_sc: DeviceBuf::alloc((rows * width) as usize * 4)?,
            width,
            ratio,
        };
        l.reset()?;
        Ok(l)
    }

    fn reset(&mut self) -> Result {
        let n = self.st_kv.bytes();
        kernels::zero(&mut self.st_kv, n)?;
        kernels::fill_row_tail(&mut self.st_sc, 1, (n / 4) as u32, 0, NEG_INF)?;
        Ok(())
    }
}

pub(super) struct LayerRt {
    comp: Option<DevLane>,
    pub n_comp: u32,
    idx: Option<DevLane>,
    /// device indexer compressed cache [cap][n_idx_dim] (read back to
    /// the host only when top-k selection fires, past 512 comp rows)
    idx_cache: DeviceBuf,
    pub n_idx_comp: u32,
}

/// deepseek4 runtime: HC stream buffers + per-layer compressor state.
pub(super) struct Dsv4Rt {
    /// 4 residual streams [n_hc][n_embd]
    hc_cur: DeviceBuf,
    /// double-duty: hc_post output AND the [n_hc*n_embd] flat-norm scratch
    hc_next: DeviceBuf,
    mix: DeviceBuf,     // [6*n_hc] control projection
    low: DeviceBuf,     // [n_out_group*rank] grouped-out low; idx-q scratch
    comp_kv: DeviceBuf, // [max width] compressor projections
    comp_sc: DeviceBuf,
    idx_w: DeviceBuf,  // [n_idx_head] indexer proj
    allowed: DeviceBuf, // [comp cap] u8 visibility mask
    /// device Sinkhorn coefficient buffers [6*n_hc] (attn / ffn halves)
    coef_attn: DeviceBuf,
    coef_ffn: DeviceBuf,
    layers: Vec<LayerRt>,
}

impl Dsv4Rt {
    pub fn new(m: &Model, ctx: u32) -> Result<Dsv4Rt> {
        let s = m.shape;
        let max_ratio_cap = m
            .compress_ratios
            .iter()
            .take(s.n_exec_layer as usize)
            .filter(|&&r| r > 0)
            .map(|&r| ctx as usize / r as usize + 2)
            .max()
            .unwrap_or(1);
        let mut layers = Vec::with_capacity(s.n_exec_layer as usize);
        for il in 0..s.n_exec_layer as usize {
            let ratio = m.compress_ratios[il];
            layers.push(LayerRt {
                comp: if ratio != 0 { Some(DevLane::new(ratio, s.head_dim)?) } else { None },
                n_comp: 0,
                idx: if ratio == 4 { Some(DevLane::new(ratio, s.n_idx_dim)?) } else { None },
                idx_cache: DeviceBuf::alloc(if ratio == 4 {
                    max_ratio_cap * s.n_idx_dim as usize * 4
                } else {
                    4
                })?,
                n_idx_comp: 0,
            });
        }
        let rank = 1024u32; // output_lora_rank (V4 Flash and Pro)
        Ok(Dsv4Rt {
            // token-major stream layout [T][n_hc][n_embd]; decode (t=1)
            // coincides with the old single-token layout
            hc_cur: DeviceBuf::alloc(T_MAX * (s.n_hc * s.n_embd) as usize * 4)?,
            hc_next: DeviceBuf::alloc(T_MAX * (s.n_hc * s.n_embd) as usize * 4)?,
            mix: DeviceBuf::alloc(T_MAX * 6 * s.n_hc as usize * 4)?,
            low: DeviceBuf::alloc(T_MAX * (s.n_out_group * rank).max(s.n_idx_head * s.n_idx_dim) as usize * 4)?,
            comp_kv: DeviceBuf::alloc(T_MAX * 2 * s.head_dim as usize * 4)?,
            comp_sc: DeviceBuf::alloc(T_MAX * 2 * s.head_dim as usize * 4)?,
            idx_w: DeviceBuf::alloc(s.n_idx_head.max(1) as usize * 4)?,
            allowed: DeviceBuf::alloc(max_ratio_cap)?,
            coef_attn: DeviceBuf::alloc(T_MAX * 6 * s.n_hc as usize * 4)?,
            coef_ffn: DeviceBuf::alloc(T_MAX * 6 * s.n_hc as usize * 4)?,
            layers,
        })
    }

    fn reset(&mut self) -> Result {
        for l in &mut self.layers {
            if let Some(c) = &mut l.comp {
                c.reset()?;
            }
            if let Some(c) = &mut l.idx {
                c.reset()?;
            }
            l.n_comp = 0;
            l.n_idx_comp = 0;
        }
        Ok(())
    }
}

/* ---- indexer top-k selection (indexer_allowed_decode_one) --------------- */

/// Score compressed rows with the QAT'd indexer query and pick top-k.
/// Returns None when every row is visible (n_comp <= top_k).
fn indexer_allowed(q: &mut [f32], weights: &[f32], idx_cache: &[f32], n_comp: usize, n_head: usize, head_dim: usize, top_k: usize, pos: u32, rope: &kernels::RopeCfg, n_rot: usize) -> Option<Vec<u8>> {
    if n_comp <= top_k {
        return None;
    }
    rope_tail_host(q, n_head, head_dim, n_rot, pos, rope);
    for h in 0..n_head {
        qat_row(&mut q[h * head_dim..(h + 1) * head_dim]);
    }
    let scale = 1.0 / ((head_dim * n_head) as f32).sqrt();
    let mut scores = vec![0f32; n_comp];
    for (c, sc) in scores.iter_mut().enumerate() {
        let kv = &idx_cache[c * head_dim..(c + 1) * head_dim];
        let mut s = 0f32;
        for h in 0..n_head {
            let qh = &q[h * head_dim..(h + 1) * head_dim];
            let dot: f32 = qh.iter().zip(kv).map(|(a, b)| a * b).sum();
            s += dot.max(0.0) * weights[h] * scale;
        }
        *sc = s;
    }
    let mut allowed = vec![0u8; n_comp];
    for _ in 0..top_k {
        let mut best = 0usize;
        let mut best_score = NEG_INF;
        for (c, &s) in scores.iter().enumerate() {
            if allowed[c] == 0 && s > best_score {
                best = c;
                best_score = s;
            }
        }
        allowed[best] = 1;
    }
    Some(allowed)
}

/* ---- forward ------------------------------------------------------------ */

impl Model {
    /// V4 forward: sequential single-token steps (rows = 0 or 1).
    pub(super) fn forward_dsv4(&self, st: &mut State, tokens: &[u32], pos0: u32, rows: u32) -> Result<Option<Vec<f32>>> {
        if tokens.is_empty() {
            return Err("empty batch".into());
        }
        if rows > 1 {
            return Err("dsv4: multi-row logits (speculative paths) not supported yet".into());
        }
        if pos0 + tokens.len() as u32 > st.ctx {
            return Err("position exceeds context".into());
        }
        let mut rt = st.dsv4.take().ok_or("dsv4 state missing")?;
        let r = self.forward_dsv4_inner(st, &mut rt, tokens, pos0, rows);
        st.dsv4 = Some(rt);
        r
    }

    fn forward_dsv4_inner(&self, st: &mut State, rt: &mut Dsv4Rt, tokens: &[u32], pos0: u32, rows: u32) -> Result<Option<Vec<f32>>> {
        let s = self.shape;
        let row = s.n_embd as usize * 4;
        if pos0 == 0 {
            rt.reset()?;
        }
        let mut pos = pos0;
        let mut last_t = 1usize;
        for chunk in tokens.chunks(T_MAX) {
            let t = chunk.len();
            // the batched path assumes no indexer masking inside the
            // chunk (fires past 512 comp rows = ctx > 2048); past that
            // boundary fall to single-token steps
            let t = if (pos + t as u32) / 4 > s.n_idx_topk { 1 } else { t };
            for sub in chunk.chunks(t) {
                let t = sub.len();
                let ids: Vec<i32> = sub.iter().map(|&x| x as i32).collect();
                st.tok.write(0, kernels::as_bytes(&ids))?;
                kernels::embed_q8_0(&mut st.cur, &self.token_embd, &st.tok, s.n_embd, s.n_vocab, t as u32)?;
                // hc_from_plain_embedding, token-major streams
                for (i, _) in sub.iter().enumerate() {
                    for h in 0..s.n_hc as usize {
                        kernels::copy_d2d(
                            &mut rt.hc_cur,
                            (i * s.n_hc as usize + h) * row,
                            &st.cur,
                            i * row,
                            row,
                        )?;
                    }
                }
                for (il, l) in self.layers.iter().enumerate() {
                    self.eval_dsv4_layer(st, rt, il, l, sub, pos, t as u32)?;
                }
                pos += t as u32;
                last_t = t;
            }
        }
        if rows == 0 {
            return Ok(None);
        }

        // output head over the LAST token's streams
        let out = self.dsv4_out.as_ref().ok_or("dsv4 output head missing")?;
        let ones = self.ones_hc.as_ref().ok_or("ones_hc missing")?;
        let hc_row = (s.n_hc * s.n_embd) as usize * 4;
        kernels::copy_d2d(&mut rt.hc_next, 0, &rt.hc_cur, (last_t - 1) * hc_row, hc_row)?;
        std::mem::swap(&mut rt.hc_cur, &mut rt.hc_next);
        kernels::rms_norm(&mut rt.hc_next, &rt.hc_cur, ones, s.n_hc * s.n_embd, 1, s.rms_eps)?;
        kernels::matmul_f32(&mut rt.mix, &out.fn_w, &rt.hc_next, s.n_hc * s.n_embd, s.n_hc, 1)?;
        let pre = rt.mix.read_f32(s.n_hc as usize)?;
        let w: Vec<f32> = pre
            .iter()
            .zip(&out.base)
            .map(|(&p, &b)| sigmoid(p * out.scale + b) + s.hc_eps)
            .collect();
        kernels::dsv4_hc_mix(&mut st.last_row, &rt.hc_cur, None, &w, None, s.n_embd, s.n_hc, 1)?;
        kernels::rms_norm(&mut st.normed, &st.last_row, &self.output_norm, s.n_embd, 1, s.rms_eps)?;
        self.head_logits(st, 1)?;
        kernels::sync()?;
        Ok(Some(st.logits.read_f32(s.n_vocab as usize)?))
    }

    /// hc_pre: flat-norm the streams, project the control vector, run
    /// the Sinkhorn split ON DEVICE into a coef buffer (was 2 host
    /// readbacks per layer per token), reduce the streams into st.cur.
    /// `ffn` picks which coef buffer holds this half's gates.
    fn dsv4_hc_pre(&self, st: &mut State, rt: &mut Dsv4Rt, fn_w: &DeviceBuf, scale: &DeviceBuf, base: &DeviceBuf, ffn: bool, t: u32) -> Result {
        let s = self.shape;
        let ones = self.ones_hc.as_ref().ok_or("ones_hc missing")?;
        // token-major streams: the flat norm is per token over 4*n_embd
        kernels::rms_norm(&mut rt.hc_next, &rt.hc_cur, ones, s.n_hc * s.n_embd, t, s.rms_eps)?;
        kernels::matmul_f32(&mut rt.mix, fn_w, &rt.hc_next, s.n_hc * s.n_embd, 6 * s.n_hc, t)?;
        let coef = if ffn { &mut rt.coef_ffn } else { &mut rt.coef_attn };
        kernels::dsv4_sinkhorn(coef, &rt.mix, scale, base, s.n_hc, s.hc_sinkhorn, s.hc_eps, t)?;
        let coef = if ffn { &rt.coef_ffn } else { &rt.coef_attn };
        kernels::dsv4_hc_mix_dev(&mut st.cur, &rt.hc_cur, None, coef, 0, -1, s.n_embd, s.n_hc, 1, t)?;
        Ok(())
    }

    /// hc_post: streams' = post*block_out + comb-mix of streams, gates
    /// read from the half's device coef buffer.
    fn dsv4_hc_post(&self, rt: &mut Dsv4Rt, block_out: &DeviceBuf, ffn: bool, t: u32) -> Result {
        let s = self.shape;
        let coef = if ffn { &rt.coef_ffn } else { &rt.coef_attn };
        kernels::dsv4_hc_mix_dev(&mut rt.hc_next, &rt.hc_cur, Some(block_out), coef, 2 * s.n_hc, s.n_hc as i32, s.n_embd, s.n_hc, s.n_hc, t)?;
        std::mem::swap(&mut rt.hc_cur, &mut rt.hc_next);
        Ok(())
    }

    fn eval_dsv4_layer(&self, st: &mut State, rt: &mut Dsv4Rt, il: usize, l: &LayerW, tokens: &[u32], pos0: u32, t: u32) -> Result {
        let s = self.shape;
        let eps = s.rms_eps;
        let Attn::Dsv4(w) = &l.attn else {
            return Err("dsv4 layer without Dsv4 attn weights".into());
        };
        let rope = rope_cfg(&s, w.ratio);
        let q_dim = s.n_head * s.head_dim;
        let hd4 = s.head_dim as usize * 4;

        // ---- attention half (matmuls/norms/rope batched)
        self.dsv4_hc_pre(st, rt, &w.hc_attn_fn, &w.hc_attn_scale, &w.hc_attn_base, false, t)?;
        kernels::rms_norm(&mut st.normed, &st.cur, &l.attn_norm, s.n_embd, t, eps)?;
        kernels::matmul_q8_0(&mut st.q_rank, &w.q_a, &st.normed, s.n_embd, s.n_lora_q, t)?;
        kernels::rms_norm(&mut st.q_rank_norm, &st.q_rank, &w.q_a_norm, s.n_lora_q, t, eps)?;
        kernels::matmul_q8_0(&mut st.q, &w.q_b, &st.q_rank_norm, s.n_lora_q, q_dim, t)?;
        kernels::gqa_head_rms_norm(&mut st.q, None, t * s.n_head, s.head_dim, eps)?;
        kernels::matmul_q8_0(&mut st.k, &w.kv, &st.normed, s.n_embd, s.head_dim, t)?;
        kernels::rms_norm(&mut st.v, &st.k, &w.kv_a_norm, s.head_dim, t, eps)?;
        kernels::dsv4_rope_tail(&mut st.q, t, s.n_head, s.head_dim, s.rot_dim, pos0, &rope, false)?;
        kernels::dsv4_rope_tail(&mut st.v, t, 1, s.head_dim, s.rot_dim, pos0, &rope, false)?;
        kernels::dsv4_fp8_sim(&mut st.v, t, s.head_dim, s.rot_dim)?;
        kernels::dsv4_f16_round(&mut st.v, t * s.head_dim)?;
        // compressor projections batched (per-token rows consumed below)
        if let Some(comp) = &w.comp {
            kernels::matmul_q8_0(&mut rt.comp_kv, &comp.kv_w, &st.normed, s.n_embd, comp.width, t)?;
            kernels::matmul_q8_0(&mut rt.comp_sc, &comp.gate_w, &st.normed, s.n_embd, comp.width, t)?;
        }

        // ---- per-token interleave: ring append -> comp step -> attend.
        // LOAD-BEARING: batched ring appends would clobber earlier
        // tokens' windows (ring cap 128 < chunk span + window).
        let mut use_mask = false;
        for i in 0..t as usize {
            let pos = pos0 + i as u32;
            let n_raw = (pos + 1).min(s.n_swa);
            let emit = w.ratio != 0 && (pos + 1) % w.ratio == 0;
            kernels::copy_d2d(
                &mut st.kcache[il],
                (pos % s.n_swa) as usize * hd4,
                &st.v,
                i * hd4,
                hd4,
            )?;
            let lrt = &mut rt.layers[il];
            if let Some(comp) = &w.comp {
                let lane = lrt.comp.as_mut().ok_or("compressor state missing")?;
                kernels::dsv4_comp_step(
                    &mut lane.st_kv, &mut lane.st_sc,
                    &mut st.vcache[il], lrt.n_comp as usize * hd4,
                    &rt.comp_kv, &rt.comp_sc, i * comp.width as usize * 4,
                    &comp.ape, &comp.norm,
                    comp.width, s.head_dim, lane.ratio, pos, emit, false, eps, &rope,
                )?;
                if emit {
                    lrt.n_comp += 1;
                }
            }
            let n_comp = rt.layers[il].n_comp;
            kernels::dsv4_attention_at(
                &mut st.heads,
                i * q_dim as usize * 4,
                &st.q,
                i * q_dim as usize * 4,
                &st.kcache[il],
                n_raw,
                (n_comp > 0).then_some(&st.vcache[il]),
                n_comp,
                None,
                &w.sinks,
                s.n_head,
                s.head_dim,
                1.0 / (s.head_dim as f32).sqrt(),
            )?;
        }
        // indexer lane: projections batched, steps per token (its cache
        // feeds SELECTION only, which the chunk gate keeps inactive for
        // batched chunks; single-token steps handle the masked regime)
        if let Some(idx) = &w.idx {
            kernels::matmul_q8_0(&mut rt.comp_kv, &idx.comp.kv_w, &st.normed, s.n_embd, idx.comp.width, t)?;
            kernels::matmul_q8_0(&mut rt.comp_sc, &idx.comp.gate_w, &st.normed, s.n_embd, idx.comp.width, t)?;
            for i in 0..t as usize {
                let pos = pos0 + i as u32;
                let emit = (pos + 1) % w.ratio == 0;
                let lrt = &mut rt.layers[il];
                let lane = lrt.idx.as_mut().ok_or("indexer state missing")?;
                let (cache, n_idx) = (&mut lrt.idx_cache, lrt.n_idx_comp);
                kernels::dsv4_comp_step(
                    &mut lane.st_kv, &mut lane.st_sc,
                    cache, n_idx as usize * s.n_idx_dim as usize * 4,
                    &rt.comp_kv, &rt.comp_sc, i * idx.comp.width as usize * 4,
                    &idx.comp.ape, &idx.comp.norm,
                    idx.comp.width, s.n_idx_dim, lane.ratio, pos, emit, true, eps, &rope,
                )?;
                if emit {
                    lrt.n_idx_comp += 1;
                }
            }
            // top-k selection: single-token regime only (t == 1 past
            // the 2048 boundary, enforced by the chunk gate)
            if t == 1 && rt.layers[il].n_idx_comp > s.n_idx_topk {
                let pos = pos0;
                kernels::matmul_q8_0(&mut rt.low, &idx.q_b, &st.q_rank_norm, s.n_lora_q, s.n_idx_head * s.n_idx_dim, 1)?;
                kernels::matmul_f32(&mut rt.idx_w, &idx.proj, &st.normed, s.n_embd, s.n_idx_head, 1)?;
                kernels::sync()?;
                let mut q = rt.low.read_f32((s.n_idx_head * s.n_idx_dim) as usize)?;
                let weights = rt.idx_w.read_f32(s.n_idx_head as usize)?;
                let lrt = &rt.layers[il];
                let idx_cache_host = lrt
                    .idx_cache
                    .read_f32(lrt.n_idx_comp as usize * s.n_idx_dim as usize)?;
                if let Some(mask) = indexer_allowed(
                    &mut q,
                    &weights,
                    &idx_cache_host,
                    lrt.n_idx_comp as usize,
                    s.n_idx_head as usize,
                    s.n_idx_dim as usize,
                    s.n_idx_topk as usize,
                    pos,
                    &rope,
                    s.rot_dim as usize,
                ) {
                    rt.allowed.write(0, &mask)?;
                    use_mask = true;
                }
            }
        }
        if use_mask {
            // masked single-token attention re-runs with the selection
            // (the unmasked pass above already wrote heads; rerun row 0)
            let n_raw = (pos0 + 1).min(s.n_swa);
            let n_comp = rt.layers[il].n_comp;
            kernels::dsv4_attention(
                &mut st.heads,
                &st.q,
                &st.kcache[il],
                n_raw,
                (n_comp > 0).then_some(&st.vcache[il]),
                n_comp,
                Some(&rt.allowed),
                &w.sinks,
                s.n_head,
                s.head_dim,
                1.0 / (s.head_dim as f32).sqrt(),
            )?;
        }

        // ---- batched tail: un-rope, grouped out, hc_post
        kernels::dsv4_rope_tail(&mut st.heads, t, s.n_head, s.head_dim, s.rot_dim, pos0, &rope, true)?;
        let rank = 1024usize;
        let group_dim = q_dim / s.n_out_group;
        // heads is [t][n_out_group] contiguous group_dim slices and low is
        // [t][n_out_group] contiguous rank rows, so all t*8 grouped
        // projections collapse into one banked launch (bitwise identical
        // to the per-(token,group) loop).
        kernels::matmul_q8_0_banked(&mut rt.low, &w.out_a, &st.heads, group_dim, rank as u32, s.n_out_group, t)?;
        kernels::matmul_q8_0(&mut st.attn_out, &l.attn_output, &rt.low, (s.n_out_group as usize * rank) as u32, s.n_embd, t)?;
        self.dsv4_hc_post(rt, &st.attn_out, false, t)?;

        // ---- ffn half: ONE router readback + ONE MoE union per chunk
        self.dsv4_hc_pre(st, rt, &w.hc_ffn_fn, &w.hc_ffn_scale, &w.hc_ffn_base, true, t)?;
        kernels::rms_norm(&mut st.normed, &st.cur, &l.ffn_norm, s.n_embd, t, eps)?;
        let Ffn::Moe { gate_inp, shexp, gate_exps, up_exps, down_exps, .. } = &l.ffn else {
            return Err("dsv4 layer without MoE ffn".into());
        };
        kernels::matmul_f32(&mut st.router_logits, gate_inp, &st.normed, s.n_embd, s.n_expert, t)?;
        let logits = st.router_logits.read_f32((t * s.n_expert) as usize)?;
        let mut selected = Vec::with_capacity((t * s.n_expert_used) as usize);
        let mut weights = Vec::with_capacity((t * s.n_expert_used) as usize);
        for (i, &tok) in tokens.iter().enumerate() {
            let (sel, wts) = route(
                &logits[i * s.n_expert as usize..(i + 1) * s.n_expert as usize],
                &w.probs_b,
                w.tid2eid.as_ref(),
                tok,
                s.n_expert_used as usize,
                s.expert_weight_scale,
            )?;
            selected.extend_from_slice(&sel);
            weights.extend_from_slice(&wts);
        }
        st.router_selected.write(0, kernels::as_bytes(&selected))?;
        st.router_weights.write(0, kernels::as_bytes(&weights))?;
        if let Some((sg, su, sd)) = shexp {
            kernels::matmul_q8_0(&mut st.gate_act, sg, &st.normed, s.n_embd, s.n_ff_exp, t)?;
            kernels::matmul_q8_0(&mut st.up_act, su, &st.normed, s.n_embd, s.n_ff_exp, t)?;
            kernels::swiglu(&mut st.ffn_mid, &st.gate_act, &st.up_act, t * s.n_ff_exp, s.clamp_exp, 1.0, 0)?;
            kernels::matmul_q8_0(&mut st.shared_out, sd, &st.ffn_mid, s.n_ff_exp, s.n_embd, t)?;
        } else {
            kernels::zero(&mut st.shared_out, (t * s.n_embd) as usize * 4)?;
        }
        kernels::quantize_q8_k(&mut st.xq, &st.normed, s.n_embd, t)?;
        self.dsv4_moe(st, &selected, gate_exps, up_exps, down_exps, 3, t)?;
        kernels::add(&mut st.ffn_out, &st.moe_out, &st.shared_out, t * s.n_embd)?;
        self.dsv4_hc_post(rt, &st.ffn_out, true, t)?;
        Ok(())
    }

    /// Routed-expert resolve + kernels for one token. A lean cousin of
    /// the shared Moe arm: VRAM cache -> host LFU -> io_uring, staged
    /// per-slab. ponytail: no tiers/prefetch/grouped here; unify with
    /// eval_layer's resolve when the dsv4 perf pass starts.
    pub(super) fn dsv4_moe(&self, st: &mut State, selected: &[i32], gate_exps: &super::ExpertTensor, up_exps: &super::ExpertTensor, down_exps: &super::ExpertTensor, act_op: u32, n_tok: u32) -> Result {
        let s = self.shape;
        let primary = kernels::get_device();
        let mut distinct: Vec<i32> = selected
            .iter()
            .copied()
            .filter(|&e| e >= 0 && (e as u32) < s.n_expert)
            .collect();
        distinct.sort_unstable();
        distinct.dedup();
        // resident-tier experts compute on their own card (the whole
        // point for the hybrid families: a 16-row verify union served
        // at VRAM speed instead of streaming over PCIe every round)
        let mut tier_map: std::collections::HashMap<i32, (usize, kernels::ExpertPtrs)> =
            std::collections::HashMap::new();
        for &e in &distinct {
            let g_off = gate_exps.abs_offset + e as u64 * gate_exps.expert_bytes;
            let u_off = up_exps.abs_offset + e as u64 * up_exps.expert_bytes;
            let d_off = down_exps.abs_offset + e as u64 * down_exps.expert_bytes;
            for (ti, t) in st.tiers.iter().enumerate() {
                if let (Some(&g), Some(&u), Some(&d)) =
                    (t.map.get(&g_off), t.map.get(&u_off), t.map.get(&d_off))
                {
                    tier_map.insert(e, (ti, kernels::ExpertPtrs { gate: g, up: u, down: d }));
                    break;
                }
            }
        }
        // CPU expert lane (PULSAR_CPU=1): same contract as the full
        // resolve - host-cached experts compute on the worker pool, no
        // fetch, no upload, partial joins moe_out after the tier gather.
        let cpu_on = st.cpu_pool.is_some()
            && n_tok <= 8
            && !st.unified
            && s.n_embd % 256 == 0
            && s.n_ff_exp % 256 == 0
            && gate_exps.quant == up_exps.quant
            && [gate_exps.quant, down_exps.quant]
                .iter()
                .all(|&q| super::cpu_tier::supported(q));
        let mut lane = super::cpu_tier::Lane::new(
            gate_exps.quant,
            down_exps.quant,
            gate_exps.row_bytes as usize,
            down_exps.row_bytes as usize,
            s.n_embd as usize,
            s.n_ff_exp as usize,
            act_op,
        );
        let mut cpu_guard: Option<super::cpu_tier::WaitGuard> = None;
        // PULSAR_CPU_STEAL=0: leave dev-cache-resident experts to the GPU
        // (see the eval_layer seam for the tradeoff)
        let cpu_steal = std::env::var("PULSAR_CPU_STEAL").ok().as_deref() != Some("0");
        if cpu_on {
            let mut pins = Vec::new();
            for &e in &distinct {
                if tier_map.contains_key(&e) {
                    continue;
                }
                let go = gate_exps.abs_offset + e as u64 * gate_exps.expert_bytes;
                let uo = up_exps.abs_offset + e as u64 * up_exps.expert_bytes;
                let dno = down_exps.abs_offset + e as u64 * down_exps.expert_bytes;
                if !cpu_steal
                    && (st.dev_cache.map.contains_key(&go)
                        || st.dev_cache.map.contains_key(&uo)
                        || st.dev_cache.map.contains_key(&dno))
                {
                    continue;
                }
                if self.mtp.as_ref().is_some_and(|mt| mt.res_map.contains_key(&go)) {
                    continue;
                }
                let (Some(gp), Some(upp), Some(dp)) = (
                    st.store.peek_ptr(go),
                    st.store.peek_ptr(uo),
                    st.store.peek_ptr(dno),
                ) else {
                    continue;
                };
                lane.add(e, gp.0, upp.0, dp.0);
                pins.extend([go, uo, dno]);
            }
            st.store.pinned = pins;
        }
        if !lane.is_empty() {
            let rw = st.router_weights.read_f32(n_tok as usize * s.n_expert_used as usize)?;
            let normed_h = st.normed.read_f32(n_tok as usize * s.n_embd as usize)?;
            let pool = st.cpu_pool.as_ref().unwrap();
            cpu_guard = Some(super::cpu_tier::WaitGuard {
                pool,
                n: lane.submit_a(pool, selected, s.n_expert_used as usize, &normed_h, &rw, n_tok as usize),
            });
        }
        let mut offsets = Vec::with_capacity(3 * distinct.len());
        for &e in &distinct {
            if tier_map.contains_key(&e) {
                // keep the census warm for resident slabs or their heat
                // freezes at placement time
                for t in [gate_exps, up_exps, down_exps] {
                    let off = t.abs_offset + e as u64 * t.expert_bytes;
                    st.dev_cache.touch.entry(off).or_insert((0, t.expert_bytes)).0 += 1;
                }
                continue;
            }
            if lane.idx.contains_key(&e) {
                // no fetch, no upload, no census touch (touch would get
                // them VRAM-seeded next load and churn the equilibrium)
                continue;
            }
            for t in [gate_exps, up_exps, down_exps] {
                offsets.push(stream::Read {
                    offset: t.abs_offset + e as u64 * t.expert_bytes,
                    len: t.expert_bytes,
                });
            }
        }
        let in_use: Vec<u64> = offsets.iter().map(|r| r.offset).collect();
        let mut resolved = std::collections::HashMap::new();
        let mut wants = Vec::new();
        for r in &offsets {
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
        let dev_cache = &mut st.dev_cache;
        let staging = &mut st.staging;
        st.store.ensure_with(&wants, |off, payload| {
            if unified {
                resolved.insert(off, payload.as_ptr() as *const std::ffi::c_void);
                return Ok(());
            }
            let p = match dev_cache.maybe_insert(off, payload, &in_use)? {
                Some(p) => p,
                None => {
                    let base = stage_base[&off];
                    staging.write(base, payload)?;
                    staging.ptr_at(base)
                }
            };
            resolved.insert(off, p);
            Ok(())
        })?;
        let mut ptrs = Vec::with_capacity(selected.len());
        let mut tptrs: Vec<Vec<kernels::ExpertPtrs>> = st
            .tiers
            .iter()
            .map(|_| vec![kernels::ExpertPtrs::NULL; selected.len()])
            .collect();
        let mut tier_slots = vec![0u64; st.tiers.len()];
        for (si, &e) in selected.iter().enumerate() {
            if e < 0 || e as u32 >= s.n_expert {
                ptrs.push(kernels::ExpertPtrs::NULL);
                continue;
            }
            if let Some(&(ti, tp)) = tier_map.get(&e) {
                ptrs.push(kernels::ExpertPtrs::NULL);
                tptrs[ti][si] = tp;
                tier_slots[ti] += 1;
                continue;
            }
            if lane.idx.contains_key(&e) {
                ptrs.push(kernels::ExpertPtrs::NULL);
                continue;
            }
            ptrs.push(kernels::ExpertPtrs {
                gate: resolved[&(gate_exps.abs_offset + e as u64 * gate_exps.expert_bytes)],
                up: resolved[&(up_exps.abs_offset + e as u64 * up_exps.expert_bytes)],
                down: resolved[&(down_exps.abs_offset + e as u64 * down_exps.expert_bytes)],
            });
        }
        st.expert_ptrs.write(0, kernels::as_bytes(&ptrs))?;

        // tier partials first: their kernels run on the other card,
        // overlapping the primary's MoE below (st.normed still holds
        // the FFN input in both hybrid layer graphs)
        //
        // Verify-size chunks route through the grouped tensor-core MoE
        // (CSR of tokens per expert; each expert block decodes to int8
        // smem ONCE per launch instead of once per row - the dp4a path
        // re-decodes iq3/iq4 codebooks per slot, which measured as THE
        // DFlash verify cost at 6.5ms/layer).
        let grouped = n_tok >= 16
            && s.n_expert_used <= 16
            && s.n_ff_exp % 256 == 0
            && 2 * gate_exps.row_bytes.max(up_exps.row_bytes) * 4 <= 49152
            && down_exps.row_bytes * 4 <= 49152
            && std::env::var_os("PULSAR_NO_GROUPED").is_none();
        let mut active = Vec::new();
        for ti in 0..st.tiers.len() {
            if tier_slots[ti] == 0 {
                continue;
            }
            let normed_bytes = (n_tok * s.n_embd) as usize * 4;
            let weight_bytes = (n_tok * s.n_expert_used) as usize * 4;
            let tier = &mut st.tiers[ti];
            tier.hits += tier_slots[ti];
            kernels::copy_across(&mut tier.xin, &st.normed, normed_bytes)?;
            kernels::copy_across(&mut tier.weights, &st.router_weights, weight_bytes)?;
            kernels::set_device(tier.dev)?;
            kernels::quantize_q8_k(&mut tier.xq, &tier.xin, s.n_embd, n_tok)?;
            let mut ran_grouped = false;
            if grouped {
                // CSR: distinct expert -> [(token << 4) | slot]
                let mut gid: std::collections::HashMap<*const std::ffi::c_void, u32> =
                    std::collections::HashMap::new();
                let mut gptrs: Vec<kernels::ExpertPtrs> = Vec::new();
                let mut members: Vec<Vec<u32>> = Vec::new();
                for (si, p) in tptrs[ti].iter().enumerate() {
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
                let n_group = gptrs.len() as u32;
                if n_group > 0 {
                    let mut starts = Vec::with_capacity(n_group as usize + 1);
                    let mut pairs = Vec::new();
                    starts.push(0u32);
                    for m in &members {
                        pairs.extend_from_slice(m);
                        starts.push(pairs.len() as u32);
                    }
                    tier.grp_ptrs.write(0, kernels::as_bytes(&gptrs))?;
                    tier.grp_starts.write(0, kernels::as_bytes(&starts))?;
                    tier.grp_pairs.write(0, kernels::as_bytes(&pairs))?;
                    kernels::moe_pair_swiglu_grouped(
                        &mut tier.mid, &tier.grp_ptrs, &tier.grp_starts, &tier.grp_pairs,
                        &tier.weights, &tier.xq,
                        s.n_embd, s.n_ff_exp, s.n_expert_used, n_group,
                        gate_exps.row_bytes, gate_exps.quant, act_op,
                    )?;
                    kernels::quantize_q8_k(&mut tier.midq, &tier.mid, s.n_ff_exp, n_tok * s.n_expert_used)?;
                    let pbytes = n_tok as usize * s.n_expert_used as usize * s.n_embd as usize * 4;
                    kernels::zero(&mut tier.grp_partial, pbytes)?;
                    kernels::moe_down_grouped(
                        &mut tier.grp_partial, &tier.grp_ptrs, &tier.grp_starts, &tier.grp_pairs,
                        &tier.midq,
                        s.n_ff_exp, s.n_embd, s.n_expert_used, n_group,
                        down_exps.row_bytes, down_exps.quant,
                    )?;
                    kernels::moe_slot_sum(&mut tier.out, &tier.grp_partial, s.n_embd, s.n_expert_used, n_tok)?;
                    ran_grouped = true;
                }
            }
            if !ran_grouped {
                tier.ptrs.write(0, kernels::as_bytes(&tptrs[ti]))?;
                kernels::moe_pair_swiglu(
                    &mut tier.mid, &tier.ptrs, &tier.weights, &tier.xq,
                    s.n_embd, s.n_ff_exp, s.n_expert_used, n_tok, gate_exps.row_bytes, gate_exps.quant, act_op,
                )?;
                kernels::quantize_q8_k(&mut tier.midq, &tier.mid, s.n_ff_exp, n_tok * s.n_expert_used)?;
                kernels::moe_down(
                    &mut tier.out, &tier.ptrs, &tier.midq,
                    s.n_ff_exp, s.n_embd, s.n_expert_used, n_tok, down_exps.row_bytes, down_exps.quant,
                )?;
            }
            kernels::set_device(primary)?;
            active.push(ti);
        }

        // router weight applies before the down projection (ds4 order);
        // act_op 3 = deepseek4 clamped silu, 0 = plain silu (qwen35)
        kernels::moe_pair_swiglu(
            &mut st.moe_mid, &st.expert_ptrs, &st.router_weights, &st.xq,
            s.n_embd, s.n_ff_exp, s.n_expert_used, n_tok, gate_exps.row_bytes, gate_exps.quant, act_op,
        )?;
        kernels::quantize_q8_k(&mut st.midq, &st.moe_mid, s.n_ff_exp, n_tok * s.n_expert_used)?;
        kernels::moe_down(
            &mut st.moe_out, &st.expert_ptrs, &st.midq,
            s.n_ff_exp, s.n_embd, s.n_expert_used, n_tok, down_exps.row_bytes, down_exps.quant,
        )?;

        // gather tier partials (blocking copy issued on the tier device
        // = ordered after its kernels). NOTE: summing partials reorders
        // float adds vs the single-device loop - the documented tier
        // drift class; PULSAR_TIERS=off restores exact.
        for ti in active {
            let n = (n_tok * s.n_embd) as usize * 4;
            let tier = &st.tiers[ti];
            kernels::set_device(tier.dev)?;
            kernels::copy_across(&mut st.tier_ret, &tier.out, n)?;
            kernels::set_device(primary)?;
            kernels::add_assign(&mut st.moe_out, &st.tier_ret, n_tok * s.n_embd)?;
        }

        // CPU-lane join: stage A ran under ensure_with + the launches
        // above; down-proj partials compute here while the GPU kernels
        // are still in flight, then one f32 upload joins moe_out.
        if !lane.is_empty() {
            drop(cpu_guard.take());
            let t_cpu = std::time::Instant::now();
            let pool = st.cpu_pool.as_ref().unwrap();
            let acc = lane.finish(pool, n_tok as usize);
            st.store.pinned.clear();
            st.cpu_hits += lane.idx.len() as u64;
            st.prof.cpu += t_cpu.elapsed();
            if st.cpu_ret.bytes() < acc.len() * 4 {
                st.cpu_ret = DeviceBuf::alloc(acc.len() * 4)?;
            }
            st.cpu_ret.write(0, kernels::as_bytes(&acc))?;
            kernels::add_assign(&mut st.moe_out, &st.cpu_ret, n_tok * s.n_embd)?;
        }
        Ok(())
    }
}

/* ---- host math tests (no GPU needed) ------------------------------------ */

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinkhorn_is_doubly_stochastic() {
        let mix: Vec<f32> = (0..24).map(|i| (i as f32 * 0.37).sin()).collect();
        let scale = [0.7f32, 1.3, 0.9];
        let base: Vec<f32> = (0..24).map(|i| (i as f32 * 0.11).cos() * 0.5).collect();
        let (pre, post, comb) = sinkhorn_split(&mix, &scale, &base, 4, 20, 1.0e-6);
        assert_eq!(pre.len(), 4);
        assert!(pre.iter().all(|&p| p > 0.0 && p < 1.1));
        assert!(post.iter().all(|&p| (0.0..=2.0).contains(&p)));
        // rows and columns of the (transposed) combine matrix sum to ~1
        for dst in 0..4 {
            let row: f32 = (0..4).map(|src| comb[dst * 4 + src]).sum();
            assert!((row - 1.0).abs() < 1.0e-3, "row {dst} sums {row}");
        }
        for src in 0..4 {
            let col: f32 = (0..4).map(|dst| comb[dst * 4 + src]).sum();
            assert!((col - 1.0).abs() < 1.0e-3, "col {src} sums {col}");
        }
    }

    #[test]
    fn route_weights_normalized_and_scaled() {
        let logits: Vec<f32> = (0..256).map(|i| ((i * 37) % 100) as f32 / 25.0 - 2.0).collect();
        let bias: Vec<f32> = (0..256).map(|i| ((i * 17) % 50) as f32 / 100.0).collect();
        let (sel, w) = route(&logits, &bias, None, 0, 6, 1.5).unwrap();
        assert_eq!(sel.len(), 6);
        let mut uniq = sel.clone();
        uniq.sort_unstable();
        uniq.dedup();
        assert_eq!(uniq.len(), 6, "distinct experts");
        let sum: f32 = w.iter().sum();
        assert!((sum - 1.5).abs() < 1.0e-4, "weights sum to the scale, got {sum}");
        // hash override wins selection but weights still come from probs
        let table: Vec<i32> = (0..6).collect();
        let (sel2, w2) = route(&logits, &bias, Some(&table.clone()), 0, 6, 1.5).unwrap();
        assert_eq!(sel2, table);
        let sum2: f32 = w2.iter().sum();
        assert!((sum2 - 1.5).abs() < 1.0e-4);
    }

    #[test]
    fn hadamard_is_involutive() {
        let mut x = [0f32; 128];
        for (i, v) in x.iter_mut().enumerate() {
            *v = (i as f32 * 0.13).sin();
        }
        let orig = x;
        hadamard128(&mut x);
        hadamard128(&mut x);
        for i in 0..128 {
            assert!((x[i] - orig[i]).abs() < 1.0e-4, "H(H(x)) != x at {i}");
        }
    }

    #[test]
    fn e4m3_roundtrip_is_idempotent_and_bounded() {
        for i in 0..10_000 {
            let v = (i as f32 - 5000.0) * 0.11;
            let once = e4m3_dec(e4m3_enc(v));
            let twice = e4m3_dec(e4m3_enc(once));
            assert_eq!(once, twice, "not idempotent at {v}");
            assert!(once.abs() <= 448.0);
            if v.abs() <= 448.0 && v != 0.0 {
                assert!(((once - v) / v).abs() < 0.07, "e4m3 rel err at {v}: {once}");
            }
        }
    }

    #[test]
    fn e2m1_prefers_even_index_on_ties() {
        // 2.5 ties between 2.0 (idx 4, even) and 3.0 (idx 5, odd) -> 2.0
        assert_eq!(e2m1_roundtrip(2.5), 2.0);
        // 1.25 ties between 1.0 (idx 2) and 1.5 (idx 3) -> 1.0
        assert_eq!(e2m1_roundtrip(1.25), 1.0);
        assert_eq!(e2m1_roundtrip(-2.5), -2.0);
        assert_eq!(e2m1_roundtrip(10.0), 6.0);
    }

    #[test]
    fn f16_rte_matches_known_values() {
        assert_eq!(f32_to_f16_rte(1.0), 0x3c00);
        assert_eq!(f32_to_f16_rte(-2.0), 0xc000);
        assert_eq!(f32_to_f16_rte(0.0), 0x0000);
        assert_eq!(f32_to_f16_rte(65504.0), 0x7bff);
        assert_eq!(f32_to_f16_rte(1.0e6), 0x7c00); // overflow -> inf
        // 1 + 2^-11 is exactly halfway between 1.0 and the next half
        // value 1+2^-10; nearest-even keeps 1.0
        assert_eq!(f32_to_f16_rte(1.0 + 2.0f32.powi(-11)), 0x3c00);
        let roundtrip = super::super::requant::f16_to_f32(f32_to_f16_rte(0.333_333_34));
        assert!((roundtrip - 0.333_333_34).abs() < 3.0e-4);
    }

    #[test]
    fn compressor_emits_on_ratio_boundaries() {
        let ratio = 4u32;
        let hd = 8u32; // small stand-in head_dim (pool math is per-dim)
        let width = 2 * hd as usize;
        let ape = vec![0.1f32; width * ratio as usize];
        let norm = vec![1.0f32; hd as usize];
        let mut lane = CompLane::new(ratio, hd);
        let rope = rope_cfg_test();
        let mut emitted = 0;
        for pos in 0..8u32 {
            let kv: Vec<f32> = (0..width).map(|j| (pos as f32 + j as f32 * 0.1).sin()).collect();
            let sc: Vec<f32> = (0..width).map(|j| (pos as f32 * 0.3 + j as f32 * 0.05).cos()).collect();
            let out = lane.step(&kv, &sc, &ape, &norm, pos, 1.0e-6, &rope, 2);
            if (pos + 1) % ratio == 0 {
                let row = out.expect("boundary should emit");
                assert_eq!(row.len(), hd as usize);
                assert!(row.iter().all(|v| v.is_finite()));
                // values are f16-rounded
                for &v in &row {
                    assert_eq!(v, f16_round(v));
                }
                emitted += 1;
            } else {
                assert!(out.is_none());
            }
        }
        assert_eq!(emitted, 2);
    }

    #[test]
    fn indexer_allowed_masks_to_top_k() {
        let n_head = 4usize;
        let hd = 128usize;
        let n_comp = 10usize;
        let top_k = 3usize;
        let mut q = vec![0f32; n_head * hd];
        for (i, v) in q.iter_mut().enumerate() {
            *v = (i as f32 * 0.017).sin();
        }
        let weights = vec![1.0f32; n_head];
        let mut cache = vec![0f32; n_comp * hd];
        for (i, v) in cache.iter_mut().enumerate() {
            *v = (i as f32 * 0.031).cos();
        }
        let rope = rope_cfg_test();
        let mask = indexer_allowed(&mut q, &weights, &cache, n_comp, n_head, hd, top_k, 7, &rope, 64)
            .expect("n_comp > top_k selects");
        assert_eq!(mask.iter().filter(|&&m| m == 1).count(), top_k);
        // all-visible fast path
        let mut q2 = vec![0f32; n_head * hd];
        assert!(indexer_allowed(&mut q2, &weights, &cache, top_k, n_head, hd, top_k, 7, &rope, 64).is_none());
    }

    fn rope_cfg_test() -> kernels::RopeCfg {
        kernels::RopeCfg {
            n_ctx_orig: 65_536,
            freq_base: 160_000.0,
            freq_scale: 1.0 / 16.0,
            ext_factor: 1.0,
            attn_factor: 1.0 / (1.0 + 0.1 * 16f32.ln()),
            beta_fast: 32.0,
            beta_slow: 1.0,
            kq_mult: 1.0,
        }
    }
}
