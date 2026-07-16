//! CPU encoders for writing recipe ggufs: BF16/F16/F32 rows in, quantized
//! blocks out. Ports of ggml-quants.c reference math (the no-imatrix
//! paths), validated by round-trip RMSE tests here and end-to-end by
//! decoding the output gguf with pulsar itself (whose CUDA dot kernels
//! read these exact layouts).
//!
//! Stage 1 formats: q8_0 (attn/dense in the ds4-style recipe) and the
//! K-quants q2_K..q6_K. Stage 2 adds iq2_xxs/iq2_xs/iq3_xxs + imatrix.

pub const QK8_0: usize = 32;
pub const QK_K: usize = 256;

// ---------------------------------------------------------------- fp helpers

#[inline]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let s = (h >> 15) as u32;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 {
            s << 31
        } else {
            // subnormal: normalize
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

#[inline]
pub fn f32_to_f16(x: f32) -> u16 {
    // round-to-nearest-even, matching ggml_fp32_to_fp16
    let bits = x.to_bits();
    let s = ((bits >> 16) & 0x8000) as u16;
    let e = ((bits >> 23) & 0xff) as i32;
    let m = bits & 0x7f_ffff;
    if e == 0xff {
        // inf/nan
        return s | 0x7c00 | if m != 0 { 0x200 } else { 0 };
    }
    let e16 = e - 127 + 15;
    if e16 >= 0x1f {
        return s | 0x7c00; // overflow -> inf
    }
    if e16 <= 0 {
        if e16 < -10 {
            return s;
        }
        // subnormal
        let m = m | 0x80_0000;
        let shift = (14 - e16) as u32;
        let half = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        let rounded = half + ((rem > halfway) as u32 | ((rem == halfway) as u32 & (half & 1)));
        return s | rounded as u16;
    }
    let half = (s as u32) << 0 | ((e16 as u32) << 10) as u32 | (m >> 13);
    let rem = m & 0x1fff;
    let rounded = half + ((rem > 0x1000) as u32 | ((rem == 0x1000) as u32 & (half & 1)));
    rounded as u16
}

/// Decode a raw source row (F32/F16/BF16 bytes) into f32.
pub fn row_to_f32(ty: gguf::TensorType, raw: &[u8], out: &mut Vec<f32>) -> Result<(), String> {
    out.clear();
    match ty {
        gguf::TensorType::F32 => {
            out.extend(raw.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())));
        }
        gguf::TensorType::F16 => {
            out.extend(raw.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap()))));
        }
        gguf::TensorType::BF16 => {
            out.extend(raw.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes(c.try_into().unwrap()))));
        }
        other => return Err(format!("unsupported source type {other:?}")),
    }
    Ok(())
}

#[inline]
fn nearest_int(x: f32) -> i32 {
    x.round() as i32
}

// ------------------------------------------------------------ ggml searches

/// ggml make_qx_quants (rmse_type 1): symmetric scale for signed quants.
fn make_qx_quants(n: usize, nmax: i32, x: &[f32], ls: &mut [i8]) -> f32 {
    let mut max = 0f32;
    let mut amax = 0f32;
    for &v in &x[..n] {
        let a = v.abs();
        if a > amax {
            amax = a;
            max = v;
        }
    }
    if amax < 1e-30 {
        ls[..n].iter_mut().for_each(|l| *l = 0);
        return 0.0;
    }
    let mut iscale = -(nmax as f32) / max;
    let mut sumlx = 0f32;
    let mut suml2 = 0f32;
    for i in 0..n {
        let l = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
        ls[i] = (l + nmax) as i8;
        let w = x[i] * x[i];
        sumlx += w * x[i] * l as f32;
        suml2 += w * (l * l) as f32;
    }
    let mut scale = if suml2 > 0.0 { sumlx / suml2 } else { 0.0 };
    let mut best = scale * sumlx;
    for is in -9..=9i32 {
        if is == 0 {
            continue;
        }
        iscale = -(nmax as f32 + 0.1 * is as f32) / max;
        let mut sumlx = 0f32;
        let mut suml2 = 0f32;
        for i in 0..n {
            let l = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
            let w = x[i] * x[i];
            sumlx += w * x[i] * l as f32;
            suml2 += w * (l * l) as f32;
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for i in 0..n {
                let l = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
                ls[i] = (nmax + l) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

/// ggml make_qkx2_quants: scale+min for unsigned quants (q2/q4/q5_K).
#[allow(clippy::too_many_arguments)]
fn make_qkx2_quants(
    n: usize,
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    ls: &mut [u8],
    the_min: &mut f32,
    laux: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    use_mad: bool,
) -> f32 {
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        min = min.min(x[i]);
        max = max.max(x[i]);
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        ls[..n].iter_mut().for_each(|l| *l = 0);
        *the_min = -min;
        return 0.0;
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_mad = 0f32;
    for i in 0..n {
        let l = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
        ls[i] = l as u8;
        let mut diff = scale * l as f32 + min - x[i];
        diff = if use_mad { diff.abs() } else { diff * diff };
        best_mad += weights[i] * diff;
    }
    if nstep < 1 {
        *the_min = -min;
        return scale;
    }
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let mut sum_l = 0f32;
        let mut sum_l2 = 0f32;
        let mut sum_xl = 0f32;
        for i in 0..n {
            let l = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            laux[i] = l as u8;
            let w = weights[i];
            sum_l += w * l as f32;
            sum_l2 += w * (l * l) as f32;
            sum_xl += w * l as f32 * x[i];
        }
        let d = sum_w * sum_l2 - sum_l * sum_l;
        if d > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = if sum_l2 > 0.0 { sum_xl / sum_l2 } else { 0.0 };
            }
            let mut mad = 0f32;
            for i in 0..n {
                let mut diff = this_scale * laux[i] as f32 + this_min - x[i];
                diff = if use_mad { diff.abs() } else { diff * diff };
                mad += weights[i] * diff;
            }
            if mad < best_mad {
                ls[..n].copy_from_slice(&laux[..n]);
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    scale
}

// --------------------------------------------------------------- q8_0

/// 32 elems -> 34 bytes: f16 d + 32x int8.
pub fn quantize_row_q8_0(x: &[f32], out: &mut Vec<u8>) {
    for blk in x.chunks(QK8_0) {
        let mut amax = 0f32;
        for &v in blk {
            amax = amax.max(v.abs());
        }
        let d = amax / 127.0;
        let id = if d > 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        for i in 0..QK8_0 {
            let v = blk.get(i).copied().unwrap_or(0.0);
            out.push(nearest_int(v * id).clamp(-127, 127) as i8 as u8);
        }
    }
}

// --------------------------------------------------------------- q2_K

/// 256 elems -> 84 bytes: scales[16] (4-bit sc | 4-bit min), qs[64] 2-bit,
/// f16 d, f16 dmin.
pub fn quantize_row_q2_k(x: &[f32], out: &mut Vec<u8>) {
    debug_assert_eq!(x.len() % QK_K, 0);
    let mut ls = [0u8; 16];
    let mut laux = [0u8; 16];
    let mut weights = [0f32; 16];
    let mut lall = [0u8; QK_K];
    for blk in x.chunks_exact(QK_K) {
        let mut scales = [0f32; 16];
        let mut mins = [0f32; 16];
        let mut max_scale = 0f32;
        let mut max_min = 0f32;
        for j in 0..16 {
            let xs = &blk[16 * j..16 * j + 16];
            for l in 0..16 {
                weights[l] = xs[l].abs();
            }
            let mut m = 0f32;
            let s = make_qkx2_quants(16, 3, xs, &weights, &mut ls, &mut m, &mut laux, -0.5, 0.1, 15, true);
            lall[16 * j..16 * j + 16].copy_from_slice(&ls);
            scales[j] = s;
            mins[j] = m;
            max_scale = max_scale.max(s);
            max_min = max_min.max(m);
        }
        let mut sc_bytes = [0u8; 16];
        let d;
        let dmin;
        if max_scale > 0.0 {
            let iscale = 15.0 / max_scale;
            for j in 0..16 {
                sc_bytes[j] = nearest_int(iscale * scales[j]).clamp(0, 15) as u8;
            }
            d = max_scale / 15.0;
        } else {
            d = 0.0;
        }
        if max_min > 0.0 {
            let iscale = 15.0 / max_min;
            for j in 0..16 {
                sc_bytes[j] |= (nearest_int(iscale * mins[j]).clamp(0, 15) as u8) << 4;
            }
            dmin = max_min / 15.0;
        } else {
            dmin = 0.0;
        }
        // requantize with the quantized scales
        for j in 0..16 {
            let dj = d * (sc_bytes[j] & 0xF) as f32;
            if dj == 0.0 {
                lall[16 * j..16 * j + 16].iter_mut().for_each(|l| *l = 0);
                continue;
            }
            let mj = dmin * (sc_bytes[j] >> 4) as f32;
            for l in 0..16 {
                let v = nearest_int((blk[16 * j + l] + mj) / dj).clamp(0, 3);
                lall[16 * j + l] = v as u8;
            }
        }
        out.extend_from_slice(&sc_bytes);
        // pack 2-bit: two 128-chunks, byte j of chunk holds elems j, j+32, j+64, j+96
        for n in 0..2 {
            for j in 0..32 {
                let base = 128 * n + j;
                let b = lall[base]
                    | (lall[base + 32] << 2)
                    | (lall[base + 64] << 4)
                    | (lall[base + 96] << 6);
                out.push(b);
            }
        }
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        out.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());
    }
}

// --------------------------------------------------------------- q3_K

/// ggml make_q3_quants with do_rmse=true (16 elems, nmax=4 -> signed -4..3).
fn make_q3_quants(n: usize, nmax: i32, x: &[f32], ls: &mut [i8]) -> f32 {
    let mut max = 0f32;
    let mut amax = 0f32;
    for &v in &x[..n] {
        let a = v.abs();
        if a > amax {
            amax = a;
            max = v;
        }
    }
    if amax == 0.0 {
        ls[..n].iter_mut().for_each(|l| *l = 0);
        return 0.0;
    }
    let iscale = -(nmax as f32) / max;
    // do_rmse path: weighted regression + greedy refinement
    let mut sumlx = 0f32;
    let mut suml2 = 0f32;
    for i in 0..n {
        let l = nearest_int(iscale * x[i]).clamp(-nmax, nmax - 1);
        ls[i] = l as i8;
        let w = x[i] * x[i];
        sumlx += w * x[i] * l as f32;
        suml2 += w * (l * l) as f32;
    }
    for _ in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let w = x[i] * x[i];
            let mut slx = sumlx - w * x[i] * ls[i] as f32;
            if slx > 0.0 {
                let mut sl2 = suml2 - w * (ls[i] as i32 * ls[i] as i32) as f32;
                let new_l = nearest_int(x[i] * sl2 / slx).clamp(-nmax, nmax - 1);
                if new_l != ls[i] as i32 {
                    slx += w * x[i] * new_l as f32;
                    sl2 += w * (new_l * new_l) as f32;
                    if sl2 > 0.0 && slx * slx * suml2 > sumlx * sumlx * sl2 {
                        ls[i] = new_l as i8;
                        sumlx = slx;
                        suml2 = sl2;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }
    for i in 0..n {
        ls[i] += nmax as i8;
    }
    if suml2 > 0.0 {
        sumlx / suml2
    } else {
        0.0
    }
}

/// 256 elems -> 110 bytes: hmask[32], qs[64], scales[12] (6-bit packed), f16 d.
pub fn quantize_row_q3_k(x: &[f32], out: &mut Vec<u8>) {
    let mut lall = [0i8; QK_K];
    for blk in x.chunks_exact(QK_K) {
        let mut scales = [0f32; 16];
        let mut max_scale = 0f32;
        let mut amax = 0f32;
        for j in 0..16 {
            let mut ls = [0i8; 16];
            scales[j] = make_q3_quants(16, 4, &blk[16 * j..16 * j + 16], &mut ls);
            lall[16 * j..16 * j + 16].copy_from_slice(&ls);
            let a = scales[j].abs();
            if a > amax {
                amax = a;
                max_scale = scales[j];
            }
        }
        let mut sc_q = [0i8; 16];
        let d;
        if max_scale != 0.0 {
            let iscale = -32.0 / max_scale;
            for j in 0..16 {
                sc_q[j] = nearest_int(iscale * scales[j]).clamp(-32, 31) as i8;
            }
            d = 1.0 / iscale;
        } else {
            d = 0.0;
        }
        let dq = {
            let h = f32_to_f16(d);
            f16_to_f32(h)
        };
        // requantize with quantized scales
        for j in 0..16 {
            let sc = sc_q[j] as f32 * dq;
            if sc == 0.0 {
                lall[16 * j..16 * j + 16].iter_mut().for_each(|l| *l = 4);
                continue;
            }
            for l in 0..16 {
                let v = nearest_int(blk[16 * j + l] / sc).clamp(-4, 3);
                lall[16 * j + l] = (v + 4) as i8;
            }
        }
        // hmask + qs
        let mut hmask = [0u8; 32];
        let mut qs = [0u8; 64];
        for n in 0..2 {
            for j in 0..32 {
                for s in 0..4 {
                    let idx = 128 * n + 32 * s + j;
                    let mut l = lall[idx] as u8; // 0..7
                    if l > 3 {
                        hmask[j] |= 1 << (4 * n + s);
                        l -= 4;
                    }
                    qs[32 * n + j] |= l << (2 * s);
                }
            }
        }
        // 6-bit scale packing (inverse of the kmask unpack)
        let mut sbytes = [0u8; 12];
        for j in 0..16 {
            let v = (sc_q[j] + 32) as u8; // 0..63
            if j < 8 {
                sbytes[j] |= v & 0xF;
            } else {
                sbytes[j - 8] |= (v & 0xF) << 4;
            }
            sbytes[8 + j % 4] |= (v >> 4) << (2 * (j / 4));
        }
        out.extend_from_slice(&hmask);
        out.extend_from_slice(&qs);
        out.extend_from_slice(&sbytes);
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    }
}

// --------------------------------------------------------------- q4_K / q5_K

fn qkx_weights_av(xs: &[f32], weights: &mut [f32]) {
    let sum_x2: f32 = xs.iter().map(|v| v * v).sum();
    let av_x = (sum_x2 / xs.len() as f32).sqrt();
    for (w, &v) in weights.iter_mut().zip(xs) {
        *w = av_x + v.abs();
    }
}

fn pack_scale_min_k4(sc_q: &[u8; 8], mn_q: &[u8; 8]) -> [u8; 12] {
    let mut s = [0u8; 12];
    for j in 0..8 {
        let ls = sc_q[j];
        let lm = mn_q[j];
        if j < 4 {
            s[j] = ls;
            s[j + 4] = lm;
        } else {
            s[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            s[j - 4] |= (ls >> 4) << 6;
            s[j] |= (lm >> 4) << 6;
        }
    }
    s
}

fn qkx2_block_scales(
    blk: &[f32],
    sub: usize,
    nmax: i32,
    rmin: f32,
    rdelta: f32,
    nstep: i32,
    lall: &mut [u8; QK_K],
) -> ([u8; 8], [u8; 8], f32, f32) {
    let n_sub = QK_K / sub; // 8 subs of 32
    let mut scales = [0f32; 8];
    let mut mins = [0f32; 8];
    let mut ls = [0u8; 32];
    let mut laux = [0u8; 32];
    let mut weights = [0f32; 32];
    let mut max_scale = 0f32;
    let mut max_min = 0f32;
    for j in 0..n_sub {
        let xs = &blk[sub * j..sub * (j + 1)];
        qkx_weights_av(xs, &mut weights[..sub]);
        let mut m = 0f32;
        let s = make_qkx2_quants(sub, nmax, xs, &weights[..sub], &mut ls[..sub], &mut m, &mut laux[..sub], rmin, rdelta, nstep, false);
        lall[sub * j..sub * (j + 1)].copy_from_slice(&ls[..sub]);
        scales[j] = s;
        mins[j] = m;
        max_scale = max_scale.max(s);
        max_min = max_min.max(m);
    }
    let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
    let mut sc_q = [0u8; 8];
    let mut mn_q = [0u8; 8];
    for j in 0..n_sub {
        sc_q[j] = nearest_int(inv_scale * scales[j]).clamp(0, 63) as u8;
        mn_q[j] = nearest_int(inv_min * mins[j]).clamp(0, 63) as u8;
    }
    let d = if max_scale > 0.0 { max_scale / 63.0 } else { 0.0 };
    let dmin = if max_min > 0.0 { max_min / 63.0 } else { 0.0 };
    (sc_q, mn_q, d, dmin)
}

fn requant_sub32(blk: &[f32], d: f32, dmin: f32, sc_q: &[u8; 8], mn_q: &[u8; 8], nmax: i32, lall: &mut [u8; QK_K]) {
    let dq = f16_to_f32(f32_to_f16(d));
    let mq = f16_to_f32(f32_to_f16(dmin));
    for j in 0..8 {
        let dj = dq * sc_q[j] as f32;
        if dj == 0.0 {
            lall[32 * j..32 * (j + 1)].iter_mut().for_each(|l| *l = 0);
            continue;
        }
        let mj = mq * mn_q[j] as f32;
        for l in 0..32 {
            let v = nearest_int((blk[32 * j + l] + mj) / dj).clamp(0, nmax);
            lall[32 * j + l] = v as u8;
        }
    }
}

/// 256 -> 144 bytes: f16 d, f16 dmin, scales[12], qs[128] 4-bit.
pub fn quantize_row_q4_k(x: &[f32], out: &mut Vec<u8>) {
    let mut lall = [0u8; QK_K];
    for blk in x.chunks_exact(QK_K) {
        let (sc_q, mn_q, d, dmin) = qkx2_block_scales(blk, 32, 15, -1.0, 0.1, 20, &mut lall);
        requant_sub32(blk, d, dmin, &sc_q, &mn_q, 15, &mut lall);
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        out.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());
        out.extend_from_slice(&pack_scale_min_k4(&sc_q, &mn_q));
        // qs: per 64-group, byte l = elem l | elem l+32 << 4
        for g in 0..4 {
            for l in 0..32 {
                out.push(lall[64 * g + l] | (lall[64 * g + 32 + l] << 4));
            }
        }
    }
}

/// 256 -> 176 bytes: f16 d, f16 dmin, scales[12], qh[32], qs[128].
pub fn quantize_row_q5_k(x: &[f32], out: &mut Vec<u8>) {
    let mut lall = [0u8; QK_K];
    for blk in x.chunks_exact(QK_K) {
        let (sc_q, mn_q, d, dmin) = qkx2_block_scales(blk, 32, 31, -0.5, 0.1, 15, &mut lall);
        requant_sub32(blk, d, dmin, &sc_q, &mn_q, 31, &mut lall);
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        out.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());
        out.extend_from_slice(&pack_scale_min_k4(&sc_q, &mn_q));
        let mut qh = [0u8; 32];
        let mut qs = [0u8; 128];
        // ggml layout: two 4-bit lanes per 64-group; high bit into qh with
        // per-group masks u1=1<<2g, u2=2<<2g
        for g in 0..4 {
            for l in 0..32 {
                let l1 = lall[64 * g + l];
                let l2 = lall[64 * g + 32 + l];
                qs[32 * g + l] = (l1 & 0xF) | ((l2 & 0xF) << 4);
                if l1 > 15 {
                    qh[l] |= 1 << (2 * g);
                }
                if l2 > 15 {
                    qh[l] |= 2 << (2 * g);
                }
            }
        }
        out.extend_from_slice(&qh);
        out.extend_from_slice(&qs);
    }
}

// --------------------------------------------------------------- q6_K

/// 256 -> 210 bytes: ql[128], qh[64], scales[16] int8, f16 d.
pub fn quantize_row_q6_k(x: &[f32], out: &mut Vec<u8>) {
    let mut lall = [0i8; QK_K];
    for blk in x.chunks_exact(QK_K) {
        let mut scales = [0f32; 16];
        let mut max_scale = 0f32;
        let mut max_abs = 0f32;
        for j in 0..16 {
            let mut ls = [0i8; 16];
            let s = make_qx_quants(16, 32, &blk[16 * j..16 * j + 16], &mut ls);
            lall[16 * j..16 * j + 16].copy_from_slice(&ls);
            scales[j] = s;
            let a = s.abs();
            if a > max_abs {
                max_abs = a;
                max_scale = s;
            }
        }
        if max_abs == 0.0 {
            out.extend_from_slice(&[0u8; 210]);
            continue;
        }
        let iscale = -128f32 / max_scale;
        let d = 1.0 / iscale;
        let dq = f16_to_f32(f32_to_f16(d));
        let mut sc_q = [0i8; 16];
        for j in 0..16 {
            sc_q[j] = nearest_int(iscale * scales[j]).min(127) as i8;
        }
        for j in 0..16 {
            let dj = dq * sc_q[j] as f32;
            if dj == 0.0 {
                lall[16 * j..16 * j + 16].iter_mut().for_each(|l| *l = 32);
                continue;
            }
            for l in 0..16 {
                let v = nearest_int(blk[16 * j + l] / dj).clamp(-32, 31);
                lall[16 * j + l] = (v + 32) as i8;
            }
        }
        // ql: 4-bit low; qh: 2-bit high; per 128-chunk
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];
        for n in 0..2 {
            for l in 0..32 {
                let base = 128 * n;
                let q1 = lall[base + l] as u8;
                let q2 = lall[base + 32 + l] as u8;
                let q3 = lall[base + 64 + l] as u8;
                let q4 = lall[base + 96 + l] as u8;
                ql[64 * n + l] = (q1 & 0xF) | ((q3 & 0xF) << 4);
                ql[64 * n + 32 + l] = (q2 & 0xF) | ((q4 & 0xF) << 4);
                qh[32 * n + l] = (q1 >> 4) | ((q2 >> 4) << 2) | ((q3 >> 4) << 4) | ((q4 >> 4) << 6);
            }
        }
        out.extend_from_slice(&ql);
        out.extend_from_slice(&qh);
        out.extend_from_slice(&sc_q.map(|v| v as u8));
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    }
}

/// Quantize one logical row into `out` as `ty`.
pub fn quantize_row(ty: gguf::TensorType, x: &[f32], out: &mut Vec<u8>) -> Result<(), String> {
    match ty {
        gguf::TensorType::Q8_0 => quantize_row_q8_0(x, out),
        gguf::TensorType::Q2K => quantize_row_q2_k(x, out),
        gguf::TensorType::Q3K => quantize_row_q3_k(x, out),
        gguf::TensorType::Q4K => quantize_row_q4_k(x, out),
        gguf::TensorType::Q5K => quantize_row_q5_k(x, out),
        gguf::TensorType::Q6K => quantize_row_q6_k(x, out),
        gguf::TensorType::F32 => out.extend(x.iter().flat_map(|v| v.to_le_bytes())),
        gguf::TensorType::F16 => out.extend(x.iter().flat_map(|v| f32_to_f16(*v).to_le_bytes())),
        other => return Err(format!("no encoder for {other:?}")),
    }
    Ok(())
}

// =================================================================== tests

#[cfg(test)]
mod tests {
    use super::*;

    // test-side dequantizers, written from the ggml layout spec (the same
    // layouts pulsar's CUDA dot kernels read)

    fn dequant_q8_0(raw: &[u8], out: &mut Vec<f32>) {
        for b in raw.chunks_exact(34) {
            let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            for i in 0..32 {
                out.push(d * (b[2 + i] as i8) as f32);
            }
        }
    }

    fn dequant_q2_k(raw: &[u8], out: &mut Vec<f32>) {
        for b in raw.chunks_exact(84) {
            let scales = &b[0..16];
            let qs = &b[16..80];
            let d = f16_to_f32(u16::from_le_bytes([b[80], b[81]]));
            let dmin = f16_to_f32(u16::from_le_bytes([b[82], b[83]]));
            let mut y = [0f32; QK_K];
            for n in 0..2 {
                for j in 0..32 {
                    let byte = qs[32 * n + j];
                    for s in 0..4 {
                        let idx = 128 * n + 32 * s + j;
                        let q = (byte >> (2 * s)) & 3;
                        let sub = idx / 16;
                        y[idx] = d * (scales[sub] & 0xF) as f32 * q as f32
                            - dmin * (scales[sub] >> 4) as f32;
                    }
                }
            }
            out.extend_from_slice(&y);
        }
    }

    fn q3_scales(sb: &[u8]) -> [i8; 16] {
        let mut sc = [0i8; 16];
        for j in 0..16 {
            let lo = if j < 8 { sb[j] & 0xF } else { sb[j - 8] >> 4 };
            let hi = (sb[8 + j % 4] >> (2 * (j / 4))) & 3;
            sc[j] = ((lo | (hi << 4)) as i32 - 32) as i8;
        }
        sc
    }

    fn dequant_q3_k(raw: &[u8], out: &mut Vec<f32>) {
        for b in raw.chunks_exact(110) {
            let hmask = &b[0..32];
            let qs = &b[32..96];
            let sc = q3_scales(&b[96..108]);
            let d = f16_to_f32(u16::from_le_bytes([b[108], b[109]]));
            let mut y = [0f32; QK_K];
            for n in 0..2 {
                for j in 0..32 {
                    let byte = qs[32 * n + j];
                    for s in 0..4 {
                        let idx = 128 * n + 32 * s + j;
                        let mut q = ((byte >> (2 * s)) & 3) as i32;
                        if hmask[j] & (1 << (4 * n + s)) == 0 {
                            q -= 4;
                        }
                        y[idx] = d * sc[idx / 16] as f32 * q as f32;
                    }
                }
            }
            out.extend_from_slice(&y);
        }
    }

    fn scale_min_k4(s: &[u8], j: usize) -> (u8, u8) {
        if j < 4 {
            (s[j] & 63, s[j + 4] & 63)
        } else {
            (
                (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4),
                (s[j + 4] >> 4) | ((s[j] >> 6) << 4),
            )
        }
    }

    fn dequant_q4_k(raw: &[u8], out: &mut Vec<f32>) {
        for b in raw.chunks_exact(144) {
            let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
            let s = &b[4..16];
            let qs = &b[16..144];
            for g in 0..4 {
                let (sc1, mn1) = scale_min_k4(s, 2 * g);
                let (sc2, mn2) = scale_min_k4(s, 2 * g + 1);
                for l in 0..32 {
                    let byte = qs[32 * g + l];
                    out.push(d * sc1 as f32 * (byte & 0xF) as f32 - dmin * mn1 as f32);
                }
                for l in 0..32 {
                    let byte = qs[32 * g + l];
                    out.push(d * sc2 as f32 * (byte >> 4) as f32 - dmin * mn2 as f32);
                }
            }
        }
    }

    fn dequant_q5_k(raw: &[u8], out: &mut Vec<f32>) {
        for b in raw.chunks_exact(176) {
            let d = f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([b[2], b[3]]));
            let s = &b[4..16];
            let qh = &b[16..48];
            let qs = &b[48..176];
            for g in 0..4 {
                let (sc1, mn1) = scale_min_k4(s, 2 * g);
                let (sc2, mn2) = scale_min_k4(s, 2 * g + 1);
                for l in 0..32 {
                    let byte = qs[32 * g + l];
                    let hi = if qh[l] & (1 << (2 * g)) != 0 { 16 } else { 0 };
                    out.push(d * sc1 as f32 * ((byte & 0xF) + hi) as f32 - dmin * mn1 as f32);
                }
                for l in 0..32 {
                    let byte = qs[32 * g + l];
                    let hi = if qh[l] & (2 << (2 * g)) != 0 { 16 } else { 0 };
                    out.push(d * sc2 as f32 * ((byte >> 4) + hi) as f32 - dmin * mn2 as f32);
                }
            }
        }
    }

    fn dequant_q6_k(raw: &[u8], out: &mut Vec<f32>) {
        for b in raw.chunks_exact(210) {
            let ql = &b[0..128];
            let qh = &b[128..192];
            let sc: Vec<i8> = b[192..208].iter().map(|&v| v as i8).collect();
            let d = f16_to_f32(u16::from_le_bytes([b[208], b[209]]));
            let mut y = [0f32; QK_K];
            for n in 0..2 {
                for l in 0..32 {
                    let base = 128 * n;
                    let h = qh[32 * n + l];
                    let q1 = ((ql[64 * n + l] & 0xF) | ((h & 3) << 4)) as i32 - 32;
                    let q2 = ((ql[64 * n + 32 + l] & 0xF) | (((h >> 2) & 3) << 4)) as i32 - 32;
                    let q3 = ((ql[64 * n + l] >> 4) | (((h >> 4) & 3) << 4)) as i32 - 32;
                    let q4 = ((ql[64 * n + 32 + l] >> 4) | (((h >> 6) & 3) << 4)) as i32 - 32;
                    y[base + l] = d * sc[(base + l) / 16] as f32 * q1 as f32;
                    y[base + 32 + l] = d * sc[(base + 32 + l) / 16] as f32 * q2 as f32;
                    y[base + 64 + l] = d * sc[(base + 64 + l) / 16] as f32 * q3 as f32;
                    y[base + 96 + l] = d * sc[(base + 96 + l) / 16] as f32 * q4 as f32;
                }
            }
            out.extend_from_slice(&y);
        }
    }

    fn rand_rows(n: usize, seed: u64) -> Vec<f32> {
        // xorshift; roughly normal-ish via sum of uniforms
        let mut s = seed.max(1);
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s as f64 / u64::MAX as f64) as f32 - 0.5
        };
        (0..n).map(|_| (0..4).map(|_| next()).sum::<f32>() * 0.5).collect()
    }

    fn rmse(a: &[f32], b: &[f32]) -> f32 {
        let s: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        (s / a.len() as f32).sqrt()
    }

    fn check(ty: gguf::TensorType, max_rel_rmse: f32) {
        let x = rand_rows(QK_K * 8, 42);
        let mut enc = Vec::new();
        quantize_row(ty, &x, &mut enc).unwrap();
        let (bs, bb) = ty.block_layout().unwrap();
        assert_eq!(enc.len() as u64, (x.len() as u64 / bs) * bb, "{ty:?} size");
        let mut dec = Vec::new();
        match ty {
            gguf::TensorType::Q8_0 => dequant_q8_0(&enc, &mut dec),
            gguf::TensorType::Q2K => dequant_q2_k(&enc, &mut dec),
            gguf::TensorType::Q3K => dequant_q3_k(&enc, &mut dec),
            gguf::TensorType::Q4K => dequant_q4_k(&enc, &mut dec),
            gguf::TensorType::Q5K => dequant_q5_k(&enc, &mut dec),
            gguf::TensorType::Q6K => dequant_q6_k(&enc, &mut dec),
            _ => unreachable!(),
        }
        let rms: f32 = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        let e = rmse(&x, &dec) / rms;
        assert!(e < max_rel_rmse, "{ty:?} rel rmse {e} >= {max_rel_rmse}");
    }

    #[test]
    fn q8_0_roundtrip() {
        check(gguf::TensorType::Q8_0, 0.01);
    }
    #[test]
    fn q2_k_roundtrip() {
        check(gguf::TensorType::Q2K, 0.30);
    }
    #[test]
    fn q3_k_roundtrip() {
        check(gguf::TensorType::Q3K, 0.15);
    }
    // thresholds sit ~15% above measured (0.070 / 0.036 on the seed-42
    // rows); the bit ladder q4->q5->q6 halves error each step, which is
    // the consistency signal that matters
    #[test]
    fn q4_k_roundtrip() {
        check(gguf::TensorType::Q4K, 0.08);
    }
    #[test]
    fn q5_k_roundtrip() {
        check(gguf::TensorType::Q5K, 0.041);
    }
    #[test]
    fn q6_k_roundtrip() {
        check(gguf::TensorType::Q6K, 0.02);
    }

    #[test]
    fn f16_conv_roundtrip() {
        for v in [0.0f32, 1.0, -1.0, 0.5, 65504.0, 1e-5, -3.14159] {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            assert!((back - v).abs() <= v.abs() * 0.001 + 1e-7, "{v} -> {back}");
        }
    }
}
