//! CPU-side expert math for the CPU expert tier: host-cache-hit experts
//! compute where their bytes live (RAM at ~70GB/s) instead of crossing
//! PCIe (~29GB/s), freeing the H2D pipe for disk-miss staging.
//!
//! v1: iq2_xxs x q8_K vec dot (covers the GLM/Hy3 ds4 recipes end to
//! end). Decode contract mirrors dev_dot_iq2_xxs_q8_K_block_lut in
//! pulsar_kernels.cu exactly: per 32-value sub-block, aux0 = 4x8-bit
//! grid indices, aux1 = 4x7-bit sign masks + 4-bit scale in the top
//! nibble; value = grid_byte(2l+1 units) * sign, block factor
//! (2*scale+1), whole-block factor 0.125 * d_x * d_y.

use crate::iq::tables;

pub const QK_K: usize = 256;
/// iq2_xxs: 2 bytes f16 d + 16 u32 per 256 values
pub const IQ2_XXS_BLOCK_BYTES: usize = 2 + 64;

/// q8_K activation row: one f32 scale + 256 i8 per block, plus the 16
/// per-16-value group sums the K-quant min terms need (ggml's bsums).
pub struct Q8KRow {
    pub d: Vec<f32>,
    pub qs: Vec<i8>,
    pub bsums: Vec<i32>,
}

/// ggml quantize_row_q8_K: d = amax/127, q = round(x/d).
pub fn quantize_row_q8_k(x: &[f32]) -> Q8KRow {
    debug_assert_eq!(x.len() % QK_K, 0);
    let nb = x.len() / QK_K;
    let mut d = Vec::with_capacity(nb);
    let mut qs: Vec<i8> = Vec::with_capacity(x.len());
    for b in x.chunks_exact(QK_K) {
        let amax = b.iter().fold(0f32, |a, &v| a.max(v.abs()));
        if amax == 0.0 {
            d.push(0.0);
            qs.extend(std::iter::repeat(0i8).take(QK_K));
            continue;
        }
        let scale = amax / 127.0;
        let inv = 127.0 / amax;
        d.push(scale);
        for &v in b {
            qs.push((v * inv).round().clamp(-127.0, 127.0) as i8);
        }
    }
    let bsums = qs
        .chunks_exact(16)
        .map(|g| g.iter().map(|&q| q as i32).sum())
        .collect();
    Q8KRow { d, qs, bsums }
}

#[inline]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let man = (bits & 0x3ff) as u32;
    let f = if exp == 0 {
        f32::from_bits(sign | 0x3880_0000) * (man as f32 / 1024.0)
    } else if exp == 31 {
        f32::from_bits(sign | 0x7f80_0000 | (man << 13))
    } else {
        f32::from_bits(sign | ((exp + 112) << 23) | (man << 13))
    };
    f
}

/// full 8-bit sign mask from the stored 7 bits (bit 7 keeps popcount even)
#[inline]
fn sign_mask(s7: u32) -> u32 {
    s7 | (((s7.count_ones() & 1) as u32) << 7)
}

/// One expert row (n columns, iq2_xxs) dotted against a q8_K activation
/// row. Dispatches to AVX2 on x86 (bitwise identical to scalar: the
/// per-block bsum is exact integer math in both paths, and the float
/// accumulation order is the same).
pub fn vec_dot_iq2_xxs_q8_k(row: &[u8], x: &Q8KRow, n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        // AVX-512 VNNI was tried and measured SLOWER (2.1 vs 4.8 GB/s
        // single-thread on the 9900X: zmm assembly from 8 scattered grid
        // loads + scalar sign-mask chain stalls it), and the aggregate is
        // RAM-bound at 12 threads anyway - AVX2 is the keeper (git log
        // has the kernel if wider tables ever change the math)
        if std::arch::is_x86_feature_detected!("avx2") {
            return unsafe { avx2::vec_dot(row, x, n) };
        }
    }
    vec_dot_iq2_xxs_q8_k_scalar(row, x, n)
}

/// Scalar reference/fallback.
pub fn vec_dot_iq2_xxs_q8_k_scalar(row: &[u8], x: &Q8KRow, n: usize) -> f32 {
    debug_assert_eq!(n % QK_K, 0);
    let t = tables();
    let nb = n / QK_K;
    debug_assert!(row.len() >= nb * IQ2_XXS_BLOCK_BYTES);
    let mut total = 0f32;
    for ibl in 0..nb {
        let blk = &row[ibl * IQ2_XXS_BLOCK_BYTES..(ibl + 1) * IQ2_XXS_BLOCK_BYTES];
        let xd = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let q8 = &x.qs[ibl * QK_K..(ibl + 1) * QK_K];
        let mut bsum = 0i32;
        for ib32 in 0..QK_K / 32 {
            let aux0 = u32::from_le_bytes([blk[2 + 8 * ib32], blk[3 + 8 * ib32], blk[4 + 8 * ib32], blk[5 + 8 * ib32]]);
            let aux1 = u32::from_le_bytes([blk[6 + 8 * ib32], blk[7 + 8 * ib32], blk[8 + 8 * ib32], blk[9 + 8 * ib32]]);
            let ls = (2 * (aux1 >> 28) + 1) as i32;
            let mut sumi = 0i32;
            for k in 0..4 {
                let g = t.grid[((aux0 >> (8 * k)) & 0xff) as usize].to_le_bytes();
                let sm = sign_mask((aux1 >> (7 * k)) & 127);
                let q8k = &q8[ib32 * 32 + 8 * k..ib32 * 32 + 8 * k + 8];
                for i in 0..8 {
                    let w = if (sm >> i) & 1 == 1 { -(g[i] as i8 as i32) } else { g[i] as i8 as i32 };
                    sumi += w * q8k[i] as i32;
                }
            }
            bsum += sumi * ls;
        }
        total += 0.125 * xd * x.d[ibl] * bsum as f32;
    }
    total
}

/// q3_K: 110 bytes per 256 - hmask[32] (high bit), qs[64] (low 2 bits),
/// scales[12] (16x 6-bit, value-32 signed), f16 d. q = lo2 - (hbit?0:4).
/// Mirrors dev_dot_q3_K_q8_K_block.
pub const Q3_K_BLOCK_BYTES: usize = 32 + 64 + 12 + 2;
/// q4_K: 144 bytes per 256 - f16 d, f16 dmin, scales[12] (8x 6-bit
/// scale+min pairs), qs[128] (nibbles). Mirrors dev_dot_q4_K_q8_K_block.
pub const Q4_K_BLOCK_BYTES: usize = 2 + 2 + 12 + 128;

/// k3_unpack_scales mirror: 16 6-bit scales from 12 packed bytes, -32.
fn k3_scales(scales: &[u8]) -> [i32; 16] {
    let mut sc = [0i32; 16];
    for (j, out) in sc.iter_mut().enumerate() {
        let s = if j < 8 {
            (scales[j] & 0x0f) | (((scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)
        } else {
            (scales[j - 8] >> 4) | (((scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)
        };
        *out = s as i32 - 32;
    }
    sc
}

/// k4_scale_min mirror: 8x (6-bit scale, 6-bit min).
fn k4_scale_min(j: usize, q: &[u8]) -> (i32, i32) {
    if j < 4 {
        ((q[j] & 63) as i32, (q[j + 4] & 63) as i32)
    } else {
        (
            ((q[j + 4] & 0x0f) | ((q[j - 4] >> 6) << 4)) as i32,
            ((q[j + 4] >> 4) | ((q[j] >> 6) << 4)) as i32,
        )
    }
}

pub fn vec_dot_q3_k_q8_k(row: &[u8], x: &Q8KRow, n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        return unsafe { avx2::vec_dot_q3k(row, x, n) };
    }
    vec_dot_q3_k_q8_k_scalar(row, x, n)
}

pub fn vec_dot_q3_k_q8_k_scalar(row: &[u8], x: &Q8KRow, n: usize) -> f32 {
    let nb = n / QK_K;
    debug_assert!(row.len() >= nb * Q3_K_BLOCK_BYTES);
    let mut total = 0f32;
    for ibl in 0..nb {
        let blk = &row[ibl * Q3_K_BLOCK_BYTES..(ibl + 1) * Q3_K_BLOCK_BYTES];
        let (hm, q3, scales) = (&blk[..32], &blk[32..96], &blk[96..108]);
        let xd = f16_to_f32(u16::from_le_bytes([blk[108], blk[109]]));
        let q8 = &x.qs[ibl * QK_K..(ibl + 1) * QK_K];
        let sc = k3_scales(scales);
        let mut isum = 0i32;
        let mut hbit = 1u8;
        let mut is = 0;
        for k in 0..2 {
            for shift in [0u8, 2, 4, 6] {
                for half in 0..2 {
                    let mut s16 = 0i32;
                    for i in 0..16 {
                        let l = half * 16 + i;
                        let mut q = ((q3[32 * k + l] >> shift) & 3) as i32;
                        if hm[l] & hbit == 0 {
                            q -= 4;
                        }
                        s16 += q * q8[128 * k + 32 * (shift as usize / 2) + l] as i32;
                    }
                    isum += sc[is] * s16;
                    is += 1;
                }
                hbit <<= 1;
            }
        }
        total += xd * x.d[ibl] * isum as f32;
    }
    total
}

pub fn vec_dot_q4_k_q8_k(row: &[u8], x: &Q8KRow, n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        return unsafe { avx2::vec_dot_q4k(row, x, n) };
    }
    vec_dot_q4_k_q8_k_scalar(row, x, n)
}

pub fn vec_dot_q4_k_q8_k_scalar(row: &[u8], x: &Q8KRow, n: usize) -> f32 {
    let nb = n / QK_K;
    debug_assert!(row.len() >= nb * Q4_K_BLOCK_BYTES);
    let mut total = 0f32;
    for ibl in 0..nb {
        let blk = &row[ibl * Q4_K_BLOCK_BYTES..(ibl + 1) * Q4_K_BLOCK_BYTES];
        let xd = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        let xmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]]));
        let (scales, q4) = (&blk[4..16], &blk[16..144]);
        let q8 = &x.qs[ibl * QK_K..(ibl + 1) * QK_K];
        let bs = &x.bsums[ibl * 16..(ibl + 1) * 16];
        let mut isum = 0i32;
        let mut msum = 0i32;
        for j in 0..4 {
            let (sc1, m1) = k4_scale_min(2 * j, scales);
            let (sc2, m2) = k4_scale_min(2 * j + 1, scales);
            let mut s1 = 0i32;
            let mut s2 = 0i32;
            for i in 0..32 {
                let v = q4[32 * j + i];
                s1 += (v & 0x0f) as i32 * q8[64 * j + i] as i32;
                s2 += (v >> 4) as i32 * q8[64 * j + 32 + i] as i32;
            }
            isum += sc1 * s1 + sc2 * s2;
            msum += m1 * (bs[4 * j] + bs[4 * j + 1]) + m2 * (bs[4 * j + 2] + bs[4 * j + 3]);
        }
        total += xd * x.d[ibl] * isum as f32 - xmin * x.d[ibl] * msum as f32;
    }
    total
}

/// AVX2 path: one ymm per 32-value sub-block. Grid bytes are unsigned
/// magnitudes (<= 43), signs applied to q8 via a +-1 byte table and
/// sign_epi8, then maddubs (max pair sum 2*43*127 < i16::MAX, no
/// saturation) + madd -> i32, scaled by ls per sub-block.
#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    /// 128 u64s: byte i = 0xFF where sign bit i of the parity-completed
    /// mask is set, else 0x01.
    const fn build_ksigns64() -> [u64; 128] {
        let mut t = [0u64; 128];
        let mut m = 0u32;
        while m < 128 {
            let full = m | ((m.count_ones() & 1) << 7);
            let mut w = 0u64;
            let mut i = 0;
            while i < 8 {
                w |= (if (full >> i) & 1 == 1 { 0xFFu64 } else { 0x01u64 }) << (8 * i);
                i += 1;
            }
            t[m as usize] = w;
            m += 1;
        }
        t
    }
    static KSIGNS64: [u64; 128] = build_ksigns64();

    /// q3_K: qu = lo2 | (hbit << 2) is unsigned 0..7 for maddubs; the
    /// implied -4 folds out exactly as 4 * sc[g] * bsums16[g], subtracted
    /// once per block (integer-exact, so still bitwise vs scalar).
    #[target_feature(enable = "avx2")]
    pub unsafe fn vec_dot_q3k(row: &[u8], x: &Q8KRow, n: usize) -> f32 {
        let nb = n / QK_K;
        debug_assert!(row.len() >= nb * Q3_K_BLOCK_BYTES);
        let low2 = _mm256_set1_epi8(3);
        let one = _mm256_set1_epi8(1);
        let mut total = 0f32;
        for ibl in 0..nb {
            let blk = row.as_ptr().add(ibl * Q3_K_BLOCK_BYTES);
            let xd = f16_to_f32(u16::from_le_bytes([*blk.add(108), *blk.add(109)]));
            let q8 = x.qs.as_ptr().add(ibl * QK_K);
            let bs = &x.bsums[ibl * 16..(ibl + 1) * 16];
            let sc = k3_scales(std::slice::from_raw_parts(blk.add(96), 12));
            let hmv = _mm256_loadu_si256(blk as *const __m256i);
            let mut corr = 0i32;
            for (g, &s) in sc.iter().enumerate() {
                corr += s * bs[g];
            }
            let mut acc = _mm256_setzero_si256();
            let mut is = 0;
            for k in 0..2i32 {
                let q3v = _mm256_loadu_si256(blk.add(32 + 32 * k as usize) as *const __m256i);
                for shift in 0..4i32 {
                    let bit = 4 * k + shift;
                    let lo = _mm256_and_si256(
                        _mm256_srl_epi16(q3v, _mm_cvtsi32_si128(2 * shift)),
                        low2,
                    );
                    let hq = _mm256_slli_epi16(
                        _mm256_and_si256(_mm256_srl_epi16(hmv, _mm_cvtsi32_si128(bit)), one),
                        2,
                    );
                    let qu = _mm256_or_si256(lo, hq);
                    let q8v = _mm256_loadu_si256(
                        q8.add((128 * k + 32 * shift) as usize) as *const __m256i
                    );
                    let d16 = _mm256_maddubs_epi16(qu, q8v);
                    let scv = _mm256_set_m128i(
                        _mm_set1_epi16(sc[is + 1] as i16),
                        _mm_set1_epi16(sc[is] as i16),
                    );
                    is += 2;
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d16, scv));
                }
            }
            let s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b0100_1110));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b1011_0001));
            let isum = _mm_cvtsi128_si32(s) - 4 * corr;
            total += xd * *x.d.get_unchecked(ibl) * isum as f32;
        }
        total
    }

    /// q4_K: nibbles are already unsigned for maddubs; mins fold through
    /// the 16-group bsums like the scalar/CUDA path.
    #[target_feature(enable = "avx2")]
    pub unsafe fn vec_dot_q4k(row: &[u8], x: &Q8KRow, n: usize) -> f32 {
        let nb = n / QK_K;
        debug_assert!(row.len() >= nb * Q4_K_BLOCK_BYTES);
        let lown = _mm256_set1_epi8(0x0f);
        let mut total = 0f32;
        for ibl in 0..nb {
            let blk = row.as_ptr().add(ibl * Q4_K_BLOCK_BYTES);
            let xd = f16_to_f32(u16::from_le_bytes([*blk, *blk.add(1)]));
            let xmin = f16_to_f32(u16::from_le_bytes([*blk.add(2), *blk.add(3)]));
            let scales = std::slice::from_raw_parts(blk.add(4), 12);
            let q8 = x.qs.as_ptr().add(ibl * QK_K);
            let bs = &x.bsums[ibl * 16..(ibl + 1) * 16];
            let mut acc = _mm256_setzero_si256();
            let mut msum = 0i32;
            for j in 0..4 {
                let (sc1, m1) = k4_scale_min(2 * j, scales);
                let (sc2, m2) = k4_scale_min(2 * j + 1, scales);
                let q4v = _mm256_loadu_si256(blk.add(16 + 32 * j) as *const __m256i);
                let lo = _mm256_and_si256(q4v, lown);
                let hi = _mm256_and_si256(_mm256_srl_epi16(q4v, _mm_cvtsi32_si128(4)), lown);
                let q8a = _mm256_loadu_si256(q8.add(64 * j) as *const __m256i);
                let q8b = _mm256_loadu_si256(q8.add(64 * j + 32) as *const __m256i);
                let d1 = _mm256_madd_epi16(_mm256_maddubs_epi16(lo, q8a), _mm256_set1_epi16(1));
                let d2 = _mm256_madd_epi16(_mm256_maddubs_epi16(hi, q8b), _mm256_set1_epi16(1));
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(d1, _mm256_set1_epi32(sc1)));
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(d2, _mm256_set1_epi32(sc2)));
                msum += m1 * (bs[4 * j] + bs[4 * j + 1]) + m2 * (bs[4 * j + 2] + bs[4 * j + 3]);
            }
            let s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b0100_1110));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b1011_0001));
            let isum = _mm_cvtsi128_si32(s);
            total += xd * *x.d.get_unchecked(ibl) * isum as f32
                - xmin * *x.d.get_unchecked(ibl) * msum as f32;
        }
        total
    }

    /// q2_K: per (k-half, shift) one 32-byte unpack ((v >> 2s) & 3 via
    /// 16-bit shifts, the cross-byte bits die on the mask), maddubs with
    /// q8, madd with the two 16-group scales split across lanes.
    #[target_feature(enable = "avx2")]
    pub unsafe fn vec_dot_q2k(row: &[u8], x: &Q8KRow, n: usize) -> f32 {
        let nb = n / QK_K;
        debug_assert!(row.len() >= nb * Q2_K_BLOCK_BYTES);
        let low2 = _mm256_set1_epi8(3);
        let mut total = 0f32;
        for ibl in 0..nb {
            let blk = row.as_ptr().add(ibl * Q2_K_BLOCK_BYTES);
            let sc = std::slice::from_raw_parts(blk, 16);
            let xd = f16_to_f32(u16::from_le_bytes([*blk.add(80), *blk.add(81)]));
            let xmin = f16_to_f32(u16::from_le_bytes([*blk.add(82), *blk.add(83)]));
            let q8 = x.qs.as_ptr().add(ibl * QK_K);
            let bs = &x.bsums[ibl * 16..(ibl + 1) * 16];
            let mut summs = 0i32;
            for j in 0..16 {
                summs += bs[j] * (sc[j] >> 4) as i32;
            }
            let mut acc = _mm256_setzero_si256();
            let mut is = 0;
            for k in 0..2 {
                let q2v = _mm256_loadu_si256(blk.add(16 + 32 * k) as *const __m256i);
                for shift in 0..4i32 {
                    let q = _mm256_and_si256(
                        _mm256_srl_epi16(q2v, _mm_cvtsi32_si128(2 * shift)),
                        low2,
                    );
                    let q8v =
                        _mm256_loadu_si256(q8.add(128 * k + 32 * shift as usize) as *const __m256i);
                    let d16 = _mm256_maddubs_epi16(q, q8v);
                    let scv = _mm256_set_m128i(
                        _mm_set1_epi16((sc[is + 1] & 0x0f) as i16),
                        _mm_set1_epi16((sc[is] & 0x0f) as i16),
                    );
                    is += 2;
                    acc = _mm256_add_epi32(acc, _mm256_madd_epi16(d16, scv));
                }
            }
            let s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b0100_1110));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b1011_0001));
            let isum = _mm_cvtsi128_si32(s);
            total += *x.d.get_unchecked(ibl) * xd * isum as f32
                - *x.d.get_unchecked(ibl) * xmin * summs as f32;
        }
        total
    }

    #[target_feature(enable = "avx2")]
    pub unsafe fn vec_dot(row: &[u8], x: &Q8KRow, n: usize) -> f32 {
        let t = tables();
        let nb = n / QK_K;
        debug_assert!(row.len() >= nb * IQ2_XXS_BLOCK_BYTES);
        let ones = _mm256_set1_epi16(1);
        let mut total = 0f32;
        for ibl in 0..nb {
            let blk = row.as_ptr().add(ibl * IQ2_XXS_BLOCK_BYTES);
            let xd = f16_to_f32(u16::from_le_bytes([*blk, *blk.add(1)]));
            let q8 = x.qs.as_ptr().add(ibl * QK_K);
            let mut acc = _mm256_setzero_si256();
            for ib32 in 0..QK_K / 32 {
                let p = blk.add(2 + 8 * ib32);
                let aux0 = (p as *const u32).read_unaligned();
                let aux1 = (p.add(4) as *const u32).read_unaligned();
                let ls = (2 * (aux1 >> 28) + 1) as i32;
                let grid = _mm256_set_epi64x(
                    t.grid[((aux0 >> 24) & 0xff) as usize] as i64,
                    t.grid[((aux0 >> 16) & 0xff) as usize] as i64,
                    t.grid[((aux0 >> 8) & 0xff) as usize] as i64,
                    t.grid[(aux0 & 0xff) as usize] as i64,
                );
                let signs = _mm256_set_epi64x(
                    KSIGNS64[((aux1 >> 21) & 127) as usize] as i64,
                    KSIGNS64[((aux1 >> 14) & 127) as usize] as i64,
                    KSIGNS64[((aux1 >> 7) & 127) as usize] as i64,
                    KSIGNS64[(aux1 & 127) as usize] as i64,
                );
                let q8v = _mm256_loadu_si256(q8.add(32 * ib32) as *const __m256i);
                let d16 = _mm256_maddubs_epi16(grid, _mm256_sign_epi8(q8v, signs));
                let d32 = _mm256_madd_epi16(d16, ones);
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(d32, _mm256_set1_epi32(ls)));
            }
            let s = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b0100_1110));
            let s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0b1011_0001));
            let bsum = _mm_cvtsi128_si32(s);
            total += 0.125 * xd * *x.d.get_unchecked(ibl) * bsum as f32;
        }
        total
    }
}

/// q2_K: 84 bytes per 256 values - scales[16] (lo nibble = scale, hi =
/// min), qs[64] (2-bit, unsigned 0..3), f16 d + f16 dmin. Mirrors
/// dev_dot_q2_K_q8_K_block in pulsar_kernels.cu: dall*isum - dmin*summs
/// with summs = sum(bsums[j] * (sc[j]>>4)).
pub const Q2_K_BLOCK_BYTES: usize = 16 + 64 + 2 + 2;

pub fn vec_dot_q2_k_q8_k(row: &[u8], x: &Q8KRow, n: usize) -> f32 {
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("avx2") {
        return unsafe { avx2::vec_dot_q2k(row, x, n) };
    }
    vec_dot_q2_k_q8_k_scalar(row, x, n)
}

/// Scalar reference/fallback.
pub fn vec_dot_q2_k_q8_k_scalar(row: &[u8], x: &Q8KRow, n: usize) -> f32 {
    debug_assert_eq!(n % QK_K, 0);
    let nb = n / QK_K;
    debug_assert!(row.len() >= nb * Q2_K_BLOCK_BYTES);
    let mut total = 0f32;
    for ibl in 0..nb {
        let blk = &row[ibl * Q2_K_BLOCK_BYTES..(ibl + 1) * Q2_K_BLOCK_BYTES];
        let (sc, q2) = (&blk[..16], &blk[16..80]);
        let xd = f16_to_f32(u16::from_le_bytes([blk[80], blk[81]]));
        let xmin = f16_to_f32(u16::from_le_bytes([blk[82], blk[83]]));
        let q8 = &x.qs[ibl * QK_K..(ibl + 1) * QK_K];
        let bs = &x.bsums[ibl * 16..(ibl + 1) * 16];
        let mut summs = 0i32;
        for j in 0..16 {
            summs += bs[j] * (sc[j] >> 4) as i32;
        }
        let mut isum = 0i32;
        let mut is = 0;
        for k in 0..2 {
            for shift in [0u8, 2, 4, 6] {
                for half in 0..2 {
                    let d = (sc[is] & 0x0f) as i32;
                    is += 1;
                    let mut sumi = 0i32;
                    for i in 0..16 {
                        let q = ((q2[32 * k + 16 * half + i] >> shift) & 3) as i32;
                        sumi += q * q8[128 * k + 32 * (shift as usize / 2) + 16 * half + i] as i32;
                    }
                    isum += d * sumi;
                }
            }
        }
        total += x.d[ibl] * xd * isum as f32 - x.d[ibl] * xmin * summs as f32;
    }
    total
}

/// Scalar dequant of an iq2_xxs row to f32 (unit-test reference only).
pub fn dequant_row_iq2_xxs(row: &[u8], n: usize, out: &mut Vec<f32>) {
    let t = tables();
    out.clear();
    for ibl in 0..n / QK_K {
        let blk = &row[ibl * IQ2_XXS_BLOCK_BYTES..(ibl + 1) * IQ2_XXS_BLOCK_BYTES];
        let xd = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]]));
        for ib32 in 0..QK_K / 32 {
            let aux0 = u32::from_le_bytes([blk[2 + 8 * ib32], blk[3 + 8 * ib32], blk[4 + 8 * ib32], blk[5 + 8 * ib32]]);
            let aux1 = u32::from_le_bytes([blk[6 + 8 * ib32], blk[7 + 8 * ib32], blk[8 + 8 * ib32], blk[9 + 8 * ib32]]);
            let db = 0.125 * xd * (2 * (aux1 >> 28) + 1) as f32;
            for k in 0..4 {
                let g = t.grid[((aux0 >> (8 * k)) & 0xff) as usize].to_le_bytes();
                let sm = sign_mask((aux1 >> (7 * k)) & 127);
                for i in 0..8 {
                    let v = db * g[i] as i8 as f32;
                    out.push(if (sm >> i) & 1 == 1 { -v } else { v });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(state: &mut u64) -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    #[test]
    fn dot_matches_dequant_reference() {
        let n = 2048;
        let mut st = 42u64;
        let src: Vec<f32> = (0..n).map(|_| lcg(&mut st)).collect();
        let act: Vec<f32> = (0..n).map(|_| lcg(&mut st)).collect();
        let ones = vec![1f32; n];
        let mut row = Vec::new();
        crate::iq::quantize_row_iq2_xxs(&src, &ones, &mut row);

        let xq = quantize_row_q8_k(&act);
        let got = vec_dot_iq2_xxs_q8_k(&row, &xq, n);

        // reference: dequantized weights x dequantized activations in f64
        let mut deq = Vec::new();
        dequant_row_iq2_xxs(&row, n, &mut deq);
        let mut reference = 0f64;
        for i in 0..n {
            let a = xq.d[i / QK_K] as f64 * xq.qs[i] as f64;
            reference += deq[i] as f64 * a;
        }
        let rel = ((got as f64 - reference) / reference.abs().max(1e-6)).abs();
        assert!(rel < 1e-4, "dot {got} vs reference {reference} (rel {rel})");
        // and the quantization itself must be sane vs the true dot
        let true_dot: f64 = src.iter().zip(&act).map(|(&a, &b)| a as f64 * b as f64).sum();
        assert!(reference.signum() == true_dot.signum() || true_dot.abs() < 1.0);
    }

    /// q2_K dequant reference straight from the format: value chunk c
    /// (16 values) uses scale sc[c], q2 byte 32*(c/8) + 16*(c%2), shift
    /// 2*((c%8)/2).
    fn dequant_q2_k(row: &[u8], n: usize, out: &mut Vec<f32>) {
        out.clear();
        for ibl in 0..n / QK_K {
            let blk = &row[ibl * Q2_K_BLOCK_BYTES..(ibl + 1) * Q2_K_BLOCK_BYTES];
            let (sc, q2) = (&blk[..16], &blk[16..80]);
            let xd = f16_to_f32(u16::from_le_bytes([blk[80], blk[81]]));
            let xmin = f16_to_f32(u16::from_le_bytes([blk[82], blk[83]]));
            for c in 0..16 {
                let (k, js, half) = (c / 8, (c % 8) / 2, c % 2);
                for i in 0..16 {
                    let q = (q2[32 * k + 16 * half + i] >> (2 * js)) & 3;
                    out.push(
                        xd * (sc[c] & 0x0f) as f32 * q as f32 - xmin * (sc[c] >> 4) as f32,
                    );
                }
            }
        }
    }

    #[test]
    fn q2k_dot_matches_dequant_and_scalar() {
        let n = 5120;
        let mut st = 99u64;
        // random bytes are a valid q2_K row; cap the f16 d/dmin exponents
        // so the reference stays finite
        let nb = n / QK_K;
        let mut row = vec![0u8; nb * Q2_K_BLOCK_BYTES];
        for b in row.iter_mut() {
            *b = (lcg(&mut st) * 127.0 + 128.0) as u8;
        }
        for ibl in 0..nb {
            let o = ibl * Q2_K_BLOCK_BYTES;
            row[o + 81] &= 0x3b; // d exponent small
            row[o + 83] &= 0x3b; // dmin exponent small
        }
        let act: Vec<f32> = (0..n).map(|_| lcg(&mut st)).collect();
        let xq = quantize_row_q8_k(&act);
        let got = vec_dot_q2_k_q8_k_scalar(&row, &xq, n);
        let mut deq = Vec::new();
        dequant_q2_k(&row, n, &mut deq);
        let mut reference = 0f64;
        for i in 0..n {
            reference += deq[i] as f64 * (xq.d[i / QK_K] as f64 * xq.qs[i] as f64);
        }
        let rel = ((got as f64 - reference) / reference.abs().max(1e-6)).abs();
        assert!(rel < 1e-4, "q2k dot {got} vs reference {reference} (rel {rel})");
        let simd = vec_dot_q2_k_q8_k(&row, &xq, n);
        assert_eq!(simd.to_bits(), got.to_bits(), "q2k simd {simd} vs scalar {got}");
    }

    #[test]
    fn q3k_dot_matches_dequant_and_scalar() {
        let n = 5120;
        let mut st = 7331u64;
        let nb = n / QK_K;
        let mut row = vec![0u8; nb * Q3_K_BLOCK_BYTES];
        for b in row.iter_mut() {
            *b = (lcg(&mut st) * 127.0 + 128.0) as u8;
        }
        for ibl in 0..nb {
            row[ibl * Q3_K_BLOCK_BYTES + 109] &= 0x3b; // keep f16 d finite/small
        }
        let act: Vec<f32> = (0..n).map(|_| lcg(&mut st)).collect();
        let xq = quantize_row_q8_k(&act);
        let got = vec_dot_q3_k_q8_k_scalar(&row, &xq, n);
        // dequant reference straight from the format
        let mut reference = 0f64;
        for ibl in 0..nb {
            let blk = &row[ibl * Q3_K_BLOCK_BYTES..(ibl + 1) * Q3_K_BLOCK_BYTES];
            let (hm, q3) = (&blk[..32], &blk[32..96]);
            let xd = f16_to_f32(u16::from_le_bytes([blk[108], blk[109]])) as f64;
            let sc = k3_scales(&blk[96..108]);
            for c in 0..16 {
                let (k, js, half) = (c / 8, (c % 8) / 2, c % 2);
                for i in 0..16 {
                    let l = half * 16 + i;
                    let mut q = ((q3[32 * k + l] >> (2 * js)) & 3) as i32;
                    if hm[l] & (1 << (4 * k + js)) == 0 {
                        q -= 4;
                    }
                    let v = xd * sc[c] as f64 * q as f64;
                    let idx = ibl * QK_K + 16 * c + i;
                    reference += v * (xq.d[idx / QK_K] as f64 * xq.qs[idx] as f64);
                }
            }
        }
        let rel = ((got as f64 - reference) / reference.abs().max(1e-6)).abs();
        assert!(rel < 1e-4, "q3k dot {got} vs reference {reference} (rel {rel})");
        let simd = vec_dot_q3_k_q8_k(&row, &xq, n);
        assert_eq!(simd.to_bits(), got.to_bits(), "q3k simd {simd} vs scalar {got}");
    }

    #[test]
    fn q4k_dot_matches_dequant_and_scalar() {
        let n = 5120;
        let mut st = 424242u64;
        let nb = n / QK_K;
        let mut row = vec![0u8; nb * Q4_K_BLOCK_BYTES];
        for b in row.iter_mut() {
            *b = (lcg(&mut st) * 127.0 + 128.0) as u8;
        }
        for ibl in 0..nb {
            row[ibl * Q4_K_BLOCK_BYTES + 1] &= 0x3b;
            row[ibl * Q4_K_BLOCK_BYTES + 3] &= 0x3b;
        }
        let act: Vec<f32> = (0..n).map(|_| lcg(&mut st)).collect();
        let xq = quantize_row_q8_k(&act);
        let got = vec_dot_q4_k_q8_k_scalar(&row, &xq, n);
        let mut reference = 0f64;
        for ibl in 0..nb {
            let blk = &row[ibl * Q4_K_BLOCK_BYTES..(ibl + 1) * Q4_K_BLOCK_BYTES];
            let xd = f16_to_f32(u16::from_le_bytes([blk[0], blk[1]])) as f64;
            let xmin = f16_to_f32(u16::from_le_bytes([blk[2], blk[3]])) as f64;
            let (scales, q4) = (&blk[4..16], &blk[16..144]);
            for j in 0..4 {
                let (sc1, m1) = k4_scale_min(2 * j, scales);
                let (sc2, m2) = k4_scale_min(2 * j + 1, scales);
                for i in 0..32 {
                    let v = q4[32 * j + i];
                    for (q, sc, m, idx) in [
                        ((v & 0x0f) as f64, sc1, m1, ibl * QK_K + 64 * j + i),
                        ((v >> 4) as f64, sc2, m2, ibl * QK_K + 64 * j + 32 + i),
                    ] {
                        let val = xd * sc as f64 * q - xmin * m as f64;
                        reference += val * (xq.d[idx / QK_K] as f64 * xq.qs[idx] as f64);
                    }
                }
            }
        }
        let rel = ((got as f64 - reference) / reference.abs().max(1e-6)).abs();
        assert!(rel < 1e-4, "q4k dot {got} vs reference {reference} (rel {rel})");
        let simd = vec_dot_q4_k_q8_k(&row, &xq, n);
        assert_eq!(simd.to_bits(), got.to_bits(), "q4k simd {simd} vs scalar {got}");
    }

    /// bsum is exact integer math in both paths and the float ops run in
    /// the same order, so SIMD must match scalar bit for bit.
    #[test]
    fn simd_matches_scalar_bitwise() {
        let n = 5120;
        let mut st = 7u64;
        let ones = vec![1f32; n];
        for trial in 0..8 {
            let src: Vec<f32> = (0..n).map(|_| lcg(&mut st) * (trial + 1) as f32).collect();
            let act: Vec<f32> = (0..n).map(|_| lcg(&mut st)).collect();
            let mut row = Vec::new();
            crate::iq::quantize_row_iq2_xxs(&src, &ones, &mut row);
            let xq = quantize_row_q8_k(&act);
            let a = vec_dot_iq2_xxs_q8_k(&row, &xq, n);
            let b = vec_dot_iq2_xxs_q8_k_scalar(&row, &xq, n);
            assert_eq!(a.to_bits(), b.to_bits(), "trial {trial}: {a} vs {b}");
        }
    }
}
