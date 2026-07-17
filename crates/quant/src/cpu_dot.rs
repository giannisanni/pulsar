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

/// q8_K activation row: one f32 scale + 256 i8 per block.
pub struct Q8KRow {
    pub d: Vec<f32>,
    pub qs: Vec<i8>,
}

/// ggml quantize_row_q8_K: d = amax/127, q = round(x/d).
pub fn quantize_row_q8_k(x: &[f32]) -> Q8KRow {
    debug_assert_eq!(x.len() % QK_K, 0);
    let nb = x.len() / QK_K;
    let mut d = Vec::with_capacity(nb);
    let mut qs = Vec::with_capacity(x.len());
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
    Q8KRow { d, qs }
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
    if std::arch::is_x86_feature_detected!("avx2") {
        return unsafe { avx2::vec_dot(row, x, n) };
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
