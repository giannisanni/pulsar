//! iq2_xxs encoder: port of ggml-quants.c quantize_row_iq2_xxs_impl and
//! iq2xs_init_impl (grid expansion, kmap, neighbour lists). Requires
//! per-column quantization weights (an imatrix); the caller falls back to
//! q2_K when none are available, mirroring llama.cpp's refusal to encode
//! iq2_xxs without one.
//!
//! Layout notes (same block the CUDA kernels read): 256 elems -> 66 bytes,
//! f16 d + 32 u16. Per 32-elem sub-block: 4 u16 = two u32s; low u32 holds
//! four 8-bit grid indices, high u32 holds 4x7-bit sign masks (parity bit
//! reconstructed) plus a 4-bit scale in the top nibble.

use std::sync::OnceLock;

pub const QK_K: usize = 256;

#[rustfmt::skip]
const KGRID_2BIT_256: [u16; 256] = [
        0,     2,     5,     8,    10,    17,    20,    32,    34,    40,    42,    65,    68,    80,    88,    97,
      100,   128,   130,   138,   162,   257,   260,   272,   277,   320,   388,   408,   512,   514,   546,   642,
     1025,  1028,  1040,  1057,  1060,  1088,  1090,  1096,  1120,  1153,  1156,  1168,  1188,  1280,  1282,  1288,
     1312,  1350,  1385,  1408,  1425,  1545,  1552,  1600,  1668,  1700,  2048,  2053,  2056,  2068,  2088,  2113,
     2116,  2128,  2130,  2184,  2308,  2368,  2562,  2580,  4097,  4100,  4112,  4129,  4160,  4192,  4228,  4240,
     4245,  4352,  4360,  4384,  4432,  4442,  4480,  4644,  4677,  5120,  5128,  5152,  5157,  5193,  5248,  5400,
     5474,  5632,  5654,  6145,  6148,  6160,  6208,  6273,  6400,  6405,  6560,  6737,  8192,  8194,  8202,  8260,
     8289,  8320,  8322,  8489,  8520,  8704,  8706,  9217,  9220,  9232,  9280,  9302,  9472,  9537,  9572,  9872,
    10248, 10272, 10388, 10820, 16385, 16388, 16400, 16408, 16417, 16420, 16448, 16456, 16470, 16480, 16513, 16516,
    16528, 16640, 16672, 16737, 16768, 16773, 16897, 16912, 16968, 16982, 17000, 17408, 17416, 17440, 17536, 17561,
    17682, 17700, 17920, 18433, 18436, 18448, 18496, 18501, 18688, 18776, 18785, 18818, 19013, 19088, 20480, 20488,
    20497, 20505, 20512, 20608, 20616, 20740, 20802, 20900, 21137, 21648, 21650, 21770, 22017, 22100, 22528, 22545,
    22553, 22628, 22848, 23048, 24580, 24592, 24640, 24680, 24832, 24917, 25112, 25184, 25600, 25605, 25872, 25874,
    25988, 26690, 32768, 32770, 32778, 32833, 32898, 33028, 33048, 33088, 33297, 33793, 33796, 33808, 33813, 33856,
    33888, 34048, 34118, 34196, 34313, 34368, 34400, 34818, 35076, 35345, 36868, 36880, 36900, 36928, 37025, 37142,
    37248, 37445, 37888, 37922, 37956, 38225, 39041, 39200, 40962, 41040, 41093, 41225, 41472, 42008, 43088, 43268,
];

const KMAP_SIZE: usize = 43692;
const NWANT: usize = 2;

pub struct Iq2Tables {
    /// 256 grid points, 8 x int8 each with values 2l+1 (encoder units).
    pub grid: Vec<u64>,
    /// packed-L index -> grid index, or -(neighbour offset + 1).
    pub kmap: Vec<i32>,
    /// [count, idx...] slices addressed via kmap.
    pub neighbours: Vec<u16>,
}

fn grid_bytes(g: u64) -> [i8; 8] {
    g.to_le_bytes().map(|b| b as i8)
}

pub fn tables() -> &'static Iq2Tables {
    static T: OnceLock<Iq2Tables> = OnceLock::new();
    T.get_or_init(|| {
        // expand: 2 bits per dim -> coordinate 2l+1
        let mut grid = Vec::with_capacity(256);
        for &k in &KGRID_2BIT_256 {
            let mut bytes = [0u8; 8];
            for (i, b) in bytes.iter_mut().enumerate() {
                let l = (k >> (2 * i)) & 0x3;
                *b = (2 * l + 1) as u8;
            }
            grid.push(u64::from_le_bytes(bytes));
        }
        let mut kmap = vec![-1i32; KMAP_SIZE];
        for (i, &g) in grid.iter().enumerate() {
            let mut index = 0u16;
            for (k, b) in g.to_le_bytes().iter().enumerate() {
                let q = ((b - 1) / 2) as u16;
                index |= q << (2 * k);
            }
            kmap[index as usize] = i as i32;
        }
        // neighbour lists: for every off-grid point, the grid points at the
        // NWANT smallest distinct distances, sorted by (d2, index)
        let mut neighbours = Vec::new();
        let mut dist: Vec<(i32, usize)> = Vec::with_capacity(256);
        for i in 0..KMAP_SIZE {
            if kmap[i] >= 0 {
                continue;
            }
            let mut pos = [0i32; 8];
            for (k, p) in pos.iter_mut().enumerate() {
                let l = ((i >> (2 * k)) & 0x3) as i32;
                *p = 2 * l + 1;
            }
            dist.clear();
            for (j, &g) in grid.iter().enumerate() {
                let pg = grid_bytes(g);
                let d2: i32 = pg.iter().zip(&pos).map(|(&a, &b)| (a as i32 - b) * (a as i32 - b)).sum();
                dist.push((d2, j));
            }
            dist.sort_unstable();
            let start = neighbours.len();
            neighbours.push(0u16); // count placeholder
            let mut n = 0u16;
            let mut d2 = dist[0].0;
            let mut nhave = 1;
            for &(d, j) in &dist {
                if d > d2 {
                    if nhave == NWANT {
                        break;
                    }
                    d2 = d;
                    nhave += 1;
                }
                neighbours.push(j as u16);
                n += 1;
            }
            neighbours[start] = n;
            kmap[i] = -((start + 1) as i32);
        }
        Iq2Tables { grid, kmap, neighbours }
    })
}

#[inline]
fn nearest_int(x: f32) -> i32 {
    x.round() as i32
}

/// ggml make_qp_quants: non-negative quants with weighted rmse refinement.
fn make_qp_quants(n: usize, nmax: i32, x: &[f32], ls: &mut [u8], qw: &[f32]) -> f32 {
    let mut max = 0f32;
    for &v in &x[..n] {
        max = max.max(v);
    }
    if max < 1e-30 {
        ls[..n].iter_mut().for_each(|l| *l = 0);
        return 0.0;
    }
    let mut iscale = nmax as f32 / max;
    let mut scale = 1.0 / iscale;
    let mut best_mse = 0f32;
    for i in 0..n {
        let l = nearest_int(iscale * x[i]).min(nmax);
        let diff = x[i] - scale * l as f32;
        best_mse += qw[i] * diff * diff;
    }
    for is in -4..=4i32 {
        if is == 0 {
            continue;
        }
        let iscale_is = (0.1 * is as f32 + nmax as f32) / max;
        let scale_is = 1.0 / iscale_is;
        let mut mse = 0f32;
        for i in 0..n {
            let l = nearest_int(iscale_is * x[i]).min(nmax);
            let diff = x[i] - scale_is * l as f32;
            mse += qw[i] * diff * diff;
        }
        if mse < best_mse {
            best_mse = mse;
            iscale = iscale_is;
        }
    }
    let mut sumlx = 0f32;
    let mut suml2 = 0f32;
    for i in 0..n {
        let l = nearest_int(iscale * x[i]).min(nmax);
        ls[i] = l as u8;
        sumlx += qw[i] * x[i] * l as f32;
        suml2 += qw[i] * (l * l) as f32;
    }
    for _ in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let w = qw[i];
            let slx = sumlx - w * x[i] * ls[i] as f32;
            let sl2 = suml2 - w * (ls[i] as i32 * ls[i] as i32) as f32;
            if slx > 0.0 && sl2 > 0.0 {
                let new_l = nearest_int(x[i] * sl2 / slx).min(nmax);
                if new_l != ls[i] as i32 {
                    let slx2 = slx + w * x[i] * new_l as f32;
                    let sl22 = sl2 + w * (new_l * new_l) as f32;
                    if slx2 * slx2 * suml2 > sumlx * sumlx * sl22 {
                        ls[i] = new_l as u8;
                        sumlx = slx2;
                        suml2 = sl22;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }
    if suml2 > 0.0 {
        sumlx / suml2
    } else {
        0.0
    }
}

fn find_best_neighbour(
    t: &Iq2Tables,
    noff: usize,
    xval: &[f32],
    waux: &[f32],
    scale: f32,
    ls: &mut [i8],
) -> usize {
    let count = t.neighbours[noff] as usize;
    let mut best_d2 = f32::MAX;
    let mut grid_index = usize::MAX;
    for &j in &t.neighbours[noff + 1..noff + 1 + count] {
        let pg = grid_bytes(t.grid[j as usize]);
        let mut d2 = 0f32;
        for i in 0..8 {
            let diff = scale * pg[i] as f32 - xval[i];
            d2 += waux[i] * diff * diff;
        }
        if d2 < best_d2 {
            best_d2 = d2;
            grid_index = j as usize;
        }
    }
    let pg = grid_bytes(t.grid[grid_index]);
    for i in 0..8 {
        ls[i] = (pg[i] - 1) / 2;
    }
    grid_index
}

/// 256 elems + per-column weights -> 66 bytes.
pub fn quantize_row_iq2_xxs(x: &[f32], qw: &[f32], out: &mut Vec<u8>) {
    debug_assert_eq!(x.len() % QK_K, 0);
    debug_assert_eq!(x.len(), qw.len());
    let t = tables();
    const K_MAX_Q: i32 = 3;
    let mut weight = [0f32; 32];
    let mut waux = [0f32; 32];
    let mut xval = [0f32; 32];
    let mut ls = [0i8; 32];
    let mut laux = [0i8; 32];
    let mut block_signs = [0u8; 4];

    for (ibl, xbl) in x.chunks_exact(QK_K).enumerate() {
        let qwbl = &qw[QK_K * ibl..QK_K * (ibl + 1)];
        let mut q2 = [0u32; 2 * (QK_K / 32)];
        let mut scales = [0f32; QK_K / 32];
        let mut max_scale = 0f32;
        let sigma2: f32 = xbl.iter().map(|v| v * v).sum::<f32>() / QK_K as f32;

        for ib in 0..QK_K / 32 {
            let xb = &xbl[32 * ib..32 * (ib + 1)];
            let qwb = &qwbl[32 * ib..32 * (ib + 1)];
            for i in 0..32 {
                weight[i] = qwb[i] * (sigma2 + xb[i] * xb[i]).sqrt();
                waux[i] = weight[i].sqrt();
            }
            for k in 0..4 {
                let mut nflip = 0;
                let mut s = 0u8;
                for i in 0..8 {
                    if xb[8 * k + i] >= 0.0 {
                        xval[8 * k + i] = xb[8 * k + i];
                    } else {
                        xval[8 * k + i] = -xb[8 * k + i];
                        nflip += 1;
                        s |= 1 << i;
                    }
                }
                if nflip % 2 == 1 {
                    let mut imin = 0;
                    let mut min = weight[8 * k] * xb[8 * k] * xb[8 * k];
                    for i in 1..8 {
                        let ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                        if ax < min {
                            min = ax;
                            imin = i;
                        }
                    }
                    xval[8 * k + imin] = -xval[8 * k + imin];
                    s ^= 1 << imin;
                }
                block_signs[k] = s & 127;
            }
            let max = xval.iter().fold(xval[0], |a, &b| a.max(b));
            if max < 1e-30 {
                scales[ib] = 0.0;
                continue;
            }
            let mut lsu8 = [0u8; 32];
            let mut scale = make_qp_quants(32, K_MAX_Q + 1, &xval, &mut lsu8, &weight);
            let eff_max = scale * K_MAX_Q as f32;
            if eff_max <= 0.0 {
                scales[ib] = 0.0;
                continue;
            }
            let mut best = 0f32;
            for is in -6..=6i32 {
                let id = (2.0 * K_MAX_Q as f32 - 1.0 + is as f32 * 0.1) / eff_max;
                let this_scale = 1.0 / id;
                for k in 0..4 {
                    for i in 0..8 {
                        let l = nearest_int(0.5 * (id * xval[8 * k + i] - 1.0)).clamp(0, K_MAX_Q - 1);
                        laux[8 * k + i] = l as i8;
                    }
                    let mut u = 0u16;
                    for i in 0..8 {
                        u |= (laux[8 * k + i] as u16) << (2 * i);
                    }
                    let gi = t.kmap[u as usize];
                    if gi < 0 {
                        find_best_neighbour(
                            t,
                            (-gi - 1) as usize,
                            &xval[8 * k..8 * k + 8],
                            &waux[8 * k..8 * k + 8],
                            this_scale,
                            &mut laux[8 * k..8 * k + 8],
                        );
                    }
                }
                let mut sumqx = 0f32;
                let mut sumq2 = 0f32;
                for i in 0..32 {
                    let q = (2 * laux[i] + 1) as f32;
                    sumqx += weight[i] * xval[i] * q;
                    sumq2 += weight[i] * q * q;
                }
                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    ls.copy_from_slice(&laux);
                }
            }
            if scale > 0.0 {
                let id = 1.0 / scale;
                for k in 0..4 {
                    let mut u = 0u16;
                    for i in 0..8 {
                        let l = nearest_int(0.5 * (id * xval[8 * k + i] - 1.0)).clamp(0, K_MAX_Q - 1);
                        u |= (l as u16) << (2 * i);
                    }
                    let gi = t.kmap[u as usize];
                    let grid_index = if gi < 0 {
                        find_best_neighbour(
                            t,
                            (-gi - 1) as usize,
                            &xval[8 * k..8 * k + 8],
                            &waux[8 * k..8 * k + 8],
                            scale,
                            &mut ls[8 * k..8 * k + 8],
                        )
                    } else {
                        gi as usize
                    };
                    let pg = grid_bytes(t.grid[grid_index]);
                    for i in 0..8 {
                        ls[8 * k + i] = (pg[i] - 1) / 2;
                    }
                }
                let mut sumqx = 0f32;
                let mut sumq2 = 0f32;
                for i in 0..32 {
                    let q = (2 * ls[i] + 1) as f32;
                    sumqx += weight[i] * xval[i] * q;
                    sumq2 += weight[i] * q * q;
                }
                if sumq2 > 0.0 {
                    scale = sumqx / sumq2;
                }
            }
            if scale < 0.0 {
                scale = -scale;
                for k in 0..4 {
                    block_signs[k] = !block_signs[k] & 127;
                }
            }
            for k in 0..4 {
                let mut u = 0u16;
                for i in 0..8 {
                    u |= (ls[8 * k + i] as u16) << (2 * i);
                }
                let gi = t.kmap[u as usize];
                assert!(gi >= 0, "off-grid point after final requant");
                q2[2 * ib] |= (gi as u32) << (8 * k);
                q2[2 * ib + 1] |= (block_signs[k] as u32) << (7 * k);
            }
            assert!(scale >= 0.0);
            scales[ib] = scale;
            max_scale = max_scale.max(scale);
        }

        if max_scale == 0.0 {
            out.extend_from_slice(&crate::f32_to_f16(0.0).to_le_bytes());
            out.extend_from_slice(&[0u8; 64]);
            continue;
        }
        let d = max_scale / 31.0;
        let id = 1.0 / d;
        for ib in 0..QK_K / 32 {
            let l = nearest_int(0.5 * (id * scales[ib] - 1.0)).clamp(0, 15);
            q2[2 * ib + 1] |= (l as u32) << 28;
        }
        out.extend_from_slice(&crate::f32_to_f16(d).to_le_bytes());
        for v in q2 {
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
}

/// llama.cpp legacy imatrix .dat: i32 n_entries, then per entry
/// (i32 name_len, name, i32 ncall, i32 nval, f32 x nval). Values are used
/// as relative weights, so the ncall normalization is irrelevant here.
pub fn read_imatrix(path: &std::path::Path) -> Result<std::collections::HashMap<String, Vec<f32>>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut at = 0usize;
    let mut i32_at = |at: &mut usize| -> Result<i32, String> {
        let v = i32::from_le_bytes(
            bytes.get(*at..*at + 4).ok_or("imatrix truncated")?.try_into().unwrap(),
        );
        *at += 4;
        Ok(v)
    };
    let n = i32_at(&mut at)?;
    if !(0..1_000_000).contains(&n) {
        return Err(format!("implausible imatrix entry count {n} (gguf-format imatrix? use the legacy .dat)"));
    }
    let mut map = std::collections::HashMap::with_capacity(n as usize);
    for _ in 0..n {
        let len = i32_at(&mut at)? as usize;
        let name = String::from_utf8(
            bytes.get(at..at + len).ok_or("imatrix truncated")?.to_vec(),
        )
        .map_err(|_| "imatrix name not utf-8")?;
        at += len;
        let _ncall = i32_at(&mut at)?;
        let nval = i32_at(&mut at)? as usize;
        let mut vals = Vec::with_capacity(nval);
        for _ in 0..nval {
            let v = f32::from_le_bytes(
                bytes.get(at..at + 4).ok_or("imatrix truncated")?.try_into().unwrap(),
            );
            at += 4;
            vals.push(v);
        }
        map.insert(name, vals);
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tables_sane() {
        let t = tables();
        assert_eq!(t.grid.len(), 256);
        assert_eq!(t.kmap.len(), KMAP_SIZE);
        // every packed grid point maps to itself
        let mut on_grid = 0;
        for &m in &t.kmap {
            if m >= 0 {
                on_grid += 1;
            } else {
                let off = (-m - 1) as usize;
                let cnt = t.neighbours[off] as usize;
                assert!(cnt > 0 && off + cnt < t.neighbours.len());
            }
        }
        assert_eq!(on_grid, 256);
        // grid coordinates are odd values 1..7
        for &g in &t.grid {
            for b in g.to_le_bytes() {
                assert!(matches!(b, 1 | 3 | 5 | 7));
            }
        }
    }

    /// Dequant mirrors ggml dequantize_row_iq2_xxs, with the premultiplied
    /// grid parsed out of the repo's CUDA table (ground truth the engine
    /// decodes with) rather than retyped.
    fn cuda_tables() -> (Vec<u64>, Vec<u8>) {
        let src = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../kernels/cuda/iq2_tables.inc"
        ))
        .expect("iq2_tables.inc");
        let grid_part = src
            .split("cuda_iq2xxs_grid[256] = {")
            .nth(1)
            .and_then(|s| s.split('}').next())
            .expect("grid block");
        let grid: Vec<u64> = grid_part
            .split(',')
            .filter_map(|tok| {
                let tok = tok.trim().strip_prefix("0x")?;
                u64::from_str_radix(tok, 16).ok()
            })
            .collect();
        assert_eq!(grid.len(), 256);
        // ksigns: value = index | parity(index) << 7 (the CUDA kernel
        // computes this on the fly; regenerate rather than parse)
        let ksigns: Vec<u8> = (0u8..128).map(|i| i | (((i.count_ones() & 1) as u8) << 7)).collect();
        (grid, ksigns)
    }

    fn dequant_iq2_xxs(raw: &[u8], out: &mut Vec<f32>) {
        let (grid, ksigns) = cuda_tables();
        for b in raw.chunks_exact(66) {
            let d = crate::f16_to_f32(u16::from_le_bytes([b[0], b[1]]));
            for ib in 0..8 {
                let base = 2 + 8 * ib;
                let a0 = u32::from_le_bytes(b[base..base + 4].try_into().unwrap());
                let a1 = u32::from_le_bytes(b[base + 4..base + 8].try_into().unwrap());
                let db = d * (0.5 + (a1 >> 28) as f32) * 0.25;
                for k in 0..4 {
                    let g = grid[((a0 >> (8 * k)) & 255) as usize].to_le_bytes();
                    let signs = ksigns[((a1 >> (7 * k)) & 127) as usize];
                    for i in 0..8 {
                        let s = if signs & (1 << i) != 0 { -1.0 } else { 1.0 };
                        out.push(db * g[i] as f32 * s);
                    }
                }
            }
        }
    }

    #[test]
    fn iq2_xxs_roundtrip() {
        let mut s = 7u64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s as f64 / u64::MAX as f64) as f32 - 0.5
        };
        let x: Vec<f32> = (0..QK_K * 8).map(|_| (0..4).map(|_| next()).sum::<f32>() * 0.5).collect();
        let qw = vec![1.0f32; x.len()];
        let mut enc = Vec::new();
        quantize_row_iq2_xxs(&x, &qw, &mut enc);
        assert_eq!(enc.len(), (x.len() / QK_K) * 66);
        let mut dec = Vec::new();
        dequant_iq2_xxs(&enc, &mut dec);
        let rms: f32 = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
        let err: f32 = (x.iter().zip(&dec).map(|(a, b)| (a - b) * (a - b)).sum::<f32>()
            / x.len() as f32)
            .sqrt();
        let rel = err / rms;
        // 2.06 bpw on gaussian-ish data: expect ~0.3-0.45 relative rmse;
        // anything above 0.6 means the encoder or packing is broken
        assert!(rel < 0.6, "iq2_xxs rel rmse {rel}");
        // and it must beat all-zeros (rel 1.0) decisively
        assert!(rel < 0.9);
    }
}
