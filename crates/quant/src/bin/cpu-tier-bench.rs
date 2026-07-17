//! Go/no-go microbench for the CPU expert tier: sustained iq2_xxs x q8_K
//! GEMV throughput across threads, working set >> L3 so reads come from
//! RAM like real host-cache hits. Target: >40GB/s aggregate on the
//! 9900X (PCIe H2D measures 28.7GB/s; below that the tier loses).
//!
//!   cpu-tier-bench [threads] [gb] [passes]

use quant::cpu_dot::{quantize_row_q8_k, vec_dot_iq2_xxs_q8_k, IQ2_XXS_BLOCK_BYTES, QK_K};

fn main() {
    let mut args = std::env::args().skip(1);
    let threads: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(
        std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8),
    );
    let gb: f64 = args.next().and_then(|a| a.parse().ok()).unwrap_or(2.0);
    let passes: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(1);

    // GLM-shaped rows: n_embd 5120 columns
    let n_cols = 5120usize;
    let row_bytes = n_cols / QK_K * IQ2_XXS_BLOCK_BYTES;
    let n_rows = ((gb * 1e9) as usize / row_bytes).max(threads);
    let total_bytes = n_rows * row_bytes;
    eprintln!(
        "cpu-tier-bench: {} rows x {} cols ({:.2} GB working set), {} threads",
        n_rows, n_cols, total_bytes as f64 / 1e9, threads
    );

    // decode never validates encoder invariants, so random bytes are a
    // faithful load (grid/sign lookups + int math all still happen)
    let mut state = 0x9e3779b97f4a7c15u64;
    let mut weights = vec![0u8; total_bytes];
    for b in weights.chunks_exact_mut(8) {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        b.copy_from_slice(&state.to_le_bytes());
    }
    // keep stored 4-bit scales small so f16 d stays sane: not needed for
    // speed, but keeps outputs finite for the checksum
    let act: Vec<f32> = (0..n_cols).map(|i| ((i % 97) as f32 - 48.0) / 48.0).collect();
    let xq = quantize_row_q8_k(&act);

    let t0 = std::time::Instant::now();
    let rows_per = n_rows.div_ceil(threads);
    let mut sink = 0f64;
    std::thread::scope(|s| {
        let mut handles = Vec::new();
        for c in 0..threads {
            let lo = c * rows_per;
            if lo >= n_rows {
                break;
            }
            let hi = ((c + 1) * rows_per).min(n_rows);
            let weights = &weights;
            let xq = &xq;
            handles.push(s.spawn(move || {
                let mut acc = 0f64;
                for _ in 0..passes {
                    for r in lo..hi {
                        let row = &weights[r * row_bytes..(r + 1) * row_bytes];
                        acc += vec_dot_iq2_xxs_q8_k(row, xq, n_cols) as f64;
                    }
                }
                acc
            }));
        }
        for h in handles {
            sink += h.join().unwrap();
        }
    });
    let dt = t0.elapsed().as_secs_f64();
    let total_bytes = total_bytes * passes;
    eprintln!(
        "cpu-tier-bench: {:.2} GB in {:.3}s = {:.1} GB/s aggregate ({:.2} GB/s/thread), checksum {:e}",
        total_bytes as f64 / 1e9,
        dt,
        total_bytes as f64 / 1e9 / dt,
        total_bytes as f64 / 1e9 / dt / threads as f64,
        sink
    );
}
