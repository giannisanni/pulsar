//! Bandwidth-adaptive hybrid miss handling (FreeToken \(q^\star\)).
//!
//! Pure policy: no CUDA, no engine state. Decode classifies host-cached
//! expert misses into a GPU cache-fill set and a CPU-execute set.
//!
//! \(q^\star \approx m \cdot B_{\mathrm{PCIe}} / B_{\mathrm{host}}\), rounded,
//! always at least one fill when \(m > 0\) so the GPU cache keeps warming.
//! Fill-set membership is recency-ranked: the most recently routed misses
//! are the ones worth a cache slot.

/// Default PCIe H2D from the reference box (Gen5 x8, measured).
pub const DEFAULT_PCIE_GBS: f64 = 28.7;
/// Default host expert-dot bandwidth from the reference box (9900X AVX2).
pub const DEFAULT_HOST_GBS: f64 = 42.0;

/// Split of one layer's host-cached misses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MissSplit {
    /// Misses to H2D into the GPU expert cache (and leave resident).
    pub fill: Vec<i32>,
    /// Misses to compute on the CPU lane (residency unchanged).
    pub cpu: Vec<i32>,
}

/// How many of `n_miss` host-cached misses should fill the GPU cache.
///
/// When host bandwidth cannot exceed the link, residual CPU bandwidth is
/// zero and every miss fills. Otherwise \(q = \mathrm{round}(m \cdot
/// B_{\mathrm{PCIe}} / B_{\mathrm{host}})\), clamped to `[1, m]`.
pub fn q_star(n_miss: usize, pcie_gbs: f64, host_gbs: f64) -> usize {
    if n_miss == 0 {
        return 0;
    }
    if !(pcie_gbs.is_finite() && pcie_gbs > 0.0) {
        return n_miss;
    }
    if !(host_gbs.is_finite() && host_gbs > pcie_gbs) {
        return n_miss;
    }
    let q = (n_miss as f64 * pcie_gbs / host_gbs).round() as usize;
    q.clamp(1, n_miss)
}

/// Rank `misses` (expert id, recency clock) and take the `q` most recent
/// as the fill set. Recency `0` means never seen: those sort last so a
/// cold miss does not evict a warm slot. Equal recency keeps input order.
pub fn split_by_recency(misses: &[(i32, u64)], q: usize) -> MissSplit {
    if misses.is_empty() {
        return MissSplit { fill: Vec::new(), cpu: Vec::new() };
    }
    let mut order: Vec<usize> = (0..misses.len()).collect();
    order.sort_by(|&a, &b| {
        misses[b]
            .1
            .cmp(&misses[a].1)
            .then(a.cmp(&b))
    });
    let q = q.min(misses.len()).max(1);
    MissSplit {
        fill: order[..q].iter().map(|&i| misses[i].0).collect(),
        cpu: order[q..].iter().map(|&i| misses[i].0).collect(),
    }
}

/// `PULSAR_PCIE_GBS` / `PULSAR_HOST_GBS`, falling back to the measured
/// reference-box pair.
pub fn bandwidths_from_env() -> (f64, f64) {
    let parse = |key: &str, default: f64| {
        std::env::var(key)
            .ok()
            .and_then(|v| v.parse::<f64>().ok())
            .filter(|v| *v > 0.0)
            .unwrap_or(default)
    };
    (parse("PULSAR_PCIE_GBS", DEFAULT_PCIE_GBS), parse("PULSAR_HOST_GBS", DEFAULT_HOST_GBS))
}

/// Host-cached experts, including VRAM hits, go to the CPU lane unless
/// the operator opts out. Default is on: on links slower than host DRAM
/// (this 3090 measured 13 GB/s H2D vs ~42 GB/s CPU) the overlap beats
/// keeping hits on the GPU. `PULSAR_CPU_STEAL=0` keeps hits on the GPU
/// and enables the q* miss split.
pub fn cpu_steal_from_env() -> bool {
    match std::env::var("PULSAR_CPU_STEAL").ok().as_deref() {
        Some("0") | Some("off") | Some("false") => false,
        _ => true,
    }
}

/// Plan fill vs CPU for host-cached misses. `PULSAR_NO_HYBRID=1` restores
/// the old "every host-cached miss goes to the CPU lane" split.
pub fn plan_host_misses(misses: &[(i32, u64)]) -> MissSplit {
    if misses.is_empty() {
        return MissSplit { fill: Vec::new(), cpu: Vec::new() };
    }
    if std::env::var_os("PULSAR_NO_HYBRID").is_some() {
        return MissSplit {
            fill: Vec::new(),
            cpu: misses.iter().map(|(id, _)| *id).collect(),
        };
    }
    let (pcie, host) = bandwidths_from_env();
    plan_host_misses_with(misses, pcie, host)
}

/// Same as [`plan_host_misses`] but with caller-supplied bandwidths (the
/// engine passes the H2D probe from `ensure_device`).
pub fn plan_host_misses_with(misses: &[(i32, u64)], pcie_gbs: f64, host_gbs: f64) -> MissSplit {
    if misses.is_empty() {
        return MissSplit { fill: Vec::new(), cpu: Vec::new() };
    }
    if std::env::var_os("PULSAR_NO_HYBRID").is_some() {
        return MissSplit {
            fill: Vec::new(),
            cpu: misses.iter().map(|(id, _)| *id).collect(),
        };
    }
    let q = q_star(misses.len(), pcie_gbs, host_gbs);
    split_by_recency(misses, q)
}

/// Least-recently-used slot indices among `slots` (offset, last_tick),
/// skipping offsets in `in_use`. Returns up to `need` victims, oldest
/// first. One pass: the caller frees whole groups from these slots.
pub fn lru_victims(slots: &[(u64, u64)], in_use: &[u64], need: usize) -> Vec<usize> {
    if need == 0 || slots.is_empty() {
        return Vec::new();
    }
    let mut cands: Vec<(usize, u64)> = slots
        .iter()
        .enumerate()
        .filter(|(_, (off, _))| !in_use.contains(off))
        .map(|(i, (_, tick))| (i, *tick))
        .collect();
    cands.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
    cands.truncate(need);
    cands.into_iter().map(|(i, _)| i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q_star_zero_misses_is_zero() {
        assert_eq!(q_star(0, 28.7, 42.0), 0);
    }

    #[test]
    fn q_star_always_fills_at_least_one() {
        assert_eq!(q_star(1, 28.7, 42.0), 1);
        assert!(q_star(2, 1.0, 100.0) >= 1);
    }

    #[test]
    fn q_star_host_not_above_pcie_fills_all() {
        assert_eq!(q_star(6, 30.0, 20.0), 6);
        assert_eq!(q_star(6, 30.0, 30.0), 6);
    }

    #[test]
    fn q_star_reference_box_is_pcie_over_host() {
        // 6 * 28.7 / 42 = 4.1 → 4
        assert_eq!(q_star(6, DEFAULT_PCIE_GBS, DEFAULT_HOST_GBS), 4);
        // 3 * 28.7 / 42 = 2.05 → 2
        assert_eq!(q_star(3, DEFAULT_PCIE_GBS, DEFAULT_HOST_GBS), 2);
    }

    #[test]
    fn split_empty() {
        let s = split_by_recency(&[], 1);
        assert!(s.fill.is_empty() && s.cpu.is_empty());
    }

    #[test]
    fn split_fills_most_recent_and_cpus_the_rest() {
        let misses = [(10, 1u64), (20, 9), (30, 3), (40, 7)];
        let s = split_by_recency(&misses, 2);
        assert_eq!(s.fill, vec![20, 40]);
        assert_eq!(s.cpu, vec![30, 10]);
    }

    #[test]
    fn split_never_seen_sorts_last() {
        let misses = [(1, 0u64), (2, 5), (3, 0)];
        let s = split_by_recency(&misses, 1);
        assert_eq!(s.fill, vec![2]);
        assert_eq!(s.cpu, vec![1, 3]);
    }

    #[test]
    fn split_q_greater_than_m_fills_all() {
        let misses = [(1, 1u64), (2, 2)];
        let s = split_by_recency(&misses, 99);
        assert_eq!(s.fill, vec![2, 1]);
        assert!(s.cpu.is_empty());
    }

    #[test]
    fn split_equal_recency_keeps_input_order() {
        let misses = [(1, 4u64), (2, 4), (3, 4)];
        let s = split_by_recency(&misses, 2);
        assert_eq!(s.fill, vec![1, 2]);
        assert_eq!(s.cpu, vec![3]);
    }

    #[test]
    fn lru_victims_skips_in_use_and_returns_oldest() {
        // slot 0 off=10 tick=5, slot 1 off=20 tick=1, slot 2 off=30 tick=9
        let slots = [(10u64, 5u64), (20, 1), (30, 9)];
        let v = lru_victims(&slots, &[10], 2);
        assert_eq!(v, vec![1, 2]); // 20 (tick 1) then 30 (tick 9); 10 is in-use
    }

    #[test]
    fn lru_victims_need_zero_or_empty() {
        assert!(lru_victims(&[(1, 1)], &[], 0).is_empty());
        assert!(lru_victims(&[], &[], 3).is_empty());
    }

    #[test]
    fn q_star_slow_pcie_fills_less() {
        // AISERVER 3090: 13.1 GB/s H2D vs 42 GB/s host → 6 * 13.1/42 ≈ 2
        assert_eq!(q_star(6, 13.1, 42.0), 2);
        assert_eq!(q_star(6, 28.7, 42.0), 4);
    }
}
