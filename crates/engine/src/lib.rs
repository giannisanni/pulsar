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

#[cfg(target_os = "linux")]
mod real {
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
        pub rope_freq_base: f32,
        pub rms_eps: f32,
        // MLA only (zero for Gqa)
        pub n_lora_q: u32,
        pub n_kv_lora: u32,
        pub qk_nope: u32,
        pub qk_rope: u32,
        pub value_mla: u32,
        pub rope_orig_ctx: u32,
        // DSA lightning indexer (zero when absent -> ctx capped at 2048)
        pub n_idx_head: u32,
        pub n_idx_dim: u32,
        pub n_idx_topk: u32,
        // YaRN (deepseek2/Kimi: factor 32, log_mult 0.1; GLM ships 1.0/off)
        pub rope_scale_factor: f32,
        pub rope_yarn_log_mult: f32,
    }

    impl Shape {
        pub fn qk_dim(&self) -> u32 {
            self.qk_nope + self.qk_rope
        }

        /// Attention output width (input of attn_output).
        fn heads_dim(&self) -> u32 {
            match self.family {
                Family::Gqa => self.n_head * self.head_dim,
                Family::Mla => self.n_head * self.value_mla,
            }
        }

        fn rope_cfg(&self) -> kernels::RopeCfg {
            // GLM-5.2 ships yarn off (scale 1.0); deepseek2/Kimi runs real
            // YaRN: freq_scale = 1/factor, attn scaled by the log-mult
            // mscale (llama.cpp deepseek2 convention). NEEDS teacher-forced
            // parity validation on the first Kimi run.
            if self.rope_scale_factor > 1.0 {
                let mscale = 1.0 + self.rope_yarn_log_mult * self.rope_scale_factor.ln();
                kernels::RopeCfg {
                    n_ctx_orig: self.rope_orig_ctx,
                    freq_base: self.rope_freq_base,
                    freq_scale: 1.0 / self.rope_scale_factor,
                    ext_factor: 1.0,
                    attn_factor: mscale,
                    beta_fast: 32.0,
                    beta_slow: 1.0,
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
                }
            }
        }
    }

    impl Shape {
        fn from_gguf(g: &Gguf) -> Result<Shape> {
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
                Some("glm-dsa") | Some("glm_dsa") => Family::Mla,
                // DeepSeek-V3 family (Kimi K2 etc.): plain MLA, no indexer
                Some("deepseek2") => Family::Mla,
                other => return Err(format!("unsupported architecture {other:?}").into()),
            };
            let n_layer = u("block_count")?;
            let nextn = u("nextn_predict_layers").unwrap_or(0);
            let n_vocab = match g.metadata.get("tokenizer.ggml.tokens") {
                Some(Value::Array(a)) => a.len() as u32,
                _ => return Err(meta_err("tokenizer.ggml.tokens")),
            };
            let mut s = Shape {
                family,
                n_embd: u("embedding_length")?,
                n_head: u("attention.head_count")?,
                n_head_kv: u("attention.head_count_kv").unwrap_or(1),
                head_dim: u("attention.key_length")?,
                n_layer,
                n_exec_layer: n_layer - nextn,
                // the ds4-lineage converter writes this KV; upstream
                // llama.cpp (AngelSlim ggufs) omits it - infer it from
                // where routed-expert tensors start
                n_leading_dense: match u("leading_dense_block_count") {
                    Ok(v) => v,
                    Err(_) => (0..u("block_count")?)
                        .find(|il| g.tensor(&format!("blk.{il}.ffn_gate_exps.weight")).is_some())
                        .ok_or_else(|| meta_err("no MoE layers found"))?,
                },
                n_expert: u("expert_count")?,
                n_expert_used: u("expert_used_count")?,
                n_ff_exp: u("expert_feed_forward_length")?,
                n_ff_dense: u("feed_forward_length")?,
                n_vocab,
                expert_weight_scale: f("expert_weights_scale")?,
                rope_freq_base: f("rope.freq_base")?,
                rms_eps: f("attention.layer_norm_rms_epsilon")?,
                n_lora_q: 0,
                n_kv_lora: 0,
                qk_nope: 0,
                qk_rope: 0,
                value_mla: 0,
                rope_orig_ctx: 0,
                n_idx_head: 0,
                n_idx_dim: 0,
                n_idx_topk: 0,
                rope_scale_factor: 1.0,
                rope_yarn_log_mult: 0.0,
            };
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
            Ok(s)
        }
    }

    /// File location of one routed expert tensor: uniform per-expert slabs.
    struct ExpertTensor {
        abs_offset: u64,
        expert_bytes: u64,
        row_bytes: u64,
        quant: u32,
    }

    impl ExpertTensor {
        fn new(g: &Gguf, t: &TensorInfo, n_expert: u32) -> Result<ExpertTensor> {
            let quant = match t.ty {
                TensorType::IQ2XXS => kernels::QUANT_IQ2_XXS,
                TensorType::Q2K => kernels::QUANT_Q2_K,
                TensorType::Q4K => kernels::QUANT_Q4_K,
                TensorType::Q5K => kernels::QUANT_Q5_K,
                TensorType::Q6K => kernels::QUANT_Q6_K,
                TensorType::Q3K => kernels::QUANT_Q3_K,
                other => return Err(format!("{}: unsupported expert type {other:?}", t.name).into()),
            };
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

    enum Ffn {
        Dense {
            gate: DeviceBuf,
            up: DeviceBuf,
            down: DeviceBuf,
        },
        Moe {
            gate_inp: DeviceBuf,
            probs_b: DeviceBuf,
            shexp_gate: DeviceBuf,
            shexp_up: DeviceBuf,
            shexp_down: DeviceBuf,
            gate_exps: ExpertTensor,
            up_exps: ExpertTensor,
            down_exps: ExpertTensor,
        },
    }

    enum Attn {
        Gqa {
            attn_q: DeviceBuf,
            attn_k: DeviceBuf,
            attn_v: DeviceBuf,
            q_norm: DeviceBuf,
            k_norm: DeviceBuf,
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
        il < n_leading_dense as usize || (il >= 6 && (il - 6) % 4 == 0)
    }

    struct LayerW {
        attn_norm: DeviceBuf,
        attn: Attn,
        attn_output: DeviceBuf,
        ffn_norm: DeviceBuf,
        ffn: Ffn,
    }

    /// The nextn/MTP draft block: predicts token t+2 from (hidden of
    /// t, embedding of t+1) through one extra transformer layer.
    struct MtpLayer {
        layer: LayerW,
        eh_proj: DeviceBuf, // q8_0 [n_embd][2*n_embd]
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
        pub shape: Shape,
        pub gguf: Gguf,
        token_embd: DeviceBuf,
        output_norm: DeviceBuf,
        output: DeviceBuf,
        layers: Vec<LayerW>,
        /// PULSAR_ATTN_GPU: second CUDA device holding ALL attn weights +
        /// KV resident (Mla only). Attention weights are read every layer
        /// every token, so residency is the one job a bandwidth-crippled
        /// PCIe link can still do: only activations cross per layer.
        pub attn_dev: Option<i32>,
        mtp: Option<MtpLayer>,
        /// Draft-chain depth (PULSAR_MTP_DEPTH, default 3): tokens
        /// speculated per round, verified together in one forward.
        pub mtp_depth: u32,
        /// (row_bytes, quant) when output.weight is a K-quant (AngelSlim
        /// ggufs keep the lm-head q6_K); None = the q8_0 fast path.
        output_kq: Option<(u64, u32)>,
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
    #[derive(Default)]
    pub struct Prof {
        pub sync: std::time::Duration,
        pub resolve: std::time::Duration,
        pub h2d: std::time::Duration,
        pub tail: std::time::Duration,
        pub calls: u64,
    }

    impl Prof {
        pub fn report(&self) -> String {
            let s = |d: std::time::Duration| d.as_secs_f64();
            format!(
                "gpu-wait {:.2}s, resolve {:.2}s (h2d {:.2}s, disk/host {:.2}s), logits-tail {:.2}s over {} layer steps",
                s(self.sync),
                s(self.resolve),
                s(self.h2d),
                s(self.resolve) - s(self.h2d),
                s(self.tail),
                self.calls
            )
        }
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
        fn spawn(path: &Path) -> Result<Prefetcher> {
            let mut fetcher = stream::fetch::Fetcher::open_with(path, 16, fetch_buf_alloc())?;
            let (req_tx, req_rx) = std::sync::mpsc::channel::<Vec<stream::Read>>();
            let (done_tx, done_rx) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                while let Ok(first) = req_rx.recv() {
                    // stale requests are useless; keep only the newest
                    let reads = req_rx.try_iter().last().unwrap_or(first);
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
    pub struct DeviceSlabCache {
        pool: DeviceBuf,
        slab_bytes: usize,
        map: std::collections::HashMap<u64, u32>,
        /// per slot: (touch count at admission, offset); u64::MAX = free
        meta: Vec<(u64, u64)>,
        /// global (touch count, slab len) per requested offset, cached or not
        touch: std::collections::HashMap<u64, (u64, u64)>,
        pub hits: u64,
        pub misses: u64,
    }

    impl DeviceSlabCache {
        fn new(budget_bytes: usize, slab_bytes: usize) -> Result<DeviceSlabCache> {
            let slots = (budget_bytes / slab_bytes.max(1)).max(1);
            Ok(DeviceSlabCache {
                pool: DeviceBuf::alloc(slots * slab_bytes)?,
                slab_bytes,
                map: std::collections::HashMap::with_capacity(slots),
                meta: vec![(0, u64::MAX); slots],
                touch: std::collections::HashMap::new(),
                hits: 0,
                misses: 0,
            })
        }

        fn slot_ptr(&self, slot: u32) -> *const std::ffi::c_void {
            self.pool.ptr_at(slot as usize * self.slab_bytes)
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

        /// Admit `payload` if it is hotter than the coldest resident (or a
        /// slot is free). Returns None when the slab is not worthy - the
        /// caller streams it through staging instead. `in_use` offsets are
        /// never evicted.
        fn maybe_insert(
            &mut self,
            offset: u64,
            payload: &[u8],
            in_use: &[u64],
        ) -> Result<Option<*const std::ffi::c_void>> {
            let freq = self.touch.get(&offset).map(|t| t.0).unwrap_or(0);
            let slot = match self.meta.iter().position(|m| m.1 == u64::MAX) {
                Some(free) => free as u32,
                None => {
                    // ponytail: O(slots) coldest-scan; heap it if slots explode
                    let Some((victim, vmeta)) = self
                        .meta
                        .iter()
                        .enumerate()
                        .filter(|(_, m)| m.1 != u64::MAX && !in_use.contains(&m.1))
                        .min_by_key(|(_, m)| m.0)
                    else {
                        return Ok(None);
                    };
                    if vmeta.0 >= freq {
                        return Ok(None); // resident is at least as hot
                    }
                    let evict_off = vmeta.1;
                    let victim = victim as u32;
                    self.map.remove(&evict_off);
                    victim
                }
            };
            let base = slot as usize * self.slab_bytes;
            self.pool.write(base, payload)?;
            self.meta[slot as usize] = (freq, offset);
            self.map.insert(offset, slot);
            Ok(Some(self.slot_ptr(slot)))
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
        fn open(path: &Path, budget: usize) -> Result<StreamingStore> {
            Ok(StreamingStore {
                fetcher: stream::fetch::Fetcher::open_with(path, 32, fetch_buf_alloc())?,
                cache: std::collections::HashMap::new(),
                used: 0,
                budget,
                tick: 0,
                hits: 0,
                misses: 0,
            })
        }

        /// Resolve every read: cached payloads go to `place(offset, bytes)`
        /// immediately, disk misses as each io_uring completion lands - so
        /// the caller's H2D uploads overlap the remaining reads. Fetched
        /// slabs enter the LFU cache afterwards.
        fn ensure_with(
            &mut self,
            wants: &[stream::Read],
            mut place: impl FnMut(u64, &[u8]) -> Result,
        ) -> Result {
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
                return Ok(());
            }
            // evict lowest (freq, tick) not wanted right now
            // ponytail: O(n) scan per eviction; heap it if the cache ever
            // holds >100k entries
            let incoming: usize = missing.iter().map(|r| r.len as usize).sum();
            while self.used + incoming > self.budget && !self.cache.is_empty() {
                let victim = self
                    .cache
                    .iter()
                    .filter(|(k, _)| !wants.iter().any(|w| w.offset == **k))
                    .min_by_key(|(_, e)| (e.freq, e.tick))
                    .map(|(k, _)| *k);
                let Some(k) = victim else { break };
                if let Some(e) = self.cache.remove(&k) {
                    self.used -= e.slab.bytes();
                }
            }
            let Self { fetcher, cache, used, tick, .. } = self;
            let mut place_err = None;
            fetcher.fetch_each(&missing, |i, slab| {
                if place_err.is_none() {
                    if let Err(e) = place(missing[i].offset, slab.payload()) {
                        place_err = Some(e);
                    }
                }
                *used += slab.bytes();
                cache.insert(
                    missing[i].offset,
                    CacheEntry { slab, freq: 1, tick: *tick },
                );
                Ok(())
            })?;
            match place_err {
                Some(e) => Err(e),
                None => Ok(()),
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

        /// Take ownership of a prefetched slab (evicting to budget).
        fn absorb(&mut self, offset: u64, slab: stream::fetch::Slab) {
            if self.cache.contains_key(&offset) {
                return;
            }
            let incoming = slab.bytes();
            while self.used + incoming > self.budget && !self.cache.is_empty() {
                let victim = self
                    .cache
                    .iter()
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

    fn warm_path(model: &Path) -> std::path::PathBuf {
        let mut p = model.as_os_str().to_owned();
        p.push(".warm");
        p.into()
    }

    /// How many header bytes to read before parsing; grows on Truncated.
    const HEAD_READ_START: usize = 32 << 20;

    pub fn parse_header(path: &Path) -> Result<(File, Gguf)> {
        let file = File::open(path)?;
        let mut n = HEAD_READ_START;
        loop {
            let mut head = vec![0u8; n];
            let got = file.read_at(&mut head, 0)?;
            head.truncate(got);
            match Gguf::parse(&head) {
                Ok(g) => return Ok((file, g)),
                Err(gguf::Error::Truncated { .. }) if got == n => n *= 2,
                Err(e) => return Err(e.into()),
            }
        }
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
                            let q0 = ((ql[l] & 0x0f) as i32 | (((qh[l] >> 0) & 3) as i32) << 4) - 32;
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

    fn read_tensor_bytes(file: &File, g: &Gguf, name: &str) -> Result<Vec<u8>> {
        let t = g.tensor(name).ok_or_else(|| meta_err(name))?;
        let bytes = t.byte_size().ok_or_else(|| meta_err(name))?;
        let mut buf = vec![0u8; bytes as usize];
        file.read_exact_at(&mut buf, g.data_offset + t.offset)?;

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

    fn upload(file: &File, g: &Gguf, name: &str) -> Result<DeviceBuf> {
        Ok(DeviceBuf::from_bytes(&read_tensor_bytes(file, g, name)?)?)
    }

    /// Big attention weights: VRAM while `vram_budget` lasts, then pinned
    /// host memory (zero-copy PCIe reads). Gqa attn always fits, so its
    /// budget is unlimited; Mla (GLM-class, ~12GB attn q8) spends a
    /// PULSAR_ATTN_VRAM_GB budget (default 5) on the tensors the caller
    /// routes here - zero-copy reads measure ~6GB/s vs VRAM's ~288GB/s, so
    /// every budgeted byte is ~50x cheaper to read each token.
    /// PULSAR_ATTN_HOST=1 forces everything pinned.
    fn upload_attn(
        file: &File,
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
        pub fn load(path: &Path) -> Result<Model> {
            let (file, gguf) = parse_header(path)?;
            let shape = Shape::from_gguf(&gguf)?;

            // the embedding table is read ~one row per token - pinned
            // host is free for it and returns ~1GB of VRAM to hot weights
            let token_embd = {
                let bytes = read_tensor_bytes(&file, &gguf, "token_embd.weight")?;
                let mut buf = if shape.family == Family::Mla {
                    DeviceBuf::alloc_pinned(bytes.len())?
                } else {
                    DeviceBuf::alloc(bytes.len())?
                };
                buf.write(0, &bytes)?;
                buf
            };
            let output_norm = upload(&file, &gguf, "output_norm.weight")?;
            let output = upload(&file, &gguf, "output.weight")?;
            let output_kq = {
                let t = gguf.tensor("output.weight").ok_or_else(|| meta_err("output.weight"))?;
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
                Family::Mla => match std::env::var("PULSAR_ATTN_GPU").ok().as_deref() {
                    Some("off") | Some("-1") => None,
                    Some(v) => v.trim().parse::<i32>().ok().filter(|&d| {
                        let ok = d != primary && d >= 0 && d < kernels::device_count();
                        if !ok {
                            eprintln!("pulsar: ignoring PULSAR_ATTN_GPU={d} (primary is {primary}, {} devices)", kernels::device_count());
                        }
                        ok
                    }),
                    None => {
                        // auto: largest-free secondary that fits the stack
                        let mut need = 0u64;
                        for il in 0..shape.n_exec_layer {
                            for suf in [
                                "attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight",
                                "attn_kv_a_mqa.weight", "attn_kv_a_norm.weight",
                                "attn_k_b.weight", "attn_v_b.weight", "attn_output.weight",
                            ] {
                                if let Some(ti) = gguf.tensor(&format!("blk.{il}.{suf}")) {
                                    need += ti.byte_size().unwrap_or(0);
                                }
                            }
                        }
                        // reserve: compact KV at ctx 4096 (both caches span
                        // n_kv_lora + qk_rope) + scratch/hop buffers + CUDA
                        // context overhead. Pinned overflow would be read
                        // over the attn card's own link, so it must FIT.
                        let kv = shape.n_exec_layer as u64
                            * 4096
                            * (shape.n_kv_lora + shape.qk_rope) as u64
                            * 4;
                        need += kv + (1 << 30);
                        let mut best: Option<(usize, i32)> = None;
                        for d in 0..kernels::device_count() {
                            if d == primary {
                                continue;
                            }
                            if let Ok((free, _)) = kernels::mem_info(d) {
                                if free as u64 >= need && best.is_none_or(|(bf, _)| free > bf) {
                                    best = Some((free, d));
                                } else if (free as u64) < need {
                                    eprintln!(
                                        "pulsar: CUDA device {d} skipped for attn ({:.1}GB free < {:.1}GB needed)",
                                        free as f64 / 1e9,
                                        need as f64 / 1e9
                                    );
                                }
                            }
                        }
                        best.map(|(free, d)| {
                            eprintln!(
                                "pulsar: auto-detected attn GPU: CUDA device {d} ({:.1}GB free, attn stack needs {:.1}GB)",
                                free as f64 / 1e9,
                                need as f64 / 1e9
                            );
                            d
                        })
                    }
                },
                Family::Gqa => None,
            };
            if let Some(d) = attn_dev {
                eprintln!("pulsar: attn weights + KV resident on CUDA device {d}");
            }

            // Mla: spend a VRAM budget on the two big per-layer attn
            // tensors (attn_output ~107MB, q_b ~36MB on GLM-5.2) - they are
            // 80%+ of the per-token pinned-host read traffic. Gqa attn is
            // small enough to always live in VRAM. With a dedicated attn
            // GPU the whole stack (~14GB q8) goes resident by default -
            // pinned overflow would be read over that card's own link.
            let env_budget = std::env::var("PULSAR_ATTN_VRAM_GB")
                .ok()
                .and_then(|v| v.parse::<i64>().ok())
                .map(|v| v << 30);
            let mut attn_vram_budget: i64 = match (shape.family, attn_dev) {
                (Family::Gqa, _) => i64::MAX,
                (Family::Mla, Some(_)) => env_budget.unwrap_or(i64::MAX),
                (Family::Mla, None) => env_budget.unwrap_or(6 << 30),
            };
            // small Mla attn tensors always go pinned (not worth budget) -
            // except on a dedicated attn GPU, where everything is resident
            let mut no_budget: i64 = if attn_dev.is_some() { i64::MAX } else { 0 };

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
                } else {
                    let exps = |suffix: &str| -> Result<ExpertTensor> {
                        let name = t(suffix);
                        let ti = gguf.tensor(&name).ok_or_else(|| meta_err(&name))?;
                        ExpertTensor::new(&gguf, ti, shape.n_expert)
                    };
                    // router bias name varies by converter: bare on the
                    // antirez Hy3/GLM files, ".bias" on others
                    let probs_b_name = if gguf.tensor(&t("exp_probs_b")).is_some() {
                        t("exp_probs_b")
                    } else {
                        t("exp_probs_b.bias")
                    };
                    Ffn::Moe {
                        gate_inp: upload(&file, &gguf, &t("ffn_gate_inp.weight"))?,
                        probs_b: upload(&file, &gguf, &probs_b_name)?,
                        shexp_gate: upload(&file, &gguf, &t("ffn_gate_shexp.weight"))?,
                        shexp_up: upload(&file, &gguf, &t("ffn_up_shexp.weight"))?,
                        shexp_down: upload(&file, &gguf, &t("ffn_down_shexp.weight"))?,
                        gate_exps: exps("ffn_gate_exps.weight")?,
                        up_exps: exps("ffn_up_exps.weight")?,
                        down_exps: exps("ffn_down_exps.weight")?,
                    }
                };
                if let Some(d) = attn_dev {
                    kernels::set_device(d)?;
                }
                let attn = match shape.family {
                    Family::Gqa => Attn::Gqa {
                        attn_q: upload_attn(&file, &gguf, &t("attn_q.weight"), &mut *attn_vram_budget)?,
                        attn_k: upload_attn(&file, &gguf, &t("attn_k.weight"), &mut *attn_vram_budget)?,
                        attn_v: upload_attn(&file, &gguf, &t("attn_v.weight"), &mut *attn_vram_budget)?,
                        q_norm: upload(&file, &gguf, &t("attn_q_norm.weight"))?,
                        k_norm: upload(&file, &gguf, &t("attn_k_norm.weight"))?,
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
                };
                let attn_output = upload_attn(&file, &gguf, &t("attn_output.weight"), &mut *attn_vram_budget)?;
                if attn_dev.is_some() {
                    kernels::set_device(primary)?;
                }
                Ok(LayerW {
                    attn_norm: upload(&file, &gguf, &t("attn_norm.weight"))?,
                    attn,
                    attn_output,
                    ffn_norm: upload(&file, &gguf, &t("ffn_norm.weight"))?,
                    ffn,
                })
            };

            let mut layers = Vec::with_capacity(shape.n_exec_layer as usize);
            for il in 0..shape.n_exec_layer {
                layers.push(load_layer(il, &mut attn_vram_budget, &mut no_budget)?);
            }

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
                        match DeviceBuf::alloc(total) {
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
                                    "pulsar: MTP draft experts resident ({:.1}GB, all {} triples)",
                                    total as f64 / 1e9,
                                    shape.n_expert
                                );
                                res_pool = pool;
                            }
                            Err(_) => eprintln!(
                                "pulsar: MTP expert residency didn't fit ({:.1}GB needed) - drafts will stream",
                                total as f64 / 1e9
                            ),
                        }
                    }
                    let m = MtpLayer {
                        layer,
                        eh_proj: upload(&file, &gguf, &nextn("eh_proj"))?,
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
                std::env::var("PULSAR_MTP_DEPTH")
                    .ok()
                    .and_then(|v| v.parse::<u32>().ok())
                    .unwrap_or(1)
                    .clamp(1, 8)
            } else {
                0
            };

            Ok(Model {
                path: path.to_path_buf(),
                shape,
                gguf,
                token_embd,
                output_norm,
                output,
                layers,
                attn_dev,
                mtp,
                mtp_depth,
                output_kq,
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
        let candidates: Vec<i32> = (0..kernels::device_count())
            .filter(|&d| d != primary && Some(d) != m.attn_dev)
            .collect();
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        let census: std::collections::HashMap<u64, u64> =
            read_census(&m.path).into_iter().map(|(off, _, count)| (off, count)).collect();
        if census.is_empty() {
            eprintln!("pulsar: no warm census yet - expert tiers idle until the next run");
            return Ok(Vec::new());
        }
        // rank whole triples by summed slab heat
        let mut triples: Vec<(u64, [ (u64, u64); 3 ])> = Vec::new();
        for l in &m.layers {
            let Ffn::Moe { gate_exps, up_exps, down_exps, .. } = &l.ffn else {
                continue;
            };
            for e in 0..s.n_expert as u64 {
                let slabs = [gate_exps, up_exps, down_exps]
                    .map(|t| (t.abs_offset + e * t.expert_bytes, t.expert_bytes));
                let heat: u64 = slabs.iter().filter_map(|(off, _)| census.get(off)).sum();
                if heat > 0 {
                    triples.push((heat, slabs));
                }
            }
        }
        triples.sort_unstable_by(|a, b| b.0.cmp(&a.0));

        let file = File::open(&m.path)?;
        let mut tiers = Vec::new();
        let mut next = triples.into_iter();
        for d in candidates {
            let Ok((free, _)) = kernels::mem_info(d) else { continue };
            let reserve: usize = 1 << 30; // scratch + CUDA context
            if free <= reserve + (1 << 30) {
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
                    mb as usize * s.n_embd as usize / kernels::Q8_K_BLOCK_ELEMS
                        * kernels::Q8_K_BLOCK_BYTES,
                )?,
                mid: DeviceBuf::alloc(mb as usize * n_used * s.n_ff_exp as usize * 4)?,
                midq: DeviceBuf::alloc(
                    mb as usize * n_used * s.n_ff_exp as usize / kernels::Q8_K_BLOCK_ELEMS
                        * kernels::Q8_K_BLOCK_BYTES,
                )?,
                out: DeviceBuf::alloc(mb as usize * s.n_embd as usize * 4)?,
                ptrs: DeviceBuf::alloc(mb as usize * n_used * std::mem::size_of::<ExpertPtrs>())?,
                weights: DeviceBuf::alloc(mb as usize * n_used * 4)?,
                hits: 0,
            };
            let mut cursor = 0usize;
            let mut slab_buf = Vec::new();
            for (_, slabs) in next.by_ref() {
                let need: usize = slabs.iter().map(|&(_, len)| len as usize).sum();
                if cursor + need > tier.pool.bytes() {
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
                "pulsar: expert tier on CUDA device {d}: {} triples ({:.1}GB) resident in {:.1}s",
                tier.map.len() / 3,
                cursor as f64 / 1e9,
                t0.elapsed().as_secs_f32()
            );
            tiers.push(tier);
        }
        Ok(tiers)
    }

    /// Per-decode device state: activation buffers, KV caches, the routed
    /// expert staging arena, and reusable host staging.
    pub struct State {
        ctx: u32,
        max_batch: u32,
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
        staging: DeviceBuf,
        expert_ptrs: DeviceBuf,
        kcache: Vec<DeviceBuf>,
        vcache: Vec<DeviceBuf>,
        logits: DeviceBuf,
        pub store: StreamingStore,
        prefetcher: Prefetcher,
        pred_logits: DeviceBuf,
        pred_selected: DeviceBuf,
        pred_weights: DeviceBuf,
        /// Cumulative per-stage wall time (PULSAR_PROFILE=1 to print).
        pub prof: Prof,
        stages: Option<[AttnStage; 2]>,
        // MLA scratch (dummies for Gqa); on the attn GPU when one is set
        q_rank: DeviceBuf,
        q_rank_norm: DeviceBuf,
        kv_raw: DeviceBuf,
        kv_norm: DeviceBuf,
        qk_low: DeviceBuf,
        mla_selected: DeviceBuf,
        // DSA indexer scratch (1-float dummies when absent): per-indexer-
        // layer K caches + q/weights/scores; selection count persists
        // across layers so non-indexer layers reuse the last list
        idx_kcache: Vec<DeviceBuf>,
        idx_kraw: DeviceBuf,
        idx_q: DeviceBuf,
        idx_w: DeviceBuf,
        idx_scores: DeviceBuf,
        idx_last_sel: u32,
        // attn-GPU hop buffers (1-float dummies otherwise): normed input
        // copied primary->attn GPU, attn output copied back
        normed_a: DeviceBuf,
        attn_out_a: DeviceBuf,
        // resident expert tiers on leftover GPUs + the primary-side
        // buffer their partial outputs are gathered into
        pub tiers: Vec<ExpertTier>,
        tier_ret: DeviceBuf,
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
        mtp_hidden: DeviceBuf,
        /// true-hidden anchor saved across a draft chain (the chain
        /// self-feeds approximate hiddens into mtp_hidden; the batched
        /// fill pass afterwards needs the pre-chain value back)
        mtp_hidden_save: DeviceBuf,
        pub mtp_drafted: u64,
        pub mtp_accepted: u64,
        /// q8_K activation scratch for a K-quant lm-head (1 f32 otherwise)
        head_xq: DeviceBuf,
        /// Unified-memory box (GB10/Spark, Jetson): host-cache slabs are
        /// device-speed, so expert resolve hands their pinned pointers to
        /// the kernels directly - no VRAM cache, no staging copies. Safe
        /// because each layer's resolve runs after a full device sync, so
        /// an evicted slab can never have in-flight readers.
        unified: bool,
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

        pub fn ctx(&self) -> u32 {
            self.ctx
        }

        /// Persist the slab popularity census so the next run starts warm.
        pub fn save_warm(&self, m: &Model) -> Result {
            let mut entries: Vec<(u64, u64, u64)> = self
                .dev_cache
                .touch
                .iter()
                .map(|(&off, &(count, len))| (count, off, len))
                .collect();
            entries.sort_unstable_by(|a, b| b.0.cmp(&a.0));
            let mut bytes = Vec::with_capacity(entries.len() * 24);
            for (count, off, len) in &entries {
                bytes.extend_from_slice(&off.to_le_bytes());
                bytes.extend_from_slice(&len.to_le_bytes());
                bytes.extend_from_slice(&count.to_le_bytes());
            }
            std::fs::write(warm_path(&m.path), bytes)?;
            Ok(())
        }

        /// Load the popularity census: hottest slabs into VRAM, the next
        /// tier into the host cache, touch counts seeded for admission.
        fn load_warm(&mut self, m: &Model) -> Result<usize> {
            let Ok(bytes) = std::fs::read(warm_path(&m.path)) else {
                return Ok(0);
            };
            let mut entries = Vec::with_capacity(bytes.len() / 24);
            for c in bytes.chunks_exact(24) {
                let off = u64::from_le_bytes(c[0..8].try_into().unwrap());
                let len = u64::from_le_bytes(c[8..16].try_into().unwrap());
                let count = u64::from_le_bytes(c[16..24].try_into().unwrap());
                entries.push((off, len, count));
            }
            // tier-resident slabs never reach the caches - don't preload them
            let in_tier =
                |off: u64| self.tiers.iter().any(|t| t.map.contains_key(&off));
            let entries: Vec<_> =
                entries.into_iter().filter(|&(off, _, _)| !in_tier(off)).collect();
            for &(off, len, count) in &entries {
                self.dev_cache.touch.insert(off, (count, len));
            }
            let dev_slots = self.dev_cache.meta.len();
            let dev_tier: Vec<stream::Read> = entries
                .iter()
                .take(dev_slots)
                .map(|&(offset, len, _)| stream::Read { offset, len })
                .collect();
            let host_budget = self.store.budget as u64;
            let mut host_bytes = 0u64;
            let host_tier: Vec<stream::Read> = entries
                .iter()
                .skip(dev_slots)
                .take_while(|&&(_, len, _)| {
                    host_bytes += len;
                    host_bytes <= host_budget
                })
                .map(|&(offset, len, _)| stream::Read { offset, len })
                .collect();
            let n = dev_tier.len() + host_tier.len();
            let dev_cache = &mut self.dev_cache;
            self.store.fetch_direct(&dev_tier, |off, payload| {
                dev_cache.maybe_insert(off, payload, &[])?;
                Ok(())
            })?;
            self.store.ensure_with(&host_tier, |_, _| Ok(()))?;
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
                    Ffn::Moe { gate_exps, up_exps, down_exps, .. } => {
                        Some(gate_exps.expert_bytes.max(up_exps.expert_bytes).max(down_exps.expert_bytes))
                    }
                    _ => None,
                })
                .max()
                .unwrap_or(0) as usize;

            // Gqa: kcache/vcache are per-head K/V. Mla: kcache is the
            // compact latent cache (kv_lora wide), vcache the rope tail.
            let (k_bytes, v_bytes) = match s.family {
                Family::Gqa => {
                    let b = s.n_head_kv as usize * ctx as usize * s.head_dim as usize * 4;
                    (b, b)
                }
                Family::Mla => (
                    ctx as usize * s.n_kv_lora as usize * 4,
                    ctx as usize * s.qk_rope as usize * 4,
                ),
            };
            // batch prefill: activations sized for max_batch tokens; the
            // logits/lm-head path stays single-row (last token only)
            // big default: each prefill chunk costs roughly one pass over
            // the expert corpus regardless of chunk size, so fewer chunks
            // win; activations at 512 cost only ~150MB
            let mb = std::env::var("PULSAR_BATCH")
                .ok()
                .and_then(|v| v.parse::<u32>().ok())
                .unwrap_or(256)
                .max(1);

            // everything the attn segment touches lives on the attn GPU
            // when one is set: KV, MLA scratch, q/heads, hop buffers
            let primary = kernels::get_device();
            if let Some(d) = m.attn_dev {
                kernels::set_device(d)?;
            }
            let mut kcache = Vec::new();
            let mut vcache = Vec::new();
            let n_kv_slots = s.n_exec_layer as usize + usize::from(m.mtp.is_some());
            for i in 0..n_kv_slots {
                let mut k = DeviceBuf::alloc(k_bytes)?;
                let mut v = DeviceBuf::alloc(v_bytes)?;
                if i == s.n_exec_layer as usize {
                    // MTP slot: position 0 is never written (no hidden
                    // before the first token) yet attention reads it -
                    // zero beats uninitialized VRAM
                    k.write(0, &vec![0u8; k_bytes])?;
                    v.write(0, &vec![0u8; v_bytes])?;
                }
                kcache.push(k);
                vcache.push(v);
            }
            let q = f32s(mb * s.n_head * s.head_dim.max(s.qk_dim()))?;
            let heads = f32s(mb * s.heads_dim().max(s.n_head * s.head_dim))?;
            let q_rank = f32s(mb * s.n_lora_q.max(1))?;
            let q_rank_norm = f32s(mb * s.n_lora_q.max(1))?;
            let kv_raw = f32s(mb * (s.n_kv_lora + s.qk_rope).max(1))?;
            let kv_norm = f32s(mb * s.n_kv_lora.max(1))?;
            let qk_low = f32s(mb * s.n_head * s.n_kv_lora.max(1))?;
            let mla_selected = DeviceBuf::alloc(mb as usize * ctx as usize * 4)?;
            // DSA indexer buffers live beside the attn stack (same device)
            let has_idx = s.n_idx_topk > 0 && s.family == Family::Mla;
            let mut idx_kcache = Vec::new();
            for il in 0..s.n_exec_layer as usize {
                idx_kcache.push(if has_idx && uses_full_indexer(il, s.n_leading_dense) {
                    DeviceBuf::alloc(ctx as usize * s.n_idx_dim as usize * 4)?
                } else {
                    f32s(1)?
                });
            }
            let idx_kraw = f32s(if has_idx { mb * s.n_idx_dim } else { 1 })?;
            let idx_q = f32s(if has_idx { mb * s.n_idx_head * s.n_idx_dim } else { 1 })?;
            let idx_w = f32s(if has_idx { mb * s.n_idx_head } else { 1 })?;
            let idx_scores = f32s(if has_idx { mb * ctx } else { 1 })?;
            let (normed_a, attn_out_a) = if m.attn_dev.is_some() {
                (f32s(mb * s.n_embd)?, f32s(mb * s.n_embd)?)
            } else {
                (f32s(1)?, f32s(1)?)
            };
            if m.attn_dev.is_some() {
                kernels::set_device(primary)?;
            }
            let tiers = build_tiers(m, mb, primary)?;
            let mut st = State {
                ctx,
                max_batch: mb,
                tok: DeviceBuf::alloc(mb as usize * 4)?,
                // spec verify reads depth+1 trailing rows
                last_row: f32s((m.mtp_depth + 1).max(2) * s.n_embd)?,
                cur: f32s(mb * s.n_embd)?,
                normed: f32s(mb * s.n_embd)?,
                q,
                k: f32s(mb * s.n_head_kv * s.head_dim)?,
                v: f32s(mb * s.n_head_kv * s.head_dim)?,
                heads,
                attn_out: f32s(mb * s.n_embd)?,
                after_attn: f32s(mb * s.n_embd)?,
                gate_act: f32s(mb * s.n_ff_dense.max(s.n_ff_exp))?,
                up_act: f32s(mb * s.n_ff_dense.max(s.n_ff_exp))?,
                ffn_mid: f32s(mb * s.n_ff_dense.max(s.n_ff_exp))?,
                ffn_out: f32s(mb * s.n_embd)?,
                shared_out: f32s(mb * s.n_embd)?,
                router_logits: f32s(mb * s.n_expert)?,
                router_selected: DeviceBuf::alloc(mb as usize * n_used * 4)?,
                router_weights: f32s(mb * s.n_expert_used)?,
                moe_mid: f32s(mb * s.n_expert_used * s.n_ff_exp)?,
                moe_out: f32s(mb * s.n_embd)?,
                xq: DeviceBuf::alloc(
                    mb as usize * s.n_embd as usize / kernels::Q8_K_BLOCK_ELEMS
                        * kernels::Q8_K_BLOCK_BYTES,
                )?,
                midq: DeviceBuf::alloc(
                    mb as usize * n_used * s.n_ff_exp as usize / kernels::Q8_K_BLOCK_ELEMS
                        * kernels::Q8_K_BLOCK_BYTES,
                )?,
                dev_cache: DeviceSlabCache::new(
                    if kernels::unified_memory() {
                        // zero-copy resolve: a separate VRAM pool would
                        // just duplicate the same physical memory
                        1
                    } else {
                        std::env::var("PULSAR_DEV_CACHE_GB")
                            .ok()
                            .and_then(|v| v.parse::<usize>().ok())
                            // with attn on its own GPU the primary has ~15GB
                            // free; 8GB measured best (32% hits), 10 OOMs
                            .unwrap_or(match (s.family, m.attn_dev) {
                                (Family::Mla, Some(_)) => 8,
                                (Family::Mla, None) => 1,
                                (Family::Gqa, _) => 3,
                            })
                            << 30
                    },
                    max_slab,
                )?,
                // grow-only: decode stages <=n_used*3 slabs; a batched
                // prefill union (up to n_expert*3) grows it on first use
                staging: DeviceBuf::alloc(n_used * 3 * max_slab)?,
                expert_ptrs: DeviceBuf::alloc(
                    mb as usize * n_used * std::mem::size_of::<ExpertPtrs>(),
                )?,
                kcache,
                vcache,
                logits: f32s((m.mtp_depth + 1).max(2) * s.n_vocab)?,
                store: StreamingStore::open(&m.path, cache_bytes)?,
                prefetcher: Prefetcher::spawn(&m.path)?,
                pred_logits: f32s(s.n_expert)?,
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
                kv_raw,
                kv_norm,
                qk_low,
                mla_selected,
                idx_kcache,
                idx_kraw,
                idx_q,
                idx_w,
                idx_scores,
                idx_last_sel: 0,
                normed_a,
                attn_out_a,
                tier_ret: if tiers.is_empty() { f32s(1)? } else { f32s(mb * s.n_embd)? },
                tiers,
                grp_ptrs: DeviceBuf::alloc(s.n_expert.max(1) as usize * std::mem::size_of::<ExpertPtrs>())?,
                grp_starts: DeviceBuf::alloc((s.n_expert as usize + 1) * 4)?,
                grp_pairs: DeviceBuf::alloc(mb as usize * n_used * 4)?,
                grp_partial: f32s(1)?, // grows on first grouped prefill
                mtp_e_raw: f32s(if m.mtp.is_some() { mb * s.n_embd } else { 1 })?,
                mtp_e: f32s(if m.mtp.is_some() { mb * s.n_embd } else { 1 })?,
                mtp_h: f32s(if m.mtp.is_some() { mb * s.n_embd } else { 1 })?,
                mtp_x: f32s(if m.mtp.is_some() { mb * 2 * s.n_embd } else { 1 })?,
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
                head_xq: if m.output_kq.is_some() {
                    DeviceBuf::alloc(
                        (m.mtp_depth + 1).max(2) as usize * s.n_embd as usize
                            / kernels::Q8_K_BLOCK_ELEMS
                            * kernels::Q8_K_BLOCK_BYTES,
                    )?
                } else {
                    f32s(1)?
                },
                unified: {
                    let u = kernels::unified_memory();
                    if u {
                        eprintln!("pulsar: unified memory detected - zero-copy expert resolve");
                    }
                    u
                },
            };
            let t0 = std::time::Instant::now();
            let warmed = st.load_warm(m)?;
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
            {
                // attention
                kernels::rms_norm(&mut st.normed, &st.cur, &l.attn_norm, s.n_embd, n_tok, eps)?;
                let mut attn_output_w: &DeviceBuf = &l.attn_output;
                match &l.attn {
                    Attn::Gqa { attn_q, attn_k, attn_v, q_norm, k_norm } => {
                        kernels::matmul_q8_0(&mut st.q, attn_q, &st.normed, s.n_embd, s.n_head * s.head_dim, n_tok)?;
                        kernels::matmul_q8_0(&mut st.k, attn_k, &st.normed, s.n_embd, s.n_head_kv * s.head_dim, n_tok)?;
                        kernels::matmul_q8_0(&mut st.v, attn_v, &st.normed, s.n_embd, s.n_head_kv * s.head_dim, n_tok)?;
                        kernels::gqa_head_rms_norm(&mut st.q, q_norm, n_tok * s.n_head, s.head_dim, eps)?;
                        kernels::gqa_head_rms_norm(&mut st.k, k_norm, n_tok * s.n_head_kv, s.head_dim, eps)?;
                        kernels::gqa_rope(&mut st.q, n_tok, s.n_head, s.head_dim, pos0, s.rope_freq_base)?;
                        kernels::gqa_rope(&mut st.k, n_tok, s.n_head_kv, s.head_dim, pos0, s.rope_freq_base)?;
                        kernels::gqa_kv_append(&mut st.kcache[il], &st.k, n_tok, s.n_head_kv, s.head_dim, st.ctx, pos0)?;
                        kernels::gqa_kv_append(&mut st.vcache[il], &st.v, n_tok, s.n_head_kv, s.head_dim, st.ctx, pos0)?;
                        kernels::gqa_attention(&mut st.heads, &st.q, &st.kcache[il], &st.vcache[il], n_tok, s.n_head, s.n_head_kv, s.head_dim, st.ctx, pos0)?;
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

                        // attn-GPU offload: hop the normed input over,
                        // run the whole segment there. Blocking copies are
                        // legacy-stream ordered on the issuing device, so
                        // producer kernels have landed before they run.
                        if let Some(d) = self.attn_dev {
                            kernels::copy_across(&mut st.normed_a, &st.normed, (n_tok * s.n_embd) as usize * 4)?;
                            kernels::set_device(d)?;
                        }
                        let xin = if self.attn_dev.is_some() { &st.normed_a } else { &st.normed };

                        let rope = s.rope_cfg();
                        let kv_raw_dim = s.n_kv_lora + s.qk_rope;
                        kernels::matmul_q8_0(&mut st.q_rank, q_a_w, xin, s.n_embd, s.n_lora_q, n_tok)?;
                        kernels::rms_norm(&mut st.q_rank_norm, &st.q_rank, q_a_norm, s.n_lora_q, n_tok, eps)?;
                        kernels::matmul_q8_0(&mut st.q, q_b_w, &st.q_rank_norm, s.n_lora_q, s.n_head * s.qk_dim(), n_tok)?;
                        kernels::mla_rope_tail(&mut st.q, n_tok, s.n_head, s.qk_dim(), s.qk_rope, pos0, &rope)?;
                        kernels::matmul_q8_0(&mut st.kv_raw, kv_a_w, xin, s.n_embd, kv_raw_dim, n_tok)?;
                        kernels::mla_kv_lora_rms_norm(&mut st.kv_norm, &st.kv_raw, kv_a_norm, n_tok, kv_raw_dim, s.n_kv_lora, eps)?;
                        kernels::mla_store_compact_kv(&mut st.kcache[il], &mut st.vcache[il], &st.kv_norm, &st.kv_raw, pos0, n_tok, st.ctx, kv_raw_dim, s.n_kv_lora, s.qk_rope)?;
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
                            kernels::matmul_q8_0(&mut st.idx_kraw, &idx.k, xin, s.n_embd, s.n_idx_dim, n_tok)?;
                            kernels::idx_store_k(&st.idx_kraw, &idx.k_norm, &idx.k_norm_b, &mut st.idx_kcache[il], pos0, n_tok, st.ctx, s.n_idx_dim, s.qk_rope, s.rms_eps, &s.rope_cfg(), 0.0, 1.0)?;
                        }
                        let n_sel = if topk == 0 || visible <= topk {
                            kernels::mla_fill_selected_range(&mut st.mla_selected, n_tok, pos0, visible, st.ctx)?;
                            st.idx_last_sel = visible;
                            visible
                        } else if is_idx_layer && indexer.is_some() {
                            let idx = indexer.as_ref().unwrap();
                            kernels::matmul_q8_0(&mut st.idx_q, &idx.q_b, &st.q_rank_norm, s.n_lora_q, s.n_idx_head * s.n_idx_dim, n_tok)?;
                            kernels::idx_rope0(&mut st.idx_q, n_tok, s.n_idx_head, s.n_idx_dim, s.qk_rope, pos0, &s.rope_cfg(), 0.0, 1.0)?;
                            // ds4 feeds proj the pre-norm residual (cur).
                            // Under attn offload cur is on the primary;
                            // borrow attn_out_a as the hop buffer - it is
                            // not written until the output projection.
                            if self.attn_dev.is_some() {
                                kernels::copy_across(&mut st.attn_out_a, &st.cur, (n_tok * s.n_embd) as usize * 4)?;
                                kernels::matmul_f32(&mut st.idx_w, &idx.proj, &st.attn_out_a, s.n_embd, s.n_idx_head, n_tok)?;
                            } else {
                                kernels::matmul_f32(&mut st.idx_w, &idx.proj, &st.cur, s.n_embd, s.n_idx_head, n_tok)?;
                            }
                            let scale = 1.0 / ((s.n_idx_dim * s.n_idx_head) as f32).sqrt();
                            if n_tok == 1 {
                                kernels::idx_score_one(&mut st.idx_scores, &st.idx_q, &st.idx_w, &st.idx_kcache[il], visible, s.n_idx_head, s.n_idx_dim, scale)?;
                                kernels::idx_topk(&mut st.mla_selected, &st.idx_scores, visible, topk)?;
                            } else {
                                // batch: every token in a post-boundary
                                // chunk has >= top_k visible rows (the
                                // forward_rows split guarantees it)
                                kernels::idx_scores_batch(&mut st.idx_scores, &st.idx_q, &st.idx_w, &st.idx_kcache[il], visible, n_tok, pos0, s.n_idx_head, s.n_idx_dim, scale)?;
                                kernels::idx_topk_batch(&mut st.mla_selected, &st.idx_scores, visible, n_tok, topk)?;
                            }
                            st.idx_last_sel = topk;
                            topk
                        } else {
                            // between indexer layers: reuse the last list
                            if st.idx_last_sel == 0 {
                                return Err("indexer selection missing (no indexer weights in gguf?)".into());
                            }
                            st.idx_last_sel
                        };
                        kernels::mla_qk_lowrank(&mut st.qk_low, &st.q, k_b_w, n_tok, s.n_head, s.n_kv_lora, s.qk_nope, s.qk_dim())?;
                        kernels::mla_attention(&mut st.heads, &st.q, &st.qk_low, &st.kcache[il], &st.vcache[il], v_b_w, &st.mla_selected, n_tok, n_sel, st.ctx, s.n_head, s.n_kv_lora, s.qk_nope, s.qk_rope, s.value_mla, &rope)?;

                        // output projection on the attn GPU, hop back,
                        // restore the primary for the ffn/expert half
                        if self.attn_dev.is_some() {
                            kernels::matmul_q8_0(&mut st.attn_out_a, attn_output_w, &st.heads, s.heads_dim(), s.n_embd, n_tok)?;
                            kernels::copy_across(&mut st.attn_out, &st.attn_out_a, (n_tok * s.n_embd) as usize * 4)?;
                            kernels::set_device(primary)?;
                        }
                    }
                }
                if self.attn_dev.is_none() {
                    kernels::matmul_q8_0(&mut st.attn_out, attn_output_w, &st.heads, s.heads_dim(), s.n_embd, n_tok)?;
                }
                kernels::add(&mut st.after_attn, &st.cur, &st.attn_out, n_tok * s.n_embd)?;

                // ffn
                kernels::rms_norm(&mut st.normed, &st.after_attn, &l.ffn_norm, s.n_embd, n_tok, eps)?;
                match &l.ffn {
                    Ffn::Dense { gate, up, down } => {
                        kernels::matmul_q8_0(&mut st.gate_act, gate, &st.normed, s.n_embd, s.n_ff_dense, n_tok)?;
                        kernels::matmul_q8_0(&mut st.up_act, up, &st.normed, s.n_embd, s.n_ff_dense, n_tok)?;
                        kernels::swiglu(&mut st.ffn_mid, &st.gate_act, &st.up_act, n_tok * s.n_ff_dense, 0.0, 1.0)?;
                        kernels::matmul_q8_0(&mut st.ffn_out, down, &st.ffn_mid, s.n_ff_dense, s.n_embd, n_tok)?;
                        kernels::add(&mut st.cur, &st.after_attn, &st.ffn_out, n_tok * s.n_embd)?;
                    }
                    Ffn::Moe { gate_inp, probs_b, shexp_gate, shexp_up, shexp_down, gate_exps, up_exps, down_exps } => {
                        kernels::matmul_f32(&mut st.router_logits, gate_inp, &st.normed, s.n_embd, s.n_expert, n_tok)?;
                        kernels::router_select(
                            &mut st.router_selected,
                            &mut st.router_weights,
                            &st.router_logits,
                            probs_b,
                            s.n_expert,
                            s.n_expert_used,
                            s.expert_weight_scale,
                            n_tok,
                        )?;

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
                            kernels::matmul_f32(&mut st.pred_logits, n_gate_inp, &st.normed, s.n_embd, s.n_expert, 1)?;
                            kernels::router_select(
                                &mut st.pred_selected,
                                &mut st.pred_weights,
                                &st.pred_logits,
                                n_probs_b,
                                s.n_expert,
                                s.n_expert_used,
                                s.expert_weight_scale,
                                1,
                            )?;
                        }

                        // shared expert: depends only on normed, so it is
                        // launched BEFORE the resolve - the GPU computes it
                        // under the disk/H2D wait
                        kernels::matmul_q8_0(&mut st.gate_act, shexp_gate, &st.normed, s.n_embd, s.n_ff_exp, n_tok)?;
                        kernels::matmul_q8_0(&mut st.up_act, shexp_up, &st.normed, s.n_embd, s.n_ff_exp, n_tok)?;
                        kernels::swiglu(&mut st.ffn_mid, &st.gate_act, &st.up_act, n_tok * s.n_ff_exp, 0.0, 1.0)?;
                        kernels::matmul_q8_0(&mut st.shared_out, shexp_down, &st.ffn_mid, s.n_ff_exp, s.n_embd, n_tok)?;
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
                        let selected = st
                            .router_selected
                            .read_i32(n_tok as usize * s.n_expert_used as usize)?;
                        if let Some((_, _, next_exps)) = &next_moe {
                            let pred = st.pred_selected.read_i32(s.n_expert_used as usize)?;
                            let mut reads = Vec::with_capacity(3 * pred.len());
                            for &e in &pred {
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
                        if n_tok > 8 && std::env::var_os("PULSAR_NO_PREFETCH").is_none() {
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
                        // absorb whatever the prefetcher finished
                        while let Ok((off, slab)) = st.prefetcher.done_rx.try_recv() {
                            st.store.absorb(off, slab);
                        }
                        // gate/up/down may use different quants (K-quant
                        // recipes put ffn_down a tier higher); staging
                        // slots are strided by the largest of the three
                        let mut distinct: Vec<i32> = selected
                            .iter()
                            .copied()
                            .filter(|&e| e >= 0 && (e as u32) < s.n_expert)
                            .collect();
                        distinct.sort_unstable();
                        distinct.dedup();
                        // resident-tier experts compute on their own card;
                        // they are never fetched, cached, or staged here
                        let tier_of = |e: i32| -> Option<(usize, ExpertPtrs)> {
                            let g = gate_exps.abs_offset + e as u64 * gate_exps.expert_bytes;
                            // resident MTP experts beat a tier copy (same
                            // device as the compute, no partial gather)
                            if self.mtp.as_ref().is_some_and(|mt| mt.res_map.contains_key(&g)) {
                                return None;
                            }
                            st.tiers.iter().enumerate().find_map(|(ti, t)| {
                                let gate = *t.map.get(&g)?;
                                Some((ti, ExpertPtrs {
                                    gate,
                                    up: *t.map.get(&(up_exps.abs_offset + e as u64 * up_exps.expert_bytes))?,
                                    down: *t.map.get(&(down_exps.abs_offset + e as u64 * down_exps.expert_bytes))?,
                                }))
                            })
                        };
                        let mut offsets =
                            Vec::with_capacity(3 * distinct.len());
                        for &e in &distinct {
                            if tier_of(e).is_some() {
                                // keep the census warm for resident slabs
                                // or their heat freezes at placement time
                                for t in [gate_exps, up_exps, down_exps] {
                                    let off = t.abs_offset + e as u64 * t.expert_bytes;
                                    st.dev_cache.touch.entry(off).or_insert((0, t.expert_bytes)).0 += 1;
                                }
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
                            // MTP draft-layer experts are fully resident on
                            // the primary - never cached, never fetched
                            if let Some(mt) = &self.mtp {
                                if let Some(&po) = mt.res_map.get(&r.offset) {
                                    resolved.insert(r.offset, mt.res_pool.ptr_at(po));
                                    continue;
                                }
                            }
                            if st.unified {
                                // zero-copy: the host cache IS device memory
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
                        let slab = gate_exps
                            .expert_bytes
                            .max(up_exps.expert_bytes)
                            .max(down_exps.expert_bytes) as usize;
                        if wants.len() * slab > st.staging.bytes() {
                            st.staging = DeviceBuf::alloc(wants.len() * slab)?;
                        }
                        let mut staged = 0usize;
                        let unified = st.unified;
                        let dev_cache = &mut st.dev_cache;
                        let staging = &mut st.staging;
                        let mut h2d = std::time::Duration::ZERO;
                        st.store.ensure_with(&wants, |off, payload| {
                            if unified {
                                // pinned host slab is device-visible at
                                // full speed; hand the pointer straight
                                // to the kernels (UVA: host ptr == dev ptr)
                                resolved.insert(off, payload.as_ptr() as *const std::ffi::c_void);
                                return Ok(());
                            }
                            let t = std::time::Instant::now();
                            let p = match dev_cache.maybe_insert(off, payload, &in_use)? {
                                Some(p) => p,
                                None => {
                                    let base = staged * slab;
                                    staged += 1;
                                    staging.write(base, payload)?;
                                    staging.ptr_at(base)
                                }
                            };
                            h2d += t.elapsed();
                            resolved.insert(off, p);
                            Ok(())
                        })?;
                        st.prof.h2d += h2d;
                        let mut ptrs = Vec::with_capacity(selected.len());
                        let mut tptrs: Vec<Vec<ExpertPtrs>> = st
                            .tiers
                            .iter()
                            .map(|_| vec![ExpertPtrs::NULL; selected.len()])
                            .collect();
                        let mut tier_slots = vec![0u64; st.tiers.len()];
                        for (si, &e) in selected.iter().enumerate() {
                            if e < 0 || e as u32 >= s.n_expert {
                                ptrs.push(ExpertPtrs::NULL);
                                continue;
                            }
                            if let Some((ti, tp)) = tier_of(e) {
                                ptrs.push(ExpertPtrs::NULL);
                                tptrs[ti][si] = tp;
                                tier_slots[ti] += 1;
                                continue;
                            }
                            let p = |t: &ExpertTensor| {
                                resolved[&(t.abs_offset + e as u64 * t.expert_bytes)]
                            };
                            ptrs.push(ExpertPtrs {
                                gate: p(gate_exps),
                                up: p(up_exps),
                                down: p(down_exps),
                            });
                        }
                        st.expert_ptrs.write(0, kernels::as_bytes(&ptrs))?;

                        // grouped batch MoE (prefill): CSR of tokens per
                        // expert so each weight row is staged in shared
                        // memory once instead of re-read per token
                        let smem_ok = 2 * gate_exps.row_bytes.max(up_exps.row_bytes) * 4 <= 49152
                            && down_exps.row_bytes * 4 <= 49152;
                        let grouped = n_tok >= 16 && s.n_expert_used <= 16 && smem_ok
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
                        st.prof.resolve += t_resolve.elapsed();
                        st.prof.calls += 1;

                        // tier partials first: their kernels run on other
                        // cards, overlapping the primary's MoE below
                        let mut active = Vec::new();
                        for ti in 0..st.tiers.len() {
                            if tier_slots[ti] == 0 {
                                continue;
                            }
                            let tier = &mut st.tiers[ti];
                            tier.hits += tier_slots[ti];
                            kernels::copy_across(&mut tier.xin, &st.normed, (n_tok * s.n_embd) as usize * 4)?;
                            kernels::copy_across(&mut tier.weights, &st.router_weights, (n_tok * s.n_expert_used) as usize * 4)?;
                            kernels::set_device(tier.dev)?;
                            tier.ptrs.write(0, kernels::as_bytes(&tptrs[ti]))?;
                            kernels::quantize_q8_k(&mut tier.xq, &tier.xin, s.n_embd, n_tok)?;
                            kernels::moe_pair_swiglu(
                                &mut tier.mid, &tier.ptrs, &tier.weights, &tier.xq,
                                s.n_embd, s.n_ff_exp, s.n_expert_used, n_tok, gate_exps.row_bytes, gate_exps.quant,
                            )?;
                            kernels::quantize_q8_k(&mut tier.midq, &tier.mid, s.n_ff_exp, n_tok * s.n_expert_used)?;
                            kernels::moe_down(
                                &mut tier.out, &tier.ptrs, &tier.midq,
                                s.n_ff_exp, s.n_embd, s.n_expert_used, n_tok, down_exps.row_bytes, down_exps.quant,
                            )?;
                            kernels::set_device(primary)?;
                            active.push(ti);
                        }

                        // routed experts: activations quantized to q8_K,
                        // integer dp4a dots (ds4's exact math)
                        if grouped && n_group > 0 {
                            kernels::moe_pair_swiglu_grouped(
                                &mut st.moe_mid, &st.grp_ptrs, &st.grp_starts, &st.grp_pairs,
                                &st.router_weights, &st.xq,
                                s.n_embd, s.n_ff_exp, s.n_expert_used, n_group, gate_exps.row_bytes, gate_exps.quant,
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
                                s.n_embd, s.n_ff_exp, s.n_expert_used, n_tok, gate_exps.row_bytes, gate_exps.quant,
                            )?;
                            kernels::quantize_q8_k(&mut st.midq, &st.moe_mid, s.n_ff_exp, n_tok * s.n_expert_used)?;
                            kernels::moe_down(
                                &mut st.moe_out, &st.expert_ptrs, &st.midq,
                                s.n_ff_exp, s.n_embd, s.n_expert_used, n_tok, down_exps.row_bytes, down_exps.quant,
                            )?;
                        }

                        // gather tier partials (blocking copy issued on the
                        // tier's device = ordered after its kernels).
                        // NOTE: summing partials reorders float adds vs the
                        // single-kernel slot loop - same drift class as
                        // batch-vs-decode; PULSAR_TIERS=off restores exact.
                        for ti in active {
                            let tier = &st.tiers[ti];
                            kernels::set_device(tier.dev)?;
                            kernels::copy_across(&mut st.tier_ret, &tier.out, (n_tok * s.n_embd) as usize * 4)?;
                            kernels::set_device(primary)?;
                            kernels::add_assign(&mut st.moe_out, &st.tier_ret, n_tok * s.n_embd)?;
                        }

                        // cur = after_attn + routed + shared (ds4's add3)
                        kernels::add(&mut st.ffn_out, &st.moe_out, &st.shared_out, n_tok * s.n_embd)?;
                        kernels::add(&mut st.cur, &st.after_attn, &st.ffn_out, n_tok * s.n_embd)?;
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
            kernels::matmul_q8_0(&mut st.cur, &mtp.eh_proj, &st.mtp_x, 2 * s.n_embd, s.n_embd, n_tok)?;
            self.eval_layer(st, self.layers.len(), &mtp.layer, n_tok, pos0, primary)
        }

        /// One MTP pass: embed `token` at `pos` against st.mtp_hidden,
        /// append the block's KV, return the greedy draft for pos+1.
        /// Clobbers st.cur.
        fn mtp_draft(&self, st: &mut State, token: u32, pos: u32) -> Result<u32> {
            self.mtp_body(st, token, pos)?;
            let mtp = self.mtp.as_ref().ok_or("mtp_draft without an MTP layer")?;
            let s = self.shape;
            kernels::rms_norm(&mut st.normed, &st.cur, &mtp.head_norm, s.n_embd, 1, s.rms_eps)?;
            self.head_logits(st, 1)?;
            kernels::sync()?;
            let logits = st.logits.read_f32(s.n_vocab as usize)?;
            if std::env::var_os("PULSAR_MTP_DEBUG").is_some() {
                let bad = logits.iter().filter(|v| !v.is_finite()).count();
                eprintln!("mtp-draft pos={pos}: logits nan={bad}, draft={}", argmax(&logits));
            }
            Ok(argmax(&logits))
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
            kernels::matmul_q8_0(&mut st.cur, &mtp.eh_proj, &st.mtp_x, 2 * s.n_embd, s.n_embd, 1)?;
            self.eval_layer(st, self.layers.len(), &mtp.layer, 1, pos, primary)
        }
    }

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
        // MTP speculative decode is greedy-only: acceptance compares the
        // draft against the verified argmax, which IS greedy sampling.
        let spec = model.mtp.is_some() && sampler.is_greedy();
        let mut pos = pos0;
        let mut logits = None;
        for chunk in prompt.chunks(st.max_batch() as usize) {
            logits = model.forward_batch(st, chunk, pos, true)?;
            if spec {
                model.mtp_prefill_fill(st, chunk.len() as u32, pos)?;
            }
            pos += chunk.len() as u32;
        }

        if spec {
            let v = model.shape.n_vocab as usize;
            let row = model.shape.n_embd as usize * 4;
            let depth_max = model.mtp_depth.max(1);
            let debug = std::env::var_os("PULSAR_MTP_DEBUG").is_some();
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

                // Draft a chain: each step self-feeds the MTP layer's own
                // output hidden (approximate but cheap - one layer/step).
                // Anchor the true pre-chain hidden for the fill pass.
                kernels::copy_d2d(&mut st.mtp_hidden_save, 0, &st.mtp_hidden, 0, row)?;
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
                let k = chain.len() - 1; // drafts in flight

                // Verify the whole chain in ONE forward: the per-layer
                // union expert fetch is what makes the extra rows cheap.
                // Greedy acceptance keeps the stream identical to plain
                // greedy decode.
                let all = model
                    .forward_rows(st, &chain, pos, (k + 1) as u32)?
                    .ok_or("no verify logits")?;
                let mut j = 0usize;
                while j < k && argmax(&all[j * v..(j + 1) * v]) == chain[j + 1] {
                    st.mtp_accepted += 1;
                    j += 1;
                }
                if debug {
                    let nans = all.iter().filter(|x| !x.is_finite()).count();
                    eprintln!("mtp: pos={pos} chain={chain:?} accepted={j}/{k} nan={nans}");
                }

                // Re-anchor the MTP cache on TRUE hiddens for the accepted
                // prefix in one batched pass: st.tok still holds the chain,
                // st.cur its verified hiddens - exactly what a prefill
                // chunk looks like to mtp_prefill_fill.
                kernels::copy_d2d(&mut st.mtp_hidden, 0, &st.mtp_hidden_save, 0, row)?;
                model.mtp_prefill_fill(st, (j + 1) as u32, pos)?;
                pos += (j + 1) as u32;
                next = argmax(&all[j * v..(j + 1) * v]);

                for &d in &chain[1..=j] {
                    if stop(d) {
                        break 'round; // forwarded, not emitted - as non-spec
                    }
                    if emitted >= max_tokens {
                        break 'round;
                    }
                    on_token(d);
                    emitted += 1;
                }
            }
            return Ok(pos);
        }

        for _ in 0..max_tokens {
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
    }
}

#[cfg(target_os = "linux")]
pub use real::*;
