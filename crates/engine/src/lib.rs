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
            // GLM-5.2 ships yarn off (scale 1.0); parameters ride along.
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

    impl Shape {
        fn from_gguf(g: &Gguf) -> Result<Shape> {
            let u = |k: &str| -> Result<u32> {
                Ok(g.arch_meta(k).and_then(Value::as_u64).ok_or_else(|| meta_err(k))? as u32)
            };
            let f = |k: &str| -> Result<f32> {
                g.arch_meta(k).and_then(Value::as_f32).ok_or_else(|| meta_err(k))
            };
            let family = match g.architecture() {
                Some("hy-v3") => Family::Gqa,
                Some("glm-dsa") => Family::Mla,
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
                n_leading_dense: u("leading_dense_block_count")?,
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
            };
            if family == Family::Mla {
                // GLM-5.2 MLA split, ds4's DS4_SHAPE_GLM52 (metadata keys
                // for the nope/rope split are not standardized; the gguf
                // values that do exist are preferred).
                s.n_lora_q = u("attention.q_lora_rank").unwrap_or(2048);
                s.n_kv_lora = u("attention.kv_lora_rank").unwrap_or(512);
                s.qk_rope = u("rope.dimension_count").unwrap_or(64);
                s.qk_nope = 192;
                s.value_mla = u("attention.value_length").unwrap_or(256);
                s.rope_orig_ctx = u("rope.scaling.original_context_length").unwrap_or(1_048_576);
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
        },
    }

    struct LayerW {
        attn_norm: DeviceBuf,
        attn: Attn,
        attn_output: DeviceBuf,
        ffn_norm: DeviceBuf,
        ffn: Ffn,
    }

    pub struct Model {
        path: std::path::PathBuf,
        pub shape: Shape,
        pub gguf: Gguf,
        token_embd: DeviceBuf,
        output_norm: DeviceBuf,
        output: DeviceBuf,
        layers: Vec<LayerW>,
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
            let mut fetcher = stream::fetch::Fetcher::open(path, 16)?;
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

    impl StreamingStore {
        fn open(path: &Path, budget: usize) -> Result<StreamingStore> {
            Ok(StreamingStore {
                fetcher: stream::fetch::Fetcher::open(path, 32)?,
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

    fn read_tensor_bytes(file: &File, g: &Gguf, name: &str) -> Result<Vec<u8>> {
        let t = g.tensor(name).ok_or_else(|| meta_err(name))?;
        let bytes = t.byte_size().ok_or_else(|| meta_err(name))?;
        let mut buf = vec![0u8; bytes as usize];
        file.read_exact_at(&mut buf, g.data_offset + t.offset)?;
        Ok(buf)
    }

    fn upload(file: &File, g: &Gguf, name: &str) -> Result<DeviceBuf> {
        Ok(DeviceBuf::from_bytes(&read_tensor_bytes(file, g, name)?)?)
    }

    /// Big attention weights: pinned host memory when PULSAR_ATTN_HOST=1
    /// (zero-copy PCIe reads - GLM-class backbones exceed VRAM), else VRAM.
    fn upload_attn(file: &File, g: &Gguf, name: &str) -> Result<DeviceBuf> {
        let bytes = read_tensor_bytes(file, g, name)?;
        let mut buf = if std::env::var_os("PULSAR_ATTN_HOST").is_some() {
            DeviceBuf::alloc_pinned(bytes.len())?
        } else {
            DeviceBuf::alloc(bytes.len())?
        };
        buf.write(0, &bytes)?;
        Ok(buf)
    }

    impl Model {
        pub fn load(path: &Path) -> Result<Model> {
            let (file, gguf) = parse_header(path)?;
            let shape = Shape::from_gguf(&gguf)?;

            let token_embd = upload(&file, &gguf, "token_embd.weight")?;
            let output_norm = upload(&file, &gguf, "output_norm.weight")?;
            let output = upload(&file, &gguf, "output.weight")?;

            let mut layers = Vec::with_capacity(shape.n_exec_layer as usize);
            for il in 0..shape.n_exec_layer {
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
                    Ffn::Moe {
                        gate_inp: upload(&file, &gguf, &t("ffn_gate_inp.weight"))?,
                        probs_b: upload(&file, &gguf, &t("exp_probs_b"))?,
                        shexp_gate: upload(&file, &gguf, &t("ffn_gate_shexp.weight"))?,
                        shexp_up: upload(&file, &gguf, &t("ffn_up_shexp.weight"))?,
                        shexp_down: upload(&file, &gguf, &t("ffn_down_shexp.weight"))?,
                        gate_exps: exps("ffn_gate_exps.weight")?,
                        up_exps: exps("ffn_up_exps.weight")?,
                        down_exps: exps("ffn_down_exps.weight")?,
                    }
                };
                let attn = match shape.family {
                    Family::Gqa => Attn::Gqa {
                        attn_q: upload_attn(&file, &gguf, &t("attn_q.weight"))?,
                        attn_k: upload_attn(&file, &gguf, &t("attn_k.weight"))?,
                        attn_v: upload_attn(&file, &gguf, &t("attn_v.weight"))?,
                        q_norm: upload(&file, &gguf, &t("attn_q_norm.weight"))?,
                        k_norm: upload(&file, &gguf, &t("attn_k_norm.weight"))?,
                    },
                    Family::Mla => Attn::Mla {
                        q_a: upload_attn(&file, &gguf, &t("attn_q_a.weight"))?,
                        q_a_norm: upload(&file, &gguf, &t("attn_q_a_norm.weight"))?,
                        q_b: upload_attn(&file, &gguf, &t("attn_q_b.weight"))?,
                        kv_a_mqa: upload_attn(&file, &gguf, &t("attn_kv_a_mqa.weight"))?,
                        kv_a_norm: upload(&file, &gguf, &t("attn_kv_a_norm.weight"))?,
                        k_b: upload_attn(&file, &gguf, &t("attn_k_b.weight"))?,
                        v_b: upload_attn(&file, &gguf, &t("attn_v_b.weight"))?,
                    },
                };
                layers.push(LayerW {
                    attn_norm: upload(&file, &gguf, &t("attn_norm.weight"))?,
                    attn,
                    attn_output: upload_attn(&file, &gguf, &t("attn_output.weight"))?,
                    ffn_norm: upload(&file, &gguf, &t("ffn_norm.weight"))?,
                    ffn,
                });
            }
            Ok(Model {
                path: path.to_path_buf(),
                shape,
                gguf,
                token_embd,
                output_norm,
                output,
                layers,
            })
        }
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
        // MLA scratch (dummies for Gqa)
        q_rank: DeviceBuf,
        q_rank_norm: DeviceBuf,
        kv_raw: DeviceBuf,
        kv_norm: DeviceBuf,
        qk_low: DeviceBuf,
        mla_selected: DeviceBuf,
    }

    impl State {
        pub fn new(m: &Model, ctx: u32) -> Result<State> {
            let gb = std::env::var("PULSAR_CACHE_GB")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(12);
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
            let max_slab = m
                .layers
                .iter()
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
            let mut kcache = Vec::new();
            let mut vcache = Vec::new();
            for _ in 0..s.n_exec_layer {
                kcache.push(DeviceBuf::alloc(k_bytes)?);
                vcache.push(DeviceBuf::alloc(v_bytes)?);
            }

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
            let mut st = State {
                ctx,
                max_batch: mb,
                tok: DeviceBuf::alloc(mb as usize * 4)?,
                last_row: f32s(s.n_embd)?,
                cur: f32s(mb * s.n_embd)?,
                normed: f32s(mb * s.n_embd)?,
                q: f32s(mb * s.n_head * s.head_dim.max(s.qk_dim()))?,
                k: f32s(mb * s.n_head_kv * s.head_dim)?,
                v: f32s(mb * s.n_head_kv * s.head_dim)?,
                heads: f32s(mb * s.heads_dim().max(s.n_head * s.head_dim))?,
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
                    std::env::var("PULSAR_DEV_CACHE_GB")
                        .ok()
                        .and_then(|v| v.parse::<usize>().ok())
                        .unwrap_or(3)
                        << 30,
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
                logits: f32s(s.n_vocab)?,
                store: StreamingStore::open(&m.path, cache_bytes)?,
                prefetcher: Prefetcher::spawn(&m.path)?,
                pred_logits: f32s(s.n_expert)?,
                pred_selected: DeviceBuf::alloc(n_used * 4)?,
                pred_weights: f32s(s.n_expert_used)?,
                q_rank: f32s(mb * s.n_lora_q.max(1))?,
                q_rank_norm: f32s(mb * s.n_lora_q.max(1))?,
                kv_raw: f32s(mb * (s.n_kv_lora + s.qk_rope).max(1))?,
                kv_norm: f32s(mb * s.n_kv_lora.max(1))?,
                qk_low: f32s(mb * s.n_head * s.n_kv_lora.max(1))?,
                mla_selected: DeviceBuf::alloc(mb as usize * ctx as usize * 4)?,
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
            let s = self.shape;
            let n_tok = tokens.len() as u32;
            if n_tok == 0 || n_tok > st.max_batch {
                return Err(format!("batch {} outside 1..={}", n_tok, st.max_batch).into());
            }
            if pos0 + n_tok > st.ctx {
                return Err("position exceeds context".into());
            }
            let eps = s.rms_eps;
            let toks_i32: Vec<i32> = tokens.iter().map(|&t| t as i32).collect();
            st.tok.write(0, kernels::as_bytes(&toks_i32))?;
            kernels::embed_q8_0(&mut st.cur, &self.token_embd, &st.tok, s.n_embd, s.n_vocab, n_tok)?;

            for (il, l) in self.layers.iter().enumerate() {
                // attention
                kernels::rms_norm(&mut st.normed, &st.cur, &l.attn_norm, s.n_embd, n_tok, eps)?;
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
                    Attn::Mla { q_a, q_a_norm, q_b, kv_a_mqa, kv_a_norm, k_b, v_b } => {
                        // ds4's GLM compact-KV decode path: q through the
                        // lora bottleneck, latent kv cached once for all
                        // heads, attention over all visible rows.
                        let rope = s.rope_cfg();
                        let kv_raw_dim = s.n_kv_lora + s.qk_rope;
                        kernels::matmul_q8_0(&mut st.q_rank, q_a, &st.normed, s.n_embd, s.n_lora_q, n_tok)?;
                        kernels::rms_norm(&mut st.q_rank_norm, &st.q_rank, q_a_norm, s.n_lora_q, n_tok, eps)?;
                        kernels::matmul_q8_0(&mut st.q, q_b, &st.q_rank_norm, s.n_lora_q, s.n_head * s.qk_dim(), n_tok)?;
                        kernels::mla_rope_tail(&mut st.q, n_tok, s.n_head, s.qk_dim(), s.qk_rope, pos0, &rope)?;
                        kernels::matmul_q8_0(&mut st.kv_raw, kv_a_mqa, &st.normed, s.n_embd, kv_raw_dim, n_tok)?;
                        kernels::mla_kv_lora_rms_norm(&mut st.kv_norm, &st.kv_raw, kv_a_norm, n_tok, kv_raw_dim, s.n_kv_lora, eps)?;
                        kernels::mla_store_compact_kv(&mut st.kcache[il], &mut st.vcache[il], &st.kv_norm, &st.kv_raw, pos0, n_tok, st.ctx, kv_raw_dim, s.n_kv_lora, s.qk_rope)?;
                        let n_sel = pos0 + n_tok;
                        kernels::mla_fill_selected_range(&mut st.mla_selected, n_tok, pos0, n_sel, st.ctx)?;
                        kernels::mla_qk_lowrank(&mut st.qk_low, &st.q, k_b, n_tok, s.n_head, s.n_kv_lora, s.qk_nope, s.qk_dim())?;
                        kernels::mla_attention(&mut st.heads, &st.q, &st.qk_low, &st.kcache[il], &st.vcache[il], v_b, &st.mla_selected, n_tok, n_sel, st.ctx, s.n_head, s.n_kv_lora, s.qk_nope, s.qk_rope, s.value_mla, &rope)?;
                    }
                }
                kernels::matmul_q8_0(&mut st.attn_out, &l.attn_output, &st.heads, s.heads_dim(), s.n_embd, n_tok)?;
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

                        // Expert resolve, batched: the union of distinct
                        // experts across all tokens fetches once. VRAM
                        // cache first, then host LFU + one io_uring batch.
                        kernels::sync()?;
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
                                    {
                                        reads.push(stream::Read { offset, len: t.expert_bytes });
                                    }
                                }
                            }
                            if !reads.is_empty() {
                                let _ = st.prefetcher.req_tx.send(reads);
                            }
                        }
                        // absorb whatever the prefetcher finished
                        while let Ok((off, slab)) = st.prefetcher.done_rx.try_recv() {
                            st.store.absorb(off, slab);
                        }
                        debug_assert_eq!(up_exps.expert_bytes, gate_exps.expert_bytes);
                        debug_assert_eq!(down_exps.expert_bytes, gate_exps.expert_bytes);
                        let mut distinct: Vec<i32> = selected
                            .iter()
                            .copied()
                            .filter(|&e| e >= 0 && (e as u32) < s.n_expert)
                            .collect();
                        distinct.sort_unstable();
                        distinct.dedup();
                        let mut offsets =
                            Vec::with_capacity(3 * distinct.len());
                        for &e in &distinct {
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
                            match st.dev_cache.get(r.offset, r.len) {
                                Some(p) => {
                                    resolved.insert(r.offset, p);
                                }
                                None => wants.push(*r),
                            }
                        }
                        let slab = gate_exps.expert_bytes as usize;
                        if wants.len() * slab > st.staging.bytes() {
                            st.staging = DeviceBuf::alloc(wants.len() * slab)?;
                        }
                        let mut staged = 0usize;
                        let dev_cache = &mut st.dev_cache;
                        let staging = &mut st.staging;
                        st.store.ensure_with(&wants, |off, payload| {
                            let p = match dev_cache.maybe_insert(off, payload, &in_use)? {
                                Some(p) => p,
                                None => {
                                    let base = staged * slab;
                                    staged += 1;
                                    staging.write(base, payload)?;
                                    staging.ptr_at(base)
                                }
                            };
                            resolved.insert(off, p);
                            Ok(())
                        })?;
                        let mut ptrs = Vec::with_capacity(selected.len());
                        for &e in &selected {
                            if e < 0 || e as u32 >= s.n_expert {
                                ptrs.push(ExpertPtrs::NULL);
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

                        // shared expert
                        kernels::matmul_q8_0(&mut st.gate_act, shexp_gate, &st.normed, s.n_embd, s.n_ff_exp, n_tok)?;
                        kernels::matmul_q8_0(&mut st.up_act, shexp_up, &st.normed, s.n_embd, s.n_ff_exp, n_tok)?;
                        kernels::swiglu(&mut st.ffn_mid, &st.gate_act, &st.up_act, n_tok * s.n_ff_exp, 0.0, 1.0)?;
                        kernels::matmul_q8_0(&mut st.shared_out, shexp_down, &st.ffn_mid, s.n_ff_exp, s.n_embd, n_tok)?;

                        // routed experts: activations quantized to q8_K,
                        // integer dp4a dots (ds4's exact math)
                        kernels::quantize_q8_k(&mut st.xq, &st.normed, s.n_embd, n_tok)?;
                        kernels::moe_pair_swiglu(
                            &mut st.moe_mid, &st.expert_ptrs, &st.router_weights, &st.xq,
                            s.n_embd, s.n_ff_exp, s.n_expert_used, n_tok, gate_exps.row_bytes, gate_exps.quant,
                        )?;
                        kernels::quantize_q8_k(&mut st.midq, &st.moe_mid, s.n_ff_exp, n_tok * s.n_expert_used)?;
                        kernels::moe_down(
                            &mut st.moe_out, &st.expert_ptrs, &st.midq,
                            s.n_ff_exp, s.n_embd, s.n_expert_used, n_tok, down_exps.row_bytes, down_exps.quant,
                        )?;

                        // cur = after_attn + routed + shared (ds4's add3)
                        kernels::add(&mut st.ffn_out, &st.moe_out, &st.shared_out, n_tok * s.n_embd)?;
                        kernels::add(&mut st.cur, &st.after_attn, &st.ffn_out, n_tok * s.n_embd)?;
                    }
                }
            }

            if !want_logits {
                return Ok(None);
            }
            let row = s.n_embd as usize * 4;
            kernels::copy_d2d(&mut st.last_row, 0, &st.cur, (n_tok as usize - 1) * row, row)?;
            kernels::rms_norm(&mut st.normed, &st.last_row, &self.output_norm, s.n_embd, 1, eps)?;
            kernels::matmul_q8_0(&mut st.logits, &self.output, &st.normed, s.n_embd, s.n_vocab, 1)?;
            kernels::sync()?;
            Ok(Some(st.logits.read_f32(s.n_vocab as usize)?))
        }
    }

    /// Prefill `prompt` at pos0 (chunked), then sample until `stop`,
    /// ctx, or max_tokens; each sampled token goes to `on_token` and is
    /// forwarded into the KV cache (including the stop token, so the
    /// context stays template-shaped for a next turn). Returns the
    /// position after everything forwarded.
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
        let mut pos = pos0;
        let mut logits = None;
        for chunk in prompt.chunks(st.max_batch() as usize) {
            logits = model.forward_batch(st, chunk, pos, true)?;
            pos += chunk.len() as u32;
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
