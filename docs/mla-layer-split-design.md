# MLA layer-split attention across GPUs (design)

Goal: spread the MLA attention stack (weights + KV + indexer cache) across
all cards instead of one `attn_dev`, so context capacity scales with total
VRAM. GLM-5.2 today: 14.4 GiB Q8_0 attn weights (~186 MB/layer over 79
layers) + KV + idx cache must fit ONE 16.6 GB card; with fp8 latent KV
(18ff73f) that caps ctx at ~20480. Split over two secondary cards the same
budget supports roughly 8x the context; fp8 idx cache (follow-up) and Q6_K
attn requant push further.

## Current shape (as of 18ff73f)

- `Model.attn_dev: Option<i32>` (832) - ONE secondary card owns the whole
  stack. Auto-picked at load (2533-2591): largest-free secondary that fits
  `sum(attn tensors) + KV@4096 + 1GB`.
- Load places all `blk.*.attn_*` weights on it (regions 2865, 3058, 3240,
  3299) and State::new switches to it before allocating the KV caches
  (4667) and every attention scratch buffer.
- Forward (MLA branch 5642-5722): hop `normed -> normed_a` onto the attn
  card, run the WHOLE segment there (q_a/q_b, kv_a, store KV, DSA indexer,
  qk_lowrank, attention, out-proj), hop `attn_out_a -> attn_out` back,
  `set_device(primary)`. One hop pair per layer.
- Precedent: qwen35-dense `layer_dev: Vec<i32>` (2627) does whole-layer
  ownership with per-layer `set_device` in the KV alloc loop
  (`dense_split` branch, 4568-4572 region) and boundary-crossing residual.

## Target shape

1. `attn_layer_dev: Vec<i32>` (per exec layer + MTP slot). `attn_dev`
   stays as a summary Option (None = all primary) for the ~10 coarse
   gates that only care "is offload on at all"; sites inside the layer
   loop switch to `attn_layer_dev[il]`.
2. Planner at load: per-layer attn byte counts (already computed in the
   auto-detect loop), greedily pack layers onto cards in layer order
   (contiguous ranges, so `mla_selected` hops only at range boundaries).
   Include the primary as a candidate only for leftover layers - its VRAM
   is expert cache. Reserve per card: that card's share of KV + idx cache
   at the requested ctx (use the real kvq_lat stride, not f32) + scratch +
   0.5 GB CUDA overhead. `PULSAR_ATTN_GPU=off` -> all primary;
   `PULSAR_ATTN_GPU=<d>` -> force single card d (today's behavior).
3. Per-device scratch: bundle the attention scratch into
   `struct MlaScratch { dev: i32, normed_a, attn_out_a, q_rank,
   q_rank_norm, q, kv_raw, kv_norm, qk_low, heads, idx_kraw, idx_q,
   idx_q16, idx_w, idx_scores, mla_selected }` (idx_scores and
   mla_selected scale with ctx - they dominate scratch, count them in the
   planner reserve). `State.attn_sc: Vec<MlaScratch>`, one per
   participating device; forward picks `let sc = &mut st.attn_sc[map]`
   by the layer's device. Gqa offload path keeps using the flat fields
   (single-card only) OR gets folded in later - do NOT refactor Gqa in
   the same commit.
4. KV/idx cache placement: generalize the `dense_split` per-layer
   `set_device` in the kcache/vcache/idx_kcache alloc loops to also fire
   when attn_layer_dev is non-uniform. MTP slot -> same card as the last
   layer range (its segment runs right after).
5. Cross-layer DSA selection dependency (THE trap): non-indexer layers
   reuse the previous layer's top-k list (`st.idx_last_sel`,
   `st.mla_selected`). With per-layer devices, when `attn_layer_dev[il] !=
   attn_layer_dev[il-1]` and layer il is a reuse layer, copy the selection
   (`n_tok * idx_last_sel * 4` bytes, ~8 KB) from the previous device's
   scratch into this device's before mla_attention. Contiguous ranges make
   this fire once per boundary per chunk. Also copy at range boundaries
   when the NEXT layer is a reuse layer regardless - cheap enough to do
   unconditionally at every device change.
6. Forward loop: replace the two `if let Some(d) = self.attn_dev` hops
   with per-layer device from `attn_layer_dev[il]`; hop-in copies
   `normed -> sc.normed_a`, hop-out unchanged. `xin` selection keys off
   `attn_layer_dev[il] != primary` instead of `attn_dev.is_some()`.
7. Weights: the load-time placement regions switch from `attn_dev` to
   `attn_layer_dev[il]` for `blk.{il}.attn_*` (+ indexer tensors of that
   layer). Shared attn tensors (none known for GLM; verify) stay primary.

## Touch-point inventory (grep attn_dev, 33 refs)

- Keep on the Option (coarse gates): 2613 banner, 2865/2869/3058/3240/
  3299/3307 load placement (become per-layer), 2881/2899/2915 VRAM budget
  bookkeeping, 3573-3575 device enumeration, 3907 pinned cap, 4476 kv_dev
  for the KV-auto default (use the layer-0 dev or max over cards), 4748
  scratch alloc (becomes MlaScratch construction), 4888, 4960 (Mla
  fast-path gate), 5166 starvation warning, 5288/5297 (?), 5441-5445 Gqa
  hop (untouched), 5614, 5646-5650 MLA hop-in, 5687 idx proj hop,
  5718-5725 hop-out + fallback out-proj.

## Validation

- check.sh must stay PASS (single-card MLA path must be bit-identical:
  when the planner lands everything on one card the code must reduce to
  today's behavior).
- GLM live: temp-0 outputs at ctx 4096 split-vs-single should match
  token-for-token if each layer's math runs identically (same device
  class); accept last-bit drift only if a kernel actually changes.
- Capacity: probe max ctx with the split; expect ~150k+ with fp8 latent
  (f32 idx cache becomes the binding term: ~40 KB/pos aggregate).
- Perf: bench.sh interleaved split-on vs split-off at ctx 512; hop count
  per layer is unchanged (one pair), so decode should be flat. The hop
  targets change cards; watch the ptrs/PCIe bucket on the Gen3 x1 card -
  if the planner puts layers there, their hops ride a straw. Planner
  should ORDER candidate cards by link bandwidth (Gen5 x8 5060Ti #1 >
  Gen4 x4 4060 Ti > Gen3 x1 5060Ti #2) and only spill to the straw when
  needed for capacity.

## Non-goals (this arc)

- Gqa layer-split (single-card offload stays).
- fp8 idx_kcache (separate follow-up, same recipe as 18ff73f).
- Dsv4/qwen35 unchanged.
