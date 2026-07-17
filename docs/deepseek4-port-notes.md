# deepseek4 (DeepSeek-V4-Flash) port notes

Working notes for task #22. Reference: antirez upstream ds4.c @ 80ebbc3
(saved at /tmp/ds4_upstream.c during recon; re-fetch via
`git -C ~/workspace/ds4 show origin/main:ds4.c`). Model on substrate:
`/mnt/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf`
(86.7GB, antirez ds4-recipe, all quant formats pulsar-native).

## Shape (from gguf metadata, arch string `deepseek4`)

- 43 layers, n_embd 4096, vocab 129280, ctx 1M (YaRN 65536 x 16,
  beta_fast 32, beta_slow 1)
- MoE: 256 experts, top-6 + 1 shared, ff_exp 2048, gating_func = 4,
  expert_weights_norm = 1, weights_scale 1.5, exp_probs_b bias [256]
- Attention: 64 heads, kv_head 1, key/value_length 512, q_lora 1024,
  output_lora 1024, rope dim 64 base 10000; compressor rope base 160000
- Indexer: 64 heads x 128, top_k 512 (pulsar's tensor-core scorer applies)
- Hyper-connections: n_hc = 4, sinkhorn_iterations = 20
  (`deepseek4.hyper_connection.sinkhorn_iterations`)
- Reference also carries: fp8 e4m3 KV quantization of the cache
  (dsv4_fp8_kv_quantize_row_inplace_cpu ~ line 2489), fp4 e2m1 +
  hadamard128 QAT on indexer activations (dsv4_indexer_qat_* ~ line 2570)

## Decoded semantics (ds4.c line refs)

### tid2eid hash routing (~7305)
First DS4_N_HASH_LAYER layers skip the learned router for SELECTION:
`selected[i] = ffn_gate_tid2eid[token_vocab_id][i]` (i32 table
[6][129280]). Expert WEIGHTS still computed from activations via
layer_hash_router_weights_one. Later layers: layer_topk_selected_experts
(router probs "sqrt(softplus(logit))", normalization AFTER the 6 winners
are known - see layer_router_probs_one ~7327). Engine impact: token ids
must reach the FFN stage (pulsar currently passes activations only).

### Routed expert activation (~7500)
clamp gate to +limit, up to +-limit, then silu(gate) * up, router weight
applied BEFORE down. New act_op in pulsar_glu (op 3: clamped-silu).
Clamp value: passed as `clamp` - find the metadata key (expert_...limit?)
in the model_get calls near 2019.

### Hyper-connections (hc_split_sinkhorn_one ~6332, caller ~6456)
Residual state = 4 streams x 4096. Per layer, per block (attn AND ffn
separately; also output_hc at the head):
- mix[24] = hc_fn^T . streams_concat(16384) (hc_attn_fn / hc_ffn_fn f16)
- gates: pre[i] = sigmoid(mix[i]*scale[0] + base[i]) + eps (4)
         post[i] = 2*sigmoid(mix[4+i]*scale[1] + base[4+i]) (4)
         comb[4][4] = row_softmax(mix[8+..]*scale[2] + base[8+..]) + eps,
         then SINKHORN row/col normalize x20 iters (doubly stochastic)
- block input x = sum_i pre[i]*stream[i] (verify exact reduction in
  caller ~6456-6550); block runs on x (4096); streams' = comb-mix of
  streams + post[i]*block_out (verify write-back shape)
- output_hc_{fn,base,scale} [16384->4],[4],[1]: final 4->1 merge before
  output_norm/lm head
- token_embd feeds the streams how? (check embed init - broadcast vs
  stream 0) - UNVERIFIED, read caller
- MTP head has its own hc_attn_fn (mtp.0.*)

### Attention (UNREAD - next)
Tensors: attn_q_a [4096->1024] + q_a_norm + attn_q_b [1024->32768(64x512)];
attn_kv [4096->512] + kv_a_norm; attn_output_a [4096->8192] +
attn_output_b [8192->4096] (chain direction unverified - dims look
inverted vs heads, READ THE CODE); attn_sinks [64] per-head sink logits;
attn_compressor_{kv,gate}[4096->1024], _ape [1024,4], _norm [512],
compressor rope base 160000. kv_head=1 (MQA-style latent like MLA).
Indexer has its own compressor (256-wide) + indexer.attn_q_b [1024->8192],
indexer.proj [4096->64].

### Indexer QAT (~2444-2580)
Keys AND queries pass hadamard128 + fp4 e2m1 quant-dequant before
scoring (dsv4_indexer_qat_row_inplace_cpu); KV cache rows additionally
fp8 e4m3 quant-dequant (dsv4_fp8_kv_quantize_row_inplace_cpu). Needed
for selection parity with QAT training. Pulsar has e4m3 device code
already (gqa fp8 KV); hadamard128 + e2m1 are new small device fns.

## Plan

1. Shape/metadata + loader (tensor map above, 41 kinds) - skeleton first
2. HC state plumbing: streams [4][4096] per token replaces cur; embed
   init; hc_mix kernel (gates + sinkhorn + mix, one small kernel);
   output_hc merge
3. V4 attention path (read reference first): MLA-descendant + sinks +
   compressor branch; fp8 KV cache REQUIRED (not optional) for parity? -
   check whether reference always quantizes (looks like yes)
4. Router: hash layers (token-id plumb) + gating_func 4 topk
5. Indexer: reuse tensor-core scorer + add hadamard128/e2m1 QAT pre-pass
6. act_op 3 clamped-silu in pulsar_glu
7. Chat template "chat-v2" - read ds4 chat code for markers; EOS
   resolution now dynamic (stop-set fix shipped)
8. MTP gguf separate - defer

Projected decode: ~8B active, ~1.7GB/token reads -> 10-15 tok/s on the
reference box.

## Session-2 recon additions

### Attention core (layer_attention_rows_one ~7045)
K and V are the SAME 512-wide latent row (score dot AND value accumulate
read kv_rows). kv_head=1. Per-head SINK logit: max/denominator include
sinks[h], no value contribution. scale 1/sqrt(512).

### Output projection (layer_grouped_out_one ~7100)
64 heads x 512 = 32768 -> 8 GROUPS of 8 heads (group_dim 4096);
attn_output_a = 8 banks of [4096 -> 1024] (tensor [4096, 8192]); concat
8x1024 = 8192 -> attn_output_b [8192 -> 4096]. Q8_0 both.

### Compressor (compressor_decode_one ~8663) - KV cache compression
Per position: kv_cur/sc_cur = attn_compressor_{kv,gate} projections of x
(q8_0 pair matvec); sc_cur += ape[:, pos % R] (additive PE). Rows land in
rolling state [2R x width] (R = compress_ratio, coff=2 when R=4). Every
R-th position: score-weighted pool (compressor_pool_decode_state) ->
rms*norm -> rope at compressed position (pos+1-R) -> fp8 e4m3 quant
(main, head_dim 512) or hadamard+fp4 QAT (indexer, head_dim 128) ->
emitted as ONE compressed cache row. So the cache = compressed history
rows + recent raw rows (+ sliding window n_swa - read attention assembly
still TODO). prefill has a batch path + compressor_finish_prefill_state.

### Hyper-connections callers (~6440-6560) - COMPLETE
- init: hc_from_plain_embedding = all 4 streams get the token embedding
- pre (per block): flat = rms_norm_no_weight(concat 4 streams, 16384) as
  ONE norm; mix[24] = hc_fn^T flat (f16); split via sinkhorn fn;
  block_in = sum_src pre[src] * streams_RAW[src]; keep post[4], comb[16]
- post: stream'[dst] = post[dst]*block_out + sum_src comb[dst + src*4] *
  stream[src]  (comb addressed [dst, src] with dst fastest)
- attn and ffn each have their own fn/scale/base; output_hc_* merges the
  4 streams before output_norm (exact merge form: check output_hc use)

### Still to read
- layer_topk_selected_experts + sqrt(softplus) probs + gating_func 4 +
  hash router weights (~7327-7460)
- attention row assembly at decode/prefill: sliding window n_swa + raw +
  compressed rows + q/kv construction (q_a/q_a_norm/q_b, kv/kv_a_norm,
  rope split 64 of 512, fp8 cache quant call sites)
- chat-v2 template markers; clamp metadata key for expert act
- output_hc merge exact form

## Session-2 final: router decoded - RECON COMPLETE

### Router (layer_router_probs_one ~7327, topk ~7370)
- probs[i] = sqrt(softplus(logit_i)), logits = ffn_gate_inp . x (f16)
- SELECTION: probs + exp_probs_b bias, top-6 (biased select, like V3
  noaux); hash layers replace selection with tid2eid[token]
- WEIGHTS: unbiased probs[selected], normalized by their sum (floor
  6.1035e-5), x expert_weights_scale (1.5)
- pulsar mapping: new softmax_mode in router_select (sqrt-softplus +
  sum-norm); bias input already exists. Hash layers: engine-side selected
  override + weights via the same probs kernel.

## Integration map (pulsar anchors, engine/src/lib.rs @ dc2928b)

- Family enum (line ~32): add Dsv4 variant; arch parse at ~166
  (Some("deepseek4") => Family::Dsv4)
- Shape: add n_hc(4), hc_sinkhorn(20), compress_ratio (READ KEY from
  metadata - grep model_get near ds4.c:2019 for the exact name),
  n_swa (attention.sliding_window), out_groups(8)/out_rank(1024),
  clamp limit key; reuse n_idx_head/n_idx_dim/indexer_top_k, n_lora_q,
  yarn fields
- LayerW (~455): add `dsv4: Option<Dsv4W>` beside ink/gemma. Dsv4W =
  { q_a, q_a_norm, q_b, kv, kv_a_norm, out_a, out_b, sinks,
    comp_{ape,kv,gate,norm}, idx_q_b, idx_proj, idx_comp_{ape,kv,gate,norm},
    hc_attn_{fn,scale,base}, hc_ffn_{fn,scale,base}, tid2eid (host Vec<i32>
    or device), exp_probs_b via Ffn::Moe.probs_b }
  output_hc_{fn,scale,base} on Model.
- Loader anchor ~1516 (`let t = |suffix|`): all 41 names in the schema
  section above; tid2eid is I32 [6][129280] - load host-side (768KB per
  layer x first-N layers; check DS4_N_HASH_LAYER value in ds4.c)
- Forward: HC streams state = 4x4096 per token (State buffers x4);
  embed -> hc_from_plain_embedding; per layer: hc_pre(attn) -> attn block
  -> hc_post -> hc_pre(ffn) -> moe -> hc_post; final output_hc merge ->
  output_norm -> lm head
- KV cache: single 512-wide latent per position, fp8 e4m3 ALWAYS (rope
  64 within row; reuse gqa fp8 helpers); compressed rows via compressor
  state machine (state [2R x width] per layer device-side); indexer cache
  128-wide with hadamard128+e2m1 QAT rows
- CUDA new: hc_mix kernel (rms16384 + matvec24 f16 + sinkhorn + weighted
  sum - can be ONE kernel, all tiny), sinks in attention softmax (extend
  gqa/new kernel over latent rows), compressor pool kernel, hadamard128 +
  e2m1 device fns, act_op 3 clamped-silu in pulsar_glu, router mode
  sqrt-softplus
- Chat template: ds4_chat_append_message ~22410 (read when wiring --chat;
  one-shot works with the dynamic stop set already)
- Attention row assembly (decode): READ ds4.c ~8563-8663 + callers for
  how raw window + compressed rows compose per query - the ONE remaining
  unread region (prefill batch path included)

## PERF PASS PHASE 1 (2026-07-17): Sinkhorn/HC-mix and the streaming
## compressor now run FULLY ON DEVICE (dsv4_sinkhorn + dsv4_hc_mix_dev
## + dsv4_comp_step kernels, both validated against serial C
## references in dsv4_selftest; host math kept as the Rust unit-test
## reference). Per-layer host syncs: ~5 -> 1 (the router logits read).
## MEASURED: short-ctx decode NEUTRAL (~7.7-8.0 - dsv4 decode is
## expert-streaming-bound, the syncs were hiding under fetch waits);
## LONG-CTX decode +41% (6.68 vs 4.74 tok/s at 2.4k ctx - the
## per-token host compressor pool was the long-ctx cost); needle ids
## BYTE-IDENTICAL; short fresh prompts show greedy near-tie drift
## (device expf vs host exp ulps - the documented drift class).
## Tiers arrived earlier the same day (5.9 -> 8.0).
##
## PHASE 2 DONE (2026-07-17, same session): chunked batched prefill
## SHIPPED. 16-chunks: matmuls/norms/rope/hc/router/shexp/MoE batched
## (ONE router readback + ONE grouped-tier MoE union per chunk-layer);
## ring append + comp steps + attention stay per-token inside the
## chunk (the load-bearing interleave); indexer masking regime (ctx >
## 2048ish) gates chunks back to single-token. MEASURED: needle
## prefill 2368 tokens 460s -> 169s (5.1 -> 14.0 tok/s, 2.7x), ids
## BYTE-IDENTICAL, decode unchanged (7.9), selftests + Hy3 gate green.
## Remaining prefill wall: the per-(token,group) grouped-out launches
## (128/layer/chunk) and per-token attention - CUDA-graph territory.
##
## PHASE 3 (2026-07-17): banked out-proj SHIPPED, and it settled the
## wall question. pulsar_q8_0_matmul_banked collapses the t*8
## per-(token,group) out-proj launches (plus their per-call prequants,
## ~256 launches/layer/chunk) into ONE prequant + ONE kernel: heads is
## [t][n_out_group] contiguous slices and low is [t][n_out_group]
## contiguous rank rows, so pseudo-row j just reads weight bank
## j % n_bank. Accumulation order matches the warp8 kernel -> BITWISE
## identical (memcmp selftest at t=1/5/16 + needle ids byte-identical).
## MEASURED: needle prefill 168.25s vs 169s = NEUTRAL. With vram cache
## at 13% hits the chunk-layer budget is eaten by expert H2D streaming,
## not launches - same lesson as phase-1 short-ctx decode. Verdict:
## dsv4 prefill is STREAMING-BOUND; CUDA-graphing the per-token
## attention/comp interleave would buy nothing here either. Launch-side
## work on this arch only pays where streaming is absent (long-ctx
## decode, resident models). Kernel kept: fewer launches, less code,
## decode canonical 8.1 tok/s (unchanged).
##
## PHASE 2 original design notes: chunked batched prefill. On-device state
## makes it feasible; layer flow = batched [hc_pre/norms/q-kv/rope/
## fp8] -> per-token sequential [ring append, comp steps, attention]
## -> batched [un-rope, grouped out, hc_post, ONE union MoE per chunk
## via dsv4_moe(n_tok) which already has grouped tier launches].
## CAREFUL: the raw SWA ring (cap 128) gets clobbered for earlier
## tokens if all appends land before attention - the per-token
## interleave above is load-bearing. Expected: prefill ~5 -> tens of
## tok/s (one expert-corpus pass per 16 tokens instead of per token).
##
## Status: SHIPPED 2026-07-16. Compiled first try; dsv4_selftest PASS;
## check.sh PASS (existing archs bit-exact); correct output on the
## first-ever run; needle recall at 2.4k ctx (compressed rows + live
## indexer masking); chat template working. bench.sh canonical:
## 5.9 tok/s warm n=64 (fastest big-MoE in the table; Hy3 is 5.3).
## Perf ceilings for the next pass: sequential prefill (~5 tok/s),
## no tiers/prefetch for this arch, ~86 host syncs/token.

### Session-3: remaining recon answers (all former unknowns closed)
- DS4_N_HASH_LAYER = 3 (`deepseek4.hash_layer_count`)
- compress ratios: `deepseek4.attention.compress_ratios` u32 array;
  Flash = layers 0,1 raw-only, then even il = 4, odd il = 128.
  ratio-4 layers carry compressor (width 1024) + indexer (width 256);
  ratio-128 layers compressor only (width 512); coff = ratio==4 ? 2 : 1
- clamp: `deepseek4.swiglu_clamp_exp` f32 array, constant 10.0
- output_hc merge: rms_norm_no_weight(flat 16384) -> output_hc_fn
  [16384->4] -> w[i] = sigmoid(pre*scale[0]+base[i]) + hc_eps ->
  weighted stream sum -> output_norm -> q8_0 head
- attention row assembly (decode): raw SWA ring (cap n_swa=128, f16
  rows, fp8-sim'd nope part) + ALL compressed rows (ratio-4 masked by
  indexer top-512 over the indexer comp cache); ratio-0 layers attend
  over the raw ring only. Sink logit per head joins max+denominator.
  Q gets a PER-HEAD WEIGHTLESS RMS NORM after q_b; heads get an
  INVERSE rope tail before the grouped output projection.
- rope: dense layers base 10000 plain; compressed layers base 160000
  YaRN 1/16 with attn_factor = 1/(1+0.1 ln 16) cancelling the helper's
  internal mscale. Adjacent-pair rotation on the LAST 64 of 512.
- fp8 sim: 64-wide blocks over the 448 nope dims, pow2 scale
  ceil(log2(amax/448)), e4m3fn nearest-even, clamp +-448. Raw AND comp
  rows also f16-round on push. Indexer rows: hadamard128 (orthonormal)
  + e2m1 {0,.5,1,1.5,2,3,4,6} 32-block pow2 QAT.
- DS4_NEG_INF = -1e30 (finite; the pool's empty-check needs it)
- chat template: bos + bare system text; <U+FF5C>User<U+FF5C>text;
  <U+FF5C>Assistant<U+FF5C> + </think> opener (ids 128803/128804,
  think 128821/128822); assistant history closes with eos

### Implementation shape (commit 3d7c4ed)
- Family::Dsv4; Attn::Dsv4(Box<Dsv4W>); Ffn::Moe reused (gate_inp
  converted f16->f32); LayerW.attn_output = attn_output_b
- engine/src/real/dsv4.rs: forward_dsv4 (single-token loop; rows<=1),
  host Sinkhorn/router/compressor/indexer in reference f32; lean
  expert resolve (dev cache + host LFU + io_uring; NO tiers/prefetch/
  grouped/MTP yet); kcache = raw ring, vcache = comp rows,
  indexer comp cache host-only
- CUDA dsv4_kernels.inc: pulsar_dsv4_hc_mix / _attention / _fp8_sim /
  _f16_round + selftest; act_op 3 in pulsar_glu; mla rope tail gained
  sin_sign + pulsar_dsv4_rope_tail(inverse)
- f16 weights: compressor/indexer q_b requantized to q8_0; router/HC
  fns/proj converted to f32; token_embd f16 -> q8_0 pinned

### Next (on substrate)
1. cargo build + cargo test -p kernels (dsv4_selftest) + engine host
   tests; fix compile fallout
2. load-test vs the real gguf (validates all 41 tensor bindings)
3. first decode: -p "Question: what is 2+2? Answer:" greedy, coherence
4. scripts/check.sh (existing archs must stay bit-exact) + bench.sh
5. ceilings noted: no tiers/prefetch/batched prefill; prefill is
   decode-speed; ~86 host syncs/token (hc mix + router readbacks);
   attn budget default 8GB resident
