# DFlash speculative decoding for Qwen3.6-35B-A3B (task #23)

References: Luce-Org/lucebox `server/src/common/{dflash_spec_decode.cpp,
draft_graph.cpp,dflash_capture.*}` (fetched during recon; the draft graph
comments are the spec). Draft model ON substrate:
`/mnt/models/qwen36-35b-a3b-dflash-Q8_0.gguf` (515MB, Anbeeld quant of
z-lab/Qwen3.6-35B-A3B-DFlash). Header: /tmp/dflash_head.bin (20MB pull).

## Draft model (arch `dflash-draft`, all verified from the header)

- 8 layers, n_embd 2048, ff 6144 (SwiGLU), 32 heads x 128, 4 KV heads,
  per-head q/k RMS norms [128], rope dim 128 (FULL head), rms_eps 1e-6
- `attention.causal = false` - bidirectional within the pass
- `dflash.block_size = 16`; `dflash.mask_token_id = 248070`
- `dflash.target_layer_ids = [1, 10, 19, 28, 37]`,
  `n_target_features = 10240` (5 x 2048)
- top-level: `dflash_fc.weight [10240 -> 2048]` q8_0,
  `dflash_hidden_norm.weight [2048]` f32, `output_norm.weight [2048]` f32
- NO token_embd, NO lm head: shares the TARGET's (embed noise ids with
  target token_embd; project draft hidden through target output.weight)
- rope: gguf carries yarn keys (factor 64, orig 4096) but the lucebox
  graph calls rope_ext with freq_scale=1, ext_factor=0 - PLAIN neox at
  base 10e6. Follow the reference, ignore the yarn keys.
- our gguf has NO is_swa layers, NO attn_gate, NO aux_hidden_norms, NO
  context_kv_layer_norm - the simple paths everywhere.

## Draft forward (build_draft_graph, exact order)

Inputs: noise_embed [2048 x 16] (target tok_embd rows of
[last_tok, MASK x 15]); target_hidden_cat [10240 x ctx_len] (feature
ring window); positions_q = [ctx..ctx+15], positions_k = [0..ctx+15]
(REBASED to the window - attention only sees position differences).

1. target_feat = rms(fc @ features, dflash_hidden_norm) [2048 x ctx]
2. h = noise_embed; per layer:
   - hn = rms(h, attn_norm)
   - Q = wq(hn) [4096x16] -> per-head rms * q_norm
   - Kctx/Vctx = wk/wv(target_feat) [512 x ctx]; Kn/Vn = wk/wv(hn) [512x16]
   - K = concat(Kctx, Kn) -> per-head rms * k_norm; V = concat raw
   - NEOX rope full 128: Q at positions_q, K at positions_k
   - attn = softmax(QK^T / sqrt(128)) V, NON-CAUSAL over all ctx+16
     keys, GQA 32/4
   - h += wo(attn); h += down(silu(gate(rms(h,post_attention_norm))) * up(..))
3. hidden_out = rms(h, output_norm) [2048 x 16]
4. draft logits = target.output @ hidden_out; draft_tok[i] = argmax row i;
   then draft_tok[0] = last_tok (row 0's prediction is discarded)

## Feature capture (target side)

After layer il's FULL residual (post-FFN add, the input to layer il+1)
for il in {1,10,19,28,37}: save cur [2048] into the feature ring at the
token's position. Capture during PREFILL (every prompt token) and during
VERIFY (all 16 rows; only accepted rows' features become durable - the
ring slots past `committed` get overwritten by the next verify anyway).
Ring: device [cap x 10240] f32, cap 2048 (lucebox DRAFT_CTX_MAX_DEFAULT);
window passed to the draft = last min(committed, cap) positions.

## Spec loop (dflash_spec_decode.cpp, chain mode)

per round, with `committed` = absolute position, `last_tok`:
1. noise = [last_tok, MASK x 15] -> embed -> draft forward -> draft_tok
   (draft_tok[0] = last_tok)
2. snapshot target recurrent state (GDN S + conv, per layer)
3. verify: target forward over draft_tok[0..15] at positions
   committed..committed+15, logits at EVERY row -> target_tok[i] =
   argmax(logits[i]) (target_tok[i] = prediction AFTER draft_tok[i])
4. accept_n = 1 + longest prefix where draft_tok[i+1] == target_tok[i];
   bonus = target_tok[accept_n-1] when accept_n < 16;
   commit = draft_tok[1..accept_n-1] + bonus (note draft_tok[0] is the
   PREVIOUS round's token - already emitted; commit_n = accept_n counts
   it, so newly emitted = accept_n - 1 + bonus... lucebox emits
   replay_tok[0..commit_n) = [draft_tok[0..accept_n), bonus]; out_all
   includes last_tok once. CAREFUL with the off-by-one: replay_tok[0] =
   last_tok = already-known token that just hadn't advanced the state.)
5. rollback: restore snapshot, replay the committed tokens (legacy) OR
   fast path: per-position state snapshots during verify -> point-restore
6. last_tok = target_tok[commit_n - 1]; committed += commit_n

## pulsar plan

Phase A - batched qwen35 forward (the enabler; also wins prefill):
- forward_qwen35 gains n_tok up to 16 (chain verify + prefill chunks):
  projections/MoE/attention run batched; GDN runs a NEW batched kernel
  that loops tokens INSIDE the launch with the state column held in
  registers (thread j owns S[:,j], 128 floats; k/q/v via smem per step).
  Optional per-position state persist -> [n_tok] snapshot ring for
  rollback (32 heads x 128 x 128 x 4B = 2MB/layer/pos; 16 pos x 30
  layers = 960MB - instead STASH per-position gdn INPUTS (q,k,v,g,beta
  ~33KB/layer/pos = 16MB) + raw qkv rows for conv state (32KB/layer/pos)
  and REPLAY only the tiny gdn steps after restore. Pre-verify snapshot
  = S (63MB) + conv (3MB), one copy).
- batched conv kernel: same trick, loop n_tok inside the launch.
- MoE: extend the lean dsv4_moe resolve to n_tok (union the 16x8
  selections, kernels already take n_tok).
- attention layers: existing gqa kernels batch already (n_tok, pos0).
- logits for all rows: head_logits(k=n_tok).
Verification gate: batched forward at n_tok=16 must reproduce the
sequential ids (modulo the documented batch-order drift class; on
identical prompts argmax should match at temp 0 in practice).

Phase B - draft engine:
- loader: tiny resident model (8 q8_0 layers + fc + norms); share the
  target's token_embd + output via the existing Model handles.
- draft attention kernel: non-causal, contiguous K/V buffers
  [ctx+16][4][128], 16 q rows, GQA map h/8... 32 q heads / 4 kv = 8
  (STANDARD block mapping here - ggml flash_attn_ext GQA semantics,
  head h reads kv head h / (32/4); NOT the GDN tile trap).
- feature fusion + per-layer K/V over the ring window each round
  (ctx x matmuls: at ctx 2048 that is 2048x2048x10240 fc = 43 GFLOP/
  round... too fat. lucebox caches ctx K/V per layer in a ring
  (DraftKvCacheRefs) and only computes the 16 noise rows per round +
  appends committed rows. v1: cap draft_ctx at 256 (fc over 256 rows =
  5 GFLOP, ~1ms) and NO kv cache; perf pass adds the ring. Acceptance
  may dip slightly vs the 2048 window.
- positions rebased to the window each round.

Phase C - spec loop in qwen35.rs (greedy-only, PULSAR_DFLASH=path env):
snapshot -> draft -> batched verify (features captured) -> accept ->
restore + gdn-replay -> emit. Reuse st.mtp_drafted/accepted counters.

## Perf math

Baseline 33.5 tok/s sequential. Verify(16) batches the matmuls/MoE that
dominate (~90% of time), so a round costs ~2-4 sequential-token
equivalents + draft (~1ms) and commits accept_n+bonus (lucebox measures
6-8 on the 27B; MoE 35B may differ). Expected 2-3x if acceptance holds.

## Status: SHIPPED as opt-in experimental (PULSAR_DFLASH=draft.gguf),
## 2026-07-16 late. All three phases landed + two live bugs fixed:
## (1) the draft is YARN-TRAINED (rope_scaling factor 64 / orig 4096 in
## the z-lab config; lucebox's 27B graph hardcodes plain rope, ours
## can't) - new neox yarn kernel, ggml attn_factor semantics;
## (2) feature capture follows the HF hidden_states[il] convention =
## the residual ENTERING layer il, not its output.
##
## MEASURED (substrate, Q3_K_XL target + Q8_0 draft, greedy n=64):
## - mechanism correct: counting prompt accepts FULL 16-blocks; natural
##   text ~2.8 tokens/round (12% of the 15 drafted); output coherent
##   and deterministic across runs
## - net throughput 6.1 tok/s vs 33.5 sequential = NET-SLOWER on this
##   box. Root cause is the MTP lesson verbatim: a 16-row verify unions
##   ~110 distinct experts/layer (~6.6GB weights/round) vs 8 for a
##   sequential token, and the qwen35 lean resolve has no tiers - the
##   union streams over PCIe every round while sequential decode rides
##   94% VRAM cache hits.
##
## TIER PASS DONE (2026-07-16 night): tier-aware lean resolve shipped
## (tier lookup + census keep-warm + cross-card launches + gather; the
## eval_layer pattern minus sinks/fused/MTP), build_tiers re-enabled
## for the hybrid families. 8725 triples / 12.2GB resident on the
## 4060 Ti, ~98% of expert slots tier-served. Also killed the batched-
## path sync tax (one-launch qkv split, ring scatter/gather kernels,
## on-device GDN coefficients + shexp gate - was ~1500 blocking d2d
## copies + 100 host readbacks per round). DFlash 6.1 -> 8.7 tok/s;
## sequential unchanged at ~35 (ids identical).
##
## MEASURED PROFILE (PULSAR_DFLASH_DEBUG timing): draft 47ms,
## verify(16) 260ms (tiers-off: 330ms - tier helps), replay
## ~18ms/token. Verify = 6.5ms/layer vs 0.6 sequential: the cost is
## IQ-QUANT DECODE COMPUTE in the dp4a MoE kernels - 128 expert-slot
## dots re-decode iq3_xxs/iq4_xs per token. Round commits ~2.8 tokens
## for ~12 sequential-tokens of work.
##
## Remaining moves, in value order:
## 1. unpack16_iq3_xxs + unpack16_iq4_xs (~40 lines each) -> the
##    task-#15 grouped-MMA MoE kernels decode each expert superblock
##    to int8 smem ONCE per launch. Est. verify 260 -> 60-90ms.
## 2. acceptance 2.8 -> 6+: Q4_K_XL target (feature noise vs the
##    bf16-trained draft), RING_CAP 2048, draft ctx-KV cache ring.
## 3. fast rollback (per-position GDN input stash) drops the replay.
## Break-even math: round_ms / 25ms-per-sequential-token < commits.
## All three land ~2x over sequential; any one alone does not flip it.
