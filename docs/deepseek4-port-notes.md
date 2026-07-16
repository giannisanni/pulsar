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
