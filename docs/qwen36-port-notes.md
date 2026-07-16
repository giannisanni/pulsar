# qwen3.6 (Qwen3.6-35B-A3B, arch `qwen35moe`) port notes

Task #21. References:
- llama.cpp `src/models/qwen35moe.cpp` + `src/models/delta-net-base.cpp`
  (fetched to scratchpad during recon; re-fetch from ggml-org/llama.cpp
  master - the arch landed as qwen35moe with delta-net-base shared with
  qwen3next/kimi-linear)
- Luce-Org/lucebox `server/` - C++/CUDA DFlash runtime on ggml for the
  same arch (`qwen35`), plus MATCHED block-diffusion draft ggufs: their
  DFlash spec decode gets 6-8 accepted tokens/round via 22-node tree
  verify (3.4x on a 3090). That acceptance regime is where pulsar's
  batch-union verify finally makes speculation pay - the MTP lesson was
  "spec pays only when the verify union is cache-absorbed", and at 17GB
  total this model IS cache-absorbed. Post-v1 follow-up.
- Model on substrate: `/mnt/models/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf`
  (16.8GB, unsloth UD quant). Local header: /tmp/qwen36_head.bin.

## Why this model (Gianni's framing)

Pulsar serves VERY modest hardware too. 35B-A3B at Q3_K_XL is 16.8GB:
fully resident on the reference box (Gemma-class speed likely), and on
a 6-8GB card (GTX 1660 tier) the experts stream - exactly pulsar's
mechanism. 262k native context via the hybrid design: only every 4th
layer keeps a KV cache; the rest carry O(1) recurrent state.

## Shape (gguf metadata, all verified from the real header)

- arch `qwen35moe`, 40 layers, n_embd 2048, vocab 248320, ctx 262144
- attention (every 4th layer, il%4==3 i.e. blk 3,7,...,39 - from
  `full_attention_interval` = 4; llama.cpp: is_recurrent(il) =
  (il+1) % 4 != 0): 16 Q heads x head_dim 256, 2 KV heads,
  rope.dimension_count 64 (PARTIAL rotary over head_dim 256),
  freq_base 10e6, rope.dimension_sections [11,11,10,0] (M-RoPE)
- GDN layers (the other 30): ssm.conv_kernel 4, state_size 128,
  group_count 16 (= n_k_heads), time_step_rank 32 (= n_v_heads),
  inner_size 4096; head_k_dim = head_v_dim = 128; key_dim 2048,
  value_dim 4096; conv_dim = 2*2048+4096 = 8192
- MoE (EVERY layer incl. attention ones): 256 experts top-8, ff_exp
  512 (tiny!), softmax router (no gating_func key, no probs bias
  tensor), NO expert_weights_scale; shared expert ff 512 with its own
  scalar sigmoid gate (`ffn_gate_inp_shexp` [2048] -> 1 logit)
- rms_eps 1e-6; add_bos false; eos 248046 (<|im_end|>), bos-slot
  248044 (<|endoftext|>); tokenizer.ggml.pre = "qwen35"
- NO nextn tensors in this gguf (MTP ships as a separate
  Qwen3.6-35B-A3B-MTP-GGUF repo; defer)
- multimodal: mmproj ggufs exist; text-only port, skip

## Tensors

Per GDN layer (30 of 40):
- attn_qkv.weight [2048 -> 8192] q8_0: contiguous [q 2048 | k 2048 | v 4096]
- attn_gate.weight [2048 -> 4096] q8_0: the "z" gate
- ssm_conv1d.weight [4, 8192] f32 (4 taps per channel, depthwise)
- ssm_dt.bias [32] f32, ssm_a [32] f32 (STORED as -exp(A_log): negative)
- ssm_alpha.weight [2048 -> 32] f32, ssm_beta.weight [2048 -> 32] f32
- ssm_norm.weight [128] f32 (per-v-head gated RMS norm weight)
- ssm_out.weight [4096 -> 2048] q8_0
- attn_norm, post_attention_norm [2048] f32

Per full-attention layer (blk 3,7,11,...,39):
- attn_q.weight [2048 -> 8192] q8_0: 16 heads x [q 256 | gate 256]
  INTERLEAVED per head (view stride 2*head_dim in llama.cpp)
- attn_k.weight / attn_v.weight [2048 -> 512] q8_0 (2 kv heads x 256)
- attn_q_norm / attn_k_norm [256] f32 (per-head WEIGHTED rms norm)
- attn_output.weight [4096 -> 2048] q8_0 (input = 16 x 256)
- attn_norm, post_attention_norm [2048] f32

Every layer (MoE):
- ffn_gate_inp.weight [2048 -> 256] f32
- ffn_gate_exps/ffn_up_exps [2048, 512, 256] iq3_xxs (a few iq4_xs)
- ffn_down_exps [512, 2048, 256] iq4_xs (a few q5_K/q6_K)
- ffn_gate_inp_shexp [2048] f32 (scalar gate),
  ffn_{gate,up,down}_shexp 512-wide q8_0

Top-level: token_embd q8_0 [2048, 248320]; output q6_K (matmul_kq
path); output_norm f32. ALL quants already have pulsar kernels
(iq3_xxs/iq4_xs/q5_K/q6_K dots exist; q8_0 dense everywhere else).

## Decoded semantics (llama.cpp reference)

### Layer skeleton (both kinds)
x -> attn_norm -> (GDN | gated attention) -> +residual ->
post_attention_norm -> MoE FFN -> +residual (pre-norm residual: FFN
residual anchors BEFORE post_attention_norm).

### Full attention layer (build_layer_attn)
1. Qfull = wq(x) [8192]; per head h: q = Qfull[h*512 .. +256],
   gate = Qfull[h*512+256 .. +256]
2. q per-head rms norm with weight attn_q_norm [256]
3. k = wk(x), reshape [256, 2]; per-head rms norm w/ attn_k_norm; v = wv(x)
4. IMRoPE on q and k: sections [11,11,10,0], is_imrope interleaved
   mode. TEXT-ONLY REDUCTION: theta_t = theta_h = theta_w for text
   tokens (all three position ids equal), so the sector logic picks
   between EQUAL thetas -> identical to plain NEOX rope with rot_dim
   64 over head_dim 256, base 10e6. Pulsar's gqa_rope (neox) serves
   as-is with rot=64. (mrope is neox-family pairing.)
5. standard causal GQA attention, scale 1/sqrt(256), full context
   (only these 10 layers hold KV: ctx x 2 heads x 256 x 2(k,v))
6. out = attn_out * sigmoid(gate) elementwise (per head), then wo.

### GDN layer (build_layer_attn_linear + delta-net AR path)
1. qkv = wqkv(x) [8192]; z = attn_gate(x) [4096]
2. beta = sigmoid(ssm_beta(x)) [32]; alpha = ssm_alpha(x) [32];
   g = ssm_a * softplus(alpha + ssm_dt.bias) [32] (g < 0)
3. conv: rolling state of last 3 qkv rows per channel; conv_input =
   [state | qkv] (4 taps); out[c] = silu(sum_t kern[c][t]*window[t]);
   state rolls forward. (pulsar's Inkling sconv is x + conv - DIFFERENT:
   no residual here, plain conv+silu. New kernel or a flag.)
4. split conv output: q [128,16], k [128,16], v [128,32] (contiguous)
5. L2-NORMALIZE q and k per head: x / sqrt(sum(x^2) + eps). NOT rms
   norm (differs by sqrt(128) factor and has no weight).
6. repeat q,k heads 16 -> 32 (each k head serves 2 v heads)
7. delta rule per v-head h, state S [128 x 128] (S[i][j], i = k-dim,
   j = v-dim), decode step (build_delta_net_autoregressive):
     q_h *= 1/sqrt(128)
     S = S * exp(g_h)                      (scalar decay)
     sk[j] = sum_i S[i][j] * k[i]          (S^T k)
     d[j] = (v[j] - sk[j]) * beta_h
     S[i][j] += k[i] * d[j]                (rank-1 update)
     o[j] = sum_i S[i][j] * q[i]           (S^T q)
8. gated norm: per head, rms_norm(o, ssm_norm[128]) * silu(z_h)
   (pulsar's swiglu kernel computes silu(gate)*up - reuse with
   gate=z, up=normed)
9. concat 32 heads -> [4096] -> ssm_out -> [2048]

State per GDN layer: S = 32 x 128 x 128 f32 = 2MB; conv state
3 x 8192 f32 = 96KB. 30 layers ~ 63MB total. Reset at pos 0.

### MoE FFN (every layer)
- router: softmax over 256 logits, top-8, renormalize selected
  (llama.cpp SOFTMAX gating + norm_w true) = pulsar router_select
  softmax mode (qwen3moe mode 1) with weight_scale 1.0
- experts: silu gated, n_ff 512 (n_ff % 256 != 0 -> grouped path
  declines automatically; 512 IS divisible by 256 - fine either way)
- shared expert: full silu FFN 512-wide, output * sigmoid(
  ffn_gate_inp_shexp . x) (a per-token SCALAR gate - new small bit;
  fold into shared_out with a scale-by-scalar or small kernel)

### Prefill
llama.cpp has a chunked GDN prefill (build_delta_net_chunking, chunk
64, cumulative-sum decay masks, triangular solve) - complex. V1 =
sequential single-token prefill like the deepseek4 port; the model is
resident so per-token cost is small. Batched full-attn layers can't
help while GDN layers serialize anyway. Chunked GDN = the perf pass
(port the chunking math or lucebox's fused kernel).

### Tokenizer
- pre "qwen35" = qwen2 regex with \p{M} added to the letter classes:
  `(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`
  -> clone pulsar's qwen2 pretokenizer, extend letter runs to accept
  marks (\p{M}) and let the punct class exclude marks. Dispatch on
  tokenizer.ggml.pre == "qwen35".
- chat: standard ChatML (<|im_start|> 248045 / <|im_end|> 248046);
  existing ChatMl style resolves with zero changes. add_bos false.
  <think> 248068 / </think> 248069 exist (hybrid-thinking model);
  one-shot works via the dynamic stop set.

## Integration map (pulsar)

- Family: new `Qwen35` variant (hybrid: per-layer recurrent flag =
  (il+1)%4 != 0; GDN layers have NO kv cache slot, attn layers no SSM
  state). Reuse Gqa kernels for the attention layers (gqa_rope rot 64,
  gqa_head_rms_norm WITH weights, gqa_kv_append/attention with
  window 0). Reuse Ffn::Moe wholesale (softmax router mode exists;
  add the scalar shexp gate).
- New CUDA (qwen35_kernels.inc, dsv4-inc pattern):
  1. `qwen35_conv_step`: depthwise conv4 + silu over 8192 channels,
     one token, rolling f32 state [3][8192] (adapt sconv, minus the
     residual add)
  2. `qwen35_gdn_step`: the AR delta rule, one block per v-head
     (128 threads = j columns; thread j owns S[:,j] strided reads);
     inputs q,k (16-head, index h/2), v, g[32], beta[32]; outputs o
     [128 x 32]; state in-place. Selftest vs host reference.
  3. `qwen35_l2_norm`: per-head x / sqrt(sum x^2 + eps) (or a mode
     flag on gqa_head_rms_norm - it's the same reduction, different
     divisor and no weight... it already supports NULL weight; add a
     `l2` flag)
  4. q/gate deinterleave: per-head strided split of the 8192-wide q
     projection (tiny copy kernel; or 32 copy_d2d's - measure later)
- Host (engine/src/real/qwen35.rs, dsv4.rs pattern): forward loop per
  token; beta/alpha/g tiny host math OR keep on GPU (alpha/beta are
  32-wide matvecs - matmul_f32 + host readback is fine at v1);
  shexp scalar gate = read 1 float, scale shared_out.
- State: per-GDN-layer DeviceBufs (S 2MB + conv 96KB), zero at pos 0.
- Loader: straightforward; all quants supported; f32 tensors small.
  attn_qkv/attn_gate/ssm_out q8_0 via upload; ssm_* f32 host or dev.
- MTP/DFlash: separate gguf, defer to the spec-decode follow-up.

## Perf expectation

17GB resident across 2x16GB (or streamed on smaller cards). A3B
active + resident weights -> Gemma-class decode plausible (tens of
tok/s); GDN state math is ~1.5M FLOPs/layer/token, negligible. The
sequential-prefill ceiling applies until chunked GDN lands.

## Status: recon COMPLETE (header verified, llama.cpp semantics read,
## all quants supported, tokenizer/chat mapped). Implementation 0%.
## Model downloading to substrate (qwen36-dl unit). Disk cleaned:
## GLM + Inkling ggufs deleted (508GB free), warm censuses kept.
