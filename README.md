# Pulsar

![pulsar](docs/assets/pulsar-poster.png)

An inference engine for giant Mixture-of-Experts models on hardware that
has no business running them. The routed experts live on NVMe and stream
per token; everything that makes decisions stays resident in VRAM. No
llama.cpp anywhere in the stack.

Successor to [NeutronStar](https://github.com/giannisanni/neutronstar),
rebuilt as its own engine in Rust + CUDA instead of a C fork. A pulsar is
a neutron star that spins fast and emits beams.

## What it does today

Nine model architectures running on consumer GPUs: **Hy3 295B**
(hy-v3, GQA), **GLM-5.2 743B** (glm-dsa, MLA + DSA sparse attention),
**Kimi K2.7 1T** (deepseek2, MLA + YaRN), **MiniMax M3** (partial
rotary, swiglu_oai), **Gemma 4 26B-A4B** (interleaved sliding-window
attention, dual GELU FFN), **TML Inkling 1T** (no rope, learned
relative-position bias, shortconv streams, sink router; supported the
day after release), **Qwen3-235B-A22B** (qwen3moe, softmax router;
correct output on its first-ever run), **DeepSeek-V4-Flash 284B**
(deepseek4: 4-stream hyper-connection residual with Sinkhorn gates,
sink attention over a sliding window plus streaming compressed KV,
fp8/fp4 cache quantization-aware sims, token-id hash routing on the
early layers; also correct output on its first-ever run), and
**Qwen3.6-35B-A3B** (qwen35moe hybrid: Gated DeltaNet linear attention
with O(1) recurrent state on 3 of every 4 layers, sigmoid-gated full
attention on the rest - 262k context with KV on only 10 of 40 layers,
needle recall verified at 45k tokens with 19.6 tok/s decode at that
depth; prefer K-quants for it: the Q4_K_XL decodes at 51.8 tok/s where
the smaller Q3_K_XL manages 36, because iq3's codebook lookups are
decode compute the simple K-quant shifts don't pay). Reference
box: RTX 5060 Ti 16GB + RTX 4060 Ti 16GB, Ryzen 9900X, 30GB RAM, one
Gen5 NVMe.

| Model | Total | Active / token | gguf | Decode, warm | vs ds4, same box |
|---|---|---|---|---|---|
| Gemma 4 26B-A4B | 26B | 4B | 16GB (Q4_K_XL) | **41 tok/s** | – |
| Qwen3.6-35B-A3B | 35B | 3B (top-8 of 256 + shared) | 22GB (Q4_K_XL) | **51.8 tok/s** | – |
| DeepSeek-V4-Flash | 284B | ~8B (top-6 of 256 + shared) | 87GB (ds4 recipe) | **8.0 tok/s** (11.4 w/ CPU lane) | – |
| Hy3 295B | 295B | 21B (top-8 of 192) | 79GB (IQ2_XXS) | **5.3 tok/s** (7.0 w/ CPU lane) | 0.64–0.70 |
| Qwen3-235B-A22B | 235B | 22B (top-8 of 128) | 83GB (Q2_K_XL) | **4.6 tok/s** | – |
| MiniMax M3 | 428B | 23B | 134GB (Q2_K_XL) | **3.4 tok/s** | – |
| GLM-5.2 | 744B | 40B | 211GB (ds4 recipe) | **1.7 tok/s** (1.9–2.8 w/ CPU lane) | 0.40 |
| TML Inkling | 975B | 41B (6 + 2 shared) | 296GB (Q2_K_XL) | **1.6 tok/s** | – |
| Kimi K2.7 Code† | ~1T | 32B | 339GB (Q2_K_XL) | **1.3 tok/s** | – |

All figures are sustained warm decode at n=64, temp 0, second run onward.
The resident tier is placed from the popularity census, which builds over
the first full run, so measure with a warm census. Shorter generations read
higher because the per-token SSD miss rate is still climbing to steady state:
Hy3 does 8.2 tok/s at n=32 versus 5.3 at n=64. Gemma is small enough that its
whole quantized weight set lives resident in the tier, so warm Gemma is
compute-bound, not streaming-bound.

† Measured before the n=64 standardization and not yet re-run (model deleted
to free disk); the sustained rate is likely a little lower than shown, as
GLM-5.2's re-measurement confirmed (2.0 -> 1.7).

DeepSeek-V4-Flash runs its state machines (sliding-window ring,
streaming KV compressor, Sinkhorn hyper-connection gates) fully on
device and prefills in batched 16-token chunks (~14 tok/s prefill,
2.7x the first port; one router readback and one expert union per
chunk-layer, with the per-token ring/compressor/attention interleave
preserved bit-exactly). Long-context retrieval verified by needle
recall at 2.4k ctx through compressed rows.

Long context on the GQA models decodes through a split-K attention
kernel: past 4k visible rows the position scan fans out across blocks
(2k rows per split, unnormalized online-softmax partials, one combine
pass) instead of serializing inside a single block per head. Measured
on Qwen3.6 at 45k tokens: needle recall correct, decode 3.7 to 19.6
tok/s (5.3x); at 10.8k tokens recall is also verified. Short contexts
take the original kernel unchanged, so bit-exact gates and the 51.8
tok/s bench are untouched. Known remaining lever: each q-head group
re-reads its shared kv-head rows, so a kv-head-centric layout has
several-fold headroom left at depth.

A CPU expert lane (opt-in, `PULSAR_CPU=1`) computes host-cache-hit
experts on the CPU instead of uploading them: AVX2 iq2_xxs and q2_K
x q8_K kernels sustain 42 GB/s across the 9900X's cores, above the
28.7 GB/s the same bytes would cost crossing PCIe, and the dots
overlap the GPU resolve. Host-cached experts stop competing for
upload bandwidth and VRAM cache slots, so both effects compound:
DeepSeek-V4-Flash measures 8.1 to 11.4 tok/s (+41%), Hy3 5.0 to 7.0
(+40%), GLM-5.2 1.6 to 2.8 on good runs. GLM's exact gain varies run
to run with how the cache ecology settles (the floor stays at
baseline). Covers every iq2_xxs/q2_K expert mix, which includes the
Q2_K_XL giants (Qwen3-235B, MiniMax M3, Kimi, Inkling).

Decode rate slides with output length on the streaming models: a longer
generation routes to a wider set of experts, so the disk-miss fraction
creeps up until the working set saturates. Hy3 measures 5.7 tok/s at n=64,
4.5 at n=128, 4.2 at n=256, converging toward a ~4 tok/s floor set by how
much of the expert working set fits in host RAM (more RAM lifts the whole
curve). n=64 is the reported standard; long outputs run nearer the floor.
Gemma is exempt: its weights are fully resident, so no disk is in the loop.

Prefill runs the quantized weights through int8 tensor cores: Hy3
**28 tok/s** (1.8× over dp4a, ds4 0.44), GLM-5.2 **15 tok/s** (2.7×).
Warm start: hot experts bulk-load in **~3s**. (ds4 = NeutronStar, the
llama.cpp-fork predecessor, on the same box.)

Decode figures are **warm-run** (second run onward). The first run is
cold while the expert-popularity census fills; only after it is written
do the host cache and resident tiers load hot, so a cold run reads far
more from disk and clocks lower, so don't benchmark the first run. On the
reference box Gemma 4 goes 28.7 tok/s cold → 41 tok/s warm (hot experts
resident on the second GPU). See the warm-start note under Quick start.

Prefill runs the quantized weights through int8 tensor cores on
sm_80+ (`mma.m16n8k32` dense GEMM + mmq-style grouped MoE that unpacks
each expert superblock to shared memory once per prefill chunk and
rescales per quant block in registers), 1.8–2.7× over the dp4a
kernels, which remain the path on older GPUs. Decode is single-token
and memory-bound, so it is deliberately untouched: ids stay
bit-identical to the dp4a path.

GLM runs contexts past its naive 2048-row ceiling via a port of the
DSA lightning indexer (top-k row selection per token), validated
against the reference engine with a long-context retrieval probe. The
indexer's batch scorer runs on tensor cores (f16 keys, m16n8k16 with the
relu-weight epilogue fused between heads): 1.9x long-prompt prefill at
4k context, byte-identical ids vs the scalar path, and the index K cache
halves to f16 (the reference indexer ships FP8 in production).

On a single RTX 4060 Ti (where NeutronStar set its numbers): Hy3 2.6,
GLM 0.56.

**Zero-config multi-GPU.** At startup pulsar *measures* each card's H2D
bandwidth (labels lie: an x8-labeled slot can train x1, a driver bug can
park a Gen5 card at Gen1, only a measurement sees that) and assigns
roles by what each card is actually good at:

- **Expert streaming** needs link bandwidth → the fastest measured card.
- **Attention residency** (MLA models: the whole ~14GB attn stack + KV
  parked on a second card) only needs capacity, weights cross the bus
  once at load, then only activations hop (2× 24KB per layer). A
  bandwidth-crippled card serves attention at full speed.
- **Expert tiers**: leftover cards are filled with the hottest expert
  triples from the warm census, and the MoE kernels *run on the card
  that holds the weights*, partial outputs gather back over PCIe. On
  the reference box the tier serves ~90% of expert computations and
  nearly doubles Hy3 decode.

Correctness is certified against ds4, not assumed: teacher-forced along
ds4's greedy path (15/16 per-position argmax agreement on Hy3, 10/12 on
GLM, every miss at a <0.09-logit tie), byte-identical greedy ids across
single-GPU vs attn-offload configurations, and bit-exact decode
determinism on a fixed code path (`--decode-consistency`, below).

## Requirements

- Linux (io_uring and CUDA are load-bearing; the workspace *compiles* on
  macOS but the engine is stubbed out there)
- One or more NVIDIA GPUs, GTX 10-series (Pascal, sm_61) or newer, the
  default build ships native code for 10/16/20, 30, and 40-series plus
  PTX that JITs on everything else (50-series Blackwell, Volta, Hopper).
  `PULSAR_CUDA_ARCH` overrides codegen targets
- CUDA toolkit with `nvcc` on PATH, plus a host compiler nvcc accepts
  (gcc-12 works; newer gcc may need `CXX=g++-12` at build time)
- Rust via [rustup](https://rustup.rs)
- The model gguf on a fast NVMe, streaming reads it at up to ~7GB/s,
  so the disk *is* the decode speed
- ~16GB system RAM for the host-side expert cache (more helps; the cache
  budget is the single biggest knob after the disk)

## Get a model

Pulsar reads standard llama.cpp ggufs: ten routed-expert quant
formats (q2_K, q3_K, q4_0, q4_K, q5_K, q5_1, q6_K, iq2_xxs, iq2_xs,
iq3_xxs, including fused gate_up tensors and non-256-multiple expert
widths), K-quant dense tensors (requantized to q8_0 at load), tied
embeddings, split -00001-of-000NN shard sets (point `-m` at the first
shard), and both converter dialects (ds4-lineage and upstream).
Known-good starters:

```sh
# Hy3 295B - 85GB, the friendlier starting point (fromBF16 = current build)
curl -L -C - -o Hy3-ds4-IQ2XXS-AttnQ8-fromBF16.gguf \
  "https://huggingface.co/giannisan/Hy3-ds4-gguf/resolve/main/Hy3-ds4-IQ2XXS-AttnQ8-fromBF16.gguf"

# GLM-5.2 743B - 197GB, needs a second 16GB GPU for the attention stack
curl -L -C - -o GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
  "https://huggingface.co/antirez/GLM-5.2-GGUF/resolve/main/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf"

# Kimi K2.7-Code 1T - 339GB in 8 shards (unsloth UD-Q2_K_XL); download
# the folder and point pulsar at shard -00001-of-00008
```

Put the file on your fastest NVMe - decode speed is read speed.

## Quick start

```sh
git clone https://github.com/giannisanni/pulsar
cd pulsar

# build (CXX only needed if your default gcc is too new for nvcc)
CXX=g++-12 cargo build --release -p engine

# run: greedy generation (multi-GPU roles auto-detected)
./target/release/pulsar-cli \
    -m /path/to/Hy3-ds4-IQ2XXS-AttnQ8.gguf \
    -p "The capital of France is" -n 64

# or: interactive chat (multi-turn, KV cache retained across turns)
./target/release/pulsar-cli -m /path/to/model.gguf --chat

# or: OpenAI-compatible server
cargo build --release -p serve
./target/release/pulsar-serve -m /path/to/model.gguf --port 11435
curl http://127.0.0.1:11435/v1/chat/completions -d '{
  "messages": [{"role": "user", "content": "Hello!"}],
  "stream": true
}'
```

First run is cold. On exit the engine writes a `<model>.gguf.warm`
sidecar (a popularity census of expert slabs); every later run bulk-loads
the hot set in a few seconds, and expert tiers (spare GPUs) fill from the
same census, so the second run is the fast one.

### CLI flags

| flag | meaning |
|---|---|
| `-m FILE` | model gguf (required) |
| `-p TEXT` | prompt (tokenized, BOS prepended) |
| `--chat` | interactive multi-turn chat (KV retained) |
| `--system TEXT` | system prompt for chat mode |
| `--temp F` / `--top-p F` / `--min-p F` / `--seed N` | sampling (chat defaults to the gguf's `general.sampling.*`; one-shot defaults to greedy) |
| `--no-bos` | don't prepend BOS |
| `--tokens 1,2,3` | feed exact token ids instead of text |
| `-n N` | tokens to generate (default 16) |
| `--ctx N` | context size (default 2048) |
| `--dump-logits FILE` | write next-token logits as JSON and exit |
| `--teacher-force` | per-position top-5 JSONL along the given ids |
| `--decode-consistency N` | decode N steps, fresh-prefill the same sequence, compare logits |

### Tuning knobs (env vars)

Everything auto-configures; these override.

| var | default | what |
|---|---|---|
| `PULSAR_GPU` | measured | CUDA index of the expert-streaming (primary) GPU |
| `PULSAR_ATTN_GPU` | auto (MLA) | attention GPU by CUDA index. MLA models auto-offload (`off` disables); GQA models are opt-in by index: a capacity shuffle that loses on 2 GPUs at short context, pays on 3+ GPUs or long context |
| `PULSAR_KV` | f32 | `fp8` stores GQA K/V as e4m3 + per-row scale (~3.9x smaller KV, longer contexts in the same VRAM). Lossy, hence opt-in: the default keeps decode bit-exact. Runs on any GPU (storage format, no fp8 hardware needed). Measured: greedy ids unchanged on short-context runs |
| `PULSAR_TIERS` | on | `off` disables resident expert tiers (also the bit-exact single-device path) |
| `PULSAR_CACHE_GB` | measured | host RAM budget for the expert LFU cache (solved from MemAvailable) |
| `PULSAR_DEV_CACHE_GB` | solved | VRAM hot-expert pool: measured free VRAM minus staging + reserve |
| `PULSAR_ATTN_VRAM_GB` | all that fits | attn VRAM budget (single-GPU MLA: default 6) |
| `PULSAR_BATCH` | solved | prefill chunk: largest whose worst-case expert staging fits a third of free VRAM |
| `PULSAR_NO_PREFETCH` | unset | set to disable the cross-layer prefetcher |
| `PULSAR_PROFILE` | unset | print per-stage wall-time profile |

¹ defaults shift with the detected topology: attn offload frees pinned
RAM (host cache 12→22) and primary VRAM (dev cache →8).

### Tests

```sh
cargo test                                        # host-side (any OS)
CXX=g++-12 cargo test -p kernels --release -- --test-threads=1   # GPU kernel selftests vs CPU references
scripts/check.sh /path/to/model.gguf              # full commit gate (build + selftests + bit-exact decode)
```

## How the streaming works

Per MoE layer, per token (or per prefill chunk as a union across the
whole batch), an expert slab resolves through:

1. **Resident tier** (spare GPUs): the hottest expert triples live
   permanently on leftover cards; their MoE compute happens *there* and
   only activations cross PCIe. Placement, not cache: no eviction.
2. **VRAM hot-set cache** (primary GPU): a fixed pool with touch-count
   admission: a slab earns a slot only by being hotter than the coldest
   resident, so the pool holds a *stable* hot set instead of thrashing.
3. **Host LFU cache**: RAM-budgeted, persisted to the `.warm` sidecar.
4. **io_uring + O_DIRECT**: misses are fetched at queue depth 32, and
   each completion is uploaded to the GPU while the remaining reads are
   still in flight.

A background thread additionally **prefetches the next layer's experts**,
predicted by running the next layer's router on the current layer's
input.

The MoE kernels never consult global state: every launch receives
explicit per-(token, slot) device pointers for gate/up/down, and a NULL
slot means "not mine", which is what makes per-card partial execution
native. Where the bytes came from is the host's problem, resolved before
launch.

## Fidelity notes

- All matmuls use ds4's exact math: activations quantized to q8_0/q8_K,
  integer dp4a dots. Logit-level parity with ds4 is within quantization
  noise.
- Batched prefill and single-token decode use different reduction
  orders, so greedy near-ties (top1−top2 < ~0.5 logits) can flip between
  them, the same class of drift ds4 has between its CUDA and Metal
  backends. `--decode-consistency N` measures it; with `PULSAR_BATCH=1`
  the two paths are identical and the comparison is bit-exact (verified:
  max |Δlogit| = 0.0).
- Expert tiers split the per-slot sum across cards, which reorders float
  adds, same drift class. `PULSAR_TIERS=off` restores the single-device
  exact path. Attention offload does NOT drift: ids are byte-identical
  with and without it.

## Status / roadmap

Done: gguf reader · io_uring disk path (parity with C at 4.8GB/s) ·
hy-v3 + glm-dsa (MLA compact-KV) forward graphs with GPU-vs-CPU kernel
selftests · from-gguf BPE tokenizer (gold-vector parity with ds4) ·
four-tier streaming · warm-cache persistence · batch prefill ·
cross-layer prefetch · measured-bandwidth GPU role assignment · MLA
attention residency on a second GPU · resident expert tiers on spare
GPUs · temp/top-p/min-p sampling · interactive chat · OpenAI-compatible
server (`pulsar-serve`: `/v1/models`, `/v1/chat/completions` with SSE
streaming; local single-user, one request at a time).

Done since: DSA lightning indexer (GLM contexts past 2048, batch scorer
on tensor cores) · Kimi K2.7/deepseek2 with llama.cpp-exact YaRN ·
split-gguf loading · MTP + draft-free n-gram speculation (built,
measured honestly: net-slower until the host cache outruns the disk;
`PULSAR_MTP=1` / `PULSAR_NGRAM=n` to experiment) · style-aware chat
templates (Hy3/Kimi/ChatML/Gemma/MiniMax/Inkling/DeepSeek) · int8
tensor-core prefill (dense GEMM + grouped MoE) · MiniMax M3, Qwen3,
Gemma 4, TML Inkling forward graphs · opt-in fp8 e4m3 KV cache
(`PULSAR_KV=fp8`) · `pulsar-quant` recipe quantizer (BF16 gguf →
ds4-style expert mixes, iq2_xxs with imatrix, per-tensor `--map`
rules; removes llama.cpp from the model-prep pipeline; shard
streaming: `--fetch-cmd`/`--delete-shards` quantize sources bigger
than the disk one shard at a time) ·
DeepSeek-V4-Flash (deepseek4): hyper-connection residual streams,
streaming compressed KV + sink attention, indexer QAT top-k, token-id
hash routing.

Also done, honestly measured: DFlash block-diffusion speculative
decoding for Qwen3.6 (the lucebox recipe: a 515MB matched draft
proposes 16 tokens conditioned on 5 captured target hidden states, one
batched target forward verifies the block, recurrent state snapshots
roll back rejections). The machinery works - structured text accepts
whole 16-blocks - but it ships opt-in experimental
(`PULSAR_DFLASH=draft.gguf`) because on the reference box it is
experimental. Four profiling rounds took it from 6.1 to 39.7 tok/s
(resident expert tiers for the hybrid families, grouped tensor-core
MoE for verify chunks, recurrence-only fast rollback that replaces the
replay forward, a token-tiled K-quant lm head), and on the iq3-heavy
Q3_K_XL target it now BEATS sequential decode on reasoning workloads
(39.7 vs 36.3, byte-identical output to plain greedy). On the faster
Q4_K_XL target sequential decode itself jumps to 51.8 tok/s and DFlash
falls behind again: the round's remaining fixed costs (a ~95ms verify
floor of per-layer launches and router readbacks, a draft whose cost
grows with the feature window) need acceptance ~7+ to amortize, and
measured acceptance is 4.3 on math, less on prose.

Not yet:

- DFlash, remaining: draft context-KV cache ring (lucebox
  DraftKvCacheRefs - caps the draft cost at long windows), CUDA-graph
  or fused launches for the verify's per-layer fixed costs, tree
  verification (DDTree) for higher acceptance per round
- deepseek4 perf pass: batched prefill (prompts currently process
  sequentially), resident tiers + cross-layer prefetch for the dsv4
  resolve, fewer host syncs on the hyper-connection gates
- tensor-core unpackers for the remaining expert formats (iq2_xs,
  iq3_xxs, q4_K, q5_1, q2_K, q3_K, the harness takes one ~40-line
  unpacker per format)

## License

MIT. The CUDA kernels derive from the
[ds4](https://github.com/antirez/ds4) lineage (MIT) and carry their
attribution:
Copyright (c) 2026 The ds4.c authors · Copyright (c) 2023–2026 The ggml
authors.
