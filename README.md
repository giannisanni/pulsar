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

Runs **Hy3 295B** (hy-v3, GQA) and **GLM-5.2 743B** (glm-dsa, MLA) on
consumer GPUs. Reference box: RTX 5060 Ti 16GB + RTX 4060 Ti 16GB,
Ryzen 9900X, 30GB RAM, one Gen5 NVMe.

| decode | pulsar | ds4/NeutronStar (reference C engine, same box) |
|---|---|---|
| Hy3 295B (85GB gguf) | **7.2 tok/s** | 0.64–0.70 |
| GLM-5.2 743B (197GB gguf) | **2.0 tok/s** | 0.40 |
| Hy3 long-prompt prefill | **7.9+ tok/s** | 0.44 |
| warm start | hot experts bulk-load in **~3s** | – |

On a single RTX 4060 Ti (where NeutronStar set its numbers): Hy3 2.6,
GLM 0.56.

**Zero-config multi-GPU.** At startup pulsar *measures* each card's H2D
bandwidth (labels lie: an x8-labeled slot can train x1, a driver bug can
park a Gen5 card at Gen1 — only a measurement sees that) and assigns
roles by what each card is actually good at:

- **Expert streaming** needs link bandwidth → the fastest measured card.
- **Attention residency** (MLA models: the whole ~14GB attn stack + KV
  parked on a second card) only needs capacity — weights cross the bus
  once at load, then only activations hop (2× 24KB per layer). A
  bandwidth-crippled card serves attention at full speed.
- **Expert tiers**: leftover cards are filled with the hottest expert
  triples from the warm census, and the MoE kernels *run on the card
  that holds the weights* — partial outputs gather back over PCIe. On
  the reference box the tier serves ~90% of expert computations and
  nearly doubles Hy3 decode.

Correctness is certified against ds4, not assumed: teacher-forced along
ds4's greedy path (15/16 per-position argmax agreement on Hy3, 10/12 on
GLM — every miss at a <0.09-logit tie), byte-identical greedy ids across
single-GPU vs attn-offload configurations, and bit-exact decode
determinism on a fixed code path (`--decode-consistency`, below).

## Requirements

- Linux (io_uring and CUDA are load-bearing; the workspace *compiles* on
  macOS but the engine is stubbed out there)
- One or more NVIDIA GPUs, Ada (sm_89) and newer by default — Blackwell
  runs via PTX JIT; `PULSAR_CUDA_ARCH` overrides codegen targets
- CUDA toolkit with `nvcc` on PATH, plus a host compiler nvcc accepts
  (gcc-12 works; newer gcc may need `CXX=g++-12` at build time)
- Rust via [rustup](https://rustup.rs)
- The model gguf on a fast NVMe — streaming reads it at up to ~7GB/s,
  so the disk *is* the decode speed
- ~16GB system RAM for the host-side expert cache (more helps; the cache
  budget is the single biggest knob after the disk)

## Get a model

Pulsar runs ds4-recipe ggufs (mixed precision: ~2-bit routed experts,
q8 attention). Two are published and known-good:

```sh
# Hy3 295B - 85GB, the friendlier starting point
curl -L -C - -o Hy3-ds4-IQ2XXS-AttnQ8.gguf \
  "https://huggingface.co/giannisan/Hy3-ds4-gguf/resolve/main/Hy3-ds4-IQ2XXS-AttnQ8.gguf"

# GLM-5.2 743B - 197GB, needs a second 16GB GPU for the attention stack
curl -L -C - -o GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf \
  "https://huggingface.co/antirez/GLM-5.2-GGUF/resolve/main/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf"
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
same census — so the second run is the fast one.

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
| `PULSAR_ATTN_GPU` | auto | CUDA index of the attention GPU (MLA models); `off` disables offload |
| `PULSAR_TIERS` | on | `off` disables resident expert tiers (also the bit-exact single-device path) |
| `PULSAR_CACHE_GB` | 12 / 22¹ | host RAM budget for the expert LFU cache |
| `PULSAR_DEV_CACHE_GB` | 3 / 1 / 8¹ | VRAM budget for the hot-expert pool on the primary |
| `PULSAR_ATTN_VRAM_GB` | all that fits | attn VRAM budget (single-GPU MLA: default 6) |
| `PULSAR_BATCH` | 256 | prefill chunk size (bigger = fewer corpus passes = faster prefill, more VRAM) |
| `PULSAR_NO_PREFETCH` | unset | set to disable the cross-layer prefetcher |
| `PULSAR_PROFILE` | unset | print per-stage wall-time profile |

¹ defaults shift with the detected topology: attn offload frees pinned
RAM (host cache 12→22) and primary VRAM (dev cache →8).

### Tests

```sh
cargo test                                        # host-side (any OS)
CXX=g++-12 cargo test -p kernels --release -- --ignored   # GPU kernel selftests vs CPU references
```

## How the streaming works

Per MoE layer, per token (or per prefill chunk as a union across the
whole batch), an expert slab resolves through:

1. **Resident tier** (spare GPUs) — the hottest expert triples live
   permanently on leftover cards; their MoE compute happens *there* and
   only activations cross PCIe. Placement, not cache: no eviction.
2. **VRAM hot-set cache** (primary GPU) — a fixed pool with touch-count
   admission: a slab earns a slot only by being hotter than the coldest
   resident, so the pool holds a *stable* hot set instead of thrashing.
3. **Host LFU cache** — RAM-budgeted, persisted to the `.warm` sidecar.
4. **io_uring + O_DIRECT** — misses are fetched at queue depth 32, and
   each completion is uploaded to the GPU while the remaining reads are
   still in flight.

A background thread additionally **prefetches the next layer's experts**,
predicted by running the next layer's router on the current layer's
input.

The MoE kernels never consult global state: every launch receives
explicit per-(token, slot) device pointers for gate/up/down, and a NULL
slot means "not mine" — which is what makes per-card partial execution
native. Where the bytes came from is the host's problem, resolved before
launch.

## Fidelity notes

- All matmuls use ds4's exact math: activations quantized to q8_0/q8_K,
  integer dp4a dots. Logit-level parity with ds4 is within quantization
  noise.
- Batched prefill and single-token decode use different reduction
  orders, so greedy near-ties (top1−top2 < ~0.5 logits) can flip between
  them — the same class of drift ds4 has between its CUDA and Metal
  backends. `--decode-consistency N` measures it; with `PULSAR_BATCH=1`
  the two paths are identical and the comparison is bit-exact (verified:
  max |Δlogit| = 0.0).
- Expert tiers split the per-slot sum across cards, which reorders float
  adds — same drift class. `PULSAR_TIERS=off` restores the single-device
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

Not yet:

- DeepSeek-family bring-up (same MLA plumbing as GLM; the DSA indexer for
  long contexts is unported — GLM runs full attention up to ctx 2048)
- MTP speculative decode (parked until batch-union expert loads, its
  measured blocker in NeutronStar)
- own BF16→quant quantizer (removes the last llama.cpp dependency from
  the model-prep pipeline)

## License

MIT. The CUDA kernels derive from the
[ds4](https://github.com/antirez/ds4) lineage (MIT) and carry their
attribution:
Copyright (c) 2026 The ds4.c authors · Copyright (c) 2023–2026 The ggml
authors.
