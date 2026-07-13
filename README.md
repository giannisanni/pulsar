# pulsar

An inference engine for giant Mixture-of-Experts models on hardware that
has no business running them. The routed experts live on NVMe and stream
per token; everything that makes decisions stays resident in VRAM. No
llama.cpp anywhere in the stack.

Successor to [NeutronStar](https://github.com/giannisanni/neutronstar),
rebuilt as its own engine in Rust + CUDA instead of a C fork. A pulsar is
a neutron star that spins fast and emits beams.

## What it does today

Runs **Hy3 295B** (hy-v3, 192 experts, ~85GB gguf) on a single
**RTX 4060 Ti 16GB**:

| | pulsar | ds4 (reference C engine, same box) |
|---|---|---|
| decode | **2.2–2.7 tok/s** | 0.64–0.70 |
| long-prompt prefill | **5.4 tok/s** | 0.44 |
| warm start | 16GB of hot experts in **~4s** | – |

Correctness is certified against ds4, not assumed: teacher-forced along
ds4's greedy path, 15/16 per-position argmax agreement (the one miss is a
0.086-logit tie), and bit-exact decode determinism on a fixed code path
(`--decode-consistency`, below).

## Requirements

- Linux (io_uring and CUDA are load-bearing; the workspace *compiles* on
  macOS but the engine is stubbed out there)
- NVIDIA GPU, Ada (sm_89) by default — edit `-arch` in
  `crates/kernels/build.rs` for other generations
- CUDA toolkit with `nvcc` on PATH, plus a host compiler nvcc accepts
  (gcc-12 works; newer gcc may need `CXX=g++-12` at build time)
- Rust via [rustup](https://rustup.rs)
- The model gguf on a fast NVMe — streaming reads it at up to ~4.8GB/s,
  so the disk *is* the decode speed
- ~16GB system RAM for the host-side expert cache (12GB default budget)

## Quick start

```sh
git clone https://github.com/giannisanni/pulsar
cd pulsar

# build (CXX only needed if your default gcc is too new for nvcc)
CXX=g++-12 cargo build --release -p engine

# run: greedy generation
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
the hot set in a few seconds and decodes noticeably faster.

### CLI flags

| flag | meaning |
|---|---|
| `-m FILE` | model gguf (required) |
| `-p TEXT` | prompt (tokenized, BOS prepended) |
| `--chat` | interactive multi-turn chat (Hy3 chat template, KV retained) |
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

| var | default | what |
|---|---|---|
| `PULSAR_CACHE_GB` | 12 | host RAM budget for the expert LFU cache |
| `PULSAR_DEV_CACHE_GB` | 3 | VRAM budget for the hot-expert pool |
| `PULSAR_BATCH` | 256 | prefill chunk size (bigger = fewer corpus passes = faster prefill, more VRAM) |
| `PULSAR_NO_PREFETCH` | unset | set to disable the cross-layer prefetcher |

### Tests

```sh
cargo test                                        # host-side (any OS)
CXX=g++-12 cargo test -p kernels --release -- --ignored   # GPU kernel selftests vs CPU references
```

## How the streaming works

Three tiers per MoE layer, resolved per token (or per prefill chunk as a
union across the whole batch):

1. **VRAM hot-set cache** — a fixed pool of expert slabs with
   touch-count admission: a slab earns a slot only by being hotter than
   the coldest resident, so the pool holds a *stable* hot set instead of
   thrashing (one token's working set is bigger than the pool).
2. **Host LFU cache** — RAM-budgeted, persisted to the `.warm` sidecar.
3. **io_uring + O_DIRECT** — misses are fetched at queue depth 32, and
   each completion is uploaded to the GPU while the remaining reads are
   still in flight.

A background thread additionally **prefetches the next layer's experts**,
predicted by running the next layer's router on the current layer's
input.

The MoE kernels never consult global state: every launch receives
explicit per-(token, slot) device pointers for gate/up/down. Where the
bytes came from is the host's problem, resolved before launch.

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

## Status / roadmap

Done: gguf reader · io_uring disk path (parity with C at 4.8GB/s) ·
hy-v3 forward graph + kernel set with GPU-vs-CPU selftests · from-gguf
BPE tokenizer (gold-vector parity with ds4) · three-tier streaming ·
warm-cache persistence · batch prefill · cross-layer prefetch ·
temp/top-p/min-p sampling · interactive chat · OpenAI-compatible server
(`pulsar-serve`: `/v1/models`, `/v1/chat/completions` with SSE
streaming; local single-user, one request at a time).

Not yet:

- other model families (GLM/DeepSeek MLA graphs — the expert/router
  kernels already cover their quants; the attention graph is the work)
- multi-GPU expert residency (2× RTX 5060 Ti target — the reason this
  engine exists)
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
