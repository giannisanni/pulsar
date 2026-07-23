# bench_kv.sh

Quick `PULSAR_KV` sweep: tok/s + greedy-id quality diff vs an f32 baseline.
Cycles every KV format through the same prompt and prints (a) decode
throughput and (b) where the greedy token stream first diverges from f32.

GPU auto-selection is lifted verbatim from [`runpulsar.sh`](./runpulsar.md)
(denylist + PCIe scoring + free-VRAM attn pick + auto `CACHE_GB` /
`ATTN_VRAM`). Keep the two in sync if you touch the topology logic.

```sh
cd /path/to/pulsar
MODEL=/home/cesar/models/qwen3moe.gguf ./docs/examples/bench_kv.sh
```

## Build prerequisites (one-time)

```sh
CXX=g++-12 cargo build --release -p engine
```

The script errors with a build hint if `pulsar-cli` is missing.

## When to use this vs `bench_kv_tf.sh`

This script compares **greedy output**. That is fast and cheap, but useless
once the engine's f32 noise floor reaches ~token #5 — one flipped argmax
cascades into ~100% mismatch via autoregressive chaos, so the *total*
mismatch count is dominated by chaos, not by KV quant error.

- **Read FIRST divergence, not the total.** A format whose first-divergence
  sits near the f32-vs-f32 noise floor is indistinguishable from
  nondeterminism → KV quant is correct. A format that diverges *much*
  earlier than the floor has a real bug.
- **For logit-level validation, use [`bench_kv_tf.sh`](./bench_kv_tf.md).**
  Teacher-forcing removes the chaos and gives top-1 agreement / mean
  |Δlogit| / top-5 Jaccard per position. Prefer it for correctness work.

## All environment variables

### Script-level

| var | default | what |
|---|---|---|
| `MODEL` | _(required)_ | path to a GQA/Qwen35 family gguf (NOT glm-dsa / GLM-5.2) |
| `PROMPT` | `List the first eight Fibonacci numbers…` | prompt text |
| `N` | `512` | tokens to generate (bigger = KV effect clearer) |
| `FMTS` | `f32 fp8 fp16 int8 q8_0 q4_0` | formats to sweep |
| `PULSAR_CLI` | `target/release/pulsar-cli` | override CLI binary path |

### GPU selection (same as runpulsar.sh)

| var | default | what |
|---|---|---|
| `PULSAR_MIN_VRAM_MB` | `8192` | min total VRAM to be a candidate |
| `PULSAR_GPU` | auto (local `0`) | force the stream-primary CUDA index |
| `PULSAR_ATTN_GPU` | auto (local `1`) | force the attention CUDA index |
| `CUDA_VISIBLE_DEVICES` | auto-remapped | restrict which physical GPUs are visible |

Pre-set both `CUDA_VISIBLE_DEVICES` and `PULSAR_GPU` to skip auto-pick:

```sh
CUDA_VISIBLE_DEVICES=2,1 PULSAR_GPU=0 PULSAR_ATTN_GPU=1 \
  MODEL=/home/cesar/models/qwen3moe.gguf ./docs/examples/bench_kv.sh
```

### Memory / CPU lane

Same budgets as `runpulsar.sh`: `PULSAR_CACHE_GB`,
`PULSAR_CACHE_HEADROOM_GB`, `PULSAR_ATTN_VRAM_GB`,
`PULSAR_ATTN_TIER_RESERVE_GB`, `PULSAR_CPU`, `PULSAR_CPU_STEAL`. The script
runs `unset PULSAR_TIERS`, so exporting `PULSAR_TIERS` from your shell has
no effect through this launcher.

## What it does

1. **warmup** — one f32 run (loads weights into the host expert cache).
2. **noise floor** — f32 again; greedy ids drift from GPU nondeterminism
   (CUDA atomics + reduction order make even the bit-exact path drift).
3. **tok/s** — one run per format in `FMTS`.
4. **quality** — greedy id stream diff vs f32, computed with `awk`:
   `X/Y differ, first at #N`. Prints `f32xf32` (noise floor) first, then
   each format.

For non-f32 formats it sets `PULSAR_KV="$fmt"` and greps stderr for the
`"$fmt KV cache on"` activation line; if missing, warns that the format
did not engage (model is not GQA/Qwen35).

## How to read the output

```
quality (greedy ids — FIRST divergence is the real signal;
          total mismatches is dominated by autoregressive chaos):
  f32xf32 3/512 differ, first at #47    ← noise floor
  f32     baseline (512 tokens)
  fp8     4/512 differ, first at #47    ← same as floor → correct
  q4_0    512/512 differ, first at #1   ← diverges at #1 → broken
```

A format is **correct** iff its first-divergence is at or beyond the noise
floor's first-divergence. Total mismatch count is noise — ignore it.

## Cookbook

```sh
# full sweep, auto topology
MODEL=/data/qwen3moe.gguf ./docs/examples/bench_kv.sh

# subset only
MODEL=/data/qwen3moe.gguf FMTS="f32 q4_0 q8_0" ./docs/examples/bench_kv.sh

# longer generation — KV effect clearer, chaos worse
MODEL=/data/qwen3moe.gguf N=2048 ./docs/examples/bench_kv.sh

# pin topology, no CPU lane
PULSAR_CPU=off CUDA_VISIBLE_DEVICES=2,1 PULSAR_GPU=0 PULSAR_ATTN_GPU=1 \
  MODEL=/data/qwen3moe.gguf ./docs/examples/bench_kv.sh
```

## Troubleshooting

| symptom | fix |
|---|---|
| `pulsar-cli not found` | `cargo build --release -p engine` |
| `WARN: <fmt> did not activate` | model is not GQA/Qwen35; PULSAR_KV is ignored by MLA / Dsv4 |
| every format diverges at ~#1–#5 | you are reading total mismatches; read first-divergence vs the `f32xf32` floor |
| first-divergence moves around run-to-run | that is the noise floor; use `bench_kv_tf.sh` for chaos-free logit comparison |
| `nvidia-smi not found` | install the NVIDIA driver / put it on `PATH` |
| `no capable GPUs` | lower `PULSAR_MIN_VRAM_MB`, or check the card is above the denylist |

## Requirements

- Linux + NVIDIA GPU (GTX 10-series / sm_61 or newer); `nvcc` on `PATH`
- `pulsar-cli` built (`CXX=g++-12 cargo build --release -p engine`)
- A GQA/Qwen35 family gguf — PULSAR_KV is a no-op on MLA / Dsv4 caches
