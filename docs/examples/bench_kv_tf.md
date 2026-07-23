# bench_kv_tf.sh

Teacher-forced `PULSAR_KV` quality sweep — the non-chaotic counterpart to
[`bench_kv.sh`](./bench_kv.md). Force-feeds the **same** token sequence
through every KV format and compares per-position logits against an f32
baseline. No autoregressive feedback → no chaos amplification, and the f32
noise floor collapses to ~100% top-1 agreement / 0.0 mean |Δlogit|.

Use this for correctness work; use `bench_kv.sh` only for a quick tok/s +
greedy-divergence smoke.

```sh
cd /path/to/pulsar
MODEL=/home/cesar/models/qwen3moe.gguf ./docs/examples/bench_kv_tf.sh
```

GPU auto-selection is lifted verbatim from [`runpulsar.sh`](./runpulsar.md).

## Build prerequisites (one-time)

```sh
CXX=g++-12 cargo build --release -p engine
```

`python3` is also required (per-position logit comparison runs inline).

## Why teacher-forcing

`bench_kv.sh` compares greedy output, where one flipped argmax at token #5
cascades into ~100% mismatch via autoregressive chaos — total mismatch
becomes noise, not signal. Teacher-forcing removes the feedback loop: each
position's logits are computed from the *same* forced context regardless
of what any format predicted, so differences are pure KV quant error.

Three signals per format, aggregated over every position:

- **top-1 agreement %** — fraction of positions where the argmax matches f32.
- **mean |Δlogit|** — average absolute logit delta at the top spot (max in parens).
- **top-5 Jaccard** — overlap of the top-5 id sets vs f32 (1.0 = identical).

The `f32xf32` noise floor should read ~100% / 0.0000 / Jac 1.00. If it
does, the method is sound and per-format numbers are directly comparable.
If the floor is not clean, the run is unstable — re-run.

## All environment variables

### Script-level

| var | default | what |
|---|---|---|
| `MODEL` | _(required)_ | path to a GQA/Qwen35 family gguf (NOT glm-dsa / GLM-5.2) |
| `PROMPT` | _(~120-token Fibonacci paragraph)_ | forced token sequence; longer = more positions |
| `PROMPT_FILE` | unset | read the prompt from this file (overrides `PROMPT`) |
| `FMTS` | `f32 fp8 fp16 int8 q8_0 q4_0` | formats to sweep |
| `PULSAR_CLI` | `target/release/pulsar-cli` | override CLI binary path |

For serious validation, point `PROMPT_FILE` at a real corpus paragraph —
more positions tighten the statistics. The default prompt gives ~100+
positions, enough for a stable mean.

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
  MODEL=/home/cesar/models/qwen3moe.gguf ./docs/examples/bench_kv_tf.sh
```

### Memory / CPU lane

Same budgets as `runpulsar.sh`: `PULSAR_CACHE_GB`,
`PULSAR_CACHE_HEADROOM_GB`, `PULSAR_ATTN_VRAM_GB`,
`PULSAR_ATTN_TIER_RESERVE_GB`, `PULSAR_CPU`, `PULSAR_CPU_STEAL`. The
script runs `unset PULSAR_TIERS`, so exporting `PULSAR_TIERS` from your
shell has no effect through this launcher.

## What it does

1. teacher-force f32 → baseline JSONL.
2. teacher-force f32 again → `f32_noise` (the noise floor).
3. teacher-force each format in `FMTS` → per-format JSONL.
4. inline `python3` compares every format's JSONL against the f32 baseline.

Output JSONL emitted by `pulsar-cli --teacher-force`, one line per
position:

```json
{"pos":N,"after":ID,"top":[[id,logit],...]}   // top-5 per position
```

For non-f32 formats the script sets `PULSAR_KV="$fmt"` and greps stderr
for the `"$fmt KV cache on"` activation line; if missing, warns that the
format did not engage (model is not GQA/Qwen35).

## How to read the output

```
quality (per-position logits vs f32 baseline; chaos-free):
  f32xf32 (noise floor):
    113 pos | top-1 agree 100.0% | mean |Δlogit| 0.0000 (max 0.000) | top-5 Jac 1.00
  fp8 vs f32:
    113 pos | top-1 agree  94.7% | mean |Δlogit| 0.0380 (max 0.911) | top-5 Jac 0.89
  fp16 vs f32:
    113 pos | top-1 agree  96.5% | mean |Δlogit| 0.0287 (max 0.704) | top-5 Jac 0.91
```

Sanity ordering for the scalar formats (no shared bug): `fp16 > int8 > fp8`,
all Jac ~0.9, all top-1 ~95%. A block format (q8_0 / q4_0) sitting far
below that — or 4-bit tied with 8-bit — signals a kernel defect, not
precision loss; investigate the dequant path.

## Cookbook

```sh
# full sweep, auto topology
MODEL=/data/qwen3moe.gguf ./docs/examples/bench_kv_tf.sh

# block formats only (the ones under investigation)
MODEL=/data/qwen3moe.gguf FMTS="f32 q4_0 q8_0" ./docs/examples/bench_kv_tf.sh

# more positions = tighter stats — point at a real corpus
PROMPT_FILE=/data/wiki.txt MODEL=/data/qwen3moe.gguf ./docs/examples/bench_kv_tf.sh

# pin topology, no CPU lane
PULSAR_CPU=off CUDA_VISIBLE_DEVICES=2,1 PULSAR_GPU=0 PULSAR_ATTN_GPU=1 \
  MODEL=/data/qwen3moe.gguf ./docs/examples/bench_kv_tf.sh
```

## Troubleshooting

| symptom | fix |
|---|---|
| `pulsar-cli not found` | `cargo build --release -p engine` |
| `python3 not found` | install python3 (logit comparison needs it) |
| `(no overlapping positions — check logs)` | `--teacher-force` emitted no JSONL; inspect `$TMP/*.log` |
| `WARN: <fmt> did not activate` | model is not GQA/Qwen35; PULSAR_KV is ignored by MLA / Dsv4 |
| noise floor not ~100% / 0.0000 | run is unstable; re-run, or pin topology |
| a format's per-position count < baseline | the format crashed mid-run and emitted fewer positions; check its log |
| `nvidia-smi not found` | install the NVIDIA driver / put it on `PATH` |
| `no capable GPUs` | lower `PULSAR_MIN_VRAM_MB`, or check the card is above the denylist |

## Requirements

- Linux + NVIDIA GPU (GTX 10-series / sm_61 or newer); `nvcc` on `PATH`
- `pulsar-cli` built (`CXX=g++-12 cargo build --release -p engine`)
- `python3` (inline logit comparison)
- A GQA/Qwen35 family gguf — PULSAR_KV is a no-op on MLA / Dsv4 caches
