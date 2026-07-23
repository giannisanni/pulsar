# runpulsar.sh

Generic Pulsar launcher. Auto-picks the best GPUs on the box, solves the
cache/attn/CPU lane budgets from measured hardware, prints the resolved
topology, then `exec`s `pulsar-cli` (generate / chat) or `pulsar-serve`
(serve).

Everything is env-driven. No flags to remember — set vars, run the script.

```sh
cd /path/to/pulsar
./docs/examples/runpulsar.sh            # MODE=generate (default)
MODE=chat   ./docs/examples/runpulsar.sh
MODE=serve  ./docs/examples/runpulsar.sh
```

## Build prerequisites (one-time)

```sh
# engine + CLI (generate, chat)
CXX=g++-12 cargo build --release -p engine

# server (serve mode only)
cargo build --release -p serve
```

The script errors with a build hint if the binary it needs is missing.

## The three modes — `MODE`

| `MODE` | binary | what it does | profile |
|---|---|---|---|
| `generate` (default) | `pulsar-cli` | one-shot: `-p "$PROMPT" -n $N`, exits | `PULSAR_PROFILE=1` forced |
| `chat` | `pulsar-cli` | interactive multi-turn, KV retained across turns | `PULSAR_PROFILE=1` forced |
| `serve` | `pulsar-serve` | OpenAI-compatible HTTP + SSE server, long-lived | left to env |

### generate

```sh
./docs/examples/runpulsar.sh
PROMPT="Explain MoE routing" N=128 ./docs/examples/runpulsar.sh
MODEL=/data/Hy3.gguf PROMPT="Hello" N=32 ./docs/examples/runpulsar.sh
```

### chat

```sh
MODE=chat ./docs/examples/runpulsar.sh
MODE=chat ./docs/examples/runpulsar.sh --system "You are concise"
```

`--system`, `--temp`, `--top-p`, etc. pass through as trailing args (see
[Pass-through args](#pass-through-args)).

### serve

```sh
cargo build --release -p serve               # once
MODE=serve ./docs/examples/runpulsar.sh
MODE=serve PORT=8080 HOST=0.0.0.0 ./docs/examples/runpulsar.sh
```

Open the **chat web UI** in a browser:

```
http://127.0.0.1:11435/        # same host:port as the API
```

The UI (dark theme, SSE streaming, sampling knobs, system prompt, stop
button) is embedded into the `pulsar-serve` binary — no separate build
step, no Node/npm. It talks to `/v1/chat/completions` on the same origin.

Or hit the API directly from another shell:

```sh
curl http://127.0.0.1:11435/v1/chat/completions -d '{
  "messages": [{"role":"user","content":"Hello!"}],
  "stream": true
}'
```

Endpoints: `/` (web UI), `/v1/models`, `/v1/chat/completions`. Single-user,
one request at a time (per engine constraint).

## All environment variables

### Script-level

| var | default | what |
|---|---|---|
| `MODE` | `generate` | `generate` \| `chat` \| `serve` |
| `MODEL` | `/home/cesar/models/GLM-5.2-…gguf` | path to the model gguf (or first shard) |
| `PROMPT` | `The capital of France is` | prompt text (generate mode only) |
| `N` | `64` | tokens to generate (generate mode only) |
| `PORT` | `11435` | serve listen port |
| `HOST` | `127.0.0.1` | serve listen host (`0.0.0.0` to expose) |
| `PULSAR_CLI` | `target/release/pulsar-cli` | override CLI binary path |
| `PULSAR_SERVE` | `target/release/pulsar-serve` | override serve binary path |

### GPU selection

The script scores every visible GPU by `pcie.link.gen.max * pcie.link.width.max`
(stable capability — idle cards downtrain to gen1, which would randomize the
pick on current link). Primary = highest PCIe score; attention = most free
VRAM among the rest. Cards below `PULSAR_MIN_VRAM_MB` or on the denylist
(GT 1030/1050/1060, 1650 Max-Q, MX150/250/330, UHD, P600/P620) are hidden.

| var | default | what |
|---|---|---|
| `PULSAR_MIN_VRAM_MB` | `8192` | min total VRAM to be a candidate |
| `PULSAR_GPU` | auto (local index `0`) | force the stream-primary CUDA index |
| `PULSAR_ATTN_GPU` | auto (local index `1`) | force the attention CUDA index |
| `CUDA_VISIBLE_DEVICES` | auto-remapped | restrict which physical GPUs are visible |

**Skip auto-pick entirely:** pre-set both `CUDA_VISIBLE_DEVICES` and
`PULSAR_GPU` — the script detects the pair and uses them verbatim:

```sh
CUDA_VISIBLE_DEVICES=0,1 PULSAR_GPU=0 PULSAR_ATTN_GPU=1 ./docs/examples/runpulsar.sh
```

### Memory / cache budgets

| var | default | what |
|---|---|---|
| `PULSAR_CACHE_GB` | auto | host RAM budget for the expert LFU cache (solved from `MemAvailable`) |
| `PULSAR_CACHE_HEADROOM_GB` | `16` | GB subtracted from `MemAvailable` before reporting cache budget |
| `PULSAR_ATTN_VRAM_GB` | auto (dual) / unset (single) | GiB of the MLA stack kept in VRAM on the attn card. `off` or `0` = full stack on attn GPU |
| `PULSAR_ATTN_TIER_RESERVE_GB` | `8` | GB to leave for expert tiers on the attn card when auto-solving |
| `PULSAR_DEV_CACHE_GB` | solved by engine | VRAM hot-expert pool. Honored if set, else engine solves it |
| `PULSAR_BATCH` | solved by engine | prefill chunk size. Honored if set, else engine solves it |

Auto attn-VRAM heuristic (dual-GPU): `budget ≈ free_gb / 2`, never below 6,
never above `free_gb − 4`, reduced if it would starve the tier reserve.

### CPU expert lane

Computes host-cache-hit experts on the CPU (AVX2) instead of uploading
them — beats PCIe on the reference box for iq2/q2_K/q3_K/q4_K tensors.

| var | default | what |
|---|---|---|
| `PULSAR_CPU` | `1` | `1` = CPU lane on. `off` or `0` = off |
| `PULSAR_CPU_STEAL` | `0` | CPU steal factor (0 = none) |

```sh
PULSAR_CPU=off ./docs/examples/runpulsar.sh     # disable CPU lane
PULSAR_CPU=4   ./docs/examples/runpulsar.sh     # explicit thread budget
```

### Engine pass-through (read at run time)

The script does **not** set these; if you export them, the engine picks
them up. Listed so you know the resolved values the script prints come
from the engine's own defaults.

| var | default | what |
|---|---|---|
| `PULSAR_KV` | `f32` | GQA K/V format: `fp8` (e4m3+scale, ~3.9×), `fp16` (~2.0×), `int8` (~4.0×), `q8_0` (~3.8×), `q4_0` (~7.6×). Lossy, opt-in |
| `PULSAR_TIERS` | `on` | resident expert tiers on spare GPUs. `off` = bit-exact single-device path |
| `PULSAR_NO_PREFETCH` | unset | set any value to disable the cross-layer prefetcher |
| `PULSAR_PROFILE` | unset | per-stage wall-time profile. **Forced to `1` in generate/chat by the script** |

> **Gotcha — `PULSAR_TIERS`:** the script runs `unset PULSAR_TIERS` before
> launch, so exporting `PULSAR_TIERS=off` from your shell has **no effect**
> through this launcher. To disable tiers, either edit the script or run
> `pulsar-cli` directly. Same shape for `PULSAR_PROFILE` in generate/chat
> (the script forces it on regardless of your env).

## Pass-through args

Anything after the script is forwarded to the binary as trailing args:

```sh
# chat with a system prompt + sampling
MODE=chat ./docs/examples/runpulsar.sh --system "Be terse" --temp 0.7 --top-p 0.9

# serve with a larger ctx
MODE=serve ./docs/examples/runpulsar.sh --ctx 8192
```

`pulsar-cli` accepts: `--chat`, `--system`, `--temp`, `--top-p`, `--min-p`,
`--seed`, `--no-bos`, `--tokens`, `--ctx`, `--dump-logits`, `--teacher-force`,
`--decode-consistency`.
`pulsar-serve` accepts: `--port`, `--host`, `--ctx`.

## What it prints

Before launch you get: GPU scan (candidate / hidden / denylisted), the
resolved topology (STREAM primary + ATTN secondary), every `PULSAR_*`
value with its provenance (`auto`, `user override`, `solved by engine`),
the run mode + exact invocation, the in-use `nvidia-smi` card table, and
a guard check.

## Safety guard

Refuses to start if the primary GPU has `< 4096 MiB` free — means another
process holds the card or the context is stuck. Free it first:

```sh
nvidia-smi
sudo nvidia-smi --gpu-reset -i <index>
```

## Cookbook

```sh
# quickest smoke test
./docs/examples/runpulsar.sh

# different model, longer output
MODEL=/data/Qwen3-235B.gguf PROMPT="Write a haiku" N=96 ./docs/examples/runpulsar.sh

# bit-exact single-device path (no tiers, no CPU lane, no profiling)
PULSAR_CPU=off ./docs/examples/runpulsar.sh        # tiers still unset→on; see gotcha

# expose the server on the LAN
MODE=serve HOST=0.0.0.0 PORT=11435 ./docs/examples/runpulsar.sh

# pin to specific physical GPUs, skip auto-pick
CUDA_VISIBLE_DEVICES=2,3 PULSAR_GPU=0 PULSAR_ATTN_GPU=1 ./docs/examples/runpulsar.sh

# bigger host cache (you have RAM to spare)
PULSAR_CACHE_GB=48 ./docs/examples/runpulsar.sh

# long-context KV squeeze (lossy)
PULSAR_KV=fp8 MODE=chat ./docs/examples/runpulsar.sh
```

## Troubleshooting

| symptom | fix |
|---|---|
| `pulsar-cli not found` | `cargo build --release -p engine` |
| `pulsar-serve not found` | `cargo build --release -p serve` |
| `nvidia-smi not found` | install the NVIDIA driver / put it on `PATH` |
| `no capable GPUs` | lower `PULSAR_MIN_VRAM_MB`, or check the card is above the denylist |
| primary `< 4 GiB free` | free the card: `nvidia-smi` / `--gpu-reset` |
| wrong GPU picked as primary | set `CUDA_VISIBLE_DEVICES` + `PULSAR_GPU` to skip auto-pick |
| attn offload hurts at short ctx | single-GPU GQA is opt-in: don't set `PULSAR_ATTN_GPU` |

## Requirements

- Linux (io_uring + CUDA are load-bearing; macOS stubs the engine)
- NVIDIA GPU, GTX 10-series (sm_61) or newer; `nvcc` on `PATH`
- Rust via <https://rustup.rs>
- `CXX=g++-12` at build time if your default gcc is too new for nvcc
- Model gguf on a fast NVMe — decode speed is read speed
