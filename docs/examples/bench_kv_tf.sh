#!/usr/bin/env bash
# Teacher-forced KV quality sweep — the non-chaotic counterpart to bench_kv.sh.
#
# bench_kv.sh compares GREEDY output, which is useless once the engine's f32
# noise floor is ~token #5 (one flipped argmax cascades into ~100% mismatch
# via autoregressive chaos). This script instead force-feeds the SAME token
# sequence through every KV format and compares per-position logits:
#
#   - no autoregressive feedback → no chaos amplification
#   - aggregates over N positions → kernel nondeterminism averages out
#   - f32-vs-f32 run gives the TRUE logit-level noise floor
#
# Reads the same JSONL that pulsar-cli --teacher-force already emits:
#   {"pos":N,"after":ID,"top":[[id,logit],...]}   (top-5 per position)
#
# Usage:
#   MODEL=/path/to/qwen3moe.gguf ./docs/examples/bench_kv_tf.sh
#   MODEL=... PROMPT_FILE=corpus.txt ./docs/examples/bench_kv_tf.sh   # more positions
#   MODEL=... FMTS="f32 fp16 q4_0" ./docs/examples/bench_kv_tf.sh      # subset
#
# Env: MODEL (required) · PROMPT | PROMPT_FILE · FMTS (default all six)
#      + same GPU-select vars as runpulsar.sh / bench_kv.sh
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CLI="${PULSAR_CLI:-$ROOT/target/release/pulsar-cli}"
[ -x "$CLI" ] || { echo "build first: cargo build --release -p engine" >&2; exit 1; }

MODEL="${MODEL:?set MODEL= to a GQA/Qwen35 family gguf (NOT glm-dsa/GLM-5.2)}"
FMTS="${FMTS:-f32 fp8 fp16 int8 q8_0 q4_0}"
MIN_VRAM_MB="${PULSAR_MIN_VRAM_MB:-8192}"

# Long-ish default prompt → enough positions for statistics (~120 tokens).
# For serious validation, point PROMPT_FILE at a real corpus paragraph.
PROMPT="${PROMPT:-List the first eight Fibonacci numbers, then explain each in one short sentence. The Fibonacci sequence starts with zero and one, and every following term is the sum of the two preceding terms. It appears across mathematics, nature, computer science, and art. Rabbit population modeling, spiral phyllotaxis in sunflowers, pinecone scales, the golden ratio, recursive algorithms, dynamic programming memos, memoized search,AVX-dominated numeric kernels, and quiescent memory access patterns all connect back to it. Write clearly and keep each explanation to a single short sentence.}"
if [ -n "${PROMPT_FILE:-}" ]; then
  PROMPT="$(cat "$PROMPT_FILE")"
fi

# ---- host expert cache (auto from MemAvailable) ----
if [ -n "${PULSAR_CACHE_GB:-}" ]; then
  CACHE_GB="$PULSAR_CACHE_GB"
else
  _AVAIL_KB=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
  _AVAIL_GB=$(( ${_AVAIL_KB:-0} / 1024 / 1024 ))
  _HEADROOM="${PULSAR_CACHE_HEADROOM_GB:-16}"
  CACHE_GB=$(( _AVAIL_GB - _HEADROOM ))
  [ "$CACHE_GB" -lt 8 ] && CACHE_GB=8
fi
ATTN_VRAM_USER="${PULSAR_ATTN_VRAM_GB-}"
CPU="${PULSAR_CPU:-1}"
CPU_STEAL="${PULSAR_CPU_STEAL:-0}"

command -v nvidia-smi >/dev/null || { echo "ERROR: nvidia-smi not found" >&2; exit 1; }
command -v python3 >/dev/null || { echo "ERROR: python3 not found (logit comparison needs it)" >&2; exit 1; }

mapfile -t GPU_ROWS < <(
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,pcie.link.gen.max,pcie.link.width.max,pcie.link.gen.current,pcie.link.width.current,compute_cap \
    --format=csv,noheader,nounits 2>/dev/null | sed 's/, /,/g'
)
[ "${#GPU_ROWS[@]}" -gt 0 ] || { echo "ERROR: no GPUs reported by nvidia-smi" >&2; exit 1; }

CAND_IDX=(); CAND_NAME=(); CAND_TOTAL=(); CAND_FREE=(); CAND_PCIE=(); CAND_CC=()  # cc major*10+minor (6.0=>60..10.0=>100); 0 if N/A
is_denylisted() {
  local u="${1^^}"
  case "$u" in
    *1030*|*1050*|*1060*|*1650\ MAX-Q*|*MX150*|*MX250*|*MX330*|*UHD*|*P600*|*P620*) return 0 ;;
  esac
  return 1
}

echo "scanning GPUs (min ${MIN_VRAM_MB} MiB total VRAM)..."
for row in "${GPU_ROWS[@]}"; do
  IFS=',' read -r idx name total free gen width cgen cwidth cc_raw <<<"$row"
  idx="${idx// /}"; name="${name# }"; total="${total// /}"; free="${free// /}"
  gen="${gen// /}"; width="${width// /}"; cgen="${cgen// /}"; cwidth="${cwidth// /}"
  [[ "$total" =~ ^[0-9]+$ ]] || total=0
  [[ "$free" =~ ^[0-9]+$ ]] || free=0
  [[ "$gen" =~ ^[0-9]+$ ]] || gen="$cgen"
  [[ "$width" =~ ^[0-9]+$ ]] || width="$cwidth"
  [[ "$gen" =~ ^[0-9]+$ ]] || gen=0
  [[ "$width" =~ ^[0-9]+$ ]] || width=0
  pcie=$(( gen * width ))
  # compute capability major*10+minor (6.0->60, 8.6->86, 10.0->100); higher SM
  # wins the stream primary — tensor-core expert kernels want sm_80+. Ranks
  # every SM from 6.0 up; 0 if nvidia-smi reports [N/A].
  cc_raw="${cc_raw// /}"
  if [[ "$cc_raw" == *.* ]]; then cc_major="${cc_raw%%.*}"; cc_minor="${cc_raw#*.}"; else cc_major="$cc_raw"; cc_minor=0; fi
  [[ "$cc_major" =~ ^[0-9]+$ ]] || cc_major=0
  [[ "$cc_minor" =~ ^[0-9]+$ ]] || cc_minor=0
  cc=$(( cc_major * 10 + cc_minor ))
  if is_denylisted "$name"; then echo "  hide  GPU $idx  $name — denylist"; continue; fi
  if [ "$total" -lt "$MIN_VRAM_MB" ]; then echo "  hide  GPU $idx  $name  (${total} < ${MIN_VRAM_MB} MiB)"; continue; fi
  echo "  cand  GPU $idx  $name  free=${free} MiB  gen${gen} x${width}  sm_${cc_raw:-?} (score=${pcie})"
  CAND_IDX+=("$idx"); CAND_NAME+=("$name"); CAND_TOTAL+=("$total")
  CAND_FREE+=("$free"); CAND_PCIE+=("$pcie"); CAND_CC+=("$cc")
done
n_cand=${#CAND_IDX[@]}
[ "$n_cand" -ge 1 ] || { echo "ERROR: no capable GPUs" >&2; exit 1; }

STREAM_I=0
for ((i = 1; i < n_cand; i++)); do
  better=0
  if [ "${CAND_CC[$i]}" -gt "${CAND_CC[$STREAM_I]}" ]; then better=1
  elif [ "${CAND_CC[$i]}" -eq "${CAND_CC[$STREAM_I]}" ]; then
    if [ "${CAND_PCIE[$i]}" -gt "${CAND_PCIE[$STREAM_I]}" ]; then better=1
    elif [ "${CAND_PCIE[$i]}" -eq "${CAND_PCIE[$STREAM_I]}" ]; then
      if [ "${CAND_FREE[$i]}" -gt "${CAND_FREE[$STREAM_I]}" ]; then better=1
      elif [ "${CAND_FREE[$i]}" -eq "${CAND_FREE[$STREAM_I]}" ] && [ "${CAND_TOTAL[$i]}" -gt "${CAND_TOTAL[$STREAM_I]}" ]; then better=1; fi
    fi
  fi
  [ "$better" -eq 1 ] && STREAM_I=$i
done
STREAM_PHYS="${CAND_IDX[$STREAM_I]}"
ATTN_I=""
if [ "$n_cand" -ge 2 ]; then
  for ((i = 0; i < n_cand; i++)); do
    [ "$i" -eq "$STREAM_I" ] && continue
    if [ -z "$ATTN_I" ] || [ "${CAND_FREE[$i]}" -gt "${CAND_FREE[$ATTN_I]}" ]; then ATTN_I=$i; fi
  done
  ATTN_PHYS="${CAND_IDX[$ATTN_I]}"
fi

if [ -n "${PULSAR_GPU:-}" ]; then
  MANUAL=1
  echo
  echo "PULSAR_GPU set — auto-pick skipped (honoring your roles)"
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && { CUDA_VISIBLE_DEVICES="$(IFS=','; echo "${CAND_IDX[*]}")"; export CUDA_VISIBLE_DEVICES; }
else
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  if [ -n "${ATTN_PHYS:-}" ]; then
    export CUDA_VISIBLE_DEVICES="${STREAM_PHYS},${ATTN_PHYS}"
    export PULSAR_GPU=0; export PULSAR_ATTN_GPU=1
  else
    export CUDA_VISIBLE_DEVICES="${STREAM_PHYS}"; export PULSAR_GPU=0; unset PULSAR_ATTN_GPU
  fi
fi
export PULSAR_CACHE_GB="$CACHE_GB"
if [ -n "$ATTN_VRAM_USER" ]; then
  [[ "$ATTN_VRAM_USER" == "off" || "$ATTN_VRAM_USER" == "0" ]] && unset PULSAR_ATTN_VRAM_GB || export PULSAR_ATTN_VRAM_GB="$ATTN_VRAM_USER"
elif [ -z "${MANUAL:-}" ] && [ -n "${ATTN_PHYS:-}" ]; then
  export PULSAR_ATTN_VRAM_GB=$(( (${CAND_FREE[$ATTN_I]} + 512) / 1024 / 2 ))
else
  unset PULSAR_ATTN_VRAM_GB
fi
unset PULSAR_TIERS 2>/dev/null || true
if [[ "$CPU" == "off" || "$CPU" == "0" ]]; then unset PULSAR_CPU; else export PULSAR_CPU="$CPU"; fi
export PULSAR_CPU_STEAL="$CPU_STEAL"

echo
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  PULSAR_GPU=$PULSAR_GPU  PULSAR_ATTN_GPU=${PULSAR_ATTN_GPU:-unset}  PULSAR_CPU=${PULSAR_CPU:-off}"
echo "PULSAR_KV will cycle: $FMTS"
echo "model: $MODEL"
echo

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# teacher-force a format → JSONL of per-position top-5 on stdout.
tf() {
  local fmt="$1" tag="${2:-$1}"
  local log="$TMP/$tag.log"
  if [ "$fmt" = "f32" ]; then
    env PULSAR_PROFILE=1 "$CLI" -m "$MODEL" -p "$PROMPT" --teacher-force > "$TMP/$tag.json" 2> "$log" || true
  else
    env PULSAR_KV="$fmt" PULSAR_PROFILE=1 "$CLI" -m "$MODEL" -p "$PROMPT" --teacher-force > "$TMP/$tag.json" 2> "$log" || true
    grep -q "$fmt KV cache on" "$log" \
      || echo "  WARN: $fmt did not activate — model is not GQA/Qwen35" >&2
  fi
  printf '  %-6s %s positions\n' "$tag" "$(wc -l < "$TMP/$tag.json")"
}

echo "teacher-forcing f32 baseline..."
tf f32
echo "teacher-forcing f32 again (noise floor — kernel nondeterminism at the logit level)..."
tf f32 f32_noise
echo
echo "teacher-forcing each KV format..."
for fmt in $FMTS; do [ "$fmt" = f32 ] && continue; tf "$fmt"; done

echo
echo "quality (per-position logits vs f32 baseline; chaos-free):"
compare() {
  python3 - "$1" "$2" <<'PY'
import json, sys
def load(p):
    d={}
    for line in open(p):
        line=line.strip()
        if not line: continue
        o=json.loads(line); top=o['top']
        d[o['pos']]=(top[0][0], float(top[0][1]), {t[0] for t in top[:5]})
    return d
a,b=load(sys.argv[1]),load(sys.argv[2])
common=sorted(set(a)&set(b))
if not common: print("  (no overlapping positions — check logs)"); sys.exit(0)
agree=sum(1 for p in common if a[p][0]==b[p][0])
d=[abs(a[p][1]-b[p][1]) for p in common]
j=[len(a[p][2]&b[p][2])/5.0 for p in common]
n=len(common)
print(f"  {n} pos | top-1 agree {100*agree/n:5.1f}% | mean |Δlogit| {sum(d)/n:.4f} (max {max(d):.3f}) | top-5 Jac {sum(j)/n:.2f}")
PY
}
echo "  f32xf32 (noise floor):"
compare "$TMP/f32.json" "$TMP/f32_noise.json"
for fmt in $FMTS; do
  [ "$fmt" = f32 ] && continue
  echo "  $fmt vs f32:"
  compare "$TMP/f32.json" "$TMP/$fmt.json"
done
