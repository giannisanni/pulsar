#!/usr/bin/env bash
# Run pulsar on AISERVER: V100 primary (stream) + 3090 attn.
# Never exposes the GTX 1060 3GB (or any other GPU) to the process.
set -euo pipefail
cd "$(dirname "$0")"

MODEL="${MODEL:-/home/cesar/models/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf}"
PROMPT="${PROMPT:-The capital of France is}"
N="${N:-64}"
# Auto-size the host expert cache: grab most of MemAvailable, leaving headroom
# for the OS, the page cache backing the model mmap, and any other GPU work on
# the box. Override with PULSAR_CACHE_GB=N; tune the reserve with
# PULSAR_CACHE_HEADROOM_GB (default 16 GiB). Bigger cache = fewer sync NVMe
# reads in ensure_with = lower resolve.host on the profile.
if [ -n "${PULSAR_CACHE_GB:-}" ]; then
  CACHE_GB="$PULSAR_CACHE_GB"
else
  _AVAIL_KB=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null)
  _AVAIL_GB=$(( ${_AVAIL_KB:-0} / 1024 / 1024 ))
  _HEADROOM="${PULSAR_CACHE_HEADROOM_GB:-16}"
  CACHE_GB=$(( _AVAIL_GB - _HEADROOM ))
  # floor: below ~8 GiB the LFU thrashes and perf falls off a cliff
  [ "$CACHE_GB" -lt 8 ] && CACHE_GB=8
  AUTO_CACHE_NOTE=" (auto: ${_AVAIL_GB}G avail - ${_HEADROOM}G headroom)"
fi
# Measured faster on AISERVER (3x A/B): cap attn VRAM on the 3090 so more
# of the MLA stack can sit pinned and the card stays less packed. Override
# with PULSAR_ATTN_VRAM_GB=off to restore "fill the attn GPU" behaviour.
ATTN_VRAM_GB="${PULSAR_ATTN_VRAM_GB:-12}"
# CPU expert lane: compute host-cache-hit experts on the CPU (AVX2 RAM
# ~42GB/s) instead of uploading them, overlapping the V100. This is the
# biggest lever for the GLM host-cache-heavy profile (host 7.59s + gpu-wait
# 8.67s in the last run). Override: PULSAR_CPU=0|off to disable, =N for N
# worker threads (engine auto-picks when "1").
CPU="${PULSAR_CPU:-1}"
# Steal=0 keeps VRAM-hot experts on the GPU and routes the CPU lane only at
# the host path, so the two don't race. Override: PULSAR_CPU_STEAL=1 to let
# the CPU lane also grab dev-cache-resident experts.
CPU_STEAL="${PULSAR_CPU_STEAL:-0}"

# ---- resolve physical indices by name; refuse unknown topology ----
mapfile -t GPU_LINES < <(nvidia-smi --query-gpu=index,name --format=csv,noheader,nounits)

V100_IDX=""
RTX_IDX=""
SKIP_IDXS=()

for line in "${GPU_LINES[@]}"; do
  idx="${line%%,*}"
  idx="${idx// /}"
  name="${line#*,}"
  name="${name# }"
  # normalize for matching
  uname="${name^^}"

  if [[ "$uname" == *"1060"* ]] || [[ "$uname" == *"1050"* ]] || [[ "$uname" == *"1030"* ]]; then
    SKIP_IDXS+=("$idx")
    echo "hiding weak GPU $idx: $name"
    continue
  fi
  if [[ "$uname" == *"V100"* ]]; then
    V100_IDX="$idx"
    echo "found V100  physical GPU $idx: $name"
    continue
  fi
  if [[ "$uname" == *"3090"* ]]; then
    RTX_IDX="$idx"
    echo "found 3090 physical GPU $idx: $name"
    continue
  fi
  # any other unexpected card: hide it
  SKIP_IDXS+=("$idx")
  echo "hiding other GPU $idx: $name"
done

if [[ -z "$V100_IDX" || -z "$RTX_IDX" ]]; then
  echo "ERROR: need both Tesla V100 and RTX 3090 visible in nvidia-smi" >&2
  echo "V100_IDX='${V100_IDX:-}'  RTX_IDX='${RTX_IDX:-}'" >&2
  nvidia-smi -L >&2 || true
  exit 1
fi

# Physical order: V100 first (becomes CUDA 0), 3090 second (CUDA 1).
# GTX 1060 and anything else are simply omitted from CUDA_VISIBLE_DEVICES,
# so the process cannot open them.
export CUDA_VISIBLE_DEVICES="${V100_IDX},${RTX_IDX}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PULSAR_GPU=0          # local 0 = V100
export PULSAR_ATTN_GPU=1     # local 1 = 3090
export PULSAR_CACHE_GB="$CACHE_GB"
if [[ "$ATTN_VRAM_GB" == "off" || "$ATTN_VRAM_GB" == "0" ]]; then
  unset PULSAR_ATTN_VRAM_GB
else
  export PULSAR_ATTN_VRAM_GB="$ATTN_VRAM_GB"
fi
# do not inherit a bad override from the parent shell
unset PULSAR_TIERS 2>/dev/null || true

# CPU expert lane (default on; "0"/"off" disables for A/B against the baseline)
if [[ "$CPU" == "off" || "$CPU" == "0" ]]; then
  unset PULSAR_CPU
else
  export PULSAR_CPU="$CPU"
fi
export PULSAR_CPU_STEAL="$CPU_STEAL"

echo
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES   # 1060 NOT included"
echo "PULSAR_GPU=$PULSAR_GPU (V100 primary / stream)"
echo "PULSAR_ATTN_GPU=$PULSAR_ATTN_GPU (3090 attn)"
echo "PULSAR_CACHE_GB=$PULSAR_CACHE_GB${AUTO_CACHE_NOTE:-}"
echo "PULSAR_ATTN_VRAM_GB=${PULSAR_ATTN_VRAM_GB:-unset (full stack on attn GPU)}"
echo "PULSAR_CPU=${PULSAR_CPU:-unset (CPU expert lane OFF)}"
echo "PULSAR_CPU_STEAL=$PULSAR_CPU_STEAL  (0 = CPU lane only on host path, GPU keeps VRAM-hot)"
echo "model: $MODEL"
echo

# nvidia-smi always lists every installed GPU; it does NOT honor
# CUDA_VISIBLE_DEVICES. Only the CUDA runtime (pulsar-cli) is masked.
# Sanity-check the mask string itself instead.
if [[ "$CUDA_VISIBLE_DEVICES" != "${V100_IDX},${RTX_IDX}" ]]; then
  echo "ERROR: bad CUDA_VISIBLE_DEVICES='$CUDA_VISIBLE_DEVICES'" >&2
  exit 1
fi
# ensure the 1060 physical index is not in the mask
for skip in "${SKIP_IDXS[@]+"${SKIP_IDXS[@]}"}"; do
  case ",${CUDA_VISIBLE_DEVICES}," in
    *",${skip},"*)
      echo "ERROR: weak/other GPU $skip leaked into CUDA_VISIBLE_DEVICES" >&2
      exit 1
      ;;
  esac
done

echo "CUDA will see (remapped):"
echo "  local GPU 0 <- physical $V100_IDX (V100, primary/stream)"
echo "  local GPU 1 <- physical $RTX_IDX (3090, attn)"
echo "physical cards in use:"
nvidia-smi -i "${V100_IDX},${RTX_IDX}" \
  --query-gpu=index,name,memory.free,pcie.link.gen.current,pcie.link.width.current \
  --format=csv
echo

# refuse to start if V100 is basically empty of free memory (common stuck state)
V100_FREE_MIB=$(nvidia-smi -i "$V100_IDX" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
if [[ "${V100_FREE_MIB:-0}" -lt 8000 ]]; then
  echo "ERROR: V100 free memory is ${V100_FREE_MIB} MiB (< 8 GiB). Free the card first:" >&2
  echo "  nvidia-smi" >&2
  echo "  # kill leftover processes, then optionally: sudo nvidia-smi --gpu-reset -i $V100_IDX" >&2
  exit 1
fi

exec env PULSAR_PROFILE=1 ./target/release/pulsar-cli \
  -m "$MODEL" \
  -p "$PROMPT" \
  -n "$N" \
  "$@"
