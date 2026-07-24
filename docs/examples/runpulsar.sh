#!/usr/bin/env bash
# Generic Pulsar launcher: auto-picks the best GPUs on the machine.
#
# Pulsar roles (see engine ensure_device / MLA attn placement):
#   PRIMARY (stream) — highest compute capability (every SM from 6.0 up ranks;
#                      expert tensor-core GEMM wants sm_80+), tie-break PCIe
#                      gen × width, then free VRAM. Expert kernels run here.
#   ATTN             — most free VRAM among the remaining capable GPUs
#
# Weak / tiny cards are hidden from CUDA_VISIBLE_DEVICES so they cannot
# become primary by accident (e.g. GTX 1060 3GB at Gen2 x1).
#
# Env overrides:
#   PULSAR_MIN_VRAM_MB     min total VRAM to be a candidate (default 8192)
#   PULSAR_GPU / PULSAR_ATTN_GPU / CUDA_VISIBLE_DEVICES
#                          if PULSAR_GPU is pre-set with CUDA_VISIBLE_DEVICES, auto-pick is skipped
#   PULSAR_CACHE_GB, PULSAR_ATTN_VRAM_GB (or =off), PULSAR_ATTN_TIER_RESERVE_GB
#   PULSAR_CPU, PULSAR_CPU_STEAL
#   MODEL, PROMPT, N
#   MODE                   generate (default) | chat | serve
#                          generate: one-shot -p PROMPT -n N
#                          chat:      pulsar-cli --chat (multi-turn, KV retained;
#                                     pass --system "..." via args)
#                          serve:     pulsar-serve --port PORT --host HOST
#                                     (build first: cargo build --release -p serve)
#   PORT, HOST             serve mode endpoint (default 11435 / 127.0.0.1)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

MODEL="${MODEL:-/home/cesar/models/GLM-5.2-UD-IQ2_XXS_RoutedIQ2XXS_blk78Q2K.gguf}"
PROMPT="${PROMPT:-The capital of France is}"
N="${N:-64}"
MIN_VRAM_MB="${PULSAR_MIN_VRAM_MB:-8192}"
MODE="${MODE:-generate}"
PORT="${PORT:-11435}"
HOST="${HOST:-127.0.0.1}"

# ---- host expert cache (auto from MemAvailable) ----
if [ -n "${PULSAR_CACHE_GB:-}" ]; then
  CACHE_GB="$PULSAR_CACHE_GB"
else
  _AVAIL_KB=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo 2>/dev/null || echo 0)
  _AVAIL_GB=$(( ${_AVAIL_KB:-0} / 1024 / 1024 ))
  _HEADROOM="${PULSAR_CACHE_HEADROOM_GB:-16}"
  CACHE_GB=$(( _AVAIL_GB - _HEADROOM ))
  [ "$CACHE_GB" -lt 8 ] && CACHE_GB=8
  AUTO_CACHE_NOTE=" (auto: ${_AVAIL_GB}G avail - ${_HEADROOM}G headroom)"
fi

# ATTN_VRAM_GB: if user set PULSAR_ATTN_VRAM_GB, honor it; else auto after GPU pick.
ATTN_VRAM_USER="${PULSAR_ATTN_VRAM_GB-}"
ATTN_VRAM_GB=""
ATTN_VRAM_NOTE=""
CPU="${PULSAR_CPU:-1}"
CPU_STEAL="${PULSAR_CPU_STEAL:-0}"

# Auto-calc attn VRAM budget (GiB) for a dual-GPU layout.
# Engine: only this many GiB of the MLA stack stay in VRAM on the attn card;
# the rest is pinned host; leftover VRAM holds resident expert tiers.
#
# Heuristic (measured max on AISERVER: 24G free → 12G budget):
#   budget ≈ free_gb / 2
#   but leave at least PULSAR_ATTN_TIER_RESERVE_GB (default 8) for tiers when free is large
#   clamp to [6, free_gb - 4]
#
# free_mb: free MiB on the attn GPU before load.
calc_attn_vram_gb() {
  local free_mb="${1:-0}"
  [[ "$free_mb" =~ ^[0-9]+$ ]] || free_mb=0
  local free_gb=$(( (free_mb + 512) / 1024 ))
  [ "$free_gb" -lt 1 ] && free_gb=1

  local tier_reserve="${PULSAR_ATTN_TIER_RESERVE_GB:-8}"
  local by_half=$(( free_gb / 2 ))
  local by_tier=$(( free_gb - tier_reserve ))
  [ "$by_tier" -lt 0 ] && by_tier=0

  # Prefer half-free (balanced stack vs tier). If free-tier_reserve is
  # smaller, use that so we never starve the tier reserve on mid-size cards.
  local budget=$by_half
  if [ "$by_tier" -gt 0 ] && [ "$by_tier" -lt "$budget" ]; then
    budget=$by_tier
  fi

  local floor=6
  local ceil=$(( free_gb - 4 ))
  [ "$ceil" -lt "$floor" ] && ceil=$floor
  [ "$budget" -lt "$floor" ] && budget=$floor
  [ "$budget" -gt "$ceil" ] && budget=$ceil
  echo "$budget"
}

# ---- GPU auto-selection ----
# Score stream candidate: theoretical PCIe-ish bandwidth ~ gen * width.
# Use the link's MAX capability, not .current: idle GPUs downtrain to gen1
# (measured: a Gen5 card reads gen1 at idle), which would randomize the
# pick on same-width boxes. Capability is a stable ORDERING; the engine
# still H2D-probes real bandwidth at runtime among visible devices.
# Score attn candidate: free MiB (capacity for MLA stack + tier residual).
command -v nvidia-smi >/dev/null || {
  echo "ERROR: nvidia-smi not found" >&2
  exit 1
}

# index, name, total_mb, free_mb, pcie_gen, pcie_width
mapfile -t GPU_ROWS < <(
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,pcie.link.gen.max,pcie.link.width.max,pcie.link.gen.current,pcie.link.width.current,compute_cap \
    --format=csv,noheader,nounits 2>/dev/null | sed 's/, /,/g'
)

if [ "${#GPU_ROWS[@]}" -eq 0 ]; then
  echo "ERROR: no GPUs reported by nvidia-smi" >&2
  nvidia-smi -L >&2 || true
  exit 1
fi

# Arrays of candidate physical indices and metadata (parallel)
CAND_IDX=()
CAND_NAME=()
CAND_TOTAL=()
CAND_FREE=()
CAND_PCIE=()   # gen * width (0 if unknown)
CAND_CC=()     # compute capability as major*10+minor (6.0=>60 ... 8.6=>86, 10.0=>100); 0 if unknown

is_denylisted() {
  # Tiny / display-class cards often mis-rank if VRAM threshold is low.
  local u="${1^^}"
  case "$u" in
    *1030*|*1050*|*1060*|*1650\ MAX-Q*|*MX150*|*MX250*|*MX330*|*UHD*|*P600*|*P620*)
      return 0 ;;
  esac
  return 1
}

echo "scanning GPUs (min ${MIN_VRAM_MB} MiB total VRAM)..."
for row in "${GPU_ROWS[@]}"; do
  IFS=',' read -r idx name total free gen width cgen cwidth cc_raw <<<"$row"
  idx="${idx// /}"
  name="${name# }"
  total="${total// /}"
  free="${free// /}"
  gen="${gen// /}"
  width="${width// /}"
  # nvidia-smi may print [N/A]
  [[ "$total" =~ ^[0-9]+$ ]] || total=0
  [[ "$free" =~ ^[0-9]+$ ]] || free=0
  cgen="${cgen// /}"
  cwidth="${cwidth// /}"
  # older drivers report [N/A] for max: fall back to the trained link
  [[ "$gen" =~ ^[0-9]+$ ]] || gen="$cgen"
  [[ "$width" =~ ^[0-9]+$ ]] || width="$cwidth"
  [[ "$gen" =~ ^[0-9]+$ ]] || gen=0
  [[ "$width" =~ ^[0-9]+$ ]] || width=0
  pcie=$(( gen * width ))

  # Compute capability, encoded major*10+minor so integer compare preserves SM
  # order across the full range (6.0 -> 60, 7.0 -> 70, 8.6 -> 86, 10.0 -> 100).
  # Higher SM wins the stream primary — tensor-core expert kernels want sm_80+
  # (Ampere INT8/INT4 mma), but every SM from 6.0 up ranks correctly, so the
  # best-capable card is always picked, never excluded by a hard threshold.
  cc_raw="${cc_raw// /}"
  if [[ "$cc_raw" == *.* ]]; then
    cc_major="${cc_raw%%.*}"
    cc_minor="${cc_raw#*.}"
  else
    cc_major="$cc_raw"
    cc_minor=0
  fi
  [[ "$cc_major" =~ ^[0-9]+$ ]] || cc_major=0
  [[ "$cc_minor" =~ ^[0-9]+$ ]] || cc_minor=0
  cc=$(( cc_major * 10 + cc_minor ))

  if is_denylisted "$name"; then
    echo "  hide  GPU $idx  $name  (${total} MiB) — denylist"
    continue
  fi
  if [ "$total" -lt "$MIN_VRAM_MB" ]; then
    echo "  hide  GPU $idx  $name  (${total} MiB < ${MIN_VRAM_MB} MiB min)"
    continue
  fi
  echo "  cand  GPU $idx  $name  free=${free} MiB  PCIe gen${gen} x${width}  sm_${cc_raw:-?} (score=${pcie})"
  CAND_IDX+=("$idx")
  CAND_NAME+=("$name")
  CAND_TOTAL+=("$total")
  CAND_FREE+=("$free")
  CAND_PCIE+=("$pcie")
  CAND_CC+=("$cc")
done

n_cand=${#CAND_IDX[@]}
if [ "$n_cand" -lt 1 ]; then
  echo "ERROR: no capable GPUs (need >= ${MIN_VRAM_MB} MiB VRAM after denylist)" >&2
  nvidia-smi -L >&2 || true
  exit 1
fi

# Pick stream primary: highest compute capability (expert tensor-core kernels
# need sm_80+), tie-break PCIe score, then free VRAM, then total VRAM.
STREAM_I=0
for ((i = 1; i < n_cand; i++)); do
  better=0
  if [ "${CAND_CC[$i]}" -gt "${CAND_CC[$STREAM_I]}" ]; then
    better=1
  elif [ "${CAND_CC[$i]}" -eq "${CAND_CC[$STREAM_I]}" ]; then
    if [ "${CAND_PCIE[$i]}" -gt "${CAND_PCIE[$STREAM_I]}" ]; then
      better=1
    elif [ "${CAND_PCIE[$i]}" -eq "${CAND_PCIE[$STREAM_I]}" ]; then
      if [ "${CAND_FREE[$i]}" -gt "${CAND_FREE[$STREAM_I]}" ]; then
        better=1
      elif [ "${CAND_FREE[$i]}" -eq "${CAND_FREE[$STREAM_I]}" ] \
        && [ "${CAND_TOTAL[$i]}" -gt "${CAND_TOTAL[$STREAM_I]}" ]; then
        better=1
      fi
    fi
  fi
  [ "$better" -eq 1 ] && STREAM_I=$i
done

STREAM_PHYS="${CAND_IDX[$STREAM_I]}"
STREAM_NAME="${CAND_NAME[$STREAM_I]}"
STREAM_FREE="${CAND_FREE[$STREAM_I]}"

# Pick attn: highest free VRAM among remaining candidates
ATTN_I=""
ATTN_PHYS=""
ATTN_NAME=""
if [ "$n_cand" -ge 2 ]; then
  for ((i = 0; i < n_cand; i++)); do
    [ "$i" -eq "$STREAM_I" ] && continue
    if [ -z "$ATTN_I" ] || [ "${CAND_FREE[$i]}" -gt "${CAND_FREE[$ATTN_I]}" ]; then
      ATTN_I=$i
    elif [ "${CAND_FREE[$i]}" -eq "${CAND_FREE[$ATTN_I]}" ] \
      && [ "${CAND_TOTAL[$i]}" -gt "${CAND_TOTAL[$ATTN_I]}" ]; then
      ATTN_I=$i
    fi
  done
  ATTN_PHYS="${CAND_IDX[$ATTN_I]}"
  ATTN_NAME="${CAND_NAME[$ATTN_I]}"
  ATTN_FREE="${CAND_FREE[$ATTN_I]}"
fi

# Manual override: PULSAR_GPU pre-set → honor user roles, skip auto-pick.
# If CUDA_VISIBLE_DEVICES is also set, use it verbatim; otherwise default
# visibility to the capable (non-denylisted) candidates in scan order, so
# PULSAR_GPU/PULSAR_ATTN_GPU index only real cards.
if [ -n "${PULSAR_GPU:-}" ]; then
  MANUAL=1
  echo
  echo "PULSAR_GPU set — auto-pick skipped (honoring your roles)"
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
    CUDA_VISIBLE_DEVICES="$(IFS=','; echo "${CAND_IDX[*]}")"
    export CUDA_VISIBLE_DEVICES
    echo "CUDA_VISIBLE_DEVICES unset — defaulting to capable cards: $CUDA_VISIBLE_DEVICES"
  fi
else
  export CUDA_DEVICE_ORDER=PCI_BUS_ID
  if [ -n "$ATTN_PHYS" ]; then
    # Remap: local 0 = stream primary, local 1 = attn
    export CUDA_VISIBLE_DEVICES="${STREAM_PHYS},${ATTN_PHYS}"
    export PULSAR_GPU=0
    export PULSAR_ATTN_GPU=1
  else
    export CUDA_VISIBLE_DEVICES="${STREAM_PHYS}"
    export PULSAR_GPU=0
    unset PULSAR_ATTN_GPU
  fi
fi

export PULSAR_CACHE_GB="$CACHE_GB"

# ---- PULSAR_ATTN_VRAM_GB: user override or auto for dual-GPU ----
if [ -n "$ATTN_VRAM_USER" ]; then
  if [[ "$ATTN_VRAM_USER" == "off" || "$ATTN_VRAM_USER" == "0" ]]; then
    unset PULSAR_ATTN_VRAM_GB
    ATTN_VRAM_NOTE=" (user: off — full stack on attn GPU)"
  else
    export PULSAR_ATTN_VRAM_GB="$ATTN_VRAM_USER"
    ATTN_VRAM_NOTE=" (user override)"
  fi
elif [ -z "${MANUAL:-}" ] && [ -n "${ATTN_PHYS:-}" ] && [ -n "${ATTN_FREE:-}" ]; then
  # Dual-GPU: balance resident MLA stack vs residual expert tier on attn card.
  ATTN_VRAM_GB="$(calc_attn_vram_gb "$ATTN_FREE")"
  export PULSAR_ATTN_VRAM_GB="$ATTN_VRAM_GB"
  _free_g=$(( (ATTN_FREE + 512) / 1024 ))
  ATTN_VRAM_NOTE=" (auto: ~${_free_g}G free on attn → budget ${ATTN_VRAM_GB}G stack, leave rest for expert tier)"
else
  # Single GPU or manual topology: let the engine default (family-specific).
  unset PULSAR_ATTN_VRAM_GB
  ATTN_VRAM_NOTE=" (engine default; set PULSAR_ATTN_VRAM_GB to override)"
fi

unset PULSAR_TIERS 2>/dev/null || true

if [[ "$CPU" == "off" || "$CPU" == "0" ]]; then
  unset PULSAR_CPU
else
  export PULSAR_CPU="$CPU"
fi
export PULSAR_CPU_STEAL="$CPU_STEAL"

echo
if [ -n "${MANUAL:-}" ]; then
  echo "manual topology (user-pinned roles):"
  echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "  PULSAR_GPU=$PULSAR_GPU (stream)   PULSAR_ATTN_GPU=${PULSAR_ATTN_GPU:-unset} (attn)"
else
  echo "selected topology:"
  echo "  STREAM primary  physical GPU $STREAM_PHYS  $STREAM_NAME  (free ${STREAM_FREE} MiB, cc ${CAND_CC[$STREAM_I]}, PCIe score ${CAND_PCIE[$STREAM_I]})"
  if [ -n "${ATTN_PHYS:-}" ]; then
    echo "  ATTN secondary  physical GPU $ATTN_PHYS  $ATTN_NAME  (free ${ATTN_FREE} MiB)"
  else
    echo "  ATTN secondary  (none — single capable GPU; Pulsar runs single-device)"
  fi
fi
echo
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "PULSAR_GPU=$PULSAR_GPU"
echo "PULSAR_ATTN_GPU=${PULSAR_ATTN_GPU:-unset}"
echo "PULSAR_CACHE_GB=$PULSAR_CACHE_GB${AUTO_CACHE_NOTE:-}"
echo "PULSAR_DEV_CACHE_GB=${PULSAR_DEV_CACHE_GB:-solved by engine (free VRAM − staging − reserve)}"
echo "PULSAR_ATTN_VRAM_GB=${PULSAR_ATTN_VRAM_GB:-unset}${ATTN_VRAM_NOTE}"
echo "PULSAR_TIERS=${PULSAR_TIERS:-on (default — unset leaves engine default)}"
echo "PULSAR_KV=${PULSAR_KV:-f32 (default; one of fp8/fp16/int8/q8_0/q4_0, lossy KV squeeze)}"
echo "PULSAR_BATCH=${PULSAR_BATCH:-solved by engine (worst-case staging vs free VRAM)}"
echo "PULSAR_NO_PREFETCH=${PULSAR_NO_PREFETCH:-unset (cross-layer prefetcher ON)}"
echo "PULSAR_PROFILE=${PULSAR_PROFILE:-unset}"
echo "PULSAR_CPU=${PULSAR_CPU:-unset (CPU expert lane OFF)}"
echo "PULSAR_CPU_STEAL=$PULSAR_CPU_STEAL"
echo "model: $MODEL"
echo
echo "run mode: $MODE"
case "$MODE" in
  generate)
    echo "  pulsar-cli  -p \"$PROMPT\"  -n $N  (PULSAR_PROFILE=1 forced)"
    ;;
  chat)
    echo "  pulsar-cli  --chat  (multi-turn, KV retained; PULSAR_PROFILE=1 forced)"
    ;;
  serve)
    echo "  pulsar-serve  --port $PORT  --host $HOST  (PULSAR_PROFILE left to env)"
    ;;
esac
echo

echo "physical cards in use:"
nvidia-smi -i "$CUDA_VISIBLE_DEVICES" \
  --query-gpu=index,name,memory.total,memory.free,pcie.link.gen.current,pcie.link.width.current \
  --format=csv
echo

# Refuse to start if primary is nearly empty (stuck context / other process)
PRIM_FREE=$(nvidia-smi -i "$STREAM_PHYS" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
PRIM_FREE=${PRIM_FREE%%.*}
if [[ "${PRIM_FREE:-0}" =~ ^[0-9]+$ ]] && [ "${PRIM_FREE:-0}" -lt 4096 ]; then
  echo "ERROR: stream GPU $STREAM_PHYS free memory is ${PRIM_FREE} MiB (< 4 GiB). Free the card first:" >&2
  echo "  nvidia-smi" >&2
  echo "  # sudo nvidia-smi --gpu-reset -i $STREAM_PHYS" >&2
  exit 1
fi

case "$MODE" in
  generate|chat)
    CLI="${PULSAR_CLI:-$ROOT/target/release/pulsar-cli}"
    if [ ! -x "$CLI" ]; then
      echo "ERROR: pulsar-cli not found at $CLI (build with: cargo build --release -p engine)" >&2
      exit 1
    fi
    ;;
  serve)
    CLI="${PULSAR_SERVE:-$ROOT/target/release/pulsar-serve}"
    if [ ! -x "$CLI" ]; then
      echo "ERROR: pulsar-serve not found at $CLI (build with: cargo build --release -p serve)" >&2
      exit 1
    fi
    ;;
  *)
    echo "ERROR: MODE='$MODE' (expected: generate | chat | serve)" >&2
    exit 1
    ;;
esac

case "$MODE" in
  generate)
    exec env PULSAR_PROFILE=1 "$CLI" -m "$MODEL" -p "$PROMPT" -n "$N" "$@"
    ;;
  chat)
    exec env PULSAR_PROFILE=1 "$CLI" -m "$MODEL" --chat "$@"
    ;;
  serve)
    echo "serving on http://${HOST}:${PORT}/v1/chat/completions (Ctrl-C to stop)"
    exec "$CLI" -m "$MODEL" --port "$PORT" --host "$HOST" "$@"
    ;;
esac
