#!/usr/bin/env bash
# Quick PULSAR_KV sweep: tok/s + greedy-id quality diff vs f32 baseline.
#
# GPU auto-selection is lifted verbatim from runpulsar.sh (denylist + compute-cap
# + PCIe scoring + free-VRAM attn pick + auto CACHE_GB / ATTN_VRAM). Keep the
# two in sync if you touch the topology logic.
#
# PULSAR_KV only touches GQA / Qwen35 family models. MLA (glm-dsa / GLM-5.2)
# and Dsv4 keep their own caches and ignore it — this script warns if a format
# fails to engage.
#
# Usage:
#   MODEL=/path/to/qwen3moe.gguf ./docs/examples/bench_kv.sh
#   MODEL=... PROMPT="..." N=512 ./docs/examples/bench_kv.sh
#   MODEL=... CUDA_VISIBLE_DEVICES=0 PULSAR_GPU=0 ./docs/examples/bench_kv.sh   # pin topology
#   MODEL=... FMTS="f32 q4_0" ./docs/examples/bench_kv.sh                        # subset
#
# Env: MODEL (required) · PROMPT · N (default 512; bigger = KV effect clearer)
#      FMTS (default "f32 fp8 fp16 int8 q8_0 q4_0")
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"

CLI="${PULSAR_CLI:-$ROOT/target/release/pulsar-cli}"
[ -x "$CLI" ] || { echo "build first: cargo build --release -p engine" >&2; exit 1; }

MODEL="${MODEL:?set MODEL= to a GQA/Qwen35 family gguf (NOT glm-dsa/GLM-5.2)}"
PROMPT="${PROMPT:-List the first eight Fibonacci numbers, then explain each in one short sentence.}"
N="${N:-512}"
FMTS="${FMTS:-f32 fp8 fp16 int8 q8_0 q4_0}"
MIN_VRAM_MB="${PULSAR_MIN_VRAM_MB:-8192}"

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

ATTN_VRAM_USER="${PULSAR_ATTN_VRAM_GB-}"
ATTN_VRAM_GB=""
ATTN_VRAM_NOTE=""
CPU="${PULSAR_CPU:-1}"
CPU_STEAL="${PULSAR_CPU_STEAL:-0}"

calc_attn_vram_gb() {
  local free_mb="${1:-0}"
  [[ "$free_mb" =~ ^[0-9]+$ ]] || free_mb=0
  local free_gb=$(( (free_mb + 512) / 1024 ))
  [ "$free_gb" -lt 1 ] && free_gb=1

  local tier_reserve="${PULSAR_ATTN_TIER_RESERVE_GB:-8}"
  local by_half=$(( free_gb / 2 ))
  local by_tier=$(( free_gb - tier_reserve ))
  [ "$by_tier" -lt 0 ] && by_tier=0

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
command -v nvidia-smi >/dev/null || {
  echo "ERROR: nvidia-smi not found" >&2; exit 1
}

mapfile -t GPU_ROWS < <(
  nvidia-smi --query-gpu=index,name,memory.total,memory.free,pcie.link.gen.max,pcie.link.width.max,pcie.link.gen.current,pcie.link.width.current,compute_cap \
    --format=csv,noheader,nounits 2>/dev/null | sed 's/, /,/g'
)

if [ "${#GPU_ROWS[@]}" -eq 0 ]; then
  echo "ERROR: no GPUs reported by nvidia-smi" >&2
  nvidia-smi -L >&2 || true
  exit 1
fi

CAND_IDX=(); CAND_NAME=(); CAND_TOTAL=(); CAND_FREE=(); CAND_PCIE=(); CAND_CC=()  # cc major*10+minor (6.0=>60..10.0=>100); 0 if N/A

is_denylisted() {
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
  idx="${idx// /}"; name="${name# }"; total="${total// /}"; free="${free// /}"
  gen="${gen// /}"; width="${width// /}"; cgen="${cgen// /}"; cwidth="${cwidth// /}"
  [[ "$total" =~ ^[0-9]+$ ]] || total=0
  [[ "$free" =~ ^[0-9]+$ ]] || free=0
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
  CAND_IDX+=("$idx"); CAND_NAME+=("$name"); CAND_TOTAL+=("$total")
  CAND_FREE+=("$free"); CAND_PCIE+=("$pcie"); CAND_CC+=("$cc")
done

n_cand=${#CAND_IDX[@]}
if [ "$n_cand" -lt 1 ]; then
  echo "ERROR: no capable GPUs (need >= ${MIN_VRAM_MB} MiB VRAM after denylist)" >&2
  nvidia-smi -L >&2 || true
  exit 1
fi

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

ATTN_I=""; ATTN_PHYS=""; ATTN_NAME=""
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

if [ -n "$ATTN_VRAM_USER" ]; then
  if [[ "$ATTN_VRAM_USER" == "off" || "$ATTN_VRAM_USER" == "0" ]]; then
    unset PULSAR_ATTN_VRAM_GB
    ATTN_VRAM_NOTE=" (user: off — full stack on attn GPU)"
  else
    export PULSAR_ATTN_VRAM_GB="$ATTN_VRAM_USER"
    ATTN_VRAM_NOTE=" (user override)"
  fi
elif [ -z "${MANUAL:-}" ] && [ -n "${ATTN_PHYS:-}" ] && [ -n "${ATTN_FREE:-}" ]; then
  ATTN_VRAM_GB="$(calc_attn_vram_gb "$ATTN_FREE")"
  export PULSAR_ATTN_VRAM_GB="$ATTN_VRAM_GB"
  _free_g=$(( (ATTN_FREE + 512) / 1024 ))
  ATTN_VRAM_NOTE=" (auto: ~${_free_g}G free on attn → budget ${ATTN_VRAM_GB}G stack)"
else
  unset PULSAR_ATTN_VRAM_GB
  ATTN_VRAM_NOTE=" (manual topology — engine default; set PULSAR_ATTN_VRAM_GB to override)"
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
echo "PULSAR_ATTN_VRAM_GB=${PULSAR_ATTN_VRAM_GB:-unset}${ATTN_VRAM_NOTE}"
echo "PULSAR_KV will cycle: $FMTS"
echo "model: $MODEL"
echo "prompt: $PROMPT"
echo "N=$N"
echo

# ---- bench ----
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# tag = tmp-file key (defaults to fmt). Lets us run f32 twice under
# different names (baseline vs noise-floor) without clobbering files.
run() {
    local fmt="$1"
    local tag="${2:-$fmt}"
    local log="$TMP/$tag.log"
    if [ "$fmt" = "f32" ]; then
        env PULSAR_PROFILE=1 "$CLI" -m "$MODEL" -p "$PROMPT" -n "$N" >/dev/null 2> "$log" || true
    else
        env PULSAR_KV="$fmt" PULSAR_PROFILE=1 "$CLI" -m "$MODEL" -p "$PROMPT" -n "$N" >/dev/null 2> "$log" || true
        grep -q "$fmt KV cache on" "$log" \
            || echo "  WARN: $fmt did not activate — model is not GQA/Qwen35" >&2
    fi
    grep -oP '\([0-9.]+ tok/s\)' "$log" | tr -d '()' | head -1 > "$TMP/$tag.tps"
    grep '^pulsar: ids' "$log" | head -1 > "$TMP/$tag.ids"
    grep -oP '\d+' "$TMP/$tag.ids" > "$TMP/$tag.nums"
    printf '  %-6s %s\n' "$tag" "$(<"$TMP/$tag.tps")"
}

echo "warmup (f32, loads weights into host cache)..."
run f32 warmup
echo

echo "noise floor (f32 again — greedy ids drift from GPU nondeterminism)..."
run f32 f32_noise
echo

echo "tok/s:"
for fmt in $FMTS; do run "$fmt"; done

echo
echo "quality (greedy ids — FIRST divergence is the real signal;"
echo "          total mismatches is dominated by autoregressive chaos):"

# Noise floor: f32 vs f32. CUDA atomics + reduction order make even the
# bit-exact path drift. A format whose first-divergence is near this floor
# is indistinguishable from nondeterminism → KV quant is correct.
# A format that diverges MUCH earlier than the floor has a real bug.
noise=$(awk 'NR==FNR{a[NR]=$1; m=NR; next} {b[FNR]=$1}
    END{ n=(m<FNR?m:FNR); c=0; first=0;
         for(i=1;i<=n;i++) if(a[i]!=b[i]){ c++; if(!first) first=i }
         printf "%d/%d differ, first at #%d", c, n, first }' \
    "$TMP/f32.nums" "$TMP/f32_noise.nums")
printf '  %-6s %s   ← noise floor (f32 vs f32)\n' "f32xf32" "$noise"

for fmt in $FMTS; do
    if [ "$fmt" = f32 ]; then
        printf '  %-6s baseline (%s tokens)\n' "$fmt" "$(wc -l < "$TMP/f32.nums")"
    else
        result=$(awk 'NR==FNR{a[NR]=$1; m=NR; next} {b[FNR]=$1}
            END{ n=(m<FNR?m:FNR); c=0; first=0;
                 for(i=1;i<=n;i++) if(a[i]!=b[i]){ c++; if(!first) first=i }
                 printf "%d/%d differ, first at #%d", c, n, first }' \
            "$TMP/f32.nums" "$TMP/$fmt.nums")
        printf '  %-6s %s\n' "$fmt" "$result"
    fi
done
