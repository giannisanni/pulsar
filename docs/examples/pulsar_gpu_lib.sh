# Shared GPU / GGUF helpers for runpulsar.sh, bench_kv.sh, bench_kv_tf.sh.
# Sourced only — not executed standalone.
#
# ATTN policy (matches engine ensure_device):
#   MLA / K3  — auto-set PULSAR_ATTN_GPU=1 when ≥2 capable GPUs (attn on the
#               fattest secondary). Safe and is what dual-GPU boxes expect.
#               With 3+ cards the engine can still use remaining GPUs for tiers;
#               set PULSAR_FORCE_ATTN=0 to leave ATTN unset and use the multi-
#               secondary layer-split planner only.
#   GQA       — opt-in only; auto-force can break some quants (matmul_q8_0 on
#               Laguna). Never auto-force.
#   Dsv4 / Qwen35 — no ATTN offload path.
#
# PULSAR_FORCE_ATTN:
#   unset|auto  — family-based policy above
#   1|yes|on    — force PULSAR_ATTN_GPU=1 when ≥2 candidates (any family)
#   0|off|no    — never force (MLA/K3 use engine default / multi-card planner)

# Print general.architecture from a GGUF (empty if missing / unreadable).
gguf_architecture() {
  local path="${1:-}"
  [ -n "$path" ] && [ -f "$path" ] || return 0
  # Prefer python3 (always available on our boxes); fall back to empty.
  command -v python3 >/dev/null 2>&1 || return 0
  python3 - "$path" <<'PY' 2>/dev/null || true
import struct, sys
path = sys.argv[1]
try:
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            sys.exit(0)
        ver = struct.unpack("<I", f.read(4))[0]
        n_tensors, n_kv = struct.unpack("<QQ", f.read(16))
        def skip(t):
            if t == 8:
                n = struct.unpack("<Q", f.read(8))[0]
                f.seek(n, 1)
            elif t == 9:
                at = struct.unpack("<I", f.read(4))[0]
                n = struct.unpack("<Q", f.read(8))[0]
                sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:8,10:1,11:1,12:8}
                if at in sizes:
                    f.seek(sizes[at] * n, 1)
                elif at == 8:
                    for _ in range(n):
                        sn = struct.unpack("<Q", f.read(8))[0]
                        f.seek(sn, 1)
                else:
                    for _ in range(n):
                        skip(at)
            else:
                sizes = {0:1,1:1,2:2,3:2,4:4,5:4,6:4,7:8,10:1,11:1,12:8}
                f.seek(sizes.get(t, 0), 1)
        for _ in range(n_kv):
            kn = struct.unpack("<Q", f.read(8))[0]
            if kn > 10_000_000:
                sys.exit(0)
            key = f.read(kn).decode("utf-8", "replace")
            t = struct.unpack("<I", f.read(4))[0]
            if key == "general.architecture" and t == 8:
                n = struct.unpack("<Q", f.read(8))[0]
                val = f.read(n).decode("utf-8", "replace")
                print(val)
                sys.exit(0)
            skip(t)
except Exception:
    pass
PY
}

# Map architecture string → family bucket: mla | k3 | gqa | primary | unknown
# "primary" = multi-GPU still useful for expert tiers, but engine never
# parks the attention stack on a secondary (Dsv4 / Qwen35 today).
gguf_attn_family() {
  local arch
  arch="$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')"
  case "$arch" in
    glm-dsa|glm_dsa|deepseek2) echo mla ;;
    kimi-k3) echo k3 ;;
    hy-v3|hy_v3|minimax-m3|minimax-m2|qwen3moe|gemma4|inkling|laguna|gpt-oss) echo gqa ;;
    deepseek4|qwen35|qwen35moe) echo primary ;;
    "") echo unknown ;;
    *) echo unknown ;;
  esac
}

# Fallback when GGUF metadata is unreadable: guess from MODEL path/name.
# Order matters: more specific patterns before broad *deepseek* / *qwen*.
gguf_attn_family_from_name() {
  local u
  u="$(echo "${1:-}" | tr '[:upper:]' '[:lower:]')"
  case "$u" in
    *deepseek*v4*|*deepseek-v4*|*dsv4*|*ds4*|*qwen35*|*qwen3.5*|*qwen3.6*|*qwen-3.5*|*qwen-3.6*)
      echo primary ;;
    *laguna*|*hy3*|*hy-v3*|*hy_v3*|*qwen3*moe*|*gemma4*|*minimax*|*inkling*|*gpt-oss*)
      echo gqa ;;
    *kimi*k3*|*kimi-k3*)
      echo k3 ;;
    *glm*|*deepseek*v3*|*deepseek-v3*|*deepseek2*|*kimi*k2*|*k2.5*|*k2.7*)
      echo mla ;;
    *)
      echo unknown ;;
  esac
}

# Resolve family for MODEL path: GGUF arch first, then filename heuristic.
resolve_attn_family() {
  local model="${1:-}"
  local arch fam
  arch="$(gguf_architecture "$model")"
  fam="$(gguf_attn_family "$arch")"
  if [ "$fam" = "unknown" ] || [ -z "$arch" ]; then
    fam="$(gguf_attn_family_from_name "$model")"
  fi
  # Export for callers that want to log arch
  GGUF_ARCH="${arch:-}"
  GGUF_ATTN_FAMILY="$fam"
  echo "$fam"
}

# Should auto-pick set PULSAR_ATTN_GPU=1?
#   0 = no, 1 = yes
# Respects PULSAR_FORCE_ATTN override; otherwise family policy.
should_force_attn() {
  local fam="${1:-unknown}"
  local n_cand="${2:-1}"
  local force="${PULSAR_FORCE_ATTN:-auto}"

  if [ "$n_cand" -lt 2 ]; then
    echo 0
    return
  fi

  case "$(echo "$force" | tr '[:upper:]' '[:lower:]')" in
    0|off|no|false|never)
      echo 0
      return
      ;;
    1|yes|on|true|force)
      echo 1
      return
      ;;
  esac

  # auto (default)
  case "$fam" in
    mla|k3)
      # Supported dual-GPU path: pin local 1 as attn (fattest secondary in CVD).
      echo 1
      ;;
    gqa)
      # Opt-in only — Laguna Q2K + forced ATTN has failed with matmul_q8_0.
      echo 0
      ;;
    primary|unknown|*)
      # Engine ignores PULSAR_ATTN_GPU for Dsv4/Qwen35 (attn stays primary).
      echo 0
      ;;
  esac
}

# Human-readable reason for the ATTN decision.
attn_policy_note() {
  local fam="${1:-unknown}"
  local forced="${2:-0}"
  local arch="${3:-}"
  local arch_s=""
  local how="${PULSAR_FORCE_ATTN:-auto}"
  [ -n "$arch" ] && arch_s=" arch=$arch"
  if [ "$forced" -eq 1 ]; then
    case "$(echo "$how" | tr '[:upper:]' '[:lower:]')" in
      1|yes|on|true|force)
        echo "PULSAR_ATTN_GPU=1 (PULSAR_FORCE_ATTN=$how)${arch_s} family=$fam"
        ;;
      *)
        echo "PULSAR_ATTN_GPU=1 (auto: $fam dual-GPU attn offload)${arch_s}"
        ;;
    esac
    return
  fi
  case "$fam" in
    mla|k3)
      echo "PULSAR_ATTN_GPU unset — need ≥2 GPUs for auto $fam offload, or set PULSAR_FORCE_ATTN=1${arch_s}"
      ;;
    gqa)
      echo "PULSAR_ATTN_GPU unset — GQA is opt-in only (PULSAR_FORCE_ATTN=1); auto off avoids matmul_q8_0${arch_s}"
      ;;
    primary)
      # deepseek4 / qwen35: engine hard-codes attn on primary (see Family::Dsv4).
      # Secondary GPUs are still used for resident expert tiers when visible.
      echo "PULSAR_ATTN_GPU intentionally unset — ${arch:-this family} keeps attention on the primary; secondary GPU(s) still take expert tiers (not a missing feature flag)"
      ;;
    *)
      echo "PULSAR_ATTN_GPU unset — unknown arch, safe default (PULSAR_FORCE_ATTN=1 only helps if the engine supports ATTN offload)${arch_s}"
      ;;
  esac
}
