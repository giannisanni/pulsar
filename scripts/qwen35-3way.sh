#!/usr/bin/env bash
# 35B three-way: no-spec vs DFlash draft vs DSpark draft. Qwen3.6-35B-A3B
# is the only fleet target WITHOUT a native nextn block, so a draft is the
# only speculative option here (and it is qwen35 family, so the draft path
# works). This decides whether the DSpark work earns its place.
set -euo pipefail
cd "$(dirname "$0")/.."

# free the cards (park the llama.cpp container by name-immune rename)
docker rename thinkingcap thinkingcap-parked 2>/dev/null || true
docker update --restart=no thinkingcap-parked 2>/dev/null || true
docker stop thinkingcap-parked >/dev/null 2>&1 || true
sleep 3

TARGET=/mnt/models/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
DFLASH=/mnt/models/Qwen3.6-35B-DFlash-draft-mine.gguf
DSPARK=/mnt/models/Qwen3.6-35B-DSpark-draft.gguf
MATH="Compute the sum of all integers n between 1 and 100 such that n^2 + n + 41 is divisible by 3. Show your work."
PROSE="Write a reflective essay about the experience of walking through a city at night, focusing on sound and memory."
N=200

echo "== build =="
cargo build --release 2>&1 | grep -E "^error" && exit 1 || true

echo "== fetch + convert 35B DSpark draft =="
DHF=/mnt/models/dspark-35b-hf
if [ ! -f "$DHF/model.safetensors" ]; then
  mkdir -p "$DHF"
  for f in config.json model.safetensors; do
    curl -sL --fail -o "$DHF/$f" \
      "https://huggingface.co/fal/Qwen3.6-35B-A3B-Magic-Prompt-FP8-DSpark/resolve/main/$f"
  done
fi
[ -f "$DSPARK" ] || python3 scripts/convert-dspark-draft.py "$DHF" "$DSPARK"

warm() { # prime the census so numbers are warm-run
  "$@" -n 8 --ctx 512 >/dev/null 2>&1 || true
}

run() { # label, env..., --, args...
  local label="$1"; shift
  local envs=(); while [ "$1" != "--" ]; do envs+=("$1"); shift; done; shift
  echo "--- $label ---"
  env "${envs[@]}" ./target/release/pulsar-cli "$@" 2>&1 | grep -E "tok/s" | tail -1
}

for mode in math prose; do
  [ "$mode" = math ] && P="$MATH" || P="$PROSE"
  echo; echo "===== $mode ====="
  # warm census for the target once
  ./target/release/pulsar-cli -m "$TARGET" --ctx 512 -p "$P" -n 8 --temp 0 >/dev/null 2>&1 || true
  run "no-spec (plain decode)" X=1 -- \
    -m "$TARGET" --ctx 4096 -p "$P" -n $N --temp 0
  run "DFlash draft" PULSAR_DFLASH="$DFLASH" -- \
    -m "$TARGET" --ctx 4096 -p "$P" -n $N --temp 0
  run "DSpark draft (heads on)" PULSAR_DFLASH="$DSPARK" -- \
    -m "$TARGET" --ctx 4096 -p "$P" -n $N --temp 0
  run "DSpark draft (conf off)" PULSAR_DFLASH="$DSPARK" PULSAR_DSPARK_CONF=off -- \
    -m "$TARGET" --ctx 4096 -p "$P" -n $N --temp 0
done

echo; echo "== identity: DFlash and DSpark greedy must match plain =="
BASE=$(./target/release/pulsar-cli -m "$TARGET" --ctx 2048 -p "$MATH" -n 48 --temp 0 2>&1 | grep ids | tail -1)
DF=$(PULSAR_DFLASH="$DFLASH" ./target/release/pulsar-cli -m "$TARGET" --ctx 2048 -p "$MATH" -n 48 --temp 0 2>&1 | grep ids | tail -1)
DS=$(PULSAR_DFLASH="$DSPARK" ./target/release/pulsar-cli -m "$TARGET" --ctx 2048 -p "$MATH" -n 48 --temp 0 2>&1 | grep ids | tail -1)
[ "$BASE" = "$DF" ] && echo "DFlash IDENTITY PASS" || echo "DFlash IDENTITY MISMATCH"
[ "$BASE" = "$DS" ] && echo "DSpark IDENTITY PASS" || echo "DSpark IDENTITY MISMATCH"

echo; echo "== restore thinkingcap =="
docker rename thinkingcap-parked thinkingcap 2>/dev/null || true
docker update --restart=unless-stopped thinkingcap 2>/dev/null || true
docker start thinkingcap >/dev/null 2>&1 || true
echo DONE
