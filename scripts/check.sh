#!/usr/bin/env bash
# Commit gate. Run on the CUDA box.
#   check.sh              build + GPU kernel selftests + unit tests
#   check.sh MODEL.gguf   + bit-exact decode consistency + census regression
set -euo pipefail
cd "$(dirname "$0")/.."

cargo build --release -q
# kernel selftests need exclusive GPU; the rest are CPU-only
cargo test --release -q -p kernels -- --test-threads=1
cargo test --release -q -p tokenizer -p gguf -p stream

if [ $# -ge 1 ]; then
    MODEL=$1
    CLI=./target/release/pulsar-cli
    PROMPT="The capital of France is"

    # decode vs fresh-prefill must be bit-exact on the fixed single-token
    # path (PULSAR_BATCH=1); any drift here is a real kernel/order bug
    out=$(PULSAR_BATCH=1 "$CLI" -m "$MODEL" --ctx 256 -p "$PROMPT" \
        --decode-consistency 4 --temp 0 2>&1) || true
    if ! echo "$out" | grep -q 'max |dlogit| 0.0000'; then
        echo "check: FAIL decode-consistency not bit-exact"
        echo "$out" | tail -4
        exit 1
    fi
    echo "check: decode-consistency bit-exact"

    # census ratchet regression (2083a6a): a short run on a mature census
    # must not raise its max count - the seed+delta ratchet raised it every
    # run, which slowly skewed tier ranking toward long-cached slabs
    if [ -f "${MODEL}.warm" ]; then
        census_max() {
            python3 -c '
import struct, sys
b = open(sys.argv[1], "rb").read()
print(max(struct.unpack("<QQQ", b[i:i+24])[2] for i in range(0, len(b), 24)))
' "$1"
        }
        before=$(census_max "${MODEL}.warm")
        if [ "$before" -ge 64 ]; then
            "$CLI" -m "$MODEL" --ctx 256 -p "$PROMPT" -n 8 --temp 0 >/dev/null 2>&1
            after=$(census_max "${MODEL}.warm")
            if [ "$after" -gt "$before" ]; then
                echo "check: FAIL census ratchet: max $before -> $after after an n=8 run"
                exit 1
            fi
            echo "check: census stable (max $before -> $after)"
        else
            echo "check: census immature (max $before), ratchet check skipped"
        fi
    fi
fi
echo "check: PASS"
