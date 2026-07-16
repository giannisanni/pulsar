#!/usr/bin/env bash
# Standardized decode benchmark. README numbers come from this script and
# nowhere else. Protocol (learned the hard way, 2026-07-15):
#   - warm census: tier placement ranks from the .warm popularity file, so
#     the first run after a census wipe is NOT a benchmark
#   - n=64 sustained: short runs read high (per-token SSD miss rate is
#     still climbing toward steady state)
#   - second warm run is the canonical number
#   - quiet box: streaming decode is disk-bound, a busy box reads ~20% low
#
# usage: bench.sh MODEL.gguf [N]
set -euo pipefail

MODEL=${1:?usage: bench.sh MODEL.gguf [N]}
N=${2:-64}
CLI=${CLI:-./target/release/pulsar-cli}
PROMPT="The three most important inventions of the twentieth century were"

load=$(awk '{print $1}' /proc/loadavg)
if awk -v l="$load" 'BEGIN{exit !(l > 1.5)}'; then
    echo "bench: WARNING 1-min load is ${load}; numbers will read low" >&2
fi

if [ ! -f "${MODEL}.warm" ]; then
    echo "bench: no census, cold run to build one (number discarded)"
    "$CLI" -m "$MODEL" --ctx 512 -p "$PROMPT" -n "$N" --temp 0 2>&1 \
        | grep "tok/s" | sed 's/^/  cold: /'
fi

echo "bench: warm run 1"
"$CLI" -m "$MODEL" --ctx 512 -p "$PROMPT" -n "$N" --temp 0 2>&1 \
    | grep "tok/s" | sed 's/^/  /'
echo "bench: warm run 2 (canonical)"
"$CLI" -m "$MODEL" --ctx 512 -p "$PROMPT" -n "$N" --temp 0 2>&1 \
    | grep "tok/s" | sed 's/^/  /'
