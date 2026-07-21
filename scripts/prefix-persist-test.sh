#!/usr/bin/env bash
# Live test: dsv4 prefix persistence across a server restart.
# Phase A: fresh serve, prefill a ~1.5K-token prompt (slow), reply, save.
# Phase B: kill, restart with the same --prefix-file, send the same
#          conversation + one more turn: prefill must cover only the
#          suffix (seconds, not minutes), and the reply must be coherent.
set -euo pipefail
cd "$(dirname "$0")/.."

docker rename thinkingcap thinkingcap-parked 2>/dev/null || true
docker update --restart=no thinkingcap-parked 2>/dev/null || true
docker stop thinkingcap-parked >/dev/null 2>&1 || true
sleep 2

M=/mnt/models/DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix.gguf
PFX=/tmp/dsv4-test.prefix
rm -f "$PFX"
LOG=/tmp/prefix-serve.log

serve() {
  PULSAR_CPU=1 nohup ./target/release/pulsar-serve -m "$M" --port 8699 --ctx 8192 \
    --prefix-file "$PFX" > "$LOG" 2>&1 &
  echo $! > /tmp/prefix-serve.pid
  until curl -s -m 2 localhost:8699/v1/models >/dev/null 2>&1; do sleep 3; done
}

# a prompt long enough that re-prefilling it would visibly cost minutes
LONGSYS=$(python3 -c "print(('You are a precise assistant. ' + 'Context paragraph: the quick brown fox jumps over the lazy dog near the riverbank while engineers benchmark inference engines on consumer hardware. ')*80)")

req() { # extra_user_turn
  python3 - "$1" <<'EOF'
import json, sys, time, urllib.request
extra = sys.argv[1]
msgs=[{"role":"system","content":open('/tmp/longsys.txt').read()},
      {"role":"user","content":"In one short sentence, what animal jumps in the context paragraph?"}]
if extra != "-":
    msgs += [{"role":"assistant","content":"A fox jumps in the context paragraph."},
             {"role":"user","content":extra}]
t0=time.time()
r=urllib.request.urlopen(urllib.request.Request("http://127.0.0.1:8699/v1/chat/completions",
    json.dumps({"model":"dsv4","messages":msgs,"max_tokens":40,"temperature":0,"stream":False}).encode(),
    {"content-type":"application/json"}), timeout=3600)
d=json.load(r)
txt=d["choices"][0]["message"]["content"]
print(f"[{time.time()-t0:.0f}s] {txt[:90]!r}")
EOF
}

echo "$LONGSYS" > /tmp/longsys.txt
echo "== phase A: fresh serve, full prefill =="
serve
req "-"
sleep 2
grep -E "prefix saved|prefix restored" "$LOG" | tail -2 || echo "(no save line yet)"
ls -la "$PFX" 2>/dev/null || echo "NO PREFIX FILE"

echo "== kill server =="
kill "$(cat /tmp/prefix-serve.pid)" 2>/dev/null || true
sleep 3

echo "== phase B: restart, suffix-only prefill =="
serve
grep -E "prefix restored|prefix file skipped" "$LOG" | tail -1
req "And what animal is lazy? One short sentence."
grep -E "prefix cache hit" "$LOG" | tail -1 || echo "(no cache-hit line)"

kill "$(cat /tmp/prefix-serve.pid)" 2>/dev/null || true
sleep 2
docker rename thinkingcap-parked thinkingcap 2>/dev/null || true
docker update --restart=unless-stopped thinkingcap 2>/dev/null || true
docker start thinkingcap >/dev/null 2>&1 || true
echo TEST-DONE
