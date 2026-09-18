#!/usr/bin/env bash
# run_matrix.sh — the real measurement. ONE pass per invocation, on purpose.
#
# You are on a fanless M3 Air. Running all three passes back to back guarantees
# the later ones measure heat, not kernels. Run one, let the machine sit for
# 10 minutes, run the next.
#
#   ./run_matrix.sh prep        # download model, print the memory plan, stop
#   ./run_matrix.sh pertoken    # pass 1: latency distribution   (~25 min)
#   ./run_matrix.sh pipelined   # pass 2: best throughput        (~25 min)
#   ./run_matrix.sh stream      # pass 3: stock mlx-lm crosscheck(~25 min)
#   ./run_matrix.sh energy      # short powermetrics pass        (~6 min)
#   ./run_matrix.sh dash        # open the dashboard
set -euo pipefail
cd "$(dirname "$0")"
VENV="${VENV:-$(cd .. && pwd)}"
PY="$VENV/bin/python"

# Verified working on this machine: coherent output, PLE protected at 8 bits by
# OptiQ's sensitivity pass. The stock mlx-community 4-bit conversions quantize
# PLE to 4 bits and are unusable; gemma-4-12B-it-4bit is additionally
# model_type gemma4_unified, which needs the compat remap in mlxutil.py.
MODEL="${MODEL:-mlx-community/gemma-4-e4b-it-OptiQ-4bit}"
PROMPTS="${PROMPTS:-128,2048,8192}"
GEN="${GEN:-256}"
RUNS="${RUNS:-5}"
COOLDOWN="${COOLDOWN:-40}"     # fanless Air. Lower it only on a Pro/mini.
OUT="${OUT:-runs.jsonl}"

usage() { sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0; }
[ $# -eq 0 ] && usage
case "${1:-}" in -h|--help|help) usage;; esac

get_gbs() {
  if [ ! -f bwprobe.json ]; then
    echo "bwprobe.json missing — run: $PY bwprobe.py --json bwprobe.json" >&2
    exit 1
  fi
  "$PY" -c "import json;print(f\"{json.load(open('bwprobe.json'))['achievable_gemv_gbs']:.1f}\")"
}

preflight() {
  echo "--- preflight ---"
  # Apple Silicon targets identical performance on battery and AC, so battery
  # alone is not disqualifying. Low Power Mode and a nearly flat battery are.
  local lpm pct src
  lpm=$(pmset -g 2>/dev/null | awk '/lowpowermode/ {print $2}')
  pct=$(pmset -g batt 2>/dev/null | grep -o '[0-9]\+%' | head -1 | tr -d '%')
  if pmset -g batt 2>/dev/null | grep -q "AC Power"; then src="AC"; else src="Battery"; fi
  echo "  power: $src ${pct:-?}%   lowpowermode=${lpm:-?}"
  if [ "${lpm:-0}" = "1" ]; then
    echo "  !! LOW POWER MODE IS ON. It caps CPU/GPU frequency — every number is fiction."
    echo "     System Settings > Battery > Low Power Mode: Never. Then re-run."
    exit 1
  fi
  if [ "$src" = "Battery" ] && [ -n "$pct" ] && [ "$pct" -lt 30 ]; then
    echo "  !! Battery under 30% — the SoC starts limiting peak power draw. Plug in."
    exit 1
  fi
  if [ "$src" = "Battery" ]; then
    echo "  note: on battery. Fine on Apple Silicon, but do not compare these runs"
    echo "        against AC runs — power_source is recorded in runs.jsonl."
  fi
  local busy
  busy=$(ps -Ao %cpu,comm | awk 'NR>1 && $1>15 {print "    " $0}' | head -6)
  if [ -n "$busy" ]; then
    echo "  processes over 15% CPU:"; echo "$busy"
    if echo "$busy" | grep -qi "xprotect\|mds\|mdworker\|backupd\|photoanalysis"; then
      echo "  ^ That is a macOS background scan (XProtect / Spotlight / Time Machine),"
      echo "    almost certainly triggered by the model files you just downloaded."
      echo "    It is transient. WAIT for it to finish — it competes for the same"
      echo "    memory bandwidth you are about to measure. Re-run this in a few minutes."
    else
      echo "  Close them. They share your memory bandwidth."
    fi
  fi
  echo "  memory:"
  vm_stat | awk '
    /Pages free/ {free=$3}
    /Pages inactive/ {inact=$3}
    /occupied by compressor/ {comp=$5}
    END {printf "    free %.1f GB   inactive %.1f GB   compressed %.1f GB\n",
         free*16384/1073741824, inact*16384/1073741824, comp*16384/1073741824}'
  echo "    pressure level: $(sysctl -n kern.memorystatus_level 2>/dev/null)  (higher is better)"
  # Reuse the tested Python rollup rather than hand-rolled awk.
  "$PY" -c "
import mlxutil as U
rows = U.top_processes()
if rows:
    print('  biggest resident processes:')
    for p in rows:
        tag = '  <- this preflight' if p.get('self') else ''
        print(f\"    {p['name']:<34}{p['rss_mb']:>7} MB{tag}\")
    other = sum(p['rss_mb'] for p in rows if not p.get('self'))
    print(f'  other apps total: {other} MB — anything over ~1 GB competes with the run.')
" 2>/dev/null || echo "  (process rollup unavailable)"
}

case "$1" in
prep)
  echo "=== downloading $MODEL (~6.8 GB, resumable) ==="
  "$VENV/bin/hf" download "$MODEL" || "$PY" -m huggingface_hub.commands.huggingface_cli download "$MODEL"
  echo
  echo "=== bandwidth probe (hidden size 2560 for e4b) ==="
  "$PY" bwprobe.py --gemv-cols "${HIDDEN:-2560}" \
    --gemv-rows 4096,16384,65536,262144 --json bwprobe.json
  GBS=$(get_gbs)
  echo
  echo "=== roofline (this loads the model and probes real KV growth) ==="
  "$PY" roofline.py --model "$MODEL" --achievable-gbs "$GBS" \
    --contexts 128,2048,8192,16384,32768 --json roofline.json
  echo
  echo "=== memory plan for the matrix (no timing) ==="
  "$PY" bench.py --model "$MODEL" --prompt-tokens "$PROMPTS" --gen-tokens "$GEN" \
    --kv-from roofline.json --print-plan
  echo
  echo "Read the plan above. If any row says EXCEEDS BUDGET, set PROMPTS to the"
  echo "lengths that fit before running a pass."
  ;;

pertoken|pipelined)
  preflight
  GBS=$(get_gbs)
  echo "=== pass: $1  ($PROMPTS x $GEN, median of $RUNS, cooldown ${COOLDOWN}s) ==="
  "$PY" bench.py --model "$MODEL" \
    --prompt-tokens "$PROMPTS" --gen-tokens "$GEN" \
    --runs "$RUNS" --warmup 1 --cooldown "$COOLDOWN" --temp 0.0 \
    --sync "$([ "$1" = pertoken ] && echo per-token || echo pipelined)" \
    --engine manual --kv-from roofline.json --achievable-gbs "$GBS" \
    --out "$OUT" --tag "baseline-$1" \
    $([ "$1" = pertoken ] && echo "--dump-prompts prompts --measure-template")
  echo
  echo "Let the machine idle ~10 minutes before the next pass."
  ;;

stream)
  preflight
  GBS=$(get_gbs)
  echo "=== pass: stock mlx-lm stream_generate ==="
  "$PY" bench.py --model "$MODEL" \
    --prompt-tokens "$PROMPTS" --gen-tokens "$GEN" \
    --runs "$RUNS" --warmup 1 --cooldown "$COOLDOWN" --temp 0.0 \
    --engine stream --kv-from roofline.json --achievable-gbs "$GBS" \
    --out "$OUT" --tag stock-stream
  ;;

energy)
  preflight
  echo "powermetrics needs sudo. Authenticating once now:"
  sudo -v
  GBS=$(get_gbs)
  "$PY" bench.py --model "$MODEL" \
    --prompt-tokens 2048 --gen-tokens "$GEN" \
    --runs 3 --warmup 1 --cooldown "$COOLDOWN" --temp 0.0 \
    --energy --kv-from roofline.json --achievable-gbs "$GBS" \
    --out "$OUT" --tag energy
  ;;

stock)
  # the reference command, run by hand against the identical prompt text
  echo "=== stock mlx_lm.generate, prompts/p2048.txt ==="
  "$VENV/bin/mlx_lm.generate" --model "$MODEL" --prompt - < prompts/p2048.txt \
    --max-tokens "$GEN" --temp 0.0 --ignore-chat-template --verbose True
  ;;

dash)
  echo "Close this before running another pass — the browser competes for bandwidth."
  "$VENV/bin/streamlit" run dashboard.py -- --runs "$OUT"
  ;;

*) usage;;
esac
