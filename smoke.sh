#!/usr/bin/env bash
# smoke.sh — prove the harness works before spending 6.8 GB and an hour on it.
#
# Runs the entire pipeline against a ~700 MB model. Everything that can be
# wrong with the harness (API names, barrier placement, cache introspection,
# JSONL schema, dashboard parsing) is wrong here too, and this costs 3 minutes
# instead of 90.
#
#   ./smoke.sh
#   ./smoke.sh mlx-community/Llama-3.2-3B-Instruct-4bit
set -euo pipefail
cd "$(dirname "$0")"
VENV="${VENV:-$(cd .. && pwd)}"
PY="$VENV/bin/python"
MODEL="${1:-mlx-community/Llama-3.2-1B-Instruct-4bit}"
OUT="smoke-runs.jsonl"

echo "########## smoke test: $MODEL ##########"
echo

echo "--- 1/4 bandwidth probe (small sweep) ---"
"$PY" bwprobe.py --max-mb 256 --reps 8 \
  --gemv-cols 2048 --gemv-rows 4096,16384,65536,262144 --json smoke-bw.json

GBS=$("$PY" -c "import json;print(f\"{json.load(open('smoke-bw.json'))['achievable_gemv_gbs']:.1f}\")")
echo
echo "--- achievable q4 GEMV: $GBS GB/s ---"

echo
echo "--- 2/4 roofline ---"
"$PY" roofline.py --model "$MODEL" --achievable-gbs "$GBS" \
  --contexts 128,512,2048,8192 --probe-lengths 64,512,1100 \
  --prefill-step 256 --json smoke-roofline.json

echo
echo "--- 3/4 bench (tiny matrix, 2 runs) ---"
"$PY" bench.py --model "$MODEL" \
  --prompt-tokens 128,512 --gen-tokens 32 \
  --runs 2 --warmup 1 --cooldown 5 --temp 0.0 \
  --sync per-token --engine manual \
  --kv-from smoke-roofline.json --achievable-gbs "$GBS" \
  --out "$OUT" --tag smoke

echo
echo "--- 3b/4 same, stock stream_generate cross-check ---"
"$PY" bench.py --model "$MODEL" \
  --prompt-tokens 128,512 --gen-tokens 32 \
  --runs 2 --warmup 1 --cooldown 5 --temp 0.0 \
  --engine stream \
  --kv-from smoke-roofline.json --achievable-gbs "$GBS" \
  --out "$OUT" --tag smoke-stream

echo
echo "--- 4/4 sanity checks on $OUT ---"
"$PY" - "$OUT" <<'PY'
import json, sys, collections
rows=[json.loads(l) for l in open(sys.argv[1]) if l.strip()]
runs=[r for r in rows if r.get("record_type")!="series_summary"]
summ=[r for r in rows if r.get("record_type")=="series_summary"]
print(f"  {len(runs)} run records, {len(summ)} series summaries")
need=["chip","model","decode_tok_s","effective_gbs","pct_of_roofline",
      "peak_memory_bytes","output_sha256","schema_version","itl_p95_ms"]
missing=[k for k in need if any(r.get(k) is None for r in runs)]
print("  missing/None fields:", missing or "none")
by=collections.defaultdict(list)
for r in runs: by[(r["engine"],r["prompt_tokens"])].append(r["decode_tok_s"])
for k,v in sorted(by.items()): print(f"  {k}: decode {min(v):.2f}-{max(v):.2f} tok/s")
m={}
for r in runs: m.setdefault((r["engine"],r["prompt_tokens"]),set()).add(r["output_sha256"])
bad=[k for k,v in m.items() if len(v)>1]
print("  greedy determinism:", "OK" if not bad else f"FAILED for {bad}")
man={k[1]:sum(v)/len(v) for k,v in by.items() if k[0]=="manual"}
strm={k[1]:sum(v)/len(v) for k,v in by.items() if k[0]=="stream"}
for p in sorted(set(man)&set(strm)):
    d=100*(man[p]-strm[p])/strm[p]
    flag="OK" if abs(d)<8 else "INVESTIGATE"
    print(f"  manual vs stock @{p}: {man[p]:.2f} vs {strm[p]:.2f} tok/s ({d:+.1f}%) {flag}")
PY

cat <<EOF

If everything above says OK, the harness is sound. Clean up with:
  rm -f smoke-runs.jsonl smoke-bw.json smoke-roofline.json

Then:  ./run_matrix.sh --help
EOF
