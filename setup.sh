#!/usr/bin/env bash
# setup.sh — one-time setup. Safe to re-run.
#
# Assumes this directory sits inside your venv (~/mlx-env/mlx-bench) so the
# venv is one level up. Override with:  VENV=/path/to/venv ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"

VENV="${VENV:-$(cd .. && pwd)}"
if [ ! -x "$VENV/bin/python" ]; then
  echo "No venv at $VENV (expected $VENV/bin/python)."
  echo "Set VENV=/path/to/venv and re-run."
  exit 1
fi
PY="$VENV/bin/python"

echo "=== venv ==="
echo "  $VENV"
"$PY" -c 'import sys; print("  python", sys.version.split()[0])'

echo
echo "=== already installed ==="
"$PY" - <<'PY'
import importlib.metadata as md
for p in ("mlx","mlx-lm","transformers","huggingface_hub","numpy","streamlit","altair","pandas"):
    try:
        print(f"  {p:<18}{md.version(p)}")
    except md.PackageNotFoundError:
        print(f"  {p:<18}MISSING")
PY

echo
echo "=== installing dashboard deps (mlx/mlx-lm left untouched) ==="
"$PY" -m pip install --quiet --upgrade streamlit altair pandas
echo "  done"

echo
echo "=== disk space (model download is ~6.8 GB) ==="
df -h "$HOME" | tail -1
echo "  HF cache: ${HF_HOME:-$HOME/.cache/huggingface}"
du -sh "${HF_HOME:-$HOME/.cache/huggingface}/hub" 2>/dev/null || echo "  (no HF cache yet)"

echo
echo "=== API check ==="
# apicheck.py was removed during the 2026-08-30 cleanup as "scaffolding" while
# setup.sh still called it — a delete made without checking references. Guarded
# rather than resurrected: what it reported (which MLX/mlx-lm APIs exist on this
# machine) is now covered by mlxutil's _resolve() shims, which fail loudly at the
# point of use instead of in a separate preflight.
if [ -f apicheck.py ]; then
  "$PY" apicheck.py
else
  echo "  apicheck.py not present — mlxutil._resolve() handles version drift at call sites"
  "$PY" -c 'import mlxutil as U; h=U.host_info(); print(f"  mlx {h["mlx_version"]} / mlx-lm {h["mlx_lm_version"]} / python {h["python"]}")'
fi

cat <<'EOF'

Next:
  ./smoke.sh                 # 3-minute end-to-end validation on a 1B model
  ./run_matrix.sh --help     # the real thing
EOF
