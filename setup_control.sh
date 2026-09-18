#!/usr/bin/env bash
# setup_control.sh — install mlx-vlm in a SEPARATE venv as an external control.
#
# WHY A SEPARATE VENV, NOT ~/mlx-env
#
# Our entire Recipe 0 baseline — every row in runs.jsonl, the 97%-of-roofline
# result, the 24.5 tok/s reference — was measured on mlx 0.32.2 / mlx-lm 0.31.3.
# Installing anything that resolves a different mlx into ~/mlx-env silently
# invalidates all of it, and there is no way to tell afterwards which numbers
# came from which build. So the control lives in its own environment.
#
# CORRECTION, measured 2026-08-28: mlx-vlm 0.6.17 does NOT depend on mlx-lm.
# A clean install of it leaves mlx_lm entirely absent. The earlier claim in
# OBJECTIVES.md that "it depends on mlx-lm and will happily upgrade it" was
# wrong. The separate venv is still correct practice — mlx-vlm does pull its own
# mlx — but the stated reason was not the real one.
#
#     bash setup_control.sh              # install / reuse
#     bash setup_control.sh --recreate   # delete and rebuild
#     bash setup_control.sh 0.6.18       # a different pinned version

set -euo pipefail

CONTROL="$HOME/mlx-vlm-control"
OURS="$HOME/mlx-env"
RECREATE=0
MLXVLM="0.6.17"
for a in "$@"; do
  case "$a" in
    --recreate) RECREATE=1 ;;
    *) MLXVLM="$a" ;;
  esac
done

# Read a distribution's version the way pip records it, not by guessing at a
# module attribute. `import mlx` exposes NO __version__, so the previous
# getattr(m,"__version__","?") probe reported "?" for mlx in EVERY environment
# and the pin gate below failed on a perfectly healthy ~/mlx-env. A check that
# raises a false alarm is worse than no check: it sends you to reinstall
# something that was never broken.
read -r -d '' VERSION_PROBE <<'GATE' || true
import sys
from importlib.metadata import version, PackageNotFoundError

def v(dist, mod=None):
    try:
        return version(dist)
    except PackageNotFoundError:
        pass
    try:
        m = __import__(mod or dist.replace("-", "_"))
        return getattr(m, "__version__", "installed, version unknown")
    except Exception as e:
        return f"MISSING ({type(e).__name__})"

names = sys.argv[1:]
expect = {}
for n in names:
    if "=" in n:
        d, e = n.split("=", 1)
        expect[d] = e
    else:
        d = n
    print(f"  {d:<14}{v(d)}")
bad = [f"{d}: expected {e}, found {v(d)}" for d, e in expect.items() if v(d) != e]
if bad:
    print()
    print("  !! PIN VIOLATION:")
    for b in bad:
        print(f"     {b}")
    sys.exit(1)
if expect:
    print("  pins intact - runs.jsonl stays comparable")
GATE

# The interpreter must match ours. Building the control venv from ~/mlx-env's
# own python guarantees it: a bare `python3 -m venv` picked up 3.14 while ours
# is 3.12.14, and drafting is CPU-bound on graph construction, so comparing
# their implementation against ours across two interpreter versions would
# confound the measurement with CPython's own performance delta.
OURPY="$("$OURS/bin/python" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])')"
echo "=== our interpreter: $OURPY ==="

if [ -d "$CONTROL" ]; then
  CTLPY="$("$CONTROL/bin/python" -c 'import sys; print("%d.%d.%d" % sys.version_info[:3])' 2>/dev/null || echo "unknown")"
  if [ "$RECREATE" = "1" ]; then
    echo "  --recreate: removing $CONTROL"
    rm -rf "$CONTROL"
  elif [ "${CTLPY%.*}" != "${OURPY%.*}" ]; then
    echo
    echo "  !! CONTROL VENV INTERPRETER MISMATCH: control $CTLPY vs ours $OURPY"
    echo "     A speed control across two CPython minor versions is not a control."
    echo "     Rebuild it:  bash setup_control.sh --recreate"
    exit 1
  else
    echo "  reusing $CONTROL (python $CTLPY)"
  fi
fi

if [ ! -d "$CONTROL" ]; then
  echo "=== creating control venv at $CONTROL from our interpreter ==="
  "$OURS/bin/python" -m venv "$CONTROL"
fi

echo
echo "=== installing mlx-vlm==$MLXVLM (this does NOT touch ~/mlx-env) ==="
"$CONTROL/bin/pip" install --quiet --upgrade pip
"$CONTROL/bin/pip" install --quiet "mlx-vlm==$MLXVLM"

echo
echo "=== versions in the CONTROL venv ==="
"$CONTROL/bin/python" -c "$VERSION_PROBE" mlx mlx-lm mlx-vlm transformers 2>/dev/null

echo
echo "=== versions in OUR venv (gated: mlx 0.32.2 / mlx-lm 0.31.3) ==="
"$OURS/bin/python" -c "$VERSION_PROBE" mlx=0.32.2 mlx-lm=0.31.3 transformers 2>/dev/null

echo
echo "next:  $CONTROL/bin/python probe_mlxvlm_api.py"
