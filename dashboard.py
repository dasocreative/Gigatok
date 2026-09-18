#!/usr/bin/env python3
"""
dashboard.py — Streamlit view over runs.jsonl.

Hard rule, enforced in code rather than by convention: a single chart axis
never mixes chips. Chip is a required filter. "Compare chips" renders separate
faceted rows with independent axes and a warning banner — because the fastest
way to fool yourself about a Metal kernel is to put an M3 and an M4 Pro on one
y-axis and read the gap as an optimization win.

Runs are also separated by sync_mode and engine, which are not comparable to
each other either, and energy passes are excluded from throughput charts by
default (powermetrics perturbs the measurement).

Run:  streamlit run dashboard.py -- --runs runs.jsonl
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

SCHEMA_VERSION = 1

st.set_page_config(page_title="MLX inference recipes", layout="wide")


def _runs_path() -> Path:
    argv = sys.argv[1:]
    if "--runs" in argv:
        return Path(argv[argv.index("--runs") + 1])
    return Path("runs.jsonl")


@st.cache_data(show_spinner=False)
def load_runs(path: str, mtime: float) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                st.warning(f"skipping malformed line {i}")
    return pd.DataFrame(rows)


path = _runs_path()
st.title("MLX inference recipes — Recipe 0 baseline")

if not path.exists():
    st.error(f"`{path}` not found. Run bench.py first.")
    st.stop()

df = load_runs(str(path), path.stat().st_mtime)
if df.empty:
    st.warning("runs.jsonl is empty.")
    st.stop()

# Separate the per-run records from the per-series summaries bench.py appends.
if "record_type" in df.columns:
    summaries = df[df["record_type"] == "series_summary"].copy()
    runs = df[df["record_type"].isna()].copy()
else:
    summaries = pd.DataFrame()
    runs = df.copy()

for col in ("chip", "model", "prompt_tokens", "decode_tok_s", "schema_version",
            "engine", "sync_mode", "run_index"):
    if col not in runs.columns:
        runs[col] = None

bad_schema = runs[runs["schema_version"] != SCHEMA_VERSION]
if not bad_schema.empty:
    st.info(f"{len(bad_schema)} run(s) from a different schema_version are hidden. "
            f"Bump the dashboard when you bump the schema.")
runs = runs[runs["schema_version"] == SCHEMA_VERSION]
if runs.empty:
    st.warning("No runs at the current schema version.")
    st.stop()

# ---------------------------------------------------------------- sidebar
st.sidebar.header("Filters")

chips = sorted(runs["chip"].dropna().unique().tolist())
compare_chips = st.sidebar.checkbox("Compare chips (faceted, never shared axes)", value=False)

if compare_chips:
    sel_chips = st.sidebar.multiselect("Chips", chips, default=chips)
    st.sidebar.warning("Faceted mode: each chip gets its own axis. Cross-chip "
                       "numbers are not a speedup — bandwidth and core count differ.")
else:
    if not chips:
        st.error("No run carries a `chip` field. Chip is required.")
        st.stop()
    sel_chips = [st.sidebar.radio("Chip (required)", chips, index=0)]

models = sorted(runs["model"].dropna().unique().tolist())
sel_models = st.sidebar.multiselect("Model", models, default=models)


def _opts(col):
    return sorted(runs[col].dropna().astype(str).unique().tolist()) if col in runs.columns else []


quants = _opts("quant_bits")
sel_quants = st.sidebar.multiselect("Quant bits", quants, default=quants)
engines = _opts("engine")
sel_engines = st.sidebar.multiselect("Engine", engines, default=engines)
syncs = _opts("sync_mode")
sel_syncs = st.sidebar.multiselect("Sync mode", syncs, default=syncs)
tags = _opts("tag")
sel_tags = st.sidebar.multiselect("Tag", tags, default=tags)
show_energy = st.sidebar.checkbox("Include energy passes in throughput charts", value=False)
show_contended = st.sidebar.checkbox(
    "Include memory-contended runs", value=False,
    help="Runs during which macOS compressed or swapped. They measure the memory "
         "subsystem, not MLX.")

f = runs[runs["chip"].isin(sel_chips) & runs["model"].isin(sel_models)]
if sel_quants and "quant_bits" in f.columns:
    f = f[f["quant_bits"].astype(str).isin(sel_quants)]
if sel_engines and "engine" in f.columns:
    f = f[f["engine"].astype(str).isin(sel_engines)]
if sel_syncs and "sync_mode" in f.columns:
    f = f[f["sync_mode"].astype(str).isin(sel_syncs)]
if sel_tags and "tag" in f.columns:
    f = f[f["tag"].astype(str).isin(sel_tags)]
if not show_energy and "energy_pass" in f.columns:
    f = f[f["energy_pass"] != True]  # noqa: E712
n_contended = 0
if "host_contended" in f.columns:
    n_contended = int((f["host_contended"] == True).sum())  # noqa: E712
    if not show_contended:
        f = f[f["host_contended"] != True]  # noqa: E712
if n_contended and not show_contended:
    st.warning(f"{n_contended} run(s) hidden: macOS compressed or swapped during them. "
               f"Close memory-hungry apps and re-run those cells.")

if f.empty:
    st.warning("No runs match the current filters.")
    st.stop()

if not compare_chips and f["chip"].nunique() > 1:
    st.error("Refusing to plot: filtered selection contains more than one chip.")
    st.stop()

if "power_source" in f.columns and f["power_source"].nunique() > 1:
    st.warning("This selection mixes AC and battery runs. Apple Silicon targets "
               "parity, but it is not guaranteed — filter to one power source "
               "before reading a small difference as real.")

f = f.copy()
f["config"] = (f["model"].astype(str).str.split("/").str[-1] + " | "
               + f["engine"].astype(str) + " | " + f["sync_mode"].astype(str))

# ---------------------------------------------------------------- headline
st.subheader("Headline")
cols = st.columns(4)
cols[0].metric("Runs shown", len(f))
cols[1].metric("Median decode tok/s", f"{f['decode_tok_s'].median():.2f}")
if "pct_of_roofline" in f.columns and f["pct_of_roofline"].notna().any():
    cols[2].metric("Median % of roofline", f"{f['pct_of_roofline'].median():.1f}%")
else:
    cols[2].metric("Median % of roofline", "n/a")
    cols[2].caption("Pass --achievable-gbs from bwprobe.py")
if "peak_memory_bytes" in f.columns and f["peak_memory_bytes"].notna().any():
    cols[3].metric("Peak memory", f"{f['peak_memory_bytes'].max() / 1024**3:.2f} GiB")

if not summaries.empty:
    sus = summaries[summaries.get("thermal_suspect") == True]  # noqa: E712
    if not sus.empty:
        st.error(f"Thermal throttling suspected in {len(sus)} series "
                 f"(monotonic decode decline > 5% first->last). "
                 f"Increase --cooldown and re-run before trusting those medians.")
    nd = summaries[summaries.get("greedy_deterministic") == False]  # noqa: E712
    if not nd.empty:
        st.error(f"{len(nd)} series produced different greedy outputs across runs. "
                 f"Losslessness checks in later recipes are meaningless until that is fixed.")


def facet(chart_fn, title: str):
    st.markdown(f"**{title}**")
    if compare_chips and f["chip"].nunique() > 1:
        for chip in sel_chips:
            sub = f[f["chip"] == chip]
            if sub.empty:
                continue
            st.caption(chip)
            st.altair_chart(chart_fn(sub), use_container_width=True)
    else:
        st.altair_chart(chart_fn(f), use_container_width=True)


# ---------------------------------------------------------------- charts
def decode_chart(d: pd.DataFrame):
    base = alt.Chart(d).encode(
        x=alt.X("prompt_tokens:O", title="prompt tokens"),
        color=alt.Color("config:N", title=None),
    )
    tips = [c for c in ("model", "prompt_tokens", "decode_tok_s", "itl_p50_ms",
                        "itl_p95_ms", "run_index", "pct_of_roofline") if c in d.columns]
    pts = base.mark_point(filled=True, size=70, opacity=0.55).encode(
        y=alt.Y("decode_tok_s:Q", title="decode tok/s", scale=alt.Scale(zero=False)),
        tooltip=tips,
    )
    med = base.mark_line(point=True, strokeWidth=2).encode(
        y=alt.Y("median(decode_tok_s):Q", title="decode tok/s", scale=alt.Scale(zero=False)),
    )
    return (pts + med).properties(height=320)


def roofline_chart(d: pd.DataFrame):
    return alt.Chart(d).mark_bar().encode(
        x=alt.X("prompt_tokens:O", title="prompt tokens"),
        y=alt.Y("median(pct_of_roofline):Q", title="% of calibrated roofline",
                scale=alt.Scale(domain=[0, 100])),
        color=alt.Color("config:N", title=None),
        xOffset="config:N",
        tooltip=["config", "prompt_tokens", "median(pct_of_roofline)", "median(effective_gbs)"],
    ).properties(height=320)


def itl_chart(d: pd.DataFrame):
    m = d.melt(id_vars=["prompt_tokens", "config"],
               value_vars=[c for c in ("itl_p50_ms", "itl_p95_ms", "itl_p99_ms") if c in d.columns],
               var_name="pct", value_name="ms")
    return alt.Chart(m).mark_bar().encode(
        x=alt.X("prompt_tokens:O", title="prompt tokens"),
        y=alt.Y("median(ms):Q", title="inter-token latency (ms)"),
        color=alt.Color("pct:N", title=None),
        xOffset="pct:N",
        row=alt.Row("config:N", title=None) if d["config"].nunique() > 1 else alt.Undefined,
    ).properties(height=220)


def ttft_chart(d: pd.DataFrame):
    m = d.melt(id_vars=["prompt_tokens", "config"],
               value_vars=[c for c in ("ttft_ms_excl_tokenize", "ttft_ms_incl_tokenize") if c in d.columns],
               var_name="which", value_name="ms")
    return alt.Chart(m).mark_bar().encode(
        x=alt.X("prompt_tokens:O", title="prompt tokens"),
        y=alt.Y("median(ms):Q", title="TTFT (ms)"),
        color=alt.Color("which:N", title=None),
        xOffset="which:N",
    ).properties(height=280)


def drift_chart(d: pd.DataFrame):
    return alt.Chart(d).mark_line(point=True).encode(
        x=alt.X("run_index:O", title="run index (thermal drift check)"),
        y=alt.Y("decode_tok_s:Q", title="decode tok/s", scale=alt.Scale(zero=False)),
        color=alt.Color("prompt_tokens:N", title="prompt tokens"),
        strokeDash="config:N",
    ).properties(height=280)


facet(decode_chart, "Decode throughput vs prompt length")
if "pct_of_roofline" in f.columns and f["pct_of_roofline"].notna().any():
    facet(roofline_chart, "% of calibrated roofline")
else:
    st.info("No %-of-roofline data: re-run bench.py with --achievable-gbs from bwprobe.py.")
facet(itl_chart, "Inter-token latency percentiles")
facet(ttft_chart, "TTFT — with and without tokenization")
facet(drift_chart, "Per-run drift within a series (thermal check)")

# ---------------------------------------------------------------- table
st.subheader("Runs")
prefer = ["timestamp_utc", "tag", "chip", "model", "quant_bits", "quant_group_size",
          "engine", "sync_mode", "prompt_tokens", "gen_tokens", "run_index",
          "ttft_ms_excl_tokenize", "prefill_tok_s", "decode_tok_s",
          "ms_per_token_median", "itl_p50_ms", "itl_p95_ms",
          "effective_gbs", "achievable_gbs", "pct_of_roofline",
          "bytes_per_token", "kv_read_bytes_per_token_mean",
          "peak_memory_bytes", "iogpu_wired_limit_mb",
          "energy_joules", "mlx_version", "mlx_lm_version", "output_sha256"]
cols_present = [c for c in prefer if c in f.columns]
st.dataframe(f[cols_present].sort_values("timestamp_utc", ascending=False),
             use_container_width=True, height=420)

if not summaries.empty:
    st.subheader("Series summaries")
    st.dataframe(summaries, use_container_width=True, height=240)

st.caption(f"source: {path.resolve()} — schema_version {SCHEMA_VERSION}")
