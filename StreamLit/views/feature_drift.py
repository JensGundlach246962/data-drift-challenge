"""
pages/feature_drift.py  –  Feature distribution comparison
Side-by-side overlaid histograms + full PSI/KS table.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.graph_objects as go
import plotly.figure_factory as ff
import numpy as np

from utils.data_loader import load_baseline, load_drift
from utils.drift_stats import compute_drift_report

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]

_COLOR_BASE = "#7c6af7"
_COLOR_PROD = "#f7796a"


def _dist_plot(baseline_vals, prod_vals, feature: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=baseline_vals, name="Baseline",
        marker_color=_COLOR_BASE, opacity=0.65,
        nbinsx=50, histnorm="probability density",
    ))
    fig.add_trace(go.Histogram(
        x=prod_vals, name="Production",
        marker_color=_COLOR_PROD, opacity=0.65,
        nbinsx=50, histnorm="probability density",
    ))
    fig.update_layout(
        barmode="overlay",
        paper_bgcolor="#12121f", plot_bgcolor="#12121f",
        font=dict(color="#8080c0", family="Space Mono", size=10),
        xaxis=dict(gridcolor="#1e1e35", title=feature),
        yaxis=dict(gridcolor="#1e1e35", title="Density"),
        legend=dict(orientation="h", y=1.1, x=0),
        margin=dict(l=10, r=10, t=30, b=10),
        height=260,
    )
    return fig


def render(selected_drift: str):
    st.markdown('<p class="section-title">Feature Distribution Analysis</p>', unsafe_allow_html=True)

    baseline = load_baseline()
    drift_df = load_drift(selected_drift)

    report = compute_drift_report(baseline, drift_df, FEATURE_COLS)

    # ── Controls ─────────────────────────────────────────────────────────────
    col_filter, col_sort, _ = st.columns([2, 2, 4])
    with col_filter:
        show_only_drifted = st.toggle("Show only drifted features", value=True)
    with col_sort:
        sort_by = st.selectbox("Sort by", ["PSI (high → low)", "Feature name"], label_visibility="visible")

    if show_only_drifted:
        display_report = report[report["drifted"]].copy()
    else:
        display_report = report.copy()

    if sort_by == "Feature name":
        display_report = display_report.sort_values("feature")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Full stats table ──────────────────────────────────────────────────────
    with st.expander("📊 Full drift statistics table", expanded=False):
        def _color_psi(val):
            if val < 0.1:   return "color: #2ecc71"
            elif val < 0.2: return "color: #f39c12"
            return "color: #e74c3c"

        styled = (
            report[["feature","psi","ks_stat","ks_pval","drifted"]]
            .style
            .format({"psi":"{:.4f}","ks_stat":"{:.4f}","ks_pval":"{:.2e}"})
            .map(_color_psi, subset=["psi"])
        )
        st.dataframe(styled, use_container_width=True, height=350)

    # ── Distribution plots grid ──────────────────────────────────────────────
    st.markdown('<p class="section-title">Distribution Plots</p>', unsafe_allow_html=True)

    features_to_show = display_report["feature"].tolist()
    if not features_to_show:
        st.markdown('<div class="placeholder-box">No drifted features found for this batch.</div>',
                    unsafe_allow_html=True)
        return

    # 2-column grid
    for i in range(0, len(features_to_show), 2):
        cols = st.columns(2, gap="medium")
        for j, col in enumerate(cols):
            if i + j >= len(features_to_show):
                break
            feat = features_to_show[i + j]
            row  = report[report["feature"] == feat].iloc[0]

            psi_val = row["psi"]
            psi_col = "#2ecc71" if psi_val < 0.1 else ("#f39c12" if psi_val < 0.2 else "#e74c3c")

            with col:
                st.markdown(
                    f"<p style='font-family:Space Mono;font-size:0.8rem;color:#c0c0ff;'>"
                    f"{feat} &nbsp;<span style='color:{psi_col};font-size:0.7rem;'>PSI={psi_val:.3f}</span></p>",
                    unsafe_allow_html=True,
                )
                fig = _dist_plot(
                    baseline[feat].dropna().values,
                    drift_df[feat].dropna().values,
                    feat,
                )
                st.plotly_chart(fig, use_container_width=True)
