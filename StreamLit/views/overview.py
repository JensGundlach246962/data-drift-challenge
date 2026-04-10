import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from utils.data_loader import load_baseline, load_drift, is_using_mock_data, DRIFT_FILES
from utils.model_stub import get_baseline_metrics, get_drift_metrics, get_all_drift_metrics
from utils.drift_stats import compute_drift_report


FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]

_BADGE = {
    "stable":   '<span class="badge badge-ok">STABLE</span>',
    "moderate": '<span class="badge badge-warn">MODERATE</span>',
    "high":     '<span class="badge badge-crit">HIGH DRIFT</span>',
}


def _overall_status(n_drifted: int, n_total: int) -> str:
    ratio = n_drifted / max(n_total, 1)
    if ratio < 0.15:
        return "stable"
    elif ratio < 0.40:
        return "moderate"
    return "high"


def render(selected_drift: str):
    st.markdown('<p class="section-title">System Overview</p>', unsafe_allow_html=True)

    # ── Mock data warning ────────────────────────────────────────────────────
    if is_using_mock_data():
        st.info(
            "📂  **No real CSV files detected** – running on synthetic mock data.  \n"
            "Drop `creditcard.csv` and `drift_1…5.csv` into the `data/` folder to use real data.",
            icon=None,
        )

    # ── Load data ────────────────────────────────────────────────────────────
    baseline  = load_baseline()
    drift_df  = load_drift(selected_drift)
    b_metrics = get_baseline_metrics()
    d_metrics = get_drift_metrics(selected_drift, drift_df)

    drift_report = compute_drift_report(baseline, drift_df, FEATURE_COLS)
    n_drifted = int(drift_report["drifted"].sum())
    status    = _overall_status(n_drifted, len(drift_report))

    # ── Status banner ────────────────────────────────────────────────────────
    col_title, col_badge = st.columns([5, 1])
    with col_title:
        st.markdown(f"### Comparing baseline vs `{selected_drift}`")
    with col_badge:
        st.markdown(f"<br>{_BADGE[status]}", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── KPI row ──────────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    def _delta(key):
        diff = d_metrics[key] - b_metrics[key]
        return f"{diff:+.3f}"

    c1.metric("F1 Score",    f"{d_metrics['f1']:.3f}",       _delta("f1"))
    c2.metric("AUC-ROC",     f"{d_metrics['auc_roc']:.3f}",  _delta("auc_roc"))
    c3.metric("Precision",   f"{d_metrics['precision']:.3f}",_delta("precision"))
    c4.metric("Recall",      f"{d_metrics['recall']:.3f}",   _delta("recall"))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Two columns: drift table + AUC trend ────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown('<p class="section-title">Top Drifted Features</p>', unsafe_allow_html=True)
        top = drift_report.head(10)[["feature", "psi", "psi_label", "ks_pval"]].copy()
        top["psi_label"] = top["psi_label"].map(lambda x: _BADGE.get(x, x))
        top["ks_pval"]   = top["ks_pval"].apply(lambda p: f"{p:.2e}")
        st.write(
            top.rename(columns={"feature":"Feature","psi":"PSI","psi_label":"Status","ks_pval":"KS p-val"})
               .to_html(escape=False, index=False),
            unsafe_allow_html=True,
        )
        st.markdown(f"<p style='font-size:0.75rem;color:#5050a0;margin-top:0.5rem;'>"
                    f"{n_drifted} / {len(drift_report)} features flagged as drifted</p>",
                    unsafe_allow_html=True)

    with right:
        st.markdown('<p class="section-title">AUC-ROC Over Batches</p>', unsafe_allow_html=True)
        all_m = get_all_drift_metrics()
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=all_m["batch"], y=all_m["auc_roc"],
            mode="lines+markers",
            line=dict(color="#7c6af7", width=2.5),
            marker=dict(size=8, color="#7c6af7", line=dict(color="#fff", width=1.5)),
            name="AUC-ROC",
        ))
        fig.add_hline(y=0.91, line_dash="dot", line_color="#e74c3c",
                      annotation_text="Alert threshold 0.91", annotation_position="bottom right",
                      annotation_font_color="#e74c3c", annotation_font_size=10)
        fig.update_layout(
            paper_bgcolor="#12121f", plot_bgcolor="#12121f",
            font=dict(color="#8080c0", family="Space Mono"),
            xaxis=dict(gridcolor="#1e1e35", tickfont=dict(size=10)),
            yaxis=dict(gridcolor="#1e1e35", range=[0.75, 1.0]),
            margin=dict(l=10, r=10, t=10, b=10),
            height=320,
        )
        st.plotly_chart(fig, use_container_width=True)
