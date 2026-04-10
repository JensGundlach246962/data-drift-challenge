"""
pages/model_performance.py  –  Model metrics across drift batches
Shows trend lines, per-batch breakdown, and feature-degradation link.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd

from utils.model_stub import get_all_drift_metrics, get_baseline_metrics, get_drift_metrics
from utils.data_loader import load_baseline, load_drift, is_using_mock_data
from utils.drift_stats import compute_drift_report

FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount"]

_METRICS = ["f1", "auc_roc", "precision", "recall"]
_LABELS  = {"f1":"F1 Score","auc_roc":"AUC-ROC","precision":"Precision","recall":"Recall"}
_COLORS  = {"f1":"#7c6af7","auc_roc":"#f7796a","precision":"#4ecdc4","recall":"#ffe66d"}

_THRESHOLDS = {"f1": 0.80, "auc_roc": 0.91, "precision": 0.80, "recall": 0.78}


def render(selected_drift: str):
    st.markdown('<p class="section-title">Model Performance Monitoring</p>', unsafe_allow_html=True)

    if is_using_mock_data():
        st.info("🤖  **Model stub active** – metrics are simulated.  \n"
                "Drop `data/model.pkl` (joblib, sklearn-compatible) to use real predictions.")

    all_m = get_all_drift_metrics()

    # ── Metric selector ───────────────────────────────────────────────────────
    selected_metrics = st.multiselect(
        "Metrics to display",
        options=_METRICS,
        default=_METRICS,
        format_func=lambda x: _LABELS[x],
    )

    # ── Trend chart ───────────────────────────────────────────────────────────
    fig = go.Figure()
    for m in selected_metrics:
        fig.add_trace(go.Scatter(
            x=all_m["batch"], y=all_m[m],
            mode="lines+markers",
            name=_LABELS[m],
            line=dict(color=_COLORS[m], width=2.5),
            marker=dict(size=9, color=_COLORS[m], line=dict(color="#fff",width=1.5)),
        ))
        # Threshold line
        fig.add_hline(
            y=_THRESHOLDS[m], line_dash="dot", line_color=_COLORS[m],
            opacity=0.35,
        )

    fig.update_layout(
        paper_bgcolor="#12121f", plot_bgcolor="#12121f",
        font=dict(color="#8080c0", family="Space Mono"),
        xaxis=dict(gridcolor="#1e1e35"),
        yaxis=dict(gridcolor="#1e1e35", range=[0.6, 1.0]),
        legend=dict(orientation="h", y=-0.15),
        margin=dict(l=10, r=10, t=20, b=10),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Dotted lines show alert thresholds. Metrics below threshold trigger retraining recommendation.")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Per-batch breakdown table ─────────────────────────────────────────────
    st.markdown('<p class="section-title">Per-Batch Breakdown</p>', unsafe_allow_html=True)

    base_m = get_baseline_metrics()
    display = all_m.copy()
    for m in _METRICS:
        display[f"{m}_delta"] = display[m] - base_m[m]

    def _fmt_delta(val):
        color = "#2ecc71" if val >= 0 else "#e74c3c"
        return f"<span style='color:{color}'>{val:+.3f}</span>"

    rows_html = ""
    for _, row in display.iterrows():
        cells = f"<td style='padding:6px 12px;font-family:Space Mono;font-size:0.8rem;color:#c0c0ff'>{row['batch']}</td>"
        for m in _METRICS:
            alert = row[m] < _THRESHOLDS[m]
            val_color = "#e74c3c" if alert else "#c0c0ff"
            cells += (
                f"<td style='padding:6px 12px;color:{val_color};font-family:Space Mono;font-size:0.8rem'>"
                f"{row[m]:.3f} {_fmt_delta(row[m+'_delta'])}</td>"
            )
        rows_html += f"<tr style='border-bottom:1px solid #1e1e35'>{cells}</tr>"

    header = "".join(
        f"<th style='padding:6px 12px;color:#5050a0;font-size:0.7rem;letter-spacing:0.08em;text-transform:uppercase;"
        f"font-weight:400;text-align:left'>{_LABELS[m]}</th>"
        for m in _METRICS
    )
    table_html = (
        f"<table style='width:100%;border-collapse:collapse;background:#12121f'>"
        f"<thead><tr><th style='padding:6px 12px;color:#5050a0;font-size:0.7rem;letter-spacing:0.08em;"
        f"text-transform:uppercase;font-weight:400;text-align:left'>Batch</th>{header}</tr></thead>"
        f"<tbody>{rows_html}</tbody></table>"
    )
    st.markdown(table_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature ↔ degradation correlation ────────────────────────────────────
    st.markdown('<p class="section-title">Feature Drift vs Performance Degradation</p>', unsafe_allow_html=True)

    baseline = load_baseline()
    drift_df = load_drift(selected_drift)
    d_metrics = get_drift_metrics(selected_drift, drift_df)
    report   = compute_drift_report(baseline, drift_df, FEATURE_COLS)
    top_drift = report.head(15)

    f1_drop = base_m["f1"] - d_metrics["f1"]

    fig2 = go.Figure(go.Bar(
        x=top_drift["feature"],
        y=top_drift["psi"],
        marker=dict(
            color=top_drift["psi"],
            colorscale=[[0,"#1e1e35"],[0.5,"#7c6af7"],[1,"#e74c3c"]],
            showscale=True,
            colorbar=dict(title="PSI", tickfont=dict(color="#8080c0"), title_font=dict(color="#8080c0")),
        ),
        text=top_drift["psi"].apply(lambda v: f"{v:.3f}"),
        textposition="outside",
        textfont=dict(color="#8080c0", size=9),
    ))
    fig2.add_annotation(
        x=0.5, y=1.08, xref="paper", yref="paper",
        text=f"F1 drop for {selected_drift}: {f1_drop:+.3f}",
        showarrow=False,
        font=dict(color="#f7796a", family="Space Mono", size=11),
    )
    fig2.update_layout(
        paper_bgcolor="#12121f", plot_bgcolor="#12121f",
        font=dict(color="#8080c0", family="Space Mono"),
        xaxis=dict(gridcolor="#1e1e35", tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(gridcolor="#1e1e35", title="PSI"),
        margin=dict(l=10, r=10, t=50, b=10),
        height=340,
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.caption("Features with high PSI are the likely drivers of performance degradation.")
