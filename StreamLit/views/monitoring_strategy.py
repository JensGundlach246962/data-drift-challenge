"""
pages/monitoring_strategy.py  –  Monitoring & Retraining Strategy Wiki
Static reference page explaining thresholds, triggers, and policy.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import plotly.graph_objects as go
import pandas as pd


def render():
    st.markdown('<p class="section-title">Monitoring & Retraining Strategy</p>', unsafe_allow_html=True)

    # ── Threshold reference ───────────────────────────────────────────────────
    st.markdown("### Drift Detection Thresholds")
    st.markdown(
        "We apply two complementary tests. Both must agree before escalating an alert level."
    )

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("#### PSI (Population Stability Index)")
        psi_data = {
            "PSI Range": ["< 0.10", "0.10 – 0.20", "> 0.20"],
            "Status":    ["✅ Stable", "⚠️ Moderate", "🚨 High Drift"],
            "Action":    ["Monitor normally", "Log & investigate", "Alert + consider retraining"],
        }
        st.dataframe(pd.DataFrame(psi_data), hide_index=True, use_container_width=True)

    with col2:
        st.markdown("#### KS Test (Kolmogorov-Smirnov)")
        ks_data = {
            "p-value":  ["≥ 0.05", "0.01 – 0.05", "< 0.01"],
            "Status":   ["✅ No drift", "⚠️ Possible drift", "🚨 Significant drift"],
            "Action":   ["Monitor normally", "Log & monitor closely", "Alert + investigate"],
        }
        st.dataframe(pd.DataFrame(ks_data), hide_index=True, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Performance thresholds ────────────────────────────────────────────────
    st.markdown("### Model Performance Thresholds")

    perf_data = {
        "Metric":    ["F1 Score",  "AUC-ROC",  "Precision", "Recall"],
        "Baseline":  ["0.863",     "0.976",    "0.882",     "0.851"],
        "⚠️ Warn":   ["< 0.83",   "< 0.94",   "< 0.84",    "< 0.81"],
        "🚨 Alert":  ["< 0.80",   "< 0.91",   "< 0.80",    "< 0.78"],
        "Action":    [
            "Schedule retraining",
            "Immediate review",
            "Bias investigation",
            "Missed fraud review",
        ],
    }
    st.dataframe(pd.DataFrame(perf_data), hide_index=True, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Decision flowchart (text-based) ──────────────────────────────────────
    st.markdown("### Automated Monitoring Decision Flow")

    flow_html = """
    <div style="background:#12121f;border:1px solid #1e1e35;border-radius:8px;padding:1.5rem;
                font-family:'Space Mono',monospace;font-size:0.78rem;line-height:2.2;color:#8080c0;">
        <div style="color:#7c6af7;font-weight:700;">① Daily drift check (scheduled job)</div>
        <div style="padding-left:1.5rem;">→ Compute PSI &amp; KS for all features</div>
        <div style="padding-left:1.5rem;">→ Compute F1 / AUC-ROC on labeled sample</div>
        <br>
        <div style="color:#7c6af7;font-weight:700;">② Evaluate thresholds</div>
        <div style="padding-left:1.5rem;">→ Any PSI &gt; 0.20  → 🚨 <span style="color:#e74c3c">HIGH DRIFT alert</span></div>
        <div style="padding-left:1.5rem;">→ Any PSI 0.10-0.20 → ⚠️ <span style="color:#f39c12">MODERATE alert</span></div>
        <div style="padding-left:1.5rem;">→ AUC-ROC &lt; 0.90   → 🚨 <span style="color:#e74c3c">PERFORMANCE alert</span></div>
        <br>
        <div style="color:#7c6af7;font-weight:700;">③ Response actions</div>
        <div style="padding-left:1.5rem;color:#2ecc71;">STABLE    → Log metrics, continue</div>
        <div style="padding-left:1.5rem;color:#f39c12;">MODERATE  → Notify team, schedule review in 3 days</div>
        <div style="padding-left:1.5rem;color:#e74c3c;">HIGH      → Trigger retraining pipeline immediately</div>
        <br>
        <div style="color:#7c6af7;font-weight:700;">④ Retraining policy</div>
        <div style="padding-left:1.5rem;">→ Retrain on rolling 90-day window of labeled data</div>
        <div style="padding-left:1.5rem;">→ Validate new model beats current on holdout set</div>
        <div style="padding-left:1.5rem;">→ Shadow deploy 7 days before full switch</div>
    </div>
    """
    st.markdown(flow_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Automated drift check code snippet ───────────────────────────────────
    st.markdown("### Automated Drift Check (Code)")
    st.markdown("Copy this into your monitoring pipeline:")

    st.code("""
import pandas as pd
import numpy as np
from scipy import stats

PSI_WARN  = 0.10
PSI_ALERT = 0.20
KS_ALPHA  = 0.05

def psi(base, prod, bins=10):
    bp = np.unique(np.percentile(base, np.linspace(0,100,bins+1)))
    b  = np.where(np.histogram(base,bp)[0]/len(base)==0, 1e-6, np.histogram(base,bp)[0]/len(base))
    p  = np.where(np.histogram(prod,bp)[0]/len(prod)==0, 1e-6, np.histogram(prod,bp)[0]/len(prod))
    return float(np.sum((p - b) * np.log(p / b)))

def run_drift_check(baseline_df, production_df, features):
    results = []
    for col in features:
        b, p   = baseline_df[col].dropna().values, production_df[col].dropna().values
        psi_v  = psi(b, p)
        ks_p   = stats.ks_2samp(b, p).pvalue
        status = "HIGH"     if psi_v > PSI_ALERT else (
                 "MODERATE" if psi_v > PSI_WARN  else "STABLE")
        results.append({"feature": col, "psi": psi_v, "ks_pval": ks_p, "status": status})

    df = pd.DataFrame(results).sort_values("psi", ascending=False)
    n_high = (df["status"] == "HIGH").sum()

    if n_high > 0:
        print(f"🚨 ALERT: {n_high} features with HIGH drift – trigger retraining")
    elif (df["status"] == "MODERATE").sum() > 0:
        print("⚠️  WARNING: moderate drift detected – monitor closely")
    else:
        print("✅ All features stable")
    return df

# Example usage
# drift_report = run_drift_check(baseline, production, feature_cols)
""", language="python")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Retraining cadence ────────────────────────────────────────────────────
    st.markdown("### Retraining Cadence Summary")
    cadence_data = {
        "Trigger":         ["High drift (PSI > 0.20)", "Performance drop (AUC < 0.90)", "Scheduled", "Manual"],
        "Frequency":       ["Immediate", "Immediate", "Monthly", "On demand"],
        "Data window":     ["Last 90 days", "Last 90 days", "Last 180 days", "Full history"],
        "Validation gate": ["AUC > 0.95", "AUC > 0.95", "AUC > 0.97", "Team review"],
    }
    st.dataframe(pd.DataFrame(cadence_data), hide_index=True, use_container_width=True)
