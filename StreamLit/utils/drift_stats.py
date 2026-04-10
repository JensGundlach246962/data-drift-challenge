"""
drift_stats.py
──────────────
Statistical drift detection: PSI and Kolmogorov-Smirnov tests.
No external libraries required beyond numpy/scipy.
"""

import numpy as np
import pandas as pd
from scipy import stats


def psi(base: np.ndarray, prod: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index.
    PSI < 0.1   → No drift
    PSI 0.1-0.2 → Moderate drift
    PSI > 0.2   → Significant drift
    """
    base = base[~np.isnan(base)]
    prod = prod[~np.isnan(prod)]
    breakpoints = np.percentile(base, np.linspace(0, 100, bins + 1))
    breakpoints = np.unique(breakpoints)

    base_pct = np.histogram(base, bins=breakpoints)[0] / len(base)
    prod_pct = np.histogram(prod, bins=breakpoints)[0] / len(prod)

    # Avoid log(0)
    base_pct = np.where(base_pct == 0, 1e-6, base_pct)
    prod_pct = np.where(prod_pct == 0, 1e-6, prod_pct)

    return float(np.sum((prod_pct - base_pct) * np.log(prod_pct / base_pct)))


def ks_test(base: np.ndarray, prod: np.ndarray) -> tuple[float, float]:
    """Returns (statistic, p_value). p < 0.05 → significant drift."""
    stat, pval = stats.ks_2samp(base, prod)
    return float(stat), float(pval)


def psi_label(score: float) -> str:
    if score < 0.1:
        return "stable"
    elif score < 0.2:
        return "moderate"
    return "high"


def compute_drift_report(baseline: pd.DataFrame, production: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Compute PSI and KS test for each feature.
    Returns a DataFrame sorted by PSI descending.
    """
    rows = []
    for col in features:
        if col not in baseline.columns or col not in production.columns:
            continue
        b = baseline[col].dropna().values
        p = production[col].dropna().values
        psi_score = psi(b, p)
        ks_stat, ks_pval = ks_test(b, p)
        rows.append({
            "feature": col,
            "psi": round(psi_score, 4),
            "psi_label": psi_label(psi_score),
            "ks_stat": round(ks_stat, 4),
            "ks_pval": round(ks_pval, 6),
            "drifted": psi_score > 0.1 or ks_pval < 0.05,
        })
    return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
