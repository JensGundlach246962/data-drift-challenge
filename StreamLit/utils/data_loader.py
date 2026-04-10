"""
data_loader.py
──────────────
Loads baseline (creditcard.csv) and drift CSVs from ./data/.
Falls back to synthetic mock data when files are not yet present,
so the dashboard runs end-to-end before the model work is done.
"""

import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")

# PCA feature names used throughout the project
FEATURE_COLS = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
LABEL_COL = "Class"

DRIFT_FILES = ["drift_1.csv", "drift_2.csv", "drift_3.csv", "drift_4.csv", "drift_5.csv"]


# ── helpers ─────────────────────────────────────────────────────────────────

def _mock_baseline(n: int = 5000) -> pd.DataFrame:
    """Lightweight synthetic stand-in for creditcard.csv."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame(rng.standard_normal((n, 28)), columns=[f"V{i}" for i in range(1, 29)])
    df["Amount"] = rng.exponential(88, n)
    df["Time"] = np.linspace(0, 172800, n)
    df[LABEL_COL] = (rng.random(n) < 0.00172).astype(int)
    return df


def _mock_drift(name: str, baseline: pd.DataFrame) -> pd.DataFrame:
    """Add synthetic drift proportional to the drift file index."""
    rng = np.random.default_rng(hash(name) % (2**31))
    idx = DRIFT_FILES.index(name) + 1          # 1-5 severity
    n = len(baseline)
    df = baseline.copy()

    # Shift top features
    for col in [f"V{i}" for i in range(1, idx * 4 + 1)]:
        df[col] = df[col] + rng.normal(0.3 * idx, 0.1, n)

    # Scale Amount
    df["Amount"] = df["Amount"] * (1 + 0.15 * idx)
    # Slightly raise fraud rate
    df[LABEL_COL] = (rng.random(n) < 0.00172 * (1 + 0.5 * idx)).astype(int)
    return df


# ── public API ───────────────────────────────────────────────────────────────

def load_baseline() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "creditcard.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return _mock_baseline()


def load_drift(name: str) -> pd.DataFrame:
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path):
        return pd.read_csv(path)
    baseline = load_baseline()
    return _mock_drift(name, baseline)


def is_using_mock_data() -> bool:
    return not os.path.exists(os.path.join(DATA_DIR, "creditcard.csv"))
