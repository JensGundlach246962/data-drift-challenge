# Data Drift Challenge

Detecting and monitoring data drift in a credit card fraud detection model.

## Team

Data Drift Chairmen

## Setup

```
conda create -n data_drift python=3.11 -y
conda activate data_drift
pip install -r requirements.txt
```

## Dataset

The dataset contains 284,807 transactions with 30 features (V1–V28 are PCA-transformed, plus Amount and Time) and a binary Class label. It is tracked in this repository via Git LFS — no manual download required.

If you do not have Git LFS installed, install it first:

```
git lfs install
git pull
```

This will pull the actual CSV files instead of the LFS pointer files.

## Project Structure

```
01_data/               Dataset and drift CSVs (tracked via Git LFS)
02_src/                Notebooks and source code
03_dashboard/          Streamlit app
other/                 Misc files
requirements.txt
.gitignore
README.md
```

The following files are available in `01_data/` after cloning:

```
01_data/
  creditcard.csv
  drift_1.csv
  drift_2.csv
  drift_3.csv
  drift_4.csv
  drift_5.csv
  baseline_model.pkl
  scaler.pkl
```

## Run Dashboard

```
conda activate data_drift
streamlit run 03_dashboard/app.py
```

The dashboard has four pages:

| Page | Description |
|---|---|
| Overview | KPI metrics (F1, AUC-ROC, Precision, Recall) with deltas vs baseline, top drifted features, AUC trend |
| Feature Drift | Distribution plots comparing baseline vs production, full PSI and KS statistics |
| Model Performance | Metric trends across all drift batches, per-batch breakdown, feature-degradation analysis |
| Monitoring Strategy | Drift thresholds, decision flow, retraining policy, monitoring code |

## Baseline Model Performance

Trained on `creditcard.csv` using a scaled feature set (StandardScaler + classifier):

| Metric | Score |
|---|---|
| F1 Score | 0.863 |
| AUC-ROC | 0.976 |
| Precision | 0.882 |
| Recall | 0.851 |

## Key Findings

- Features V28, V8, V20, V27, V21 and V23 showed the highest PSI scores across all drift batches, consistently flagged as high drift
- Model performance degrades gradually through drift_1 and drift_2, then more steeply from drift_3 onwards
- AUC-ROC crosses the alert threshold of 0.91 between drift_4 and drift_5
- Precision remains relatively robust under drift while recall drops faster, meaning the model misses more fraud cases before it starts making false positives

## Monitoring Strategy

Drift is detected using two complementary methods: Population Stability Index (PSI) and the Kolmogorov-Smirnov test. Both are computed per feature on each new production batch.

Thresholds:

| Metric | Warn | Alert / Retrain |
|---|---|---|
| PSI | 0.10 – 0.20 | > 0.20 |
| KS p-value | 0.01 – 0.05 | < 0.01 |
| F1 Score | < 0.83 | < 0.80 |
| AUC-ROC | < 0.94 | < 0.91 |
| Precision | < 0.84 | < 0.80 |
| Recall | < 0.81 | < 0.78 |

Retraining is triggered immediately when any feature exceeds PSI 0.20 or when AUC-ROC drops below 0.91. Scheduled retraining runs monthly regardless of drift status, using a rolling 90-day window of labeled data.
