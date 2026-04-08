# Data Drift Challenge

Detecting and monitoring data drift in a credit card fraud detection model.

## Team
- [Data drift chairmen]

## Setup

```
conda create -n data_drift python=3.11 -y
conda activate data_drift
pip install -r requirements.txt
```

## Dataset

Download from Kaggle: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Place `creditcard.csv` in the `01_data/` folder (not tracked by git).

## Project Structure

```
├── 01_data/               # Dataset (gitignored)
├── 02_src/                # Notebooks and source code
├── 03_dashboard/          # Streamlit app
├── other/                 # Misc files
├── requirements.txt
├── .gitignore
└── README.md
```

## Run Dashboard

```
conda activate data_drift
streamlit run 03_dashboard/app.py
```

## Key Findings

[To be completed]

## Monitoring Strategy

[To be completed]
