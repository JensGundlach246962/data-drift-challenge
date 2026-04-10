import streamlit as st

st.set_page_config(
    page_title="Drift Sentinel",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;600&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  /* Dark sidebar */
  section[data-testid="stSidebar"] {
      background: #0d0d14;
      border-right: 1px solid #1e1e2e;
  }
  section[data-testid="stSidebar"] * { color: #c9c9e0 !important; }
  section[data-testid="stSidebar"] .stSelectbox label,
  section[data-testid="stSidebar"] .stRadio label { color: #7070a0 !important; font-size: 0.75rem; letter-spacing: 0.08em; text-transform: uppercase; }

  /* Main background */
  .main { background: #080811; }
  .block-container { padding-top: 2rem; }

  /* Metric cards */
  div[data-testid="metric-container"] {
      background: #12121f;
      border: 1px solid #1e1e35;
      border-radius: 8px;
      padding: 1rem 1.2rem;
  }
  div[data-testid="metric-container"] label { color: #6060a0 !important; font-size: 0.7rem; letter-spacing: 0.1em; text-transform: uppercase; }
  div[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #e0e0ff; font-family: 'Space Mono', monospace; font-size: 1.6rem; }
  div[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.8rem; }

  /* Status badge */
  .badge { display:inline-block; padding:3px 10px; border-radius:20px; font-size:0.72rem; font-family:'Space Mono',monospace; font-weight:700; letter-spacing:0.05em; }
  .badge-ok   { background:#0d2b1f; color:#2ecc71; border:1px solid #2ecc71; }
  .badge-warn { background:#2b220d; color:#f39c12; border:1px solid #f39c12; }
  .badge-crit { background:#2b0d0d; color:#e74c3c; border:1px solid #e74c3c; }

  /* Section headers */
  .section-title {
      font-family: 'Space Mono', monospace;
      color: #5050c0;
      font-size: 0.72rem;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      border-bottom: 1px solid #1e1e35;
      padding-bottom: 0.5rem;
      margin-bottom: 1.2rem;
  }

  /* Info box */
  .placeholder-box {
      background: #12121f;
      border: 1px dashed #2e2e50;
      border-radius: 8px;
      padding: 2rem;
      text-align: center;
      color: #404070;
      font-family: 'Space Mono', monospace;
      font-size: 0.8rem;
  }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Drift Sentinel")
    st.markdown("<hr style='border-color:#1e1e2e;margin:0.5rem 0 1rem'>", unsafe_allow_html=True)

    page = st.radio(
        "NAVIGATE",
        ["Overview", "Feature Drift", "Model Performance", "Monitoring Strategy"],
        label_visibility="visible",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<p style='font-size:0.7rem;color:#333355;'>DATA WINDOW</p>", unsafe_allow_html=True)
    selected_drift = st.selectbox(
        "Production batch",
        ["drift_1.csv", "drift_2.csv", "drift_3.csv", "drift_4.csv", "drift_5.csv"],
        label_visibility="collapsed",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        "<p style='font-size:0.65rem;color:#222240;font-family:Space Mono,monospace;'>"
        "Baseline: creditcard.csv<br>284,807 transactions</p>",
        unsafe_allow_html=True,
    )

# ── Route pages ─────────────────────────────────────────────────────────────
import importlib.util, os

def load_page(filename):
    path = os.path.join(os.path.dirname(__file__), "views", filename)
    spec = importlib.util.spec_from_file_location(filename, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

if page == "Overview":
    load_page("overview.py").render(selected_drift)
elif page == "Feature Drift":
    load_page("feature_drift.py").render(selected_drift)
elif page == "Model Performance":
    load_page("model_performance.py").render(selected_drift)
elif page == "Monitoring Strategy":
    load_page("monitoring_strategy.py").render()
