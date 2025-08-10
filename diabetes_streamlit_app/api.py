# api.py
import os
import math
import requests
import pandas as pd
import streamlit as st  # NEW

# Prefer Streamlit secret, then env var, then localhost (for local dev)
API = (
    st.secrets.get("BACKEND_URL")
    or os.getenv("API_BASE")
    or "http://127.0.0.1:8000"
).rstrip("/")  # keep it clean for f"{API}/..."

def get_health():
    return requests.get(f"{API}/health", timeout=20)

def get_labels(limit=1000, max_pages=50):
    """Paginate /labels to get a DataFrame."""
    rows = []
    for page in range(max_pages):
        offset = page * limit
        r = requests.get(f"{API}/labels/", params={"limit": limit, "offset": offset}, timeout=60)
        if not r.ok:
            raise RuntimeError(f"/labels error {r.status_code}: {r.text}")
        batch = r.json()
        if not batch:
            break
        rows.extend(batch)
        if len(batch) < limit:
            break
    return pd.DataFrame(rows)

def get_correlation(cols_csv: str):
    return requests.get(f"{API}/stats/correlation", params={"cols": cols_csv}, timeout=120)

def get_bivariate(x: str, y: str):
    return requests.get(f"{API}/stats/bivariate", params={"x": x, "y": y}, timeout=120)

def get_feature_importance(target="DR_ICDR"):
    return requests.get(f"{API}/ml/feature-importance", params={"target": target}, timeout=300)
