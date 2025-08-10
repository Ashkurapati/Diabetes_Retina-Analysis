# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from pathlib import Path
from api import get_health, get_labels, get_correlation, get_bivariate, get_feature_importance

st.set_page_config(page_title="Retina Labels EDA", layout="wide")

# Config (tweak names here if yours differ) 
DEFAULT_CSV = "retina_labels.csv"
TARGET = "DR_ICDR"
AGE_COL = "patient_age"
DIAB_TIME_COL = "diabetes_time_y"  # change if your column name differs

# Data loader (cached) 
@st.cache_data
def load_df_from_path(path: str):
    df = pd.read_csv(path)
    return df

@st.cache_data
def load_df_from_file(file):
    return pd.read_csv(file)

# Load data: CSV (local) or API (backend) 
st.sidebar.header("Data")
data_source = st.sidebar.radio("Data source", ["CSV (local)", "API (backend)"], index=0)

df = None

if data_source == "API (backend)":
    # Optional quick health check
    try:
        from api import get_health, get_labels  # make sure you created api.py
        if not get_health().ok:
            st.error("Backend not healthy. Start FastAPI on port 8000.")
            st.stop()
    except Exception as e:
        st.error(f"Could not reach backend: {e}")
        st.stop()

    with st.spinner("Fetching data from backend..."):
        try:
            df = get_labels()   # pulls /labels pages and returns a DataFrame
        except Exception as e:
            st.error(f"API error: {e}")
            st.stop()

else:  # CSV (local)
    from pathlib import Path
    default_path = Path(DEFAULT_CSV)

    uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
    if uploaded is not None:
        df = load_df_from_file(uploaded)
    elif default_path.exists():
        df = load_df_from_path(str(default_path))

# Guard: must have data
if df is None or df.empty:
    st.info(f"No data yet. Choose **API (backend)** or provide **{DEFAULT_CSV}** / upload a CSV.")
    st.stop()

st.success("Data loaded ")
with st.expander("Preview data", expanded=False):
    st.dataframe(df.head())


#  Main nav 
st.title("Diabetes / Retina Labels Analysis")

mode = st.sidebar.radio(
    "Choose section",
    ["Bivariate analysis", "Correlation matrix", "Feature importance"],
)

# Helpers 
def ensure_numeric(df, cols):
    return [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]

def draw_corr(df, method="pearson"):
    num_cols = ensure_numeric(df, df.columns)
    if len(num_cols) < 2:
        st.warning("Need at least 2 numeric columns for a correlation matrix.")
        return
    corr = df[num_cols].corr(method=method).fillna(0)

    fig = ff.create_annotated_heatmap(
        z=np.round(corr.values, 2),
        x=list(corr.columns),
        y=list(corr.index),
        showscale=True,
        colorscale="Viridis",
        hoverinfo="z",
    )
    fig.update_layout(height=700, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig, use_container_width=True)

def draw_age_bivariate():
    if AGE_COL not in df.columns:
        st.error(f"Column '{AGE_COL}' not found.")
        return
    st.subheader("Distribution of Patient Age")
    st.plotly_chart(px.histogram(df, x=AGE_COL, nbins=30), use_container_width=True)

    if TARGET in df.columns:
        st.subheader(f"Patient Age vs {TARGET}")
        st.plotly_chart(px.violin(df, x=TARGET, y=AGE_COL, box=True, points="all"), use_container_width=True)

def draw_diab_time_bivariate():
    if DIAB_TIME_COL not in df.columns:
        st.error(f"Column '{DIAB_TIME_COL}' not found.")
        return
    st.subheader("Distribution of Diabetes Duration")
    st.plotly_chart(px.histogram(df, x=DIAB_TIME_COL, nbins=30), use_container_width=True)

    if TARGET in df.columns:
        st.subheader(f"Diabetes Duration vs {TARGET}")
        st.plotly_chart(px.box(df, x=TARGET, y=DIAB_TIME_COL, points="all"), use_container_width=True)

def compute_feature_importance(df, target):
    if target not in df.columns:
        st.error(f"Target '{target}' not in data.")
        return

    X = df.drop(columns=[target])
    y = df[target]

    # Basic cleaning: drop rows with NA in y; keep aligned X
    valid = y.notna()
    X, y = X.loc[valid], y.loc[valid]

    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in numeric_cols]

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    model = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    pipe = Pipeline([("prep", pre), ("rf", model)])

    strat = y if y.nunique() < 20 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=strat
    )
    pipe.fit(X_train, y_train)
    acc = pipe.score(X_test, y_test)

    rf = pipe.named_steps["rf"]
    # If there are no categorical features, skip grab
    cat_feature_names = []
    if len(cat_cols) > 0:
        ohe = pipe.named_steps["prep"].named_transformers_["cat"]
        cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
    feature_names = numeric_cols + cat_feature_names

    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    return acc, importances

# Views 
if mode == "Correlation matrix":
    st.header("Correlation matrix")
    method = st.selectbox("Method", ["pearson", "spearman", "kendall"], index=0)

    if data_source == "API (backend)":
        # pick some numeric columns to send
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        default_cols = ",".join(numeric_cols[:6]) if numeric_cols else ""
        cols_csv = st.text_input("Columns to correlate (comma-separated)", default_cols)

        if st.button("Compute correlation (API)"):
            with st.spinner("Computing via backend..."):
                r = get_correlation(cols_csv)
            if not r.ok:
                st.error(f"Error {r.status_code}: {r.text}")
            else:
                matrix = r.json().get("matrix", {})
                if not matrix:
                    st.info("Empty result. Check column names.")
                else:
                    st.dataframe(pd.DataFrame(matrix))
    else:
        # your local CSV version
        draw_corr(df, method)

elif mode == "Feature importance":
    st.header("Feature importance")
    st.caption("Random Forest (works with numeric + categorical; auto one-hot encodes categories).")

    targets = [TARGET] + [c for c in df.columns if c != TARGET] if TARGET in df.columns else list(df.columns)
    default_target_index = 0 if TARGET in df.columns else 0
    target_col = st.selectbox("Target column", targets, index=default_target_index)

    if data_source == "API (backend)":
        if st.button("Compute (API)"):
            with st.spinner("Training on backend..."):
                r = get_feature_importance(target_col)
            if not r.ok:
                st.error(f"Error {r.status_code}: {r.text}")
            else:
                imps = r.json().get("importances", [])
                if not imps:
                    st.info("No importances returned. Target may have <2 classes.")
                else:
                    df_imp = pd.DataFrame(imps)  # columns: feature, importance
                    st.dataframe(df_imp)
                    # SAFE slider bounds
                    k_max = int(max(1, min(30, len(df_imp))))
                    k_default = int(min(15, k_max))
                    top_k = st.slider("Show top k features", 1, k_max, k_default, key="topk_api")
                    fig = px.bar(df_imp.head(top_k).iloc[::-1], x="importance", y="feature", orientation="h")
                    fig.update_layout(yaxis_title="Feature", xaxis_title="Importance", margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig, use_container_width=True)

    else:
        if st.button("Compute"):
            with st.spinner("Fitting model and computing importances..."):
                res = compute_feature_importance(df, target_col)
            if res:
                acc, importances = res
                st.success(f"Hold-out accuracy: {acc:.3f}")

                # SAFE slider bounds
                k_max = int(max(1, min(30, len(importances))))
                k_default = int(min(15, k_max))
                top_k = st.slider("Show top k features", 1, k_max, k_default, key="topk_local")

                fig = px.bar(importances.head(top_k).iloc[::-1])
                fig.update_layout(yaxis_title="Feature", xaxis_title="Importance", margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(importances.rename("importance"))

else:
    #Bivariate analysis (robust, works with CSV or API df) 
    st.header("Bivariate analysis")

    numeric_cols = sorted([c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])])
    all_cols = sorted(df.columns)

    if not numeric_cols:
        st.warning("No numeric columns found in the data.")
    else:
        # Sensible defaults
        default_x = AGE_COL if AGE_COL in numeric_cols else numeric_cols[0]
        default_y = TARGET if TARGET in all_cols else all_cols[0]

        x_col = st.selectbox(
            "Numeric X column",
            numeric_cols,
            index=numeric_cols.index(default_x) if default_x in numeric_cols else 0,
        )
        y_col = st.selectbox(
            "Group/Target (categorical or numeric)",
            all_cols,
            index=all_cols.index(default_y) if default_y in all_cols else 0,
        )

        chart_type = st.radio("Chart type", ["Box", "Violin"], index=0, horizontal=True)
        force_numeric = st.checkbox(f"Force {x_col} to numeric (coerce)", value=True)

        # Build plotting frame safely
        df_plot = df[[x_col, y_col]].copy()

        if force_numeric:
            df_plot[x_col] = pd.to_numeric(df_plot[x_col], errors="coerce")

        # If Y is numeric with many uniques, bin it so box/violin makes sense
        if pd.api.types.is_numeric_dtype(df_plot[y_col]) and df_plot[y_col].nunique(dropna=True) > 20:
            bin_choice = st.radio(
                "Y is numeric — group by:",
                ["Raw numeric", "Quartiles (q=4)", "Deciles (q=10)"],
                index=1,
                horizontal=True,
            )
            if bin_choice != "Raw numeric":
                q = 4 if "Quartiles" in bin_choice else 10
                df_plot[y_col] = pd.qcut(df_plot[y_col], q=q, duplicates="drop")

        # Drop missing rows after coercion/binning
        df_plot = df_plot.dropna(subset=[x_col, y_col])

        if df_plot.empty:
            st.error("No rows left to plot after cleaning. Try a different X/Y or uncheck 'Force numeric'.")
        else:
            # Distribution of X
            st.subheader(f"Distribution of {x_col}")
            st.plotly_chart(px.histogram(df_plot, x=x_col, nbins=30), use_container_width=True)

            # X vs Y (box or violin)
            st.subheader(f"{x_col} vs {y_col}")
            if chart_type == "Box":
                st.plotly_chart(px.box(df_plot, x=y_col, y=x_col, points="all"), use_container_width=True)
            else:
                st.plotly_chart(px.violin(df_plot, x=y_col, y=x_col, box=True, points="all"), use_container_width=True)
