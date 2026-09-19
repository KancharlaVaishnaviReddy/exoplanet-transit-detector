"""
app.py
-------
Streamlit web UI for the Exoplanet Detection project.

Run with:  streamlit run app.py
Then open the browser link it prints (usually http://localhost:8501).

Features:
  - Pick a sample light curve (confirmed transit / eclipsing binary / quiet star)
    or upload your own CSV of a light curve
  - Interactive plot of the light curve (raw + flattened)
  - Model prediction with confidence, shown as a clear visual verdict
  - Extracted feature table (depth, duration, etc.)
  - Model performance dashboard (confusion matrix, ROC curve, feature importance)
  - Batch mode: run the model on the whole dataset and browse results
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import plotly.graph_objects as go

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from features import extract_features, flatten_curve, FEATURE_NAMES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "exoplanet_model.joblib")
METRICS_PATH = os.path.join(BASE_DIR, "models", "model_metrics.json")
DATA_PATH = os.path.join(BASE_DIR, "data", "light_curves.csv")
SAMPLE_PATH = os.path.join(BASE_DIR, "samples", "sample_light_curves.csv")

st.set_page_config(
    page_title="Exoplanet Detector",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- styling --
st.markdown("""
<style>
    .main { background-color: #0b0f19; }
    .stApp { background: radial-gradient(circle at 20% 0%, #131a2a 0%, #0b0f19 60%); }
    h1, h2, h3 { color: #e2e8f0; }
    .verdict-box {
        padding: 1.4rem; border-radius: 12px; text-align: center;
        font-size: 1.4rem; font-weight: 700; margin-bottom: 1rem;
    }
    .planet-yes { background: linear-gradient(135deg,#065f46,#059669); color: white; }
    .planet-no { background: linear-gradient(135deg,#7f1d1d,#b91c1c); color: white; }
    .metric-card {
        background: #131a2a; border: 1px solid #1f2937; border-radius: 10px;
        padding: 0.8rem 1rem; text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    bundle = joblib.load(MODEL_PATH)
    return bundle["model"]


@st.cache_data
def load_metrics():
    if not os.path.exists(METRICS_PATH):
        return {}
    with open(METRICS_PATH) as f:
        return json.load(f)


@st.cache_data
def load_samples():
    return pd.read_csv(SAMPLE_PATH)


@st.cache_data
def load_full_dataset():
    return pd.read_csv(DATA_PATH)


def get_flux_cols(df):
    return [c for c in df.columns if c.startswith("flux_")]


def predict_one(model, flux):
    feats = extract_features(flux)
    X = pd.DataFrame([feats])[FEATURE_NAMES]
    prob = model.predict_proba(X)[0, 1]
    pred = int(prob >= 0.5)
    return pred, prob, feats


def plot_light_curve(flux, flat, title="Light Curve"):
    t = np.arange(len(flux))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=flux, mode="lines", name="Raw flux",
        line=dict(color="#64748b", width=1),
        opacity=0.6,
    ))
    fig.add_trace(go.Scatter(
        x=t, y=flat, mode="lines", name="Flattened flux",
        line=dict(color="#38bdf8", width=2),
    ))
    fig.update_layout(
        title=title,
        template="plotly_dark",
        paper_bgcolor="#0b0f19", plot_bgcolor="#0b0f19",
        xaxis_title="Time (cadence index)",
        yaxis_title="Normalized brightness",
        height=420,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🪐 Exoplanet Detector")
st.sidebar.caption("ML-based transit detection from light curve photometry")

page = st.sidebar.radio(
    "Navigate",
    ["🔭 Detect a Transit", "📊 Model Performance", "📁 Batch Explorer", "ℹ️ About"],
)

model = load_model()
metrics = load_metrics()

if model is None:
    st.error(
        "No trained model found. Please run the training pipeline first:\n\n"
        "```\npython data/generate_data.py\npython src/train_model.py\n```"
    )
    st.stop()

# --------------------------------------------------------------- main page
if page == "🔭 Detect a Transit":
    st.title("Exoplanet Transit Detector")
    st.write(
        "Feed a star's brightness-over-time curve into the model and it will "
        "flag whether the periodic dimming pattern looks like a genuine "
        "planet transit, a false-positive (eclipsing binary), or just noise."
    )

    col_input, col_result = st.columns([1.1, 1])

    with col_input:
        st.subheader("1. Choose a light curve")
        source_choice = st.radio(
            "Input source", ["Use a sample light curve", "Upload my own CSV"],
            horizontal=True,
        )

        flux = None
        label_truth = None
        source_truth = None

        if source_choice == "Use a sample light curve":
            samples = load_samples()
            options = [
                f"#{row.id} — true label: {'Exoplanet' if row.label == 1 else 'No planet'} "
                f"({row.source})"
                for row in samples.itertuples()
            ]
            idx = st.selectbox("Sample", range(len(options)), format_func=lambda i: options[i])
            row = samples.iloc[idx]
            flux_cols = get_flux_cols(samples)
            flux = row[flux_cols].values.astype(float)
            label_truth = int(row["label"])
            source_truth = row["source"]
        else:
            st.caption(
                "CSV format: a single row (or column) of numeric flux values, "
                "e.g. columns flux_0, flux_1, ... flux_199 — or any single "
                "column of brightness measurements over time."
            )
            uploaded = st.file_uploader("Upload light curve CSV", type=["csv"])
            if uploaded is not None:
                udf = pd.read_csv(uploaded)
                flux_cols = get_flux_cols(udf)
                if flux_cols:
                    flux = udf.iloc[0][flux_cols].values.astype(float)
                else:
                    # fall back: treat first numeric column as the flux series
                    numeric_cols = udf.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) == 0:
                        st.error("No numeric columns found in the uploaded file.")
                    else:
                        flux = udf[numeric_cols[0]].values.astype(float)

        run_btn = st.button("🚀 Run Detection", type="primary", use_container_width=True,
                             disabled=flux is None)

    with col_result:
        st.subheader("2. Result")
        if flux is not None and run_btn:
            pred, prob, feats = predict_one(model, flux)
            flat = flatten_curve(flux)

            if pred == 1:
                st.markdown(
                    f'<div class="verdict-box planet-yes">🪐 EXOPLANET SIGNAL DETECTED<br>'
                    f'<span style="font-size:1rem;font-weight:400">confidence: {prob*100:.1f}%</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="verdict-box planet-no">🚫 No Transit Detected<br>'
                    f'<span style="font-size:1rem;font-weight:400">exoplanet probability: {prob*100:.1f}%</span></div>',
                    unsafe_allow_html=True,
                )

            if label_truth is not None:
                match = "✅ matches" if pred == label_truth else "⚠️ differs from"
                st.caption(
                    f"Ground truth for this sample: "
                    f"**{'Exoplanet' if label_truth == 1 else 'No planet'}** "
                    f"({source_truth}) — model prediction {match} ground truth."
                )

            st.progress(min(max(prob, 0.0), 1.0))

            m1, m2, m3 = st.columns(3)
            m1.markdown(f'<div class="metric-card"><b>{feats["depth"]*100:.2f}%</b><br>Transit depth</div>', unsafe_allow_html=True)
            m2.markdown(f'<div class="metric-card"><b>{feats["dip_width"]:.0f} pts</b><br>Dip duration</div>', unsafe_allow_html=True)
            m3.markdown(f'<div class="metric-card"><b>{feats["depth_to_width_ratio"]:.4f}</b><br>Depth/width ratio</div>', unsafe_allow_html=True)

            with st.expander("See all extracted features"):
                st.dataframe(pd.DataFrame([feats]).T.rename(columns={0: "value"}),
                             use_container_width=True)
        elif flux is not None:
            st.info("Click **Run Detection** to classify this light curve.")
        else:
            st.info("Select or upload a light curve to get started.")

    if flux is not None:
        st.subheader("3. Light curve visualization")
        flat = flatten_curve(flux)
        st.plotly_chart(plot_light_curve(flux, flat), use_container_width=True)
        st.caption(
            "Grey = raw brightness measurements. Blue = flattened curve after "
            "removing slow stellar-variability trends (same technique "
            "astronomers use before searching for transits)."
        )

# --------------------------------------------------------------- perf page
elif page == "📊 Model Performance":
    st.title("Model Performance")
    st.write("Random Forest classifier trained on engineered light-curve features.")

    if metrics:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
        c2.metric("Precision", f"{metrics['precision']*100:.1f}%")
        c3.metric("Recall", f"{metrics['recall']*100:.1f}%")
        c4.metric("F1 Score", f"{metrics['f1_score']*100:.1f}%")
        c5.metric("ROC AUC", f"{metrics['roc_auc']:.3f}")

        st.info(
            "🔎 **Why precision & recall matter more than accuracy here:** "
            "astronomers care a lot about **false positives** (wasting expensive "
            "follow-up telescope time on a non-planet) and about **false negatives** "
            "(missing a real planet). Precision tells us how many of our 'exoplanet' "
            "calls are actually real; recall tells us how many real planets we caught."
        )

    st.subheader("Diagnostic plots")
    p1, p2 = st.columns(2)
    cm_path = os.path.join(BASE_DIR, "models", "confusion_matrix.png")
    roc_path = os.path.join(BASE_DIR, "models", "roc_curve.png")
    fi_path = os.path.join(BASE_DIR, "models", "feature_importance.png")

    if os.path.exists(cm_path):
        p1.image(cm_path, caption="Confusion Matrix", use_container_width=True)
    if os.path.exists(roc_path):
        p2.image(roc_path, caption="ROC Curve", use_container_width=True)
    if os.path.exists(fi_path):
        st.image(fi_path, caption="Which features drive the model's decisions", use_container_width=True)

# ------------------------------------------------------------ batch explorer
elif page == "📁 Batch Explorer":
    st.title("Batch Explorer")
    st.write("Run the model across the full dataset and browse / filter predictions.")

    df = load_full_dataset()
    flux_cols = get_flux_cols(df)

    n_limit = st.slider("Number of light curves to score", 50, len(df), 300, step=50)
    sub = df.head(n_limit).copy()

    if st.button("Run batch prediction", type="primary"):
        with st.spinner("Scoring light curves..."):
            from features import extract_features_batch
            X_feat = extract_features_batch(sub[flux_cols].values)[FEATURE_NAMES]
            probs = model.predict_proba(X_feat)[:, 1]
            preds = (probs >= 0.5).astype(int)
            sub["predicted_label"] = preds
            sub["exoplanet_probability"] = probs
            sub["correct"] = sub["predicted_label"] == sub["label"]

        st.session_state["batch_results"] = sub

    if "batch_results" in st.session_state:
        sub = st.session_state["batch_results"]
        acc = sub["correct"].mean()
        st.metric("Batch accuracy", f"{acc*100:.1f}%", help="On this subset")

        filt = st.multiselect(
            "Filter by predicted class",
            ["Exoplanet", "No planet"], default=["Exoplanet", "No planet"],
        )
        wanted = []
        if "Exoplanet" in filt: wanted.append(1)
        if "No planet" in filt: wanted.append(0)

        view = sub[sub["predicted_label"].isin(wanted)][
            ["id", "source", "label", "predicted_label", "exoplanet_probability", "correct"]
        ].rename(columns={
            "label": "true_label",
        }).sort_values("exoplanet_probability", ascending=False)

        st.dataframe(view, use_container_width=True, height=420)

        csv = view.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download results as CSV", csv, "batch_predictions.csv", "text/csv")

# --------------------------------------------------------------------- about
else:
    st.title("About this project")
    st.markdown("""
This project simulates the real method NASA's Kepler / TESS pipelines use to
find exoplanets: watching a star's brightness over time and looking for the
small, periodic dip caused by a planet passing in front of it (a **transit**).

**Pipeline**
1. **Data**: Light curves (brightness vs. time). This build ships with a
   realistic *simulated* dataset generated by `data/generate_data.py` so the
   whole project runs immediately, offline. It models three cases: genuine
   planet transits, eclipsing-binary false positives, and quiet stars.
2. **Preprocessing**: `src/features.py` flattens each curve with a rolling
   median filter to remove slow stellar-variability trends — the same
   technique used by the `lightkurve` library on real Kepler data.
3. **Feature engineering**: transit depth, dip duration/width, skewness,
   kurtosis, depth-to-width ratio, etc.
4. **Model**: a Random Forest classifier (`src/train_model.py`), evaluated
   with precision/recall/F1/ROC-AUC — the metrics astronomers actually care
   about, since false positives waste follow-up telescope time.
5. **UI**: this Streamlit app — pick or upload a light curve, get an instant
   verdict with confidence, inspect the extracted features, and browse batch
   results.

**Using real NASA data instead**
Swap in real light curves from:
- NASA Exoplanet Archive — https://exoplanetarchive.ipac.caltech.edu
- MAST (Kepler / K2 / TESS) — https://archive.stsci.edu
- `lightkurve` Python package — purpose-built for downloading & processing
  Kepler/TESS light curves

Format your CSV with columns `flux_0 ... flux_N` (one row per light curve,
label column optional) and drop it in as `data/real_data.csv`, then re-run
`src/train_model.py` pointed at that file.
""")
