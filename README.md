# 🪐 Exoplanet Detector — Machine Learning Transit Detection

Detect exoplanets the way NASA's Kepler/TESS pipelines do: by watching a
star's brightness over time and spotting the small, periodic dip caused by
a planet passing in front of it. This project includes a full pipeline
(data → features → model → evaluation) and a **web UI** — no console output,
just click and see results in your browser.

This build ships with a realistic **simulated** light-curve dataset so it
runs immediately and fully offline (see "Using real NASA data" below to
swap in the real thing).

---

## What you get

- `data/generate_data.py` — generates 3,000 labeled light curves (genuine
  transits, eclipsing-binary false positives, and quiet stars)
- `src/features.py` — flattens curves and extracts transit features (depth,
  duration, skewness, etc.), the same style of preprocessing the `lightkurve`
  library does for real Kepler data
- `src/train_model.py` — trains a Random Forest classifier and saves
  accuracy/precision/recall/ROC-AUC plots
- `app.py` — the **Streamlit web app**: pick or upload a light curve, get an
  instant "Exoplanet / No planet" verdict with confidence, view the model's
  performance dashboard, and run batch predictions
- Pre-trained model included in `models/` so the app works the moment you
  install dependencies — no waiting required (but you can retrain anytime)

---

## Requirements

- **Python 3.9 or newer** (check with `python3 --version` or `python --version`)
- About 200 MB free disk space for dependencies

Don't have Python? Download it from https://www.python.org/downloads/
(on Windows, tick **"Add Python to PATH"** during install).

---

## Installation & running

### Option A — one-click scripts (easiest)

**Mac / Linux**
```bash
cd exoplanet-detector
chmod +x run.sh
./run.sh
```

**Windows**
```
cd exoplanet-detector
run.bat
```

This creates a virtual environment, installs everything, trains the model
if needed, and launches the app — your browser should open automatically to
`http://localhost:8501`. If it doesn't, just open that link manually.

### Option B — manual steps (any OS)

```bash
# 1. Go into the project folder
cd exoplanet-detector

# 2. (Recommended) create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional — a trained model is already included)
#    Regenerate data and retrain from scratch if you want:
python data/generate_data.py
python src/train_model.py

# 5. Launch the app
streamlit run app.py
```

Your browser opens to **http://localhost:8501** — that's the whole UI, no
terminal/console output needed once it's running.

To stop the app, go back to the terminal and press `Ctrl+C`.

---

## Using the app

- **🔭 Detect a Transit** — pick a sample star's light curve (or upload your
  own CSV) and click "Run Detection" to get an instant verdict with a
  confidence score, an interactive plot of the brightness curve, and the
  extracted transit features (depth, duration, etc.)
- **📊 Model Performance** — accuracy, precision, recall, F1, ROC-AUC, plus
  confusion matrix / ROC curve / feature-importance charts
- **📁 Batch Explorer** — run the model across hundreds of light curves at
  once, filter results, and download predictions as CSV
- **ℹ️ About** — explains the method and how to plug in real NASA data

### Uploading your own light curve
CSV with columns `flux_0, flux_1, ... flux_N` (one row = one light curve),
or a single column of brightness values over time.

---

## Using real NASA data instead of the simulation

The synthetic dataset mimics real transit physics closely enough to learn
and demo the method, but for a research-grade or "real data" version:

1. **NASA Exoplanet Archive**: https://exoplanetarchive.ipac.caltech.edu
2. **MAST (Kepler / K2 / TESS)**: https://archive.stsci.edu
3. **`lightkurve`** Python package — built specifically for downloading and
   processing Kepler/TESS light curves:
   ```bash
   pip install lightkurve
   ```
   ```python
   import lightkurve as lk
   search = lk.search_lightcurve("Kepler-10", mission="Kepler")
   lc = search.download()
   lc.flatten().plot()
   ```
4. Save your real light curves as `data/real_data.csv` with columns
   `flux_0 ... flux_N` (+ optional `label` column), then point
   `src/train_model.py`'s `DATA_PATH` at that file and re-run it.

---

## Project structure

```
exoplanet-detector/
├── app.py                  # Streamlit UI (run this)
├── requirements.txt
├── run.sh / run.bat        # one-click launch scripts
├── data/
│   ├── generate_data.py    # synthetic light-curve generator
│   └── light_curves.csv    # generated dataset (3,000 curves)
├── src/
│   ├── features.py         # flattening + feature extraction
│   └── train_model.py      # trains & evaluates the Random Forest
├── models/
│   ├── exoplanet_model.joblib   # pre-trained model (ready to use)
│   ├── model_metrics.json
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
└── samples/
    └── sample_light_curves.csv  # curated demo examples for the UI
```

---

## Troubleshooting

- **"python3: command not found"** → try `python` instead of `python3`, or
  reinstall Python and check "Add to PATH" (Windows).
- **"streamlit: command not found"** → make sure your virtual environment is
  activated (`source venv/bin/activate` / `venv\Scripts\activate`), then
  `pip install -r requirements.txt` again.
- **Port already in use** → run `streamlit run app.py --server.port 8502`
  and open `http://localhost:8502` instead.
- **Browser didn't open automatically** → manually visit the URL printed in
  the terminal (typically `http://localhost:8501`).

---

## Report-writing tip

NASA's own vetting pipelines use machine learning too (e.g. Google's
AstroNet, used to help validate Kepler candidates) — mentioning that in
your report shows this project mirrors real research methodology, not just
a classroom exercise.
