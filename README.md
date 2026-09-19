# 🪐 Exoplanet Transit Detector

**Machine Learning–Based Exoplanet Detection from Light Curve Photometry**

A complete, end-to-end machine learning pipeline and interactive web
application that detects exoplanets using the **transit method** — the same
technique behind NASA's Kepler and TESS space telescope missions. The
system analyzes a star's brightness over time, identifies periodic dips
caused by an orbiting planet, and classifies the signal as a genuine
transit or a false positive.

---

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [How the Transit Method Works](#how-the-transit-method-works)
- [Project Architecture](#project-architecture)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Model Performance](#model-performance)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Using Real NASA Data](#using-real-nasa-data)
- [Limitations](#limitations)
- [Future Scope](#future-scope)
- [Author](#author)
- [License](#license)

---

## Overview

This project simulates and automates the exoplanet vetting process used by
astronomers: given a light curve (a time series of a star's brightness),
the system determines whether the pattern of dimming is consistent with a
planet transiting its host star, or whether it is caused by something else
entirely — an eclipsing binary star system, instrument noise, or natural
stellar variability.

The pipeline includes data generation, signal preprocessing, feature
engineering, model training and evaluation, and a full interactive web
interface for real-time classification — built to mirror, at a smaller
scale, the same workflow used by NASA's own automated vetting systems
(e.g. Google's AstroNet model, applied to Kepler mission candidates).

---

## Motivation

NASA's Kepler mission alone monitored over 150,000 stars continuously for
years, producing millions of individual light curves. Manually inspecting
this volume of data is infeasible, and the signal of interest — a transit
— is subtle, typically dimming a star's brightness by less than 1%. This
makes exoplanet detection a natural and well-documented application of
machine learning, and one where the stakes of classification errors are
concrete: a false positive wastes costly follow-up telescope time, while a
false negative means a real discovery is missed.

---

## How the Transit Method Works

When a planet's orbit is aligned such that it passes directly between its
star and an observer, it blocks a small fraction of the star's light. This
produces a **periodic, U-shaped dip** in the star's measured brightness
over time. By analyzing:

- **Depth** of the dip → planet's relative size
- **Duration** of the dip → orbital geometry and speed
- **Periodicity** → orbital period

astronomers can infer the presence and rough characteristics of a planet
without ever directly imaging it.

---

## Project Architecture

```
Raw Light Curve
      │
      ▼
Preprocessing (rolling-median flattening — removes stellar variability)
      │
      ▼
Feature Extraction (depth, duration, skewness, kurtosis, etc.)
      │
      ▼
Random Forest Classifier
      │
      ▼
Prediction: Exoplanet Transit / No Transit  (+ confidence score)
      │
      ▼
Streamlit Web Interface (visualization + interaction)
```

---

## Dataset

Real Kepler/TESS light curves are distributed as large FITS files via
NASA's MAST archive and require an internet connection to retrieve. To
keep this project fully self-contained, reproducible, and runnable
offline, a **physically-informed synthetic dataset** of 3,000 light curves
is generated programmatically (`data/generate_data.py`), modeling three
realistic categories:

| Class | Description | Proportion |
|---|---|---|
| Genuine transit | Shallow (0.5–3%), smooth U-shaped dip | 35% |
| Eclipsing binary (false positive) | Deep (4–15%), sharp V-shaped dip | 20% |
| Quiet star | No dip; noise + stellar variability only | 45% |

Each curve also incorporates simulated stellar variability (low-frequency
brightness drift from starspot rotation) and Gaussian instrument noise, to
reflect the noise characteristics of real photometric data.

> The system is designed to accept real NASA data as a drop-in replacement
> — see [Using Real NASA Data](#using-real-nasa-data).

---

## Methodology

**1. Preprocessing**
A rolling-median filter removes slow-moving stellar variability trends
from each light curve, isolating the transit signal — equivalent to the
`.flatten()` operation in NASA's own `lightkurve` Python package.

**2. Feature Engineering**
Twelve numerical features are extracted per light curve, including:
transit depth, dip duration/width, depth-to-width ratio, skewness,
kurtosis, and standard distributional statistics (mean, median, standard
deviation, min, max, range).

**3. Model Training**
A **Random Forest Classifier** (300 estimators, balanced class weighting)
is trained on the extracted features. This model was selected for its
strong performance on structured/tabular features, resistance to
overfitting, fast training time, and interpretability — an important
consideration given the need to explain feature importance in a research
context.

**4. Evaluation**
The model is evaluated on a held-out 20% test split using accuracy,
precision, recall, F1-score, and ROC-AUC — with particular emphasis on
precision and recall, which more directly reflect the real-world costs of
false positives and false negatives in astronomical vetting.

---

## Model Performance

| Metric | Score |
|---|---|
| Accuracy | 97.5% |
| Precision | 97.9% |
| Recall | 94.5% |
| F1-Score | 96.2% |
| ROC-AUC | 0.99 |

*(Evaluated on a held-out test set of 600 light curves, trained on 2,400.)*

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.9+ |
| Data Processing | NumPy, Pandas, SciPy |
| Machine Learning | scikit-learn (Random Forest) |
| Visualization | Matplotlib, Plotly |
| Web Interface | Streamlit |
| Model Persistence | Joblib |

---

## Project Structure

```
exoplanet-transit-detector/
├── app.py                       # Streamlit web application (entry point)
├── requirements.txt             # Python dependencies
├── run.sh / run.bat             # One-click setup & launch scripts
├── README.md
├── data/
│   ├── generate_data.py         # Synthetic light curve generator
│   └── light_curves.csv         # Generated dataset (3,000 samples)
├── src/
│   ├── features.py              # Preprocessing & feature extraction
│   └── train_model.py           # Model training & evaluation
├── models/                      # Pre-trained model & evaluation artifacts
│   ├── exoplanet_model.joblib
│   ├── model_metrics.json
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
└── samples/
    └── sample_light_curves.csv  # Curated examples for the UI
```

---

## Installation

### Prerequisites
- Python 3.9 or newer
- ~200 MB free disk space

### Quick Start

```bash
# Clone the repository
git clone https://github.com/<your-username>/exoplanet-transit-detector.git
cd exoplanet-transit-detector

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Launch the application
streamlit run app.py
```

Alternatively, use the included one-click scripts:

```bash
./run.sh          # macOS / Linux
run.bat           # Windows
```

The application will open automatically at `http://localhost:8501`.

To retrain the model from scratch:

```bash
python data/generate_data.py
python src/train_model.py
```

---

## Usage

The web application provides four sections:

- **Detect a Transit** — select a sample or upload a light curve to receive
  an instant classification with confidence score and visualization
- **Model Performance** — view accuracy, precision, recall, F1, ROC-AUC,
  confusion matrix, and feature importance
- **Batch Explorer** — run predictions across the full dataset, filter
  results, and export as CSV
- **About** — methodology summary and references

---

## Using Real NASA Data

This project is designed to accommodate real observational data as a
drop-in replacement for the synthetic dataset:

1. **NASA Exoplanet Archive** — https://exoplanetarchive.ipac.caltech.edu
2. **MAST Portal** (Kepler / K2 / TESS) — https://archive.stsci.edu
3. **`lightkurve`** — official Python package for retrieving and
   processing Kepler/TESS light curves:
   ```bash
   pip install lightkurve
   ```

Format retrieved data as a CSV with columns `flux_0 ... flux_N` (one row
per light curve, with an optional `label` column), save it as
`data/real_data.csv`, and re-run `src/train_model.py` pointed at that file.

---

## Limitations

- Trained on physics-informed **synthetic** data rather than raw NASA
  telescope data; real light curves contain additional noise sources
  (cosmic ray artifacts, instrumental systematics, observation gaps) not
  fully captured here
- Performs **binary classification** (transit vs. non-transit) rather than
  the multi-class disposition scheme (candidate / confirmed / false
  positive) used in official NASA vetting pipelines
- Feature-based Random Forest approach, rather than a deep learning model
  operating on raw flux (as used in NASA's AstroNet)

---

## Future Scope

- Integrate real Kepler/TESS light curves via the `lightkurve` API
- Implement a 1D CNN / LSTM model trained directly on raw flux sequences
- Extend to multi-class classification matching NASA's official candidate
  disposition categories
- Add automated period-folding and phase-detection for multi-transit
  signals

---

## Author

**Kancharla Vaishnavi Reddy**

---

## License

This project is released under the MIT License. Feel free to use, modify,
and build upon it for academic or personal projects.
