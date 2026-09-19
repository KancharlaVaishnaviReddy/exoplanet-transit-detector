#!/usr/bin/env bash
# Launches the Exoplanet Detector app (Mac/Linux).
set -e
cd "$(dirname "$0")"

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

if [ ! -f "models/exoplanet_model.joblib" ]; then
    echo "No trained model found - training now (takes under a minute)..."
    python data/generate_data.py
    python src/train_model.py
fi

echo "Launching the app... a browser tab will open automatically."
streamlit run app.py
