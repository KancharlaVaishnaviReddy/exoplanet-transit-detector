@echo off
REM Launches the Exoplanet Detector app (Windows).
cd /d "%~dp0"

if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat
pip install -q --upgrade pip
pip install -q -r requirements.txt

if not exist models\exoplanet_model.joblib (
    echo No trained model found - training now, this takes under a minute...
    python data\generate_data.py
    python src\train_model.py
)

echo Launching the app... a browser tab will open automatically.
streamlit run app.py
pause
