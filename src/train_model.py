"""
train_model.py
----------------
Trains a Random Forest classifier to distinguish real exoplanet transits
from false positives (eclipsing binaries) and quiet stars, using the
engineered features from features.py.

Run:  python src/train_model.py
Produces: models/exoplanet_model.joblib, models/model_metrics.json,
          models/confusion_matrix.png, models/roc_curve.png,
          models/feature_importance.png
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, roc_auc_score, classification_report
)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from features import extract_features_batch, FEATURE_NAMES

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "light_curves.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    flux_cols = [c for c in df.columns if c.startswith("flux_")]
    X_raw = df[flux_cols].values
    y = df["label"].values
    return X_raw, y, df


def main():
    print("Loading dataset...")
    X_raw, y, df = load_data()

    print("Extracting features from light curves...")
    X_feat = extract_features_batch(X_raw)
    X_feat = X_feat[FEATURE_NAMES]

    X_train, X_test, y_train, y_test = train_test_split(
        X_feat, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest classifier...")
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }
    print("\n=== Model Performance ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print("\n" + classification_report(y_test, y_pred, target_names=["No Planet", "Exoplanet"]))

    # Save metrics
    with open(os.path.join(MODELS_DIR, "model_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Planet", "Exoplanet"])
    ax.set_yticklabels(["No Planet", "Exoplanet"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14)
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(os.path.join(MODELS_DIR, "confusion_matrix.png"), dpi=120)
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, label=f"AUC = {metrics['roc_auc']:.3f}", color="#7c3aed")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve"); ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(MODELS_DIR, "roc_curve.png"), dpi=120)
    plt.close(fig)

    # Feature importance
    importances = clf.feature_importances_
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(np.array(FEATURE_NAMES)[order], importances[order], color="#0ea5e9")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(MODELS_DIR, "feature_importance.png"), dpi=120)
    plt.close(fig)

    # Save model
    model_path = os.path.join(MODELS_DIR, "exoplanet_model.joblib")
    joblib.dump({"model": clf, "feature_names": FEATURE_NAMES}, model_path)
    print(f"\nModel saved -> {model_path}")


if __name__ == "__main__":
    main()
