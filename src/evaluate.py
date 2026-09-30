"""Evaluate the trained model and create error-analysis output."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)


def main():
    X_test = np.load(PROCESSED / "X_test_final.npy")
    y_test = np.load(PROCESSED / "y_test.npy")
    model = joblib.load(MODELS / "hotel_cancellation_model.joblib")

    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }

    with open(OUTPUTS / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    errors = pd.DataFrame({
        "actual": y_test,
        "predicted": pred,
        "predicted_probability": proba,
    })
    errors["error"] = (errors["actual"] != errors["predicted"]).astype(int)
    errors.to_csv(OUTPUTS / "error_analysis.csv", index=False)

    print(classification_report(y_test, pred, zero_division=0))
    print(json.dumps(metrics, indent=2))
    print("Evaluation complete and error analysis files saved!")


if __name__ == "__main__":
    main()
