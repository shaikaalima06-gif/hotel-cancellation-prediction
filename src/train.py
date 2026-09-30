"""Train the baseline Random Forest model."""
from pathlib import Path
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)

RANDOM_STATE = 42


def main():
    X_train = np.load(PROCESSED / "X_train_final.npy")
    y_train = np.load(PROCESSED / "y_train.npy")

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    joblib.dump(model, MODELS / "hotel_cancellation_model.joblib")

    metadata = {
        "model_name": "RandomForest",
        "version": "1.0.0",
        "random_state": RANDOM_STATE,
        "n_estimators": 200,
        "class_weight": "balanced",
    }
    with open(MODELS / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Model training complete and saved to disk!")


if __name__ == "__main__":
    main()
