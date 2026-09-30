"""Verify that preprocessing is deterministic and saved artifacts are usable."""
from pathlib import Path
import hashlib
import json
import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / "data" / "processed"
M = ROOT / "models"


def sha256_array(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    required = [
        P / "X_train_final.npy",
        P / "X_test_final.npy",
        P / "y_train.npy",
        P / "y_test.npy",
        P / "dataset_metadata.json",
        M / "preprocessor.joblib",
        M / "hotel_cancellation_model.joblib",
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Required file(s) not found:\n" + "\n".join(missing))

    with open(P / "dataset_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)

    x_train = np.load(P / "X_train_final.npy")
    x_test = np.load(P / "X_test_final.npy")
    y_train = np.load(P / "y_train.npy")
    y_test = np.load(P / "y_test.npy")

    model = joblib.load(M / "hotel_cancellation_model.joblib")
    preprocessor = joblib.load(M / "preprocessor.joblib")

    predictions = model.predict(x_test)

    checks = {
        "train_shape_matches_metadata": list(x_train.shape) == metadata["processed_train_shape"],
        "test_shape_matches_metadata": list(x_test.shape) == metadata["processed_test_shape"],
        "train_target_length_ok": len(y_train) == x_train.shape[0],
        "test_target_length_ok": len(y_test) == x_test.shape[0],
        "model_prediction_length_ok": len(predictions) == len(y_test),
        "preprocessor_loaded": preprocessor is not None,
        "model_loaded": model is not None,
        "x_train_sha256": sha256_array(P / "X_train_final.npy"),
        "x_test_sha256": sha256_array(P / "X_test_final.npy"),
    }

    passed = all(v for k, v in checks.items() if isinstance(v, bool))
    print(json.dumps(checks, indent=2))
    print("[PASS] Reproducibility validation successful!" if passed
          else "[FAIL] Reproducibility validation failed!")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
