"""
Preprocess hotel booking data.

Target:
    is_canceled

Important leakage protection:
    reservation_status and reservation_status_date are excluded because they
    describe the final reservation outcome/status.
"""
from pathlib import Path
import json
import hashlib
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "hotel_bookings.csv"
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

TARGET = "is_canceled"
LEAKAGE_COLUMNS = ["reservation_status", "reservation_status_date"]
RANDOM_STATE = 42


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical),
    ])


def main():
    PROCESSED.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    if not RAW.exists():
        raise FileNotFoundError(f"Dataset not found: {RAW}")

    df = pd.read_csv(RAW)

    # Basic type cleanup.
    if "children" in df.columns:
        df["children"] = df["children"].fillna(0)

    # Do not use post-outcome fields as predictors.
    X = df.drop(columns=[TARGET] + LEAKAGE_COLUMNS, errors="ignore")
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(X_train)

    # Fit only on training data to prevent data leakage.
    X_train_final = preprocessor.fit_transform(X_train)
    X_test_final = preprocessor.transform(X_test)

    # Keep a dense representation for simple reproducibility artifacts.
    X_train_final = X_train_final.toarray() if hasattr(X_train_final, "toarray") else X_train_final
    X_test_final = X_test_final.toarray() if hasattr(X_test_final, "toarray") else X_test_final

    np.save(PROCESSED / "X_train_final.npy", X_train_final)
    np.save(PROCESSED / "X_test_final.npy", X_test_final)
    np.save(PROCESSED / "y_train.npy", y_train.to_numpy())
    np.save(PROCESSED / "y_test.npy", y_test.to_numpy())
    joblib.dump(preprocessor, MODELS / "preprocessor.joblib")

    metadata = {
        "dataset": RAW.name,
        "dataset_sha256": sha256_file(RAW),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "target": TARGET,
        "target_distribution": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "test_size": 0.20,
        "random_state": RANDOM_STATE,
        "dropped_leakage_columns": LEAKAGE_COLUMNS,
        "feature_columns": X.columns.tolist(),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "processed_train_shape": list(X_train_final.shape),
        "processed_test_shape": list(X_test_final.shape),
        "missing_values": {str(k): int(v) for k, v in df.isna().sum().items() if int(v) > 0},
    }

    with open(PROCESSED / "dataset_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("Preprocessing completed successfully!")
    print(f"Raw shape: {df.shape}")
    print(f"Processed train shape: {X_train_final.shape}")
    print(f"Processed test shape: {X_test_final.shape}")
    print("Saved processed data and metadata.")


if __name__ == "__main__":
    main()
