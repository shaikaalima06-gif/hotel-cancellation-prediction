"""Experiment 4: compare Logistic Regression, Decision Tree and Random Forest with MLflow."""
from pathlib import Path
import json
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "hotel_bookings.csv"
DB = ROOT / "mlflow.db"

TARGET = "is_canceled"
DROP = ["reservation_status", "reservation_status_date"]
RANDOM_STATE = 42


def make_preprocessor(X, scale_numeric=True):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()

    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    cat_steps = [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]

    return ColumnTransformer([
        ("num", Pipeline(num_steps), numeric),
        ("cat", Pipeline(cat_steps), categorical),
    ])


def main():
    mlflow.set_tracking_uri(f"sqlite:///{DB.as_posix()}")
    mlflow.set_experiment("Hotel_Cancellation_Prediction")

    df = pd.read_csv(RAW)
    df["children"] = df["children"].fillna(0)

    X = df.drop(columns=[TARGET] + DROP, errors="ignore")
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, C=1.0, random_state=RANDOM_STATE),
        "DecisionTree": DecisionTreeClassifier(max_depth=12, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=200, max_depth=None, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    for name, estimator in models.items():
        scale = name == "LogisticRegression"
        pipe = Pipeline([
            ("preprocessor", make_preprocessor(X_train, scale_numeric=scale)),
            ("model", estimator),
        ])

        with mlflow.start_run(run_name=name) as run:
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            proba = pipe.predict_proba(X_test)[:, 1]

            metrics = {
                "accuracy": accuracy_score(y_test, pred),
                "precision": precision_score(y_test, pred, zero_division=0),
                "recall": recall_score(y_test, pred, zero_division=0),
                "f1": f1_score(y_test, pred, zero_division=0),
                "roc_auc": roc_auc_score(y_test, proba),
            }

            mlflow.log_param("model", name)
            mlflow.log_param("test_size", 0.20)
            mlflow.log_param("random_state", RANDOM_STATE)
            mlflow.log_param("dropped_columns", ",".join(DROP))
            if hasattr(estimator, "get_params"):
                for key, value in estimator.get_params().items():
                    if isinstance(value, (str, int, float, bool)) or value is None:
                        try:
                            mlflow.log_param(f"model_{key}", value)
                        except Exception:
                            pass

            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(pipe, artifact_path="model")
            mlflow.log_artifact(str(ROOT / "data" / "processed" / "dataset_metadata.json"))

            print(f"{name}: {json.dumps(metrics)} | run_id={run.info.run_id}")

    print("\nMLflow tracking completed.")


if __name__ == "__main__":
    main()
