import os
import json
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)


# ============================================================
# PATHS
# ============================================================

X_TRAIN_PATH = "data/processed/X_train.npy"
X_TEST_PATH = "data/processed/X_test.npy"
Y_TRAIN_PATH = "data/processed/y_train.npy"
Y_TEST_PATH = "data/processed/y_test.npy"

MODEL_DIR = "models"
OUTPUT_DIR = "outputs"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD PROCESSED DATA
# ============================================================

print("Loading processed data...")

X_train = np.load(X_TRAIN_PATH)
X_test = np.load(X_TEST_PATH)
y_train = np.load(Y_TRAIN_PATH)
y_test = np.load(Y_TEST_PATH)

print(f"Training data shape: {X_train.shape}")
print(f"Testing data shape: {X_test.shape}")


# ============================================================
# DEFINE MODELS
# ============================================================

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
}


# ============================================================
# TRAIN AND EVALUATE
# ============================================================

results = []

for name, model in models.items():

    print("\n" + "=" * 50)
    print(f"Training {name}...")
    print("=" * 50)

    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    # Save model
    filename = name.lower().replace(" ", "_") + "_model.pkl"
    model_path = os.path.join(MODEL_DIR, filename)

    joblib.dump(model, model_path)

    print(f"Model saved: {model_path}")

    results.append({
        "model": name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "roc_auc": roc_auc
    })


# ============================================================
# SAVE MODEL COMPARISON
# ============================================================

comparison_path = os.path.join(
    OUTPUT_DIR,
    "model_comparison.json"
)

with open(comparison_path, "w") as f:
    json.dump(results, f, indent=4)

print("\nModel comparison saved to:")
print(comparison_path)

print("\n" + "=" * 50)
print("MODEL TRAINING COMPLETED SUCCESSFULLY")
print("=" * 50)