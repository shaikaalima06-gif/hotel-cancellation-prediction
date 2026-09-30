import os
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# PATHS
# ============================================================

X_TEST_PATH = "data/processed/X_test.npy"
Y_TEST_PATH = "data/processed/y_test.npy"

MODEL_PATH = "models/random_forest_model.pkl"

OUTPUT_DIR = "outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

print("Loading test data...")

X_test = np.load(X_TEST_PATH)
y_test = np.load(Y_TEST_PATH)

print(f"Test data shape: {X_test.shape}")


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading Random Forest model...")

model = joblib.load(MODEL_PATH)

print("Model loaded successfully.")


# ============================================================
# PREDICTIONS
# ============================================================

print("Generating predictions...")

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_test,
    y_prob
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 50)
print("RANDOM FOREST EVALUATION")
print("=" * 50)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_test,
    y_pred,
    target_names=[
        "Not Cancelled",
        "Cancelled"
    ],
    zero_division=0
)

print("\nClassification Report:")
print(report)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# SAVE CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")

plt.xticks(
    [0, 1],
    ["Not Cancelled", "Cancelled"]
)

plt.yticks(
    [0, 1],
    ["Not Cancelled", "Cancelled"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.savefig(cm_path)

plt.close()

print(f"Confusion matrix saved: {cm_path}")


# ============================================================
# ERROR ANALYSIS
# ============================================================

errors = np.where(y_test != y_pred)[0]

error_data = pd.DataFrame({
    "test_index": errors,
    "actual": y_test[errors],
    "predicted": y_pred[errors],
    "predicted_probability": y_prob[errors]
})

error_path = os.path.join(
    OUTPUT_DIR,
    "error_analysis.csv"
)

error_data.to_csv(
    error_path,
    index=False
)

print(f"Error analysis saved: {error_path}")
print(f"Number of incorrect predictions: {len(errors)}")


# ============================================================
# SAVE EVALUATION REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "evaluation_report.txt"
)

with open(report_path, "w") as f:

    f.write("HOTEL BOOKING CANCELLATION PREDICTION\n")
    f.write("Random Forest Evaluation Report\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Accuracy : {accuracy:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1 Score : {f1:.4f}\n")
    f.write(f"ROC-AUC  : {roc_auc:.4f}\n\n")

    f.write("Classification Report:\n")
    f.write(report)

    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))

    f.write("\n\nIncorrect Predictions:\n")
    f.write(str(len(errors)))

print(f"Evaluation report saved: {report_path}")


print("\n" + "=" * 50)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 50)