import os
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ============================================================
# CONFIGURATION
# ============================================================

DATA_PATH = "data/raw/hotel_bookings.csv"

PROCESSED_DIR = "data/processed"

X_TRAIN_PATH = os.path.join(PROCESSED_DIR, "X_train.npy")
X_TEST_PATH = os.path.join(PROCESSED_DIR, "X_test.npy")
Y_TRAIN_PATH = os.path.join(PROCESSED_DIR, "y_train.npy")
Y_TEST_PATH = os.path.join(PROCESSED_DIR, "y_test.npy")

SCALER_PATH = "models/scaler.pkl"
METADATA_PATH = os.path.join(PROCESSED_DIR, "dataset_metadata.json")


# ============================================================
# CREATE REQUIRED DIRECTORIES
# ============================================================

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)


# ============================================================
# 1. LOAD DATASET
# ============================================================

print("Loading hotel booking dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Dataset loaded successfully.")
print(f"Initial shape: {df.shape}")


# ============================================================
# 2. REMOVE DUPLICATES
# ============================================================

duplicate_count = df.duplicated().sum()

print(f"Duplicate records found: {duplicate_count}")

df = df.drop_duplicates()

print(f"Shape after removing duplicates: {df.shape}")


# ============================================================
# 3. HANDLE MISSING VALUES
# ============================================================

print("Handling missing values...")

# Categorical columns
categorical_columns = df.select_dtypes(
    include=["object"]
).columns

for column in categorical_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna("Unknown")


# Numerical columns
numerical_columns = df.select_dtypes(
    include=["int64", "float64"]
).columns

for column in numerical_columns:
    if df[column].isnull().sum() > 0:
        df[column] = df[column].fillna(df[column].median())


# ============================================================
# 4. FIX DATA TYPES
# ============================================================

if "children" in df.columns:
    df["children"] = pd.to_numeric(
        df["children"],
        errors="coerce"
    )

    df["children"] = df["children"].fillna(
        df["children"].median()
    )


# ============================================================
# 5. REMOVE TARGET LEAKAGE COLUMNS
# ============================================================

leakage_columns = [
    "reservation_status",
    "reservation_status_date"
]

existing_leakage_columns = [
    column
    for column in leakage_columns
    if column in df.columns
]

if existing_leakage_columns:
    df = df.drop(
        columns=existing_leakage_columns
    )

print(
    f"Removed leakage columns: "
    f"{existing_leakage_columns}"
)


# ============================================================
# 6. HANDLE OUTLIERS
# ============================================================

def cap_outliers(dataframe, column):

    if column not in dataframe.columns:
        return dataframe

    Q1 = dataframe[column].quantile(0.25)
    Q3 = dataframe[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    dataframe[column] = dataframe[column].clip(
        lower=lower_limit,
        upper=upper_limit
    )

    return dataframe


# Apply outlier treatment to ADR
if "adr" in df.columns:
    df = cap_outliers(df, "adr")

print("Outlier treatment completed.")


# ============================================================
# 7. SEPARATE FEATURES AND TARGET
# ============================================================

if "is_canceled" not in df.columns:
    raise ValueError(
        "Target column 'is_canceled' was not found."
    )

y = df["is_canceled"]

X = df.drop(
    columns=["is_canceled"]
)


# ============================================================
# 8. ENCODE CATEGORICAL FEATURES
# ============================================================

print("Encoding categorical variables...")

X = pd.get_dummies(
    X,
    drop_first=True
)

print(
    f"Number of features after encoding: {X.shape[1]}"
)


# ============================================================
# 9. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print(
    f"Training data shape: {X_train.shape}"
)

print(
    f"Testing data shape: {X_test.shape}"
)


# ============================================================
# 10. FEATURE SCALING
# ============================================================

print("Scaling features...")

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(
    X_train
)

X_test_scaled = scaler.transform(
    X_test
)


# ============================================================
# 11. SAVE PROCESSED DATA
# ============================================================

np.save(
    X_TRAIN_PATH,
    X_train_scaled
)

np.save(
    X_TEST_PATH,
    X_test_scaled
)

np.save(
    Y_TRAIN_PATH,
    y_train.to_numpy()
)

np.save(
    Y_TEST_PATH,
    y_test.to_numpy()
)


# ============================================================
# 12. SAVE SCALER
# ============================================================

import joblib

joblib.dump(
    scaler,
    SCALER_PATH
)


# ============================================================
# 13. SAVE METADATA
# ============================================================

metadata = {
    "dataset": "Hotel Booking Cancellation",
    "target_column": "is_canceled",

    "original_rows": int(
        duplicate_count + len(df)
    ),

    "processed_rows": int(
        len(df)
    ),

    "number_of_features": int(
        X.shape[1]
    ),

    "train_rows": int(
        X_train.shape[0]
    ),

    "test_rows": int(
        X_test.shape[0]
    ),

    "test_size": 0.20,

    "random_state": 42,

    "stratified_split": True,

    "outlier_method": "IQR clipping",

    "outlier_column": "adr",

    "removed_leakage_columns":
        existing_leakage_columns
}


with open(
    METADATA_PATH,
    "w"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# ============================================================
# COMPLETION MESSAGE
# ============================================================

print("\n========================================")
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("========================================")

print(f"Processed data saved in: {PROCESSED_DIR}")
print(f"Scaler saved in: {SCALER_PATH}")
print(f"Metadata saved in: {METADATA_PATH}")