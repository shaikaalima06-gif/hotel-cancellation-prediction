# LAB 1: ML Project Repository Initialization, Baseline Model Development and Evaluation

## 1. Aim

To initialize a machine learning project repository, develop baseline classification models using the Hotel Booking Demand dataset, evaluate their performance, compare different algorithms, and maintain project files using Git version control.

## 2. Dataset Description

- Dataset: `hotel_bookings.csv`
- Total records: 119,390
- Original columns: 32
- Target variable: `is_canceled`
- Class 0 (Not Cancelled): 75,166 records
- Class 1 (Cancelled): 44,224 records

The dataset contains hotel reservation information, including lead time, customer details, room types, booking channels, and cancellation status.

## 3. Data Preprocessing

The following preprocessing operations were performed:

1. Loaded the dataset using Pandas.
2. Replaced missing values in the `children` column with 0.
3. Removed `reservation_status` and `reservation_status_date` to avoid target leakage.
4. Separated independent features and the target variable.
5. Split the dataset into 80% training data and 20% testing data.
6. Used stratified sampling with random state 42.
7. Applied median imputation to numerical features.
8. Applied most-frequent imputation to categorical features.
9. Applied StandardScaler to numerical features.
10. Applied OneHotEncoder to categorical features.

After preprocessing:

- Training records: 95,512
- Testing records: 23,878
- Processed training features: 252
- Processed testing features: 252

## 4. Baseline Models Implemented

Three classification algorithms were trained and evaluated:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

## 5. Model Evaluation

The models were evaluated using accuracy, precision, recall, F1-score, and ROC-AUC.

| Metric | Logistic Regression | Decision Tree | Random Forest |
|---|---:|---:|---:|
| Accuracy | 81.84% | 85.35% | 89.55% |
| Precision | 81.14% | 82.39% | 86.66% |
| Recall | 66.41% | 76.89% | 84.84% |
| F1-score | 73.04% | 79.54% | 85.74% |
| ROC-AUC | 89.62% | 92.75% | 95.95% |

These values are from the model comparison experiment.

## 6. Confusion Matrix - Random Forest

The standalone Random Forest evaluation produced the following confusion matrix:

| Actual / Predicted | Not Cancelled | Cancelled |
|---|---:|---:|
| Not Cancelled | 13,871 | 1,162 |
| Cancelled | 1,334 | 7,511 |

- True negatives: 13,871
- False positives: 1,162
- False negatives: 1,334
- True positives: 7,511
- Correct predictions: 21,382
- Incorrect predictions: 2,496
- Total test predictions: 23,878

## 7. Error Analysis

The standalone Random Forest evaluation achieved approximately 89.55% accuracy.

The error analysis identified two types of incorrect predictions:

- False positives (1,162): Bookings that were not cancelled but were predicted as cancelled.
- False negatives (1,334): Bookings that were cancelled but were predicted as not cancelled.

False negatives are particularly important because the model fails to identify bookings that will actually be cancelled.

Possible improvements include feature engineering, hyperparameter tuning, improved class-specific evaluation, and testing alternative classification thresholds.

Note: The model comparison metrics and standalone confusion matrix were produced by separate evaluation runs. Small differences in their metrics are therefore possible.

## 8. Preprocessing Limitations and Observations

The preprocessing workflow has the following limitations:

1. Missing values are handled using fixed imputation strategies, which may not capture the actual meaning of missing information.
2. The dataset contains more non-cancelled bookings than cancelled bookings, so class distribution should be considered when evaluating predictions.
3. One-hot encoding increases the number of input features to 252.
4. StandardScaler is applied to numerical features, but future experiments could compare alternative preprocessing strategies.
5. The evaluation uses a stratified random split. A time-based split could provide additional insight into performance on future bookings.
6. The current model comparison is based on the selected algorithms and configurations; additional hyperparameter tuning may improve performance.

## 9. Model Selection

Random Forest was selected as the best-performing model because it achieved the highest scores across the evaluated metrics in the model comparison experiment.

Its comparison results were:

- Accuracy: 89.55%
- Precision: 86.66%
- Recall: 84.84%
- F1-score: 85.74%
- ROC-AUC: 95.95%

## 10. Model Serialization and Version Information

The trained model and preprocessing objects are saved using Joblib.

- Model file: `models/hotel_cancellation_model.joblib`
- Preprocessor file: `models/preprocessor.joblib`
- Model metadata: `models/model_metadata.json`
- Model version recorded in metadata: 1.0.0

A separate Pickle serialization demonstration is implemented in `src/pickle_serialization_demo.py`. The demonstration verifies model serialization and deserialization without retaining an additional large model copy.

## 11. Repository Structure

The project is organized into modular directories:

- `data/raw/` - Original dataset.
- `data/processed/` - Processed training and testing arrays.
- `src/` - Preprocessing, training, evaluation, and validation scripts.
- `pipelines/` - Automated experiment and pipeline scripts.
- `models/` - Saved models and model metadata.
- `outputs/` - Evaluation metrics, comparison reports, and analysis results.
- `notebooks/` - Jupyter notebooks.
- `.github/workflows/` - GitHub Actions workflows.
- `docs/` - Project documentation.

## 12. Git Version Control and Collaborative Workflow

Git is used to maintain project history and track source-code and documentation changes.

The project uses the following branches:

- `main`
- `preprocessing`
- `training`
- `validation`
- `deployment`
- `experimentation`
- `rollback-practice`

The following Git activities were performed:

1. Created and maintained separate branches for different development responsibilities.
2. Recorded changes using meaningful commit messages.
3. Simulated independent developer workflows for training, validation, deployment, and experimentation.
4. Merged the independent branches into `main`.
5. Intentionally created and resolved a merge conflict between `main` and `preprocessing`.
6. Practised rollback by reverting a temporary documentation commit.
7. Reviewed the Git commit graph to verify the development and merge history.
8. Pushed the updated `main` branch to GitHub.

Supporting documentation:

- `docs/GIT_WORKFLOW_LAB1.md`
- `docs/GIT_CONFLICT_PRACTICE.md`
- `docs/DEVELOPER_TRAINING_WORK.md`
- `docs/DEVELOPER_VALIDATION_WORK.md`
- `docs/DEVELOPER_DEPLOYMENT_WORK.md`
- `docs/DEVELOPER_EXPERIMENTATION_WORK.md`
- `docs/PREPROCESSING_WORKFLOW.md`

## 13. Result

The Hotel Booking Cancellation Prediction baseline system was successfully developed and evaluated. Three classification algorithms were compared, and Random Forest achieved the best performance in the model comparison experiment, with approximately 89.55% accuracy and 95.95% ROC-AUC.

Model serialization, preprocessing documentation, Git version control, branching, merging, conflict resolution, rollback, and independent developer workflow exercises were also demonstrated.

The completed baseline provides a foundation for subsequent MLOps laboratory exercises involving experiment tracking, model registry, pipeline engineering, CI/CD, deployment, and monitoring.
