\# LAB 1: ML Project Repository Initialization, Baseline Model Development and Evaluation



\## 1. Aim



To initialize a machine learning project repository, develop baseline classification models using the Hotel Booking Demand dataset, evaluate their performance, compare different algorithms, and maintain project files using Git version control.



\## 2. Dataset Description



\- Dataset: `hotel\_bookings.csv`

\- Total records: 119,390

\- Original columns: 32

\- Target variable: `is\_canceled`

\- Class 0: Not Cancelled – 75,166 records

\- Class 1: Cancelled – 44,224 records



The dataset contains hotel reservation information, including lead time, customer details, room types, booking channels, and cancellation status.



\## 3. Data Preprocessing



The following preprocessing operations were performed:



1\. Loaded the dataset using Pandas.

2\. Handled missing values in the `children` column by replacing them with 0.

3\. Removed `reservation\_status` and `reservation\_status\_date` to avoid target leakage.

4\. Separated independent features and the target variable.

5\. Split the dataset into 80% training data and 20% testing data.

6\. Used stratified sampling with random state 42.

7\. Applied median imputation to numerical features.

8\. Applied most-frequent imputation to categorical features.

9\. Applied StandardScaler to numerical features.

10\. Applied OneHotEncoder to categorical features.



After preprocessing:

\- Training records: 95,512

\- Testing records: 23,878

\- Processed training features: 252

\- Processed testing features: 252



\## 4. Baseline Models Implemented



Three classification algorithms were trained and evaluated:



1\. Logistic Regression

2\. Decision Tree Classifier

3\. Random Forest Classifier



\## 5. Model Evaluation



The models were evaluated using accuracy, precision, recall, F1-score, and ROC-AUC.



| Metric | Logistic Regression | Decision Tree | Random Forest |

|---|---:|---:|---:|

| Accuracy | 81.84% | 85.35% | 89.55% |

| Precision | 81.14% | 82.39% | 86.66% |

| Recall | 66.41% | 76.89% | 84.84% |

| F1-score | 73.04% | 79.54% | 85.74% |

| ROC-AUC | 89.62% | 92.75% | 95.95% |



\## 6. Confusion Matrix – Random Forest



| Actual / Predicted | Not Cancelled | Cancelled |

|---|---:|---:|

| Not Cancelled | 13,871 | 1,162 |

| Cancelled | 1,334 | 7,511 |



Correct predictions: 21,382



Incorrect predictions: 2,496



Total test predictions: 23,878



\## 7. Error Analysis



The Random Forest model achieved 89.55% accuracy on the test dataset.



Out of 23,878 predictions:

\- 21,382 predictions were correct.

\- 2,496 predictions were incorrect.



The model successfully identifies many cancelled and non-cancelled bookings. Further improvement can focus on reducing false predictions and improving cancellation detection.



\## 8. Model Selection



Random Forest was selected as the best-performing model because it achieved the highest accuracy, precision, recall, F1-score, and ROC-AUC among the three tested algorithms.



Its evaluation results were:

\- Accuracy: 89.55%

\- Precision: 86.66%

\- Recall: 84.84%

\- F1-score: 85.74%

\- ROC-AUC: 95.95%



\## 9. Model Serialization and Version Information



The trained model and preprocessing objects are saved using Joblib.



\- Model file: `models/hotel\_cancellation\_model.joblib`

\- Preprocessor file: `models/preprocessor.joblib`

\- Model metadata: `models/model\_metadata.json`

\- Model version recorded in metadata: 1.0.0



Pickle-format model files are also present in the models directory from earlier experiments.



\## 10. Repository Structure



The project is organized into modular directories:



\- `data/raw/` – Original dataset.

\- `data/processed/` – Processed training and testing arrays.

\- `src/` – Preprocessing, training, evaluation, and validation scripts.

\- `pipelines/` – Automated experiment and pipeline scripts.

\- `models/` – Saved models and model metadata.

\- `outputs/` – Evaluation metrics, comparison reports, and analysis results.

\- `notebooks/` – Jupyter notebooks.

\- `.github/workflows/` – GitHub Actions workflows.

\- `docs/` – Project documentation.



\## 11. Git Version Control



Git is used to maintain the project history and track source-code and documentation changes.



The project uses the `main` branch and is connected to a GitHub remote repository.



Changes are recorded through commits and synchronized using Git push and pull operations.



Branching, merging, merge-conflict resolution, rollback, and collaborative workflow exercises are to be documented separately as part of the remaining Lab 1 requirements.



\## 12. Result



The Hotel Booking Cancellation Prediction baseline system was successfully developed and evaluated. Three classification algorithms were compared, and Random Forest achieved the best performance with 89.55% accuracy and 95.95% ROC-AUC.



The experiment provides a baseline for further model tracking, validation, registry management, deployment, and monitoring in subsequent MLOps laboratory exercises.

