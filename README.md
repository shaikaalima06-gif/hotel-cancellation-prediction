# Hotel Cancellation Prediction

Machine-learning project for predicting whether a hotel booking will be cancelled.

## Dataset
Place `hotel_bookings.csv` in `data/raw/`.

Target: `is_canceled`

The preprocessing workflow removes `reservation_status` and `reservation_status_date` because they are post-outcome fields and can cause target leakage.

## Experiments

1. Dataset preparation and documentation
2. Baseline classification and evaluation
3. Git version control and collaboration
4. MLflow experiment tracking and reproducibility
5. Schema/data validation and idempotent preprocessing
6. MLflow model registry and version management

## Main commands

```powershell
python -m pip install -r requirements.txt

python src/validate_data.py
python pipelines/run_lab3_baseline.py
python pipelines/run_lab4_tracking.py
python pipelines/run_lab5_validation.py
python pipelines/run_lab6_registry.py
```

## MLflow UI

After running Experiment 4:

```powershell
mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
```

Open http://127.0.0.1:5000 in a browser.

## Git

```powershell
git add .
git commit -m "Add hotel cancellation ML pipeline"
git push -u origin main
```
