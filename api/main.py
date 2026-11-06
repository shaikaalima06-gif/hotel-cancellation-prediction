
from pathlib import Path
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict


# Project paths
ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"

MODEL_PATH = MODELS / "hotel_cancellation_model.joblib"
PREPROCESSOR_PATH = MODELS / "preprocessor.joblib"
METADATA_PATH = MODELS / "model_metadata.json"


# Verify deployment artifacts
for path in [MODEL_PATH, PREPROCESSOR_PATH, METADATA_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")


# Load model, preprocessor and metadata
model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    metadata = json.load(file)

FEATURE_COLUMNS = preprocessor.feature_names_in_.tolist()


# Create FastAPI application
app = FastAPI(
    title="Hotel Cancellation Prediction API",
    description="Predict whether a hotel booking will be cancelled.",
    version=metadata["version"]
)


# Input validation
class BookingInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hotel: str
    lead_time: float
    arrival_date_year: int
    arrival_date_month: str
    arrival_date_week_number: int
    arrival_date_day_of_month: int
    stays_in_weekend_nights: int
    stays_in_week_nights: int
    adults: int
    children: float
    babies: int
    meal: str
    country: str | None = None
    market_segment: str
    distribution_channel: str
    is_repeated_guest: int
    previous_cancellations: int
    previous_bookings_not_canceled: int
    reserved_room_type: str
    assigned_room_type: str
    booking_changes: int
    deposit_type: str
    agent: float | None = None
    company: float | None = None
    days_in_waiting_list: int
    customer_type: str
    adr: float
    required_car_parking_spaces: int
    total_of_special_requests: int


@app.get("/")
def home():
    return {
        "message": "Hotel Cancellation Prediction API is running",
        "model": metadata["model_name"],
        "version": metadata["version"]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "preprocessor_loaded": preprocessor is not None
    }


@app.post("/predict")
def predict(booking: BookingInput):
    try:
        # Convert incoming data into a DataFrame
        input_data = pd.DataFrame(
            [booking.model_dump()],
            columns=FEATURE_COLUMNS
        )

        # Apply the saved preprocessing pipeline
        transformed_data = preprocessor.transform(input_data)

        # Generate prediction
        prediction = int(model.predict(transformed_data)[0])

        # Calculate cancellation probability
        probabilities = model.predict_proba(transformed_data)[0]
        cancellation_probability = float(probabilities[1])

        return {
            "prediction": prediction,
            "result": (
                "Cancelled" if prediction == 1
                else "Not Cancelled"
            ),
            "cancellation_probability": round(
                cancellation_probability, 4
            ),
            "model": metadata["model_name"],
            "model_version": metadata["version"]
        }

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(error)}"
        )
