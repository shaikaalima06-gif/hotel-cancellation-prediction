
"""Demonstrate Pickle serialization and deserialization of the trained model."""

from pathlib import Path
import pickle
import joblib

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "hotel_cancellation_model.joblib"
TEMP_PICKLE_PATH = ROOT / "models" / "pickle_demo_temp.pkl"


def main():
    # Load the existing trained model using Joblib
    model = joblib.load(MODEL_PATH)

    # Serialize the model using Python Pickle
    with open(TEMP_PICKLE_PATH, "wb") as file:
        pickle.dump(model, file)

    print("Model successfully serialized using Pickle.")

    # Deserialize the model using Pickle
    with open(TEMP_PICKLE_PATH, "rb") as file:
        loaded_model = pickle.load(file)

    print("Model successfully deserialized using Pickle.")
    print("Model type:", type(loaded_model).__name__)

    # Remove the temporary demonstration file
    TEMP_PICKLE_PATH.unlink()
    print("Temporary Pickle file removed.")


if __name__ == "__main__":
    main()
