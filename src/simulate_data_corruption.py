
"""Lab 9: Simulate corruption in a test copy of the hotel dataset."""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "data" / "raw" / "hotel_bookings.csv"
TEST_DIR = ROOT / "data" / "test"
CORRUPTED_FILE = TEST_DIR / "hotel_bookings_corrupted.csv"


def main():
    print("\n=== LAB 9: DATA CORRUPTION SIMULATION ===")

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Original dataset not found: {RAW_FILE}"
        )

    TEST_DIR.mkdir(parents=True, exist_ok=True)

    # Load only 1,000 rows to create a small test dataset.
    df = pd.read_csv(RAW_FILE, nrows=1000)

    print(f"Original test data shape: {df.shape}")

    # Deliberately remove a required column.
    df = df.drop(columns=["hotel"])

    # Save the corrupted copy separately.
    df.to_csv(CORRUPTED_FILE, index=False)

    print("\n[CORRUPTION SIMULATED]")
    print("Removed required column: hotel")
    print(f"Corrupted dataset saved to: {CORRUPTED_FILE}")
    print(f"Corrupted data shape: {df.shape}")

    print("\nOriginal dataset remains unchanged.")


if __name__ == "__main__":
    main()
