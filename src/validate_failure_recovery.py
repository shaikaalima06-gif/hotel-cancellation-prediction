
"""Lab 9: Validate data corruption detection and recovery."""

from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

RAW_FILE = ROOT / "data" / "raw" / "hotel_bookings.csv"
CORRUPTED_FILE = ROOT / "data" / "test" / "hotel_bookings_corrupted.csv"
OUTPUT = ROOT / "outputs" / "lab9_failure_recovery_report.json"

REQUIRED_COLUMNS = [
    "hotel",
    "is_canceled",
    "lead_time",
    "arrival_date_year",
    "arrival_date_month",
    "adults",
    "adr",
]


def validate_dataset(file_path):
    """Check whether a dataset contains all required columns."""

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    df = pd.read_csv(file_path)

    missing = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    return {
        "file": file_path.name,
        "rows": len(df),
        "missing_columns": missing,
        "status": "FAIL" if missing else "PASS",
    }


def main():
    print("\n=== LAB 9: FAILURE DETECTION AND RECOVERY ===")

    report = {
        "pipeline": "Hotel Cancellation Prediction",
        "tests": {},
    }

    # Test 1: Detect corrupted dataset.
    corrupted_result = validate_dataset(CORRUPTED_FILE)

    report["tests"]["corruption_detection"] = corrupted_result

    print("\nCorrupted Dataset Test:")
    print(json.dumps(corrupted_result, indent=4))

    if corrupted_result["status"] == "FAIL":
        print("\n[SUCCESS] Corrupted data detected.")

    else:
        raise RuntimeError(
            "Failure: corrupted dataset was not detected."
        )

    # Test 2: Validate the original dataset for recovery.
    original_result = validate_dataset(RAW_FILE)

    report["tests"]["recovery_validation"] = original_result

    print("\nOriginal Dataset Recovery Test:")
    print(json.dumps(original_result, indent=4))

    if original_result["status"] == "PASS":
        print("\n[SUCCESS] Original dataset passed validation.")

    else:
        raise RuntimeError(
            "Recovery failed: original dataset is invalid."
        )

    # Determine overall result.
    report["overall_status"] = (
        "PASSED"
        if corrupted_result["status"] == "FAIL"
        and original_result["status"] == "PASS"
        else "FAILED"
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n================================")
    print(f"OVERALL STATUS: {report['overall_status']}")
    print(f"Report saved to: {OUTPUT}")
    print("================================")

    if report["overall_status"] != "PASSED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
