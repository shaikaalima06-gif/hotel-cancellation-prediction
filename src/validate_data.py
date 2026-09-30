"""Schema/data quality validation for the raw hotel dataset."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "hotel_bookings.csv"
OUTPUT = ROOT / "outputs" / "data_validation_report.json"
OUTPUT.parent.mkdir(exist_ok=True)

EXPECTED_COLUMNS = [
    "hotel", "is_canceled", "lead_time", "arrival_date_year",
    "arrival_date_month", "arrival_date_week_number", "arrival_date_day_of_month",
    "stays_in_weekend_nights", "stays_in_week_nights", "adults", "children",
    "babies", "meal", "country", "market_segment", "distribution_channel",
    "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "reserved_room_type",
    "assigned_room_type", "booking_changes", "deposit_type", "agent",
    "company", "days_in_waiting_list", "customer_type", "adr",
    "required_car_parking_spaces", "total_of_special_requests",
    "reservation_status", "reservation_status_date"
]

ALLOWED = {
    "hotel": {"Resort Hotel", "City Hotel"},
    "deposit_type": {"No Deposit", "Refundable", "Non Refund"},
    "customer_type": {"Transient", "Contract", "Transient-Party", "Group"},
}


def main():
    df = pd.read_csv(RAW)
    report = {
        "file": RAW.name,
        "rows": len(df),
        "columns": len(df.columns),
        "missing_columns": [c for c in EXPECTED_COLUMNS if c not in df.columns],
        "unexpected_columns": [c for c in df.columns if c not in EXPECTED_COLUMNS],
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_values": {c: int(v) for c, v in df.isna().sum().items() if v > 0},
        "invalid_categorical_values": {},
        "negative_value_counts": {},
    }

    for col, allowed in ALLOWED.items():
        invalid = sorted(set(df[col].dropna().unique()) - allowed)
        if invalid:
            report["invalid_categorical_values"][col] = [str(x) for x in invalid]

    numeric_cols = df.select_dtypes(include="number").columns
    for col in numeric_cols:
        count = int((df[col] < 0).sum())
        if count:
            report["negative_value_counts"][col] = count

    report["status"] = (
        "PASS" if not report["missing_columns"]
        and not report["unexpected_columns"]
        and not report["invalid_categorical_values"]
        else "FAIL"
    )

    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    if report["status"] == "FAIL":
        raise SystemExit("Data validation failed. Review the report.")


if __name__ == "__main__":
    main()
