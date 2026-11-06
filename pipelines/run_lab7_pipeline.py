
"""Lab 7: Automated end-to-end MLOps pipeline for hotel cancellation prediction."""

from pathlib import Path
from datetime import datetime
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "lab7_pipeline_report.json"

STAGES = [
    ("Data Validation", ROOT / "src" / "validate_data.py"),
    ("Data Preprocessing", ROOT / "src" / "preprocess.py"),
    ("Model Training", ROOT / "src" / "train.py"),
    ("Model Evaluation", ROOT / "src" / "evaluate.py"),
    ("Reproducibility Validation",
     ROOT / "src" / "validate_reproducibility.py"),
    ("MLflow Experiment Tracking",
     ROOT / "pipelines" / "run_lab4_tracking.py"),
    ("Model Registration",
     ROOT / "pipelines" / "run_lab6_registry.py"),
]


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "pipeline": "Hotel Cancellation Prediction MLOps",
        "dataset": "hotel_bookings.csv",
        "start_time": datetime.now().isoformat(),
        "stages": [],
        "overall_status": "RUNNING",
    }

    print("\n==========================================")
    print(" HOTEL CANCELLATION MLOPS PIPELINE")
    print("==========================================")

    for stage_name, script_path in STAGES:
        print(f"\n[RUNNING] {stage_name}")
        start = time.perf_counter()

        try:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                cwd=ROOT,
                check=False,
            )
            exit_code = result.returncode

        except Exception as error:
            print(f"[ERROR] {error}")
            exit_code = 1

        duration = round(time.perf_counter() - start, 2)

        stage_result = {
            "stage": stage_name,
            "status": "PASSED" if exit_code == 0 else "FAILED",
            "exit_code": exit_code,
            "duration_seconds": duration,
        }

        report["stages"].append(stage_result)

        print(f"[{stage_result['status']}] {stage_name}")

        if exit_code != 0:
            print("[STOPPED] Pipeline stopped because a stage failed.")
            break

    passed = all(
        stage["status"] == "PASSED"
        for stage in report["stages"]
    )

    report["overall_status"] = "PASSED" if passed else "FAILED"
    report["end_time"] = datetime.now().isoformat()

    with open(OUTPUT, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n==========================================")
    print(f"OVERALL STATUS: {report['overall_status']}")
    print(f"Report saved to: {OUTPUT}")
    print("==========================================")

    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
