
"""Lab 9: Automated failure handling and recovery tests."""

from pathlib import Path
from datetime import datetime
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]

OUTPUT = ROOT / "outputs" / "lab9_pipeline_report.json"

STAGES = [
    (
        "Data Corruption Simulation",
        ROOT / "src" / "simulate_data_corruption.py",
    ),
    (
        "Failure Detection and Recovery Validation",
        ROOT / "src" / "validate_failure_recovery.py",
    ),
]


def run_stage(stage_name, script_path):
    """Execute one stage and capture its output and errors."""

    print(f"\n[RUNNING] {stage_name}")

    start = time.perf_counter()

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

        exit_code = result.returncode
        stdout = result.stdout
        stderr = result.stderr

    except Exception as error:
        exit_code = 1
        stdout = ""
        stderr = str(error)

    duration = round(time.perf_counter() - start, 2)

    status = "PASSED" if exit_code == 0 else "FAILED"

    print(stdout)

    if stderr:
        print("[ERROR DETAILS]")
        print(stderr)

    print(f"[{status}] {stage_name}")

    return {
        "stage": stage_name,
        "status": status,
        "exit_code": exit_code,
        "duration_seconds": duration,
        "error": stderr if exit_code != 0 else None,
    }


def main():
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "pipeline": "Hotel Cancellation Prediction - Lab 9",
        "start_time": datetime.now().isoformat(),
        "stages": [],
        "overall_status": "RUNNING",
    }

    print("\n==========================================")
    print(" LAB 9 FAILURE HANDLING PIPELINE")
    print("==========================================")

    for stage_name, script_path in STAGES:

        stage_result = run_stage(stage_name, script_path)

        report["stages"].append(stage_result)

        if stage_result["status"] == "FAILED":
            print("\n[STOPPED] A stage failed.")
            break

    all_passed = (
        len(report["stages"]) == len(STAGES)
        and all(
            stage["status"] == "PASSED"
            for stage in report["stages"]
        )
    )

    report["overall_status"] = (
        "PASSED" if all_passed else "FAILED"
    )

    report["end_time"] = datetime.now().isoformat()

    with open(OUTPUT, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=4)

    print("\n==========================================")
    print(f"OVERALL STATUS: {report['overall_status']}")
    print(f"Report saved to: {OUTPUT}")
    print("==========================================")

    if not all_passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
