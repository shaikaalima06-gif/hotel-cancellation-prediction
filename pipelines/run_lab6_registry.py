"""Experiment 6: register a tracked model in MLflow Model Registry."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    subprocess.run(
        [sys.executable, str(ROOT / "src" / "register_model.py")],
        cwd=ROOT,
        check=True,
    )
    print("[SUCCESS] Model registry workflow completed!")
