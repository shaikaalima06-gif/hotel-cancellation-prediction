"""Experiment 2 / Lab 3: baseline end-to-end pipeline."""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(script):
    print(f"\n===== Running {script.name} =====")
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


if __name__ == "__main__":
    run(ROOT / "src" / "validate_data.py")
    run(ROOT / "src" / "preprocess.py")
    run(ROOT / "src" / "train.py")
    run(ROOT / "src" / "evaluate.py")
    run(ROOT / "src" / "validate_reproducibility.py")
    print("\n[SUCCESS] Experiment 2 / Lab 3 pipeline completed!")
