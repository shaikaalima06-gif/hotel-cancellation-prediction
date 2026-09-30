"""Experiment 5: data validation + idempotent preprocessing checks."""
import subprocess
import sys
from pathlib import Path
import hashlib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def run(script):
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


if __name__ == "__main__":
    run(ROOT / "src" / "validate_data.py")
    run(ROOT / "src" / "preprocess.py")

    files = [
        ROOT / "data" / "processed" / "X_train_final.npy",
        ROOT / "data" / "processed" / "X_test_final.npy",
        ROOT / "data" / "processed" / "y_train.npy",
        ROOT / "data" / "processed" / "y_test.npy",
    ]
    first = {str(p.name): sha(p) for p in files}

    run(ROOT / "src" / "preprocess.py")
    second = {str(p.name): sha(p) for p in files}

    print("First hashes :", first)
    print("Second hashes:", second)

    if first != second:
        raise SystemExit("[FAIL] Preprocessing is not deterministic/idempotent.")
    print("[PASS] Validation and idempotent preprocessing completed successfully!")
