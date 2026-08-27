"""CLI: python scripts/run_preprocessing.py"""
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.services.training_service import prepare_dataset  # noqa: E402


def main():
    train, val, test, label_space, pipeline, _ = prepare_dataset()
    print(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}  Labels: {len(label_space)}")
    print("Saved preprocessing artifacts to data/artifacts/")


if __name__ == "__main__":
    main()
