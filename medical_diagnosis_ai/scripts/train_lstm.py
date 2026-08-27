"""CLI: python scripts/train_lstm.py"""
import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.services.training_service import prepare_dataset, train_lstm  # noqa: E402

if __name__ == "__main__":
    train, val, test, label_space, pipeline, cbn = prepare_dataset()
    result = train_lstm(train, val, test, label_space, pipeline, cbn)
    print(json.dumps(result, indent=2))
