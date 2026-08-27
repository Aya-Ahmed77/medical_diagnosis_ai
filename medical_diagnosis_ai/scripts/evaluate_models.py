"""CLI: python scripts/evaluate_models.py -- trains+evaluates all 4 models and
prints a comparison table, then reports the selected best model."""
import sys, os, json
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app.services.training_service import train_all_models  # noqa: E402
from app.database.schemas import ModelsRepository  # noqa: E402
from app.models.model_selector import select_best_model, explain_selection  # noqa: E402

if __name__ == "__main__":
    summary = train_all_models()
    print(json.dumps(summary, indent=2))

    docs = ModelsRepository().list_all()
    docs_with_metrics = [d for d in docs if d.get("metrics")]
    best = select_best_model(docs_with_metrics)
    print("\n" + explain_selection(best, docs_with_metrics))
