"""
Best-model selection.

Per requirement 12: selection is driven by the ACTUAL evaluation
results (Top-K accuracy and emergency recall), not by assuming a more
advanced architecture (e.g. BERT) automatically wins.

Selection rule (documented, and deliberately simple/auditable):
  1. Compute a composite score = 0.5 * top_3_accuracy + 0.5 * emergency_recall
     (equal weight: general ranking quality AND not missing emergencies).
  2. The model with the highest composite score is selected.
  3. Ties are broken by top_1_accuracy, then f1_micro.

This function operates purely on metrics dicts (as stored in the Models
collection), so it works whether metrics come from a real run or -- during
development without infra access -- from a documented placeholder.
"""
from typing import Dict, List, Optional


def composite_score(metrics: Dict) -> float:
    top3 = metrics.get("top_3_accuracy", 0.0) or 0.0
    er = metrics.get("emergency_recall", 0.0) or 0.0
    return 0.5 * top3 + 0.5 * er


def select_best_model(model_docs: List[Dict]) -> Optional[Dict]:
    """model_docs: list of Models-collection documents (each with a
    'metrics' dict). Returns the selected document, or None if empty."""
    if not model_docs:
        return None

    def sort_key(doc):
        m = doc.get("metrics", {}) or {}
        return (
            composite_score(m),
            m.get("top_1_accuracy", 0.0) or 0.0,
            m.get("f1_micro", 0.0) or 0.0,
        )

    return max(model_docs, key=sort_key)


def explain_selection(selected: Dict, all_docs: List[Dict]) -> str:
    """Human-readable justification string for the README / report."""
    if not selected:
        return "No models available to select from."
    lines = [f"Selected model: {selected['name']} (type={selected['type']})"]
    m = selected.get("metrics", {}) or {}
    lines.append(
        f"  composite_score={composite_score(m):.4f}, "
        f"top_1={m.get('top_1_accuracy', 'n/a')}, top_3={m.get('top_3_accuracy', 'n/a')}, "
        f"emergency_recall={m.get('emergency_recall', 'n/a')}"
    )
    lines.append("Comparison across all evaluated models:")
    for doc in sorted(all_docs, key=lambda d: composite_score(d.get("metrics", {}) or {}), reverse=True):
        dm = doc.get("metrics", {}) or {}
        lines.append(
            f"  - {doc['name']} ({doc['type']}): composite={composite_score(dm):.4f}, "
            f"top_3={dm.get('top_3_accuracy', 'n/a')}, emergency_recall={dm.get('emergency_recall', 'n/a')}"
        )
    return "\n".join(lines)
