import json
import os
import re
from typing import Any, Dict, Optional


DEMO_RAGAS_PATH = os.getenv(
    "DEMO_RAGAS_PATH",
    "./evaluation_results/demo_ragas_scores.json",
)


def _normalize_query(query: str) -> str:
    normalized = query.strip().lower()
    return re.sub(r"\s+", " ", normalized)


def _overall_level(faithfulness: float, answer_relevancy: float) -> str:
    if faithfulness >= 0.75 and answer_relevancy >= 0.75:
        return "High"
    if faithfulness >= 0.50 and answer_relevancy >= 0.50:
        return "Medium"
    return "Low"


def _load_demo_scores() -> Dict[str, Any]:
    if not os.path.exists(DEMO_RAGAS_PATH):
        return {}

    with open(DEMO_RAGAS_PATH, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, dict) and "scores" in payload:
        return payload["scores"]

    return payload if isinstance(payload, dict) else {}


def get_demo_ragas_scores(query: str) -> Optional[Dict[str, Any]]:
    scores_by_query = _load_demo_scores()
    item = scores_by_query.get(_normalize_query(query))
    if not item:
        return None

    faithfulness = item.get("faithfulness")
    answer_relevancy = item.get("answer_relevancy")
    if faithfulness is None or answer_relevancy is None:
        return None

    faithfulness = round(float(faithfulness), 4)
    answer_relevancy = round(float(answer_relevancy), 4)

    return {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "overall": item.get("overall")
        or _overall_level(faithfulness, answer_relevancy),
        "source": item.get("source", "precomputed_demo_cache"),
    }


def attach_demo_ragas_scores(state: Any, query: str) -> Any:
    state.ragas_scores = get_demo_ragas_scores(query)
    if state.ragas_scores:
        state.debug["ragas_scores_source"] = state.ragas_scores.get("source")
    return state
