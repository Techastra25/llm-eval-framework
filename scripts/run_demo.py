"""
Entry point: runs three simulated models (strong / partial / weak tier)
through the full pipeline and writes results to SQLite + CSV.

Run from the project root:
    python -m scripts.run_demo

To use real models instead, swap the `models` list below for
OpenAIModel(...) / AnthropicModel(...) instances -- see
src/models/openai_model.py and src/models/anthropic_model.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluators.judge import RubricJudge
from src.models.mock_model import MockModel
from src.pipeline import EvaluationPipeline
from src.storage import Storage


def main() -> None:
    models = [
        MockModel(name="model-alpha-strong", skill_tier="strong"),
        MockModel(name="model-beta-partial", skill_tier="partial"),
        MockModel(name="model-gamma-weak", skill_tier="weak"),
    ]

    root = Path(__file__).resolve().parent.parent
    results_dir = root / "results"
    results_dir.mkdir(exist_ok=True)

    judge = RubricJudge()  # heuristic judge by default; pass judge_model=... for real LLM-as-judge
    storage = Storage(db_path=str(results_dir / "eval_results.db"))
    pipeline = EvaluationPipeline(models=models, judge=judge, storage=storage)

    dataset_path = str(root / "data" / "sample_prompts.jsonl")
    run_id = pipeline.run(dataset_path)
    pipeline.export_csv(run_id, out_path=str(results_dir / "eval_results.csv"))

    print(f"Run complete. run_id={run_id}")
    print("Results written to results/eval_results.db and results/eval_results.csv")


if __name__ == "__main__":
    main()
