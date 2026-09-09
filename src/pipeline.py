"""
Orchestrator: runs a dataset through one or more models, scores every
response with both cheap heuristic metrics and the rubric judge, and
persists everything. This is the file that ties the whole framework
together -- if you want to understand the system in one read, start here.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

from src.evaluators.judge import RubricJudge
from src.evaluators.metrics import exact_match, hallucination_score, token_f1, toxicity_score
from src.models.base import BaseModelClient
from src.storage import EvalRecord, Storage


class EvaluationPipeline:
    def __init__(self, models: list[BaseModelClient], judge: RubricJudge, storage: Storage):
        self.models = models
        self.judge = judge
        self.storage = storage

    def load_dataset(self, path: str) -> list[dict]:
        items = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    def run(self, dataset_path: str) -> str:
        """Runs every model against every prompt in the dataset. Returns the run_id."""
        dataset = self.load_dataset(dataset_path)
        run_id = uuid.uuid4().hex[:10]

        for model in self.models:
            for item in dataset:
                prompt = item["prompt"]
                reference = item["reference"]
                context = item.get("context", "")
                category = item.get("category", "general")
                prompt_id = item["id"]

                response = model.generate(prompt)
                rubric = self.judge.score(prompt, response.text, reference, context)

                record = EvalRecord(
                    run_id=run_id,
                    model_name=model.model_name,
                    prompt_id=prompt_id,
                    category=category,
                    prediction=response.text,
                    reference=reference,
                    token_f1=token_f1(response.text, reference),
                    exact_match=exact_match(response.text, reference),
                    hallucination_score=hallucination_score(response.text, context) if context else 0.0,
                    toxicity_score=toxicity_score(response.text),
                    rubric_correctness=rubric.correctness,
                    rubric_completeness=rubric.completeness,
                    rubric_clarity=rubric.clarity,
                    rubric_safety=rubric.safety,
                    rubric_overall=rubric.overall,
                    latency_ms=response.latency_ms,
                    cost_usd=response.cost_usd,
                )
                self.storage.insert(record)

        return run_id

    def export_csv(self, run_id: str, out_path: str = "results/eval_results.csv") -> None:
        import csv

        rows = self.storage.fetch_all(run_id=run_id)
        if not rows:
            return
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
