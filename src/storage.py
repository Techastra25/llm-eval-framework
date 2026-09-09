"""
Persistence layer -- SQLite, chosen deliberately.

Why SQLite (and not Postgres/Mongo/etc.)
------------------------------------------
This is a portfolio-scale evaluation framework meant to run on a laptop
with zero setup. SQLite needs no server, ships with Python, and the
resulting `.db` file can be committed or shared directly. The
`Storage` class below is the only place that knows this -- every other
module talks to it through plain method calls, so migrating to Postgres
later (see README "Future Work") means rewriting this one file, not
the pipeline or the dashboard.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    model_name TEXT NOT NULL,
    prompt_id TEXT NOT NULL,
    category TEXT,
    prediction TEXT,
    reference TEXT,
    token_f1 REAL,
    exact_match REAL,
    hallucination_score REAL,
    toxicity_score REAL,
    rubric_correctness REAL,
    rubric_completeness REAL,
    rubric_clarity REAL,
    rubric_safety REAL,
    rubric_overall REAL,
    latency_ms REAL,
    cost_usd REAL,
    human_reviewed INTEGER DEFAULT 0,
    human_agrees_with_judge INTEGER
);
"""


@dataclass
class EvalRecord:
    run_id: str
    model_name: str
    prompt_id: str
    category: str
    prediction: str
    reference: str
    token_f1: float
    exact_match: float
    hallucination_score: float
    toxicity_score: float
    rubric_correctness: float
    rubric_completeness: float
    rubric_clarity: float
    rubric_safety: float
    rubric_overall: float
    latency_ms: float
    cost_usd: float
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()


class Storage:
    def __init__(self, db_path: str = "results/eval_results.db"):
        self.db_path = db_path
        with self._connect() as conn:
            conn.execute(SCHEMA)

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def insert(self, record: EvalRecord) -> None:
        fields = asdict(record)
        columns = ", ".join(fields.keys())
        placeholders = ", ".join(["?"] * len(fields))
        with self._connect() as conn:
            conn.execute(
                f"INSERT INTO eval_results ({columns}) VALUES ({placeholders})",
                list(fields.values()),
            )

    def fetch_all(self, run_id: str | None = None) -> list[dict]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            if run_id:
                rows = conn.execute("SELECT * FROM eval_results WHERE run_id = ?", (run_id,)).fetchall()
            else:
                rows = conn.execute("SELECT * FROM eval_results").fetchall()
            return [dict(r) for r in rows]

    def mark_human_review(self, record_id: int, agrees_with_judge: bool) -> None:
        """Human-in-the-loop: a reviewer confirms or overrides the automated judge score."""
        with self._connect() as conn:
            conn.execute(
                "UPDATE eval_results SET human_reviewed = 1, human_agrees_with_judge = ? WHERE id = ?",
                (int(agrees_with_judge), record_id),
            )
