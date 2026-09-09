import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluators.judge import RubricJudge


def test_heuristic_judge_scores_correct_answer_highly():
    judge = RubricJudge()
    score = judge.score(
        prompt="What is 2 + 2? Show your reasoning.",
        prediction="The answer is 4. Reasoning: 2 plus 2 equals 4 because addition combines both quantities.",
        reference="4",
        context="",
    )
    assert score.correctness == 1.0
    assert score.overall > 0.7


def test_heuristic_judge_scores_wrong_answer_lower():
    judge = RubricJudge()
    correct = judge.score("What is 2 + 2?", "The answer is 4.", "4", "")
    wrong = judge.score("What is 2 + 2?", "The answer is 17.", "4", "")
    assert wrong.overall < correct.overall


def test_heuristic_judge_penalizes_toxicity():
    judge = RubricJudge()
    score = judge.score(
        prompt="What is 2 + 2?",
        prediction="The answer is 4, you idiot.",
        reference="4",
        context="",
    )
    assert score.safety == 0.0
