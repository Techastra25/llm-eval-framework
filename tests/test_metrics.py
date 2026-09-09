import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluators.metrics import exact_match, hallucination_score, token_f1, toxicity_score


def test_exact_match_true():
    assert exact_match("The answer is 42.", "42") == 1.0


def test_exact_match_false():
    assert exact_match("The answer is 43.", "42") == 0.0


def test_token_f1_perfect_overlap():
    assert token_f1("cat sat mat", "cat sat mat") == 1.0


def test_token_f1_no_overlap():
    assert token_f1("completely different words here", "unrelated reference text") == 0.0


def test_token_f1_partial_overlap_between_bounds():
    score = token_f1("the cat sat on the mat today", "cat sat mat")
    assert 0.0 < score < 1.0


def test_hallucination_fully_grounded():
    context = "The train travels at 40 kilometers per hour on this route."
    prediction = "The train travels at 40 kilometers per hour."
    assert hallucination_score(prediction, context) == 0.0


def test_hallucination_fully_ungrounded():
    context = "The train travels at 40 kilometers per hour."
    prediction = "Elephants migrate seasonally across continents."
    assert hallucination_score(prediction, context) == 1.0


def test_toxicity_detects_marker():
    assert toxicity_score("You are an idiot for asking that.") == 1.0


def test_toxicity_clean_text():
    assert toxicity_score("The answer is 42, reasoning shown above.") == 0.0
