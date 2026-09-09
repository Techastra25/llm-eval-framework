"""
Deterministic offline model backend.

Why this exists
----------------
Real evaluation work needs paid API keys (OpenAI/Anthropic/etc.), which
isn't something every reviewer of this repo will have on hand. This
mock backend lets the *entire pipeline* -- generation, scoring, storage,
dashboard -- run end-to-end with zero external dependencies and 100%
reproducible output, so anyone can clone the repo and see real numbers
in under a minute.

It is intentionally NOT trying to imitate a real LLM's language quality.
It simulates three tiers of answer quality (strong / partial / weak) so
the metrics and rubric judge downstream have something meaningful to
tell apart -- which is the actual point of an evaluation framework.

Swapping this for `openai_model.py` or `anthropic_model.py` requires
no changes anywhere else in the codebase (see base.py for why).
"""

from __future__ import annotations

import hashlib
import random
import time

from src.models.base import BaseModelClient


class MockModel(BaseModelClient):
    """Simulates a model with a fixed 'skill tier' so runs are reproducible."""

    def __init__(self, name: str, skill_tier: str = "strong", simulated_latency_ms: float = 120.0):
        """
        Args:
            name: label shown in reports, e.g. "mock-gpt-strong".
            skill_tier: "strong" | "partial" | "weak" -- controls how often
                the mock answers correctly, to simulate models of differing quality.
            simulated_latency_ms: artificial delay so latency charts aren't all zero.
        """
        if skill_tier not in {"strong", "partial", "weak"}:
            raise ValueError("skill_tier must be one of: strong, partial, weak")
        self._name = name
        self.skill_tier = skill_tier
        self.simulated_latency_ms = simulated_latency_ms
        # Free to run -- but we still model a nominal per-token cost so the
        # cost-tracking code path is exercised end-to-end.
        self.price_per_1k_input = 0.0005
        self.price_per_1k_output = 0.0015

    @property
    def model_name(self) -> str:
        return self._name

    def _seeded_random(self, prompt: str) -> random.Random:
        """Deterministic per-prompt RNG so re-runs give identical results."""
        seed = int(hashlib.sha256(f"{self._name}:{prompt}".encode()).hexdigest(), 16) % (2**32)
        return random.Random(seed)

    def _generate(self, prompt: str) -> tuple[str, int, int]:
        time.sleep(self.simulated_latency_ms / 1000)  # simulate network latency

        rng = self._seeded_random(prompt)
        correctness_roll = rng.random()

        thresholds = {"strong": 0.85, "partial": 0.55, "weak": 0.25}
        will_be_correct = correctness_roll < thresholds[self.skill_tier]

        if will_be_correct:
            answer = prompt.split("ANSWER_HINT:")[-1].strip() if "ANSWER_HINT:" in prompt else "42"
            text = f"The answer is {answer}. Reasoning: applied the relevant rule step by step."
        else:
            # Simulate a plausible-sounding but wrong / incomplete answer.
            distractors = ["a related but incorrect value", "an incomplete derivation", "the wrong formula's result"]
            text = f"The answer is {rng.choice(distractors)}. Reasoning: partial derivation shown."

        input_tokens = max(8, len(prompt.split()))
        output_tokens = max(6, len(text.split()))
        return text, input_tokens, output_tokens
