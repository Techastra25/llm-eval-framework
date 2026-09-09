"""
Base interface every model backend must implement.

Why this exists
----------------
The whole framework is built around one idea: the evaluation logic
(metrics, rubric judge, storage, dashboard) should never know or care
which LLM produced a response. It only needs a consistent shape of
output. That means new model backends (OpenAI, Anthropic, a local
Hugging Face model, or a company's internal endpoint) can be added by
writing one small adapter class -- nothing else in the pipeline changes.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ModelResponse:
    """Normalized response returned by every model backend."""

    text: str
    latency_ms: float
    input_tokens: int
    output_tokens: int
    cost_usd: float
    model_name: str


class BaseModelClient(ABC):
    """Every model backend (real API or local) implements this contract."""

    #: USD price per 1K input/output tokens. Override in subclasses.
    price_per_1k_input: float = 0.0
    price_per_1k_output: float = 0.0

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...

    @abstractmethod
    def _generate(self, prompt: str) -> tuple[str, int, int]:
        """Return (text, input_tokens, output_tokens). Implemented by subclasses."""
        raise NotImplementedError

    def generate(self, prompt: str) -> ModelResponse:
        """Times the call and computes cost, wrapping whatever `_generate` returns."""
        start = time.perf_counter()
        text, in_tok, out_tok = self._generate(prompt)
        latency_ms = (time.perf_counter() - start) * 1000

        cost = (in_tok / 1000) * self.price_per_1k_input + (out_tok / 1000) * self.price_per_1k_output

        return ModelResponse(
            text=text,
            latency_ms=round(latency_ms, 2),
            input_tokens=in_tok,
            output_tokens=out_tok,
            cost_usd=round(cost, 6),
            model_name=self.model_name,
        )
