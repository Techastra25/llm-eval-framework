"""
Real Anthropic backend.

Same story as openai_model.py: not exercised in the demo run since it
needs a paid API key, but drop-in compatible with the rest of the
pipeline. Set ANTHROPIC_API_KEY and swap it in for MockModel.
"""

from __future__ import annotations

import os

from src.models.base import BaseModelClient

# Approximate public pricing as of early 2026, per 1K tokens (USD).
_PRICING = {
    "claude-sonnet-4-6": (0.003, 0.015),
}


class AnthropicModel(BaseModelClient):
    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        try:
            import anthropic  # imported lazily so the package is optional
        except ImportError as exc:
            raise ImportError(
                "The 'anthropic' package is required for AnthropicModel. Install with: pip install anthropic"
            ) from exc

        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Set ANTHROPIC_API_KEY (env var) or pass api_key= explicitly.")

        self._client = anthropic.Anthropic(api_key=key)
        self._model = model
        self.price_per_1k_input, self.price_per_1k_output = _PRICING.get(model, (0.0, 0.0))

    @property
    def model_name(self) -> str:
        return self._model

    def _generate(self, prompt: str) -> tuple[str, int, int]:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text if response.content else ""
        return text, response.usage.input_tokens, response.usage.output_tokens
