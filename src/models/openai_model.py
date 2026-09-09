"""
Real OpenAI backend.

Not exercised in the demo run (no API key is bundled with this repo --
and shouldn't be). Wire it up by setting OPENAI_API_KEY and swapping
MockModel for OpenAIModel in scripts/run_demo.py. Nothing else changes,
by design (see src/models/base.py).
"""

from __future__ import annotations

import os

from src.models.base import BaseModelClient

# Approximate public pricing as of early 2026, per 1K tokens (USD).
# Update to match whichever model you actually call.
_PRICING = {
    "gpt-4o": (0.0025, 0.010),
    "gpt-4o-mini": (0.00015, 0.0006),
}


class OpenAIModel(BaseModelClient):
    def __init__(self, model: str = "gpt-4o-mini", api_key: str | None = None):
        try:
            from openai import OpenAI  # imported lazily so the package is optional
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAIModel. Install with: pip install openai"
            ) from exc

        key = api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Set OPENAI_API_KEY (env var) or pass api_key= explicitly.")

        self._client = OpenAI(api_key=key)
        self._model = model
        self.price_per_1k_input, self.price_per_1k_output = _PRICING.get(model, (0.0, 0.0))

    @property
    def model_name(self) -> str:
        return self._model

    def _generate(self, prompt: str) -> tuple[str, int, int]:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
        )
        text = response.choices[0].message.content or ""
        usage = response.usage
        return text, usage.prompt_tokens, usage.completion_tokens
