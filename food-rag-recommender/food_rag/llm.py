"""Pluggable LLM backends: IBM watsonx.ai (Granite), any OpenAI-compatible API, or none.

Every backend exposes `generate(prompt) -> str` and `name`. If no credentials are
configured, `get_llm()` returns None and the RAG layer uses a template fallback,
so the app always works, even without API keys.
"""

from __future__ import annotations

import logging
from typing import Optional, Protocol

import requests

from . import config

log = logging.getLogger(__name__)


class LLM(Protocol):
    name: str

    def generate(self, prompt: str) -> str: ...


class WatsonxLLM:
    """IBM Granite via watsonx.ai, the model used in the original lab."""

    def __init__(self) -> None:
        from ibm_watsonx_ai import Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference

        self.name = f"watsonx · {config.WATSONX_MODEL_ID}"
        self._model = ModelInference(
            model_id=config.WATSONX_MODEL_ID,
            credentials=Credentials(url=config.WATSONX_URL, api_key=config.WATSONX_APIKEY),
            project_id=config.WATSONX_PROJECT_ID,
            params={"max_new_tokens": config.LLM_MAX_TOKENS},
        )

    def generate(self, prompt: str) -> str:
        response = self._model.generate(prompt=prompt)
        return response["results"][0]["generated_text"].strip()


class OpenAICompatibleLLM:
    """Any /chat/completions endpoint: OpenAI, Groq, Together, Ollama, LM Studio, etc."""

    def __init__(self) -> None:
        self.name = f"openai-compatible · {config.OPENAI_MODEL}"
        self._url = config.OPENAI_BASE_URL.rstrip("/") + "/chat/completions"
        self._headers = {"Authorization": f"Bearer {config.OPENAI_API_KEY}"}

    def generate(self, prompt: str) -> str:
        payload = {
            "model": config.OPENAI_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": config.LLM_MAX_TOKENS,
            "temperature": 0.4,
        }
        resp = requests.post(self._url, json=payload, headers=self._headers, timeout=60)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()


def get_llm() -> Optional[LLM]:
    """Pick a backend from LLM_PROVIDER (auto | watsonx | openai | none)."""
    provider = config.LLM_PROVIDER
    try:
        if provider == "watsonx" or (provider == "auto" and config.WATSONX_APIKEY and config.WATSONX_PROJECT_ID):
            return WatsonxLLM()
        if provider == "openai" or (provider == "auto" and config.OPENAI_API_KEY):
            return OpenAICompatibleLLM()
    except Exception as exc:  # missing SDK, bad credentials, ...
        log.warning("LLM backend '%s' unavailable, using template fallback: %s", provider, exc)
    return None
