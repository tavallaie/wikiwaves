"""Thin OpenAI-compatible LLM client — shared across all services."""

from __future__ import annotations

import os

import requests
from dotenv import load_dotenv
from loguru import logger


class LLMError(Exception):
    """Raised when an LLM API call fails."""


class LLMClient:
    """Minimal client for any OpenAI-compatible chat-completions endpoint."""

    DEFAULT_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_MODEL = "gpt-4o-mini"
    DEFAULT_TIMEOUT = 60.0

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        env_path: str | None = None,
    ) -> None:
        if env_path:
            load_dotenv(env_path)
        elif os.path.exists(".env"):
            load_dotenv()

        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")
        self.model = model or os.getenv("LLM_MODEL", self.DEFAULT_MODEL)
        self.timeout = timeout

        logger.debug(f"LLMClient initialised (model={self.model}, base={self.base_url})")

    def chat(self, messages: list[dict[str, str]], temperature: float = 0.3) -> str:
        """Send a chat-completion request and return the assistant's text."""
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }

        url = f"{self.base_url}/chat/completions"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(f"LLM request failed: {exc}") from exc

        try:
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            logger.error(f"Unexpected LLM response structure: {resp.text[:500]}")
            raise LLMError(f"Could not parse LLM response: {exc}") from exc
