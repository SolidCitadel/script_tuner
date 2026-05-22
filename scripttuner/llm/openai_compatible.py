"""OpenAI-compatible LLM client.

Provider-agnostic: works with OpenAI, OpenRouter, Together, Groq, local vLLM, etc.
Authentication and endpoint are read from environment variables that the OpenAI SDK
automatically recognizes:

- ``OPENAI_API_KEY``: API key for any provider
- ``OPENAI_BASE_URL``: Provider endpoint (omit for OpenAI default)
"""

from __future__ import annotations

from typing import Any

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


class OpenAICompatibleClient:
    """`LLMClient` Protocol impl using the OpenAI Python SDK.

    The SDK reads ``OPENAI_API_KEY`` and ``OPENAI_BASE_URL`` from the environment.
    Transient errors (429, 5xx, network) are retried by the SDK up to ``max_retries``
    times with exponential backoff. Permanent failures raise.
    """

    def __init__(self, *, model: str, max_retries: int = 3) -> None:
        self._client = OpenAI(max_retries=max_retries)
        self._model = model

    def complete(self, system: str, user: str) -> tuple[str, dict[str, Any]]:
        messages: list[ChatCompletionMessageParam] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
        )
        choice = resp.choices[0]
        content = choice.message.content
        if content is None:
            raise ValueError(
                f"LLM returned no content (finish_reason={choice.finish_reason})"
            )
        metadata: dict[str, Any] = {
            "response_model": resp.model,
            "finish_reason": choice.finish_reason,
        }
        usage = resp.usage
        if usage is not None:
            metadata["prompt_tokens"] = usage.prompt_tokens
            metadata["completion_tokens"] = usage.completion_tokens
            metadata["total_tokens"] = usage.total_tokens
        return content, metadata
