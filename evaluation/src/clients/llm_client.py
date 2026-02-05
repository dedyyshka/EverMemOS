from __future__ import annotations

import asyncio
from typing import Any, Dict, Optional

import httpx

from evaluation.src.clients.types import LLMClientConfig, ClientConfigError


class LLMClientError(RuntimeError):
    """Ошибка запроса к LLM."""


class LLMClient:
    """Единый LLM клиент для eval (OpenAI‑compatible)."""

    def __init__(self, config: LLMClientConfig):
        self.config = config
        if not self.config.base_url:
            raise ClientConfigError("LLM base_url не задан.")
        if not self.config.model:
            raise ClientConfigError("LLM model не задан.")

        self._client = httpx.AsyncClient(
            base_url=self.config.base_url.rstrip("/"),
            timeout=self.config.timeout_seconds,
            verify=False,
        )

    async def close(self) -> None:
        await self._client.aclose()

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _build_payload(
        self,
        prompt: str,
        temperature: Optional[float],
        max_tokens: Optional[int],
        extra_body: Optional[Dict[str, Any]],
        response_format: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": (
                temperature if temperature is not None else self.config.temperature
            ),
        }

        final_max_tokens = (
            max_tokens if max_tokens is not None else self.config.max_tokens
        )
        if final_max_tokens is not None:
            payload["max_tokens"] = final_max_tokens

        if response_format is not None:
            payload["response_format"] = response_format

        if extra_body:
            payload.update(extra_body)

        if self.config.provider == "openrouter" and self.config.openrouter_provider:
            order = [
                p.strip()
                for p in self.config.openrouter_provider.split(",")
                if p.strip()
            ]
            if order:
                payload["provider"] = {"order": order, "allow_fallbacks": False}

        return payload

    async def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Сгенерировать ответ по строковому prompt."""
        headers = self._build_headers()
        payload = self._build_payload(
            prompt, temperature, max_tokens, extra_body, response_format
        )

        last_error: Exception | None = None
        for attempt in range(self.config.max_retries):
            try:
                response = await self._client.post(
                    "/chat/completions", headers=headers, json=payload
                )
                if response.status_code >= 400:
                    try:
                        error_body = response.json()
                        message = error_body.get("error", {}).get(
                            "message", response.text
                        )
                    except Exception:
                        message = response.text
                    raise LLMClientError(
                        f"LLM HTTP {response.status_code}: {message}"
                    )

                data = response.json()
                choices = data.get("choices", [])
                if not choices:
                    raise LLMClientError("LLM response missing choices.")
                message = choices[0].get("message", {})
                content = message.get("content")
                if content is None:
                    raise LLMClientError("LLM response missing content.")
                return str(content)
            except Exception as exc:
                last_error = exc
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                raise

        raise LLMClientError(f"LLM request failed: {last_error}")
