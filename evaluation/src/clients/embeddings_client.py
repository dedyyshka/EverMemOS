from __future__ import annotations

import asyncio
from typing import List, Optional, Tuple

import numpy as np
from openai import AsyncOpenAI

from agentic_layer.vectorize_interface import UsageInfo, VectorizeError
from evaluation.src.clients.types import EmbeddingsClientConfig, ClientConfigError


class EmbeddingsClient:
    """Единый клиент эмбеддингов для eval (OpenAI‑compatible)."""

    def __init__(self, config: EmbeddingsClientConfig):
        self.config = config
        if not self.config.base_url:
            raise ClientConfigError("Embeddings base_url не задан.")
        if not self.config.model:
            raise ClientConfigError("Embeddings model не задан.")

        self._client = AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
            timeout=self.config.timeout_seconds,
        )
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        # Для deepinfra можно передавать dimensions, для остальных — только client‑side truncation
        self._pass_dimensions = self.config.provider == "deepinfra"
        self._truncate_client_side = self.config.provider in ("neuro", "openrouter")

    async def close(self) -> None:
        await self._client.close()

    async def _make_request(
        self,
        texts: List[str],
        instruction: Optional[str],
        is_query: bool,
    ):
        if not self.config.model:
            raise VectorizeError("Embedding model is not configured.")

        if is_query:
            default_instruction = (
                "Given a search query, retrieve relevant passages that answer the query"
            )
            final_instruction = instruction or default_instruction
            formatted_texts = [
                f"Instruct: {final_instruction}\nQuery: {text}" for text in texts
            ]
        else:
            formatted_texts = texts

        request_kwargs = {
            "model": self.config.model,
            "input": formatted_texts,
            "encoding_format": self.config.encoding_format,
        }

        if self._pass_dimensions and self.config.dimensions > 0:
            request_kwargs["dimensions"] = self.config.dimensions

        async with self._semaphore:
            for attempt in range(self.config.max_retries):
                try:
                    return await self._client.embeddings.create(**request_kwargs)
                except Exception as exc:
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    raise VectorizeError(f"Embeddings request failed: {exc}") from exc

    def _parse_embeddings_response(self, response) -> List[np.ndarray]:
        if not response.data:
            raise VectorizeError("Invalid embeddings response: missing data")

        embeddings = []
        for item in response.data:
            emb = np.array(item.embedding, dtype=np.float32)
            if self._truncate_client_side:
                if (
                    self.config.dimensions
                    and self.config.dimensions > 0
                    and len(emb) > self.config.dimensions
                ):
                    emb = emb[: self.config.dimensions]
            embeddings.append(emb)
        return embeddings

    async def get_embedding(
        self, text: str, instruction: Optional[str] = None, is_query: bool = False
    ) -> np.ndarray:
        response = await self._make_request([text], instruction, is_query)
        return self._parse_embeddings_response(response)[0]

    async def get_embedding_with_usage(
        self, text: str, instruction: Optional[str] = None, is_query: bool = False
    ) -> Tuple[np.ndarray, Optional[UsageInfo]]:
        response = await self._make_request([text], instruction, is_query)
        embeddings = self._parse_embeddings_response(response)
        usage_info = UsageInfo.from_openai_usage(response.usage) if response.usage else None
        return embeddings[0], usage_info

    async def get_embeddings(
        self, texts: List[str], instruction: Optional[str] = None, is_query: bool = False
    ) -> List[np.ndarray]:
        if not texts:
            return []
        if len(texts) <= self.config.batch_size:
            response = await self._make_request(texts, instruction, is_query)
            return self._parse_embeddings_response(response)

        embeddings: List[np.ndarray] = []
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i : i + self.config.batch_size]
            response = await self._make_request(batch_texts, instruction, is_query)
            embeddings.extend(self._parse_embeddings_response(response))
            if i + self.config.batch_size < len(texts):
                await asyncio.sleep(0.1)
        return embeddings

    async def get_embeddings_batch(
        self,
        text_batches: List[List[str]],
        instruction: Optional[str] = None,
        is_query: bool = False,
    ) -> List[List[np.ndarray]]:
        tasks = [
            self.get_embeddings(batch, instruction, is_query) for batch in text_batches
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        embeddings_batches = []
        for result in results:
            if isinstance(result, Exception):
                embeddings_batches.append([])
            else:
                embeddings_batches.append(result)
        return embeddings_batches

    def get_model_name(self) -> str:
        return self.config.model
