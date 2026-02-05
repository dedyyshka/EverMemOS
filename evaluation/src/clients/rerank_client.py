from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

import httpx

from agentic_layer.rerank_deepinfra import DeepInfraRerankConfig, DeepInfraRerankService
from agentic_layer.rerank_interface import RerankError
from api_specs.memory_models import MemoryType
from evaluation.src.clients.types import RerankClientConfig, ClientConfigError


class RerankClient:
    """Единый клиент rerank для eval."""

    def __init__(self, config: RerankClientConfig):
        self.config = config
        if not self.config.base_url:
            raise ClientConfigError("Rerank base_url не задан.")
        if not self.config.model:
            raise ClientConfigError("Rerank model не задан.")

        self._mode = "deepinfra" if self.config.provider == "deepinfra" else "tei"
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)

        if self._mode == "deepinfra":
            di_config = DeepInfraRerankConfig(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                model=self.config.model,
                timeout=int(self.config.timeout_seconds),
                max_retries=self.config.max_retries,
                batch_size=self.config.batch_size,
                max_concurrent_requests=self.config.max_concurrent_requests,
            )
            self._deepinfra = DeepInfraRerankService(di_config)
            self._client = None
        else:
            self._deepinfra = None
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url.rstrip("/"),
                timeout=self.config.timeout_seconds,
                verify=False,
            )

    async def close(self) -> None:
        if self._deepinfra:
            await self._deepinfra.close()
        if self._client:
            await self._client.aclose()

    def _build_headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.config.api_key:
            headers["Authorization"] = f"Bearer {self.config.api_key}"
        return headers

    def _extract_text_from_hit(self, hit: Dict[str, Any]) -> str:
        """Извлечь текст из hit по типу памяти."""
        source = hit.get("_source", hit)
        memory_type = hit.get("memory_type", "")

        match memory_type:
            case MemoryType.EPISODIC_MEMORY.value:
                episode = source.get("episode", "")
                if episode:
                    return f"Episode Memory: {episode}"
            case MemoryType.FORESIGHT.value:
                foresight = source.get("foresight", "") or source.get("content", "")
                evidence = source.get("evidence", "")
                if foresight:
                    if evidence:
                        return f"Foresight: {foresight} (Evidence: {evidence})"
                    return f"Foresight: {foresight}"
            case MemoryType.EVENT_LOG.value:
                atomic_fact = source.get("atomic_fact", "")
                if atomic_fact:
                    return f"Atomic Fact: {atomic_fact}"

        if source.get("episode"):
            return source["episode"]
        if source.get("atomic_fact"):
            return source["atomic_fact"]
        if source.get("foresight"):
            return source["foresight"]
        if source.get("content"):
            return source["content"]
        if source.get("summary"):
            return source["summary"]
        if source.get("subject"):
            return source["subject"]
        return str(hit)

    def _normalize_scores(self, scores: List[float], num_documents: int) -> List[Dict[str, Any]]:
        if len(scores) < num_documents:
            scores.extend([0.0] * (num_documents - len(scores)))
        scores = scores[:num_documents]

        indexed_scores = [(i, score) for i, score in enumerate(scores)]
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results = []
        for rank, (original_index, score) in enumerate(indexed_scores):
            results.append({"index": original_index, "score": score, "rank": rank})
        return results

    async def _tei_rerank_batch(
        self,
        query: str,
        documents: List[str],
        instruction: Optional[str],
    ) -> List[float]:
        if not self._client:
            raise RerankError("Rerank HTTP client is not initialized.")

        query_text = f"{instruction}\n\n{query}" if instruction else query
        payload = {
            "query": query_text,
            "texts": documents,
        }
        headers = self._build_headers()

        async with self._semaphore:
            for attempt in range(self.config.max_retries):
                try:
                    response = await self._client.post("/rerank", json=payload, headers=headers)
                    if response.status_code >= 400:
                        raise RerankError(f"HTTP {response.status_code}: {response.text}")

                    data = response.json()
                    if isinstance(data, list):
                        scores = [0.0] * len(documents)
                        for item in data:
                            idx = item.get("index")
                            score = item.get("score", item.get("relevance_score", 0.0))
                            if idx is not None and 0 <= idx < len(scores):
                                scores[idx] = score
                        return scores
                    if "results" in data:
                        scores = [0.0] * len(documents)
                        for item in data["results"]:
                            idx = item.get("index")
                            score = item.get("score", item.get("relevance_score", 0.0))
                            if idx is not None and 0 <= idx < len(scores):
                                scores[idx] = score
                        return scores
                    if "scores" in data:
                        return list(data["scores"])

                    raise RerankError("Invalid rerank response format.")
                except Exception as exc:
                    if attempt < self.config.max_retries - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    raise RerankError(f"Rerank request failed: {exc}") from exc

        return [0.0] * len(documents)

    async def rerank_documents(
        self, query: str, documents: List[str], instruction: Optional[str] = None
    ) -> Dict[str, Any]:
        if not documents:
            return {"results": []}

        if self._mode == "deepinfra":
            return await self._deepinfra.rerank_documents(query, documents, instruction)

        batch_size = self.config.batch_size if self.config.batch_size > 0 else 10
        batches = [
            documents[i : i + batch_size] for i in range(0, len(documents), batch_size)
        ]

        tasks = [
            self._tei_rerank_batch(query, batch, instruction) for batch in batches
        ]
        results = await asyncio.gather(*tasks)

        all_scores: List[float] = []
        for scores in results:
            all_scores.extend(scores)

        return {"results": self._normalize_scores(all_scores, len(documents))}

    async def rerank_memories(
        self,
        query: str,
        hits: List[Dict[str, Any]],
        top_k: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if not hits:
            return []

        documents = [self._extract_text_from_hit(hit) for hit in hits]
        if not documents:
            return []

        result = await self.rerank_documents(query, documents, instruction)
        if "results" not in result:
            raise RerankError("Invalid rerank response format.")

        score_map = {}
        for item in result["results"]:
            index = item.get("index")
            score = item.get("score", item.get("relevance_score", 0.0))
            if index is not None:
                score_map[index] = score

        reranked_hits = []
        for i, hit in enumerate(hits):
            if i in score_map:
                hit_copy = hit.copy()
                hit_copy["score"] = score_map[i]
                reranked_hits.append(hit_copy)

        reranked_hits.sort(key=lambda x: x.get("score", 0.0), reverse=True)

        if top_k is not None and top_k > 0:
            reranked_hits = reranked_hits[:top_k]

        return reranked_hits

    def get_model_name(self) -> str:
        return self.config.model
