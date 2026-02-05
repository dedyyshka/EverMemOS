#!/usr/bin/env python
"""
Smoke-тест клиентов для neuro (LLM/Embeddings/Rerank).
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any, Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from evaluation.src.clients.factory import (  # noqa: E402
    build_embeddings_client,
    build_llm_client,
    build_rerank_client,
)


def load_env_file(env_path: Path) -> None:
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def require_env(keys: List[str]) -> None:
    missing = [key for key in keys if not os.getenv(key)]
    if missing:
        raise RuntimeError(
            "Отсутствуют обязательные переменные окружения: " + ", ".join(missing)
        )


async def run_llm(llm_client) -> None:
    prompt = "Reply with 'ok' only."
    result = await llm_client.generate(prompt=prompt, temperature=0)
    print(f"LLM result: {result[:200]}")


async def run_embeddings(emb_client) -> None:
    text = "hello world"
    emb = await emb_client.get_embedding(text)
    emb_list = list(emb) if emb is not None else []
    print(f"Embeddings: dim={len(emb_list)}, head={emb_list[:5]}")


async def run_rerank(rerank_client) -> None:
    query = "What is machine learning?"
    documents = [
        "Machine learning is a subset of AI.",
        "Python is a programming language.",
        "Deep learning uses neural networks.",
    ]

    doc_result: Dict[str, Any] = await rerank_client.rerank_documents(
        query=query,
        documents=documents,
        instruction="Given a question and a passage, determine relevance.",
    )
    print(f"Rerank documents results: {doc_result.get('results', [])[:3]}")

    hits = []
    for idx, doc in enumerate(documents):
        hits.append(
            {
                "id": f"doc_{idx}",
                "_source": {"episode": doc},
                "memory_type": "episodic_memory",
                "score": 1.0,
            }
        )

    hit_result = await rerank_client.rerank_memories(
        query=query,
        hits=hits,
        top_k=2,
        instruction="Given a question and a passage, determine relevance.",
    )
    top_scores = [h.get("score", 0.0) for h in hit_result]
    print(f"Rerank memories top scores: {top_scores}")


async def main() -> int:
    load_env_file(PROJECT_ROOT / ".env")
    require_env(["LLM_API_KEY", "VECTORIZE_API_KEY", "RERANK_API_KEY"])

    llm_config = {
        "provider": "neuro",
        "base_url": "https://foundation-models.api.cloud.ru/v1",
        "model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "api_key": os.getenv("LLM_API_KEY"),
        "temperature": 0.3,
        "max_tokens": 512,
        "timeout_seconds": 120.0,
        "max_retries": 2,
    }
    embeddings_config = {
        "provider": "neuro",
        "base_url": "http://192.168.0.236:5007/v1",
        "model": "deepvk/USER-bge-m3",
        "api_key": os.getenv("VECTORIZE_API_KEY"),
        "dimensions": 1024,
        "encoding_format": "float",
        "timeout_seconds": 30.0,
        "max_retries": 2,
        "batch_size": 8,
        "max_concurrent_requests": 2,
    }
    rerank_config = {
        "provider": "neuro",
        "base_url": "http://192.168.0.236:5004",
        "model": "BAAI/bge-reranker-v2-m3",
        "api_key": os.getenv("RERANK_API_KEY"),
        "timeout_seconds": 30.0,
        "max_retries": 2,
        "batch_size": 4,
        "max_concurrent_requests": 2,
    }

    llm_client = build_llm_client(llm_config)
    emb_client = build_embeddings_client(embeddings_config)
    rerank_client = build_rerank_client(rerank_config)

    try:
        await run_llm(llm_client)
        await run_embeddings(emb_client)
        await run_rerank(rerank_client)
    finally:
        await llm_client.close()
        await emb_client.close()
        await rerank_client.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
