from __future__ import annotations

from typing import Any, Dict

from evaluation.src.clients.embeddings_client import EmbeddingsClient
from evaluation.src.clients.llm_client import LLMClient
from evaluation.src.clients.rerank_client import RerankClient
from evaluation.src.clients.types import (
    ClientConfigError,
    EmbeddingsClientConfig,
    LLMClientConfig,
    ProviderName,
    RerankClientConfig,
)


ALLOWED_PROVIDERS = {"neuro", "openrouter", "deepinfra"}


def _require(config: Dict[str, Any], key: str, ctx: str) -> Any:
    value = config.get(key)
    if value is None or value == "":
        raise ClientConfigError(f"{ctx}: отсутствует обязательный параметр '{key}'.")
    return value


def _provider(config: Dict[str, Any], ctx: str) -> ProviderName:
    provider = _require(config, "provider", ctx)
    if provider not in ALLOWED_PROVIDERS:
        raise ClientConfigError(
            f"{ctx}: недопустимый provider '{provider}'. "
            f"Разрешены: {', '.join(sorted(ALLOWED_PROVIDERS))}."
        )
    return provider  # type: ignore[return-value]


def build_llm_client(config: Dict[str, Any]) -> LLMClient:
    provider = _provider(config, "LLM")
    cfg = LLMClientConfig(
        provider=provider,
        base_url=_require(config, "base_url", "LLM"),
        model=_require(config, "model", "LLM"),
        api_key=_require(config, "api_key", "LLM"),
        temperature=config.get("temperature", 0.3),
        max_tokens=config.get("max_tokens"),
        timeout_seconds=config.get("timeout_seconds", 120.0),
        max_retries=config.get("max_retries", 3),
        openrouter_provider=config.get("openrouter_provider"),
    )
    return LLMClient(cfg)


def build_embeddings_client(config: Dict[str, Any]) -> EmbeddingsClient:
    provider = _provider(config, "Embeddings")
    cfg = EmbeddingsClientConfig(
        provider=provider,
        base_url=_require(config, "base_url", "Embeddings"),
        model=_require(config, "model", "Embeddings"),
        api_key=_require(config, "api_key", "Embeddings"),
        dimensions=config.get("dimensions", 0),
        encoding_format=config.get("encoding_format", "float"),
        timeout_seconds=config.get("timeout_seconds", 30.0),
        max_retries=config.get("max_retries", 3),
        batch_size=config.get("batch_size", 10),
        max_concurrent_requests=config.get("max_concurrent_requests", 5),
    )
    return EmbeddingsClient(cfg)


def build_rerank_client(config: Dict[str, Any]) -> RerankClient:
    provider = _provider(config, "Rerank")
    cfg = RerankClientConfig(
        provider=provider,
        base_url=_require(config, "base_url", "Rerank"),
        model=_require(config, "model", "Rerank"),
        api_key=_require(config, "api_key", "Rerank"),
        timeout_seconds=config.get("timeout_seconds", 30.0),
        max_retries=config.get("max_retries", 3),
        batch_size=config.get("batch_size", 10),
        max_concurrent_requests=config.get("max_concurrent_requests", 5),
    )
    return RerankClient(cfg)
