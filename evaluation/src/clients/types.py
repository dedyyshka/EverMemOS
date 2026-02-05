from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


ProviderName = Literal["neuro", "openrouter", "deepinfra"]


class ClientConfigError(ValueError):
    """Ошибка в конфигурации клиента."""


@dataclass
class LLMClientConfig:
    provider: ProviderName
    base_url: str
    model: str
    api_key: str
    temperature: float = 0.3
    max_tokens: int | None = None
    timeout_seconds: float = 120.0
    max_retries: int = 3
    openrouter_provider: str | None = None


@dataclass
class EmbeddingsClientConfig:
    provider: ProviderName
    base_url: str
    model: str
    api_key: str
    dimensions: int = 0
    encoding_format: str = "float"
    timeout_seconds: float = 30.0
    max_retries: int = 3
    batch_size: int = 10
    max_concurrent_requests: int = 5


@dataclass
class RerankClientConfig:
    provider: ProviderName
    base_url: str
    model: str
    api_key: str
    timeout_seconds: float = 30.0
    max_retries: int = 3
    batch_size: int = 10
    max_concurrent_requests: int = 5
