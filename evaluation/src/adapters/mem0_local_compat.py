from __future__ import annotations

import os
from typing import Any, Dict, Optional


def _clean_dict(values: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def map_provider_for_mem0(
    provider: Optional[str],
    base_url: Optional[str],
    override: Optional[str] = None,
) -> str:
    if override:
        return override

    provider_name = (provider or "").strip().lower()
    if provider_name == "vllm":
        return "vllm"
    if provider_name in {"openai", "openrouter", "neuro", "deepinfra"}:
        return "openai"

    if base_url and "vllm" in base_url.lower():
        return "vllm"

    return "openai"


def build_mem0_llm_config(
    llm_config: Dict[str, Any],
    *,
    provider_override: Optional[str] = None,
) -> Dict[str, Any]:
    provider = map_provider_for_mem0(
        llm_config.get("provider"),
        llm_config.get("base_url"),
        provider_override,
    )
    base_fields = _clean_dict(
        {
            "model": llm_config.get("model"),
            "api_key": llm_config.get("api_key"),
            "temperature": llm_config.get("temperature"),
            "max_tokens": llm_config.get("max_tokens"),
            "top_p": llm_config.get("top_p"),
            "top_k": llm_config.get("top_k"),
            "enable_vision": llm_config.get("enable_vision"),
            "vision_details": llm_config.get("vision_details"),
        }
    )

    if provider == "vllm":
        config = dict(base_fields)
        config["vllm_base_url"] = llm_config.get("base_url")
        return {"provider": provider, "config": _clean_dict(config)}

    config = dict(base_fields)
    config["openai_base_url"] = llm_config.get("base_url")
    return {"provider": "openai", "config": _clean_dict(config)}


def build_mem0_embeddings_config(
    vectorize_config: Dict[str, Any],
    *,
    provider_override: Optional[str] = None,
) -> Dict[str, Any]:
    provider = map_provider_for_mem0(
        vectorize_config.get("provider"),
        vectorize_config.get("base_url"),
        provider_override,
    )
    if provider != "openai":
        provider = "openai"

    config = _clean_dict(
        {
            "model": vectorize_config.get("model"),
            "api_key": vectorize_config.get("api_key"),
            "openai_base_url": vectorize_config.get("base_url"),
            "embedding_dims": vectorize_config.get("dimensions"),
        }
    )
    return {"provider": provider, "config": config}


def build_mem0_vector_store_config(
    *,
    persist_dir: str,
    namespace: str,
    embedding_dims: int,
    qdrant_host: Optional[str] = None,
    qdrant_port: Optional[int] = None,
    qdrant_url: Optional[str] = None,
    qdrant_api_key: Optional[str] = None,
) -> Dict[str, Any]:
    qdrant_path = os.path.join(persist_dir, "qdrant")
    vector_store_config: Dict[str, Any] = {
        "provider": "qdrant",
        "config": {
            "collection_name": namespace,
            "embedding_model_dims": embedding_dims,
        },
    }
    if qdrant_host and qdrant_port:
        vector_store_config["config"]["host"] = qdrant_host
        vector_store_config["config"]["port"] = qdrant_port
        return vector_store_config
    if qdrant_url and qdrant_api_key:
        vector_store_config["config"]["url"] = qdrant_url
        vector_store_config["config"]["api_key"] = qdrant_api_key
        return vector_store_config

    vector_store_config["config"]["path"] = qdrant_path
    vector_store_config["config"]["on_disk"] = True
    return vector_store_config


def build_mem0_config(
    *,
    mem0_settings: Dict[str, Any],
    llm_config: Dict[str, Any],
    vectorize_config: Dict[str, Any],
) -> Dict[str, Any]:
    persist_dir = mem0_settings.get("persist_dir", ".")
    namespace = mem0_settings.get("namespace", "mem0_local")
    embedding_dims = vectorize_config.get("dimensions")
    if embedding_dims is None:
        raise ValueError("vectorize.dimensions обязателен для mem0 local.")

    llm_override = mem0_settings.get("llm_provider_override") or None
    embeddings_override = mem0_settings.get("embeddings_provider_override") or None

    history_db_path = mem0_settings.get("history_db_path") or os.path.join(
        persist_dir, "history.db"
    )
    qdrant_host = mem0_settings.get("qdrant_host")
    qdrant_port = mem0_settings.get("qdrant_port")
    qdrant_url = mem0_settings.get("qdrant_url")
    qdrant_api_key = mem0_settings.get("qdrant_api_key")

    config: Dict[str, Any] = {
        "llm": build_mem0_llm_config(llm_config, provider_override=llm_override),
        "embedder": build_mem0_embeddings_config(
            vectorize_config, provider_override=embeddings_override
        ),
        "vector_store": build_mem0_vector_store_config(
            persist_dir=persist_dir,
            namespace=namespace,
            embedding_dims=embedding_dims,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_api_key,
        ),
        "history_db_path": history_db_path,
    }

    return config


def apply_mem0_runtime_patch() -> None:
    """
    Резерв: патчи mem0 на случай жёсткой валидации.
    Сейчас не используется.
    """
    return None
