#!/usr/bin/env python
"""
Сбор мета-информации по LLM/Embeddings/Reranker API для последующей интеграции.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Конфигурация API фиксируется в тесте (провайдеры не указываем).
LLM_CONFIG = {
    "base_url": "https://foundation-models.api.cloud.ru/v1",
    "model": "Qwen/Qwen3-Next-80B-A3B-Instruct",
}

EMBEDDINGS_CONFIG = {
    "base_url": "http://192.168.0.236:5007/v1",
    "model": "deepvk/USER-bge-m3",
    "dimensions": 1024,
}

RERANK_CONFIG = {
    "base_url": "http://192.168.0.236:5004",
    "model": "BAAI/bge-reranker-v2-m3",
}


def require_env(keys: list[str]) -> None:
    missing = [key for key in keys if not os.getenv(key)]
    if missing:
        print(
            "❌ Отсутствуют обязательные переменные окружения: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        sys.exit(1)


def build_url(base_url: str, path: str) -> str:
    base = base_url.rstrip("/")
    suffix = path.lstrip("/")
    return f"{base}/{suffix}"


def probe_get(url: str, headers: Dict[str, str], timeout: float) -> Dict[str, Any]:
    start = time.perf_counter()
    try:
        import ssl
        import urllib.request

        request = urllib.request.Request(url, headers=headers, method="GET")
        ssl_context = ssl._create_unverified_context()
        response = urllib.request.urlopen(request, timeout=timeout, context=ssl_context)
        latency_ms = (time.perf_counter() - start) * 1000.0
        status_code = getattr(response, "status", None)
        raw_body = response.read()
        text = raw_body.decode("utf-8", errors="replace")
        result: Dict[str, Any] = {
            "url": url,
            "status_code": status_code,
            "ok": status_code is not None and 200 <= status_code < 300,
            "latency_ms": round(latency_ms, 2),
        }
        try:
            result["json"] = json.loads(text)
        except Exception:
            result["text_preview"] = text[:2000]
        return result
    except Exception as exc:
        latency_ms = (time.perf_counter() - start) * 1000.0
        return {
            "url": url,
            "ok": False,
            "latency_ms": round(latency_ms, 2),
            "error": str(exc),
        }


def collect_section(
    name: str,
    provider: str | None,
    base_url: str,
    api_key: str | None,
    model: str | None,
    extra: Dict[str, Any] | None,
    timeout: float,
) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    endpoints = {
        "health": build_url(base_url, "health"),
        "models": build_url(base_url, "models"),
        "info": build_url(base_url, "info"),
    }

    return {
        "name": name,
        "config": {
            "base_url": base_url,
            "model": model,
            "api_key_present": bool(api_key),
            **(extra or {}),
        },
        "endpoints": {
            "health": probe_get(endpoints["health"], headers, timeout),
            "models": probe_get(endpoints["models"], headers, timeout),
            "info": probe_get(endpoints["info"], headers, timeout),
        },
    }


def build_output_path(output_arg: str | None) -> Path:
    if output_arg:
        return Path(output_arg).expanduser()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "outputs" / f"neuro_api_meta_{timestamp}.json"


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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Collect meta info from LLM/Embeddings/Reranker APIs."
    )
    parser.add_argument(
        "--output",
        help="Путь для сохранения отчёта (JSON). По умолчанию outputs/neuro_api_meta_*.json",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Таймаут запросов (секунды). По умолчанию 15.",
    )
    parser.add_argument(
        "--no-dotenv",
        action="store_true",
        help="Не загружать .env из корня проекта.",
    )
    args = parser.parse_args()

    if not args.no_dotenv:
        load_env_file(PROJECT_ROOT / ".env")

    require_env(["LLM_API_KEY", "VECTORIZE_API_KEY", "RERANK_API_KEY"])

    llm_api_key = os.getenv("LLM_API_KEY")
    vectorize_api_key = os.getenv("VECTORIZE_API_KEY")
    rerank_api_key = os.getenv("RERANK_API_KEY")

    report: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "project_root": str(PROJECT_ROOT),
        "llm": collect_section(
            name="llm",
            provider=None,
            base_url=LLM_CONFIG["base_url"],
            api_key=llm_api_key,
            model=LLM_CONFIG["model"],
            extra=None,
            timeout=args.timeout,
        ),
        "embeddings": collect_section(
            name="embeddings",
            provider=None,
            base_url=EMBEDDINGS_CONFIG["base_url"],
            api_key=vectorize_api_key,
            model=EMBEDDINGS_CONFIG["model"],
            extra={"dimensions": EMBEDDINGS_CONFIG["dimensions"]},
            timeout=args.timeout,
        ),
        "reranker": collect_section(
            name="reranker",
            provider=None,
            base_url=RERANK_CONFIG["base_url"],
            api_key=rerank_api_key,
            model=RERANK_CONFIG["model"],
            extra=None,
            timeout=args.timeout,
        ),
    }

    output_path = build_output_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"OK: Отчёт сохранён: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
