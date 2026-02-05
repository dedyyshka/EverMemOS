import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Явно включаем UTF-8, чтобы не падать на эмодзи в логах
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Добавляем пути до проекта до любых импортов из src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from common_utils.load_env import setup_environment

from evaluation.src.adapters.registry import create_adapter
from evaluation.src.core.data_models import Conversation, Message
from evaluation.src.utils.config import load_yaml


def _set_vectorize_env_from_config(config: dict) -> None:
    vectorize_cfg = config.get("vectorize", {})
    provider = vectorize_cfg.get("provider", "")

    # В memory_layer поддерживаются только vllm/deepinfra, маппим neuro->vllm
    provider_map = {
        "neuro": "vllm",
        "openrouter": "vllm",
        "deepinfra": "deepinfra",
    }
    mapped_provider = provider_map.get(provider, provider)

    os.environ["VECTORIZE_PROVIDER"] = mapped_provider
    os.environ["VECTORIZE_MODEL"] = str(vectorize_cfg.get("model", ""))
    os.environ["VECTORIZE_BASE_URL"] = str(vectorize_cfg.get("base_url", ""))
    os.environ["VECTORIZE_API_KEY"] = str(vectorize_cfg.get("api_key", ""))
    os.environ["VECTORIZE_DIMENSIONS"] = str(vectorize_cfg.get("dimensions", 0))
    os.environ["VECTORIZE_FALLBACK_PROVIDER"] = "none"


def _build_smoke_conversation() -> Conversation:
    base_time = datetime.utcnow()
    messages = [
        Message(
            speaker_id="alice",
            speaker_name="Alice",
            content="Привет, Боб! Я обожаю суши и рамен.",
            timestamp=base_time,
            metadata={},
        ),
        Message(
            speaker_id="bob",
            speaker_name="Bob",
            content="Здорово! А я люблю пиццу и пасту.",
            timestamp=base_time + timedelta(seconds=30),
            metadata={},
        ),
        Message(
            speaker_id="alice",
            speaker_name="Alice",
            content="Кстати, мой любимый ресторан — Sakura.",
            timestamp=base_time + timedelta(seconds=60),
            metadata={},
        ),
        Message(
            speaker_id="bob",
            speaker_name="Bob",
            content="Хорошо, давай как-нибудь зайдем туда вместе.",
            timestamp=base_time + timedelta(seconds=90),
            metadata={},
        ),
    ]

    return Conversation(
        conversation_id="smoke_0",
        messages=messages,
        metadata={"speaker_a": "Alice", "speaker_b": "Bob"},
    )


async def run_smoke() -> None:
    project_root = PROJECT_ROOT
    setup_environment(load_env_file_name=".env", check_env_var="MONGODB_HOST")

    system_config_path = (
        project_root / "evaluation" / "config" / "systems" / "evermemos.yaml"
    )
    system_config = load_yaml(str(system_config_path))

    # Лёгкий режим поиска (без agentic‑LLM и rerank), но с embeddings+BM25
    system_config.setdefault("search", {})
    system_config["search"]["mode"] = "lightweight"
    system_config["search"]["lightweight_search_mode"] = "hybrid"
    system_config["dataset_name"] = "smoke"

    # env-поддержка для memory_layer (event_log embeddings)
    _set_vectorize_env_from_config(system_config)

    output_dir = project_root / "evaluation" / "results" / "smoke-evermemos"
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter = create_adapter(system_config["adapter"], system_config, output_dir=output_dir)

    try:
        conversation = _build_smoke_conversation()
        index_metadata = await adapter.add([conversation], output_dir=output_dir)

        search_result = await adapter.search(
            query="Что любит есть Алиса?",
            conversation_id=conversation.conversation_id,
            index=index_metadata,
            conversation=conversation,
        )

        print("✅ Smoke add/search завершён")
        print(f"   Найдено результатов: {len(search_result.results)}")
        if search_result.results:
            print(f"   Top1 score: {search_result.results[0].get('score')}")
    finally:
        if hasattr(adapter, "close") and callable(getattr(adapter, "close")):
            await adapter.close()


if __name__ == "__main__":
    asyncio.run(run_smoke())
