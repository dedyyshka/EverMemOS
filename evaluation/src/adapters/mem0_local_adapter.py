"""
Mem0 Local Adapter - adapt mem0 library (local) for evaluation framework.
"""

from __future__ import annotations

import asyncio
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from rich.console import Console

from evaluation.src.adapters.online_base import OnlineAPIAdapter
from evaluation.src.adapters.registry import register_adapter
from evaluation.src.core.data_models import Conversation, SearchResult
from evaluation.src.clients.factory import build_rerank_client
from evaluation.src.adapters.mem0_local_compat import build_mem0_config


@register_adapter("mem0_local")
class Mem0LocalAdapter(OnlineAPIAdapter):
    """
    Mem0 local adapter (mem0ai).
    """

    def __init__(self, config: dict, output_dir: Path = None):
        super().__init__(config, output_dir)

        try:
            from mem0 import AsyncMemory
        except ImportError:
            raise ImportError(
                "Mem0 client not installed. Please install: pip install mem0ai"
            )

        self._mem0_cls = AsyncMemory
        self._memory = None
        self._memory_lock = asyncio.Lock()

        self.console = Console()

        add_cfg = config.get("add", {})
        search_cfg = config.get("search", {})

        self.batch_size = int(add_cfg.get("batch_size", 2))
        self.max_retries = int(add_cfg.get("max_retries", 5))
        self.max_content_length = int(add_cfg.get("max_content_length", 12000))
        self.add_interval = float(add_cfg.get("add_interval", 0.0))
        self.search_interval = float(search_cfg.get("search_interval", 0.0))
        self.post_add_wait_seconds = float(add_cfg.get("post_add_wait_seconds", 0.0))
        self.clean_before_add = bool(add_cfg.get("clean_before_add", False))

        rerank_cfg = config.get("rerank", {})
        self.rerank_client = build_rerank_client(rerank_cfg) if rerank_cfg else None

        mem0_settings = dict(config.get("mem0", {}) or {})
        self._mem0_settings = mem0_settings
        if mem0_settings.get("suppress_logs", True):
            for logger_name in (
                "mem0",
                "mem0.memory",
                "mem0.vector_stores",
                "mem0.vector_stores.qdrant",
            ):
                logging.getLogger(logger_name).setLevel(logging.WARNING)
        persist_dir = mem0_settings.get("persist_dir")
        if not persist_dir:
            persist_dir = str(self.output_dir / "mem0_storage")
            mem0_settings["persist_dir"] = persist_dir

        namespace = mem0_settings.get("namespace")
        if not namespace:
            namespace = self._derive_namespace()
            mem0_settings["namespace"] = namespace

        os.makedirs(persist_dir, exist_ok=True)

        self._mem0_config = build_mem0_config(
            mem0_settings=mem0_settings,
            llm_config=config.get("llm", {}),
            vectorize_config=config.get("vectorize", {}),
        )

        print(f"   Mem0 persist_dir: {persist_dir}")
        print(f"   Mem0 namespace: {namespace}")
    def _derive_namespace(self) -> str:
        dataset_name = self.config.get("dataset_name") or "dataset"
        system_name = self.config.get("adapter") or "mem0_local"
        output_name = self.output_dir.name
        prefix = f"{dataset_name}-{system_name}"
        if output_name.startswith(prefix):
            suffix = output_name[len(prefix):].lstrip("-")
            if suffix:
                return f"{dataset_name}-{suffix}"
        return dataset_name

    async def _get_memory(self):
        if self._memory is None:
            async with self._memory_lock:
                if self._memory is None:
                    self._memory = await self._mem0_cls.from_config(self._mem0_config)
        return self._memory

    def _convert_timestamp_to_display_timezone(self, timestamp_str: str) -> str:
        if not timestamp_str:
            return timestamp_str
        try:
            dt = datetime.fromisoformat(timestamp_str)
            dt_display = dt.astimezone(None)
            return dt_display.strftime("%Y-%m-%d %H:%M:%S")
        except Exception as exc:
            self.console.print(
                f"⚠️  Failed to convert timestamp '{timestamp_str}': {exc}",
                style="yellow",
            )
            return timestamp_str

    async def prepare(self, conversations: List[Conversation], **kwargs) -> None:
        if not self.clean_before_add:
            self.console.print("   ⏭️  Skipping data cleanup (clean_before_add=false)", style="dim")
            return

        memory = await self._get_memory()
        user_ids_to_clean = set()
        for conv in conversations:
            speaker_a = conv.metadata.get("speaker_a", "")
            speaker_b = conv.metadata.get("speaker_b", "")
            need_dual = self._need_dual_perspective(speaker_a, speaker_b)
            user_ids_to_clean.add(self._extract_user_id(conv, speaker="speaker_a"))
            if need_dual:
                user_ids_to_clean.add(self._extract_user_id(conv, speaker="speaker_b"))

        self.console.print(f"\n🗑️  Cleaning data for {len(user_ids_to_clean)} user(s)...", style="yellow")
        cleaned_count = 0
        failed_count = 0

        for user_id in user_ids_to_clean:
            try:
                await memory.delete_all(user_id=user_id)
                cleaned_count += 1
                self.console.print(f"   ✅ Cleaned: {user_id}", style="green")
            except Exception as exc:
                failed_count += 1
                self.console.print(f"   ⚠️  Failed to clean {user_id}: {exc}", style="yellow")

        self.console.print(
            f"\n✅ Cleanup completed: {cleaned_count} succeeded, {failed_count} failed",
            style="bold green",
        )

    async def _add_user_messages(
        self,
        conv: Conversation,
        messages: List[Dict[str, Any]],
        speaker: str,
        **kwargs,
    ) -> Any:
        progress = kwargs.get("progress")
        task_id = kwargs.get("task_id")
        output_console = progress.console if progress is not None else self.console
        user_id = self._extract_user_id(conv, speaker=speaker)

        truncated_count = 0
        for msg in messages:
            if len(msg["content"]) > self.max_content_length:
                msg["content"] = msg["content"][: self.max_content_length]
                truncated_count += 1

        speaker_name = conv.metadata.get(speaker, speaker)
        is_fake_timestamp = conv.messages[0].metadata.get("is_fake_timestamp", False) if conv.messages else False

        output_console.print(
            f"   📤 Adding for {speaker_name} ({user_id}): {len(messages)} messages",
            style="dim",
        )
        if is_fake_timestamp:
            output_console.print("   ⚠️  Using fake timestamp", style="yellow")
        if truncated_count > 0:
            output_console.print(
                f"   ⚠️  Truncated {truncated_count} messages (>{self.max_content_length} chars)",
                style="yellow",
            )

        memory = await self._get_memory()

        for i in range(0, len(messages), self.batch_size):
            batch_messages = messages[i : i + self.batch_size]
            metadata = None
            if i < len(conv.messages) and conv.messages[i].timestamp:
                metadata = {"created_at": conv.messages[i].timestamp.isoformat()}
            for attempt in range(self.max_retries):
                try:
                    await memory.add(
                        messages=batch_messages,
                        user_id=user_id,
                        metadata=metadata,
                    )
                    if progress is not None and task_id is not None:
                        progress.update(task_id, advance=len(batch_messages))
                    if self.add_interval > 0:
                        await asyncio.sleep(self.add_interval)
                    break
                except Exception as exc:
                    if attempt < self.max_retries - 1:
                        output_console.print(
                            f"   ⚠️  [{speaker_name} (user_id={user_id})] Retry {attempt + 1}/{self.max_retries}: {exc}",
                            style="yellow",
                        )
                        await asyncio.sleep(2**attempt)
                    else:
                        output_console.print(
                            f"   ❌ [{speaker_name} (user_id={user_id})] Failed after {self.max_retries} retries: {exc}",
                            style="red",
                        )
                        raise exc

        return None

    async def _post_add_process(self, add_results: List[Any], **kwargs) -> None:
        if self.post_add_wait_seconds > 0:
            await asyncio.sleep(self.post_add_wait_seconds)
    async def _search_single_user(
        self,
        query: str,
        conversation_id: str,
        user_id: str,
        top_k: int,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        if self.search_interval > 0:
            await asyncio.sleep(self.search_interval)

        memory = await self._get_memory()

        try:
            raw_results = await memory.search(
                query=query,
                user_id=user_id,
                limit=top_k,
                rerank=False,
            )
        except Exception as exc:
            self.console.print(f"❌ Mem0 local search error: {exc}", style="red")
            return []

        results = []
        for memory_item in raw_results.get("results", []):
            created_at_original = memory_item.get("created_at", "")
            created_at_display = self._convert_timestamp_to_display_timezone(created_at_original)
            content = memory_item.get("memory", "")
            if created_at_display:
                content = f"{created_at_display}: {content}"

            results.append(
                {
                    "content": content,
                    "score": memory_item.get("score", 0.0),
                    "user_id": user_id,
                    "metadata": {
                        "id": memory_item.get("id", ""),
                        "created_at": created_at_original,
                        "created_at_display": created_at_display,
                        "memory": memory_item.get("memory", ""),
                        "user_id": memory_item.get("user_id", ""),
                    },
                }
            )

        return await self._rerank_results(query, results, top_k)

    async def _rerank_results(
        self, query: str, results: List[Dict[str, Any]], top_k: int
    ) -> List[Dict[str, Any]]:
        if not self.rerank_client or not results:
            return results

        try:
            reranked = await self.rerank_client.rerank_memories(
                query=query, hits=results, top_k=top_k
            )
            return reranked or results
        except Exception as exc:
            self.console.print(f"⚠️  Rerank failed, using original results: {exc}", style="yellow")
            return results

    def _build_single_search_result(
        self,
        query: str,
        conversation_id: str,
        results: List[Dict[str, Any]],
        user_id: str,
        top_k: int,
        **kwargs,
    ) -> SearchResult:
        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=results,
            retrieval_metadata={
                "system": "mem0_local",
                "top_k": top_k,
                "dual_perspective": False,
                "user_ids": [user_id],
            },
        )

    def _build_dual_search_result(
        self,
        query: str,
        conversation_id: str,
        all_results: List[Dict[str, Any]],
        results_a: List[Dict[str, Any]],
        results_b: List[Dict[str, Any]],
        speaker_a: str,
        speaker_b: str,
        speaker_a_user_id: str,
        speaker_b_user_id: str,
        top_k: int,
        **kwargs,
    ) -> SearchResult:
        speaker_a_memories_text = "\n".join([r["content"] for r in results_a]) if results_a else "(No memories found)"
        speaker_b_memories_text = "\n".join([r["content"] for r in results_b]) if results_b else "(No memories found)"

        template = self._prompts["online_api"].get("templates", {}).get("default", "")
        formatted_context = template.format(
            speaker_1=speaker_a,
            speaker_1_memories=speaker_a_memories_text,
            speaker_2=speaker_b,
            speaker_2_memories=speaker_b_memories_text,
        )

        return SearchResult(
            query=query,
            conversation_id=conversation_id,
            results=all_results,
            retrieval_metadata={
                "system": "mem0_local",
                "top_k": top_k,
                "dual_perspective": True,
                "user_ids": [speaker_a_user_id, speaker_b_user_id],
                "formatted_context": formatted_context,
                "speaker_a_memories_count": len(results_a),
                "speaker_b_memories_count": len(results_b),
            },
        )

    def _get_answer_prompt(self) -> str:
        return self._prompts["online_api"]["default"]["answer_prompt_mem0"]

    def get_system_info(self) -> Dict[str, Any]:
        return {
            "name": "Mem0Local",
            "type": "local",
            "description": "Mem0 local adapter (mem0ai)",
            "adapter": "Mem0LocalAdapter",
        }
