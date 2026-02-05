from __future__ import annotations

import sys
from pathlib import Path


def _apply_patch(text: str) -> tuple[str, bool]:
    start_marker = "        returned_memories = []\n        try:\n            memory_tasks = []\n"
    end_marker = '        except Exception as e:\n            logger.error(f"Error in memory processing loop (async): {e}")\n'

    start_idx = text.find(start_marker)
    if start_idx == -1:
        return text, False

    end_idx = text.find(end_marker, start_idx)
    if end_idx == -1:
        return text, False

    end_idx += len(end_marker)

    replacement = (
        "        returned_memories = []\n"
        "        try:\n"
        "            for resp in new_memories_with_actions.get(\"memory\", []):\n"
        "                logger.info(resp)\n"
        "                try:\n"
        "                    action_text = resp.get(\"text\")\n"
        "                    if not action_text:\n"
        "                        continue\n"
        "                    event_type = resp.get(\"event\")\n"
        "\n"
        "                    if event_type == \"ADD\":\n"
        "                        result_id = await self._create_memory(\n"
        "                            data=action_text,\n"
        "                            existing_embeddings=new_message_embeddings,\n"
        "                            metadata=deepcopy(metadata),\n"
        "                        )\n"
        "                        returned_memories.append({\"id\": result_id, \"memory\": action_text, \"event\": event_type})\n"
        "                    elif event_type == \"UPDATE\":\n"
        "                        result_id = await self._update_memory(\n"
        "                            memory_id=temp_uuid_mapping[resp[\"id\"]],\n"
        "                            data=action_text,\n"
        "                            existing_embeddings=new_message_embeddings,\n"
        "                            metadata=deepcopy(metadata),\n"
        "                        )\n"
        "                        returned_memories.append(\n"
        "                            {\n"
        "                                \"id\": result_id,\n"
        "                                \"memory\": action_text,\n"
        "                                \"event\": event_type,\n"
        "                                \"previous_memory\": resp.get(\"old_memory\"),\n"
        "                            }\n"
        "                        )\n"
        "                    elif event_type == \"DELETE\":\n"
        "                        result_id = await self._delete_memory(\n"
        "                            memory_id=temp_uuid_mapping[resp.get(\"id\")]\n"
        "                        )\n"
        "                        returned_memories.append({\"id\": result_id, \"memory\": action_text, \"event\": event_type})\n"
        "                    elif event_type == \"NONE\":\n"
        "                        memory_id = temp_uuid_mapping.get(resp.get(\"id\"))\n"
        "                        if memory_id and (metadata.get(\"agent_id\") or metadata.get(\"run_id\")):\n"
        "                            async def update_session_ids(mem_id, meta):\n"
        "                                existing_memory = await asyncio.to_thread(self.vector_store.get, vector_id=mem_id)\n"
        "                                updated_metadata = deepcopy(existing_memory.payload)\n"
        "                                if meta.get(\"agent_id\"):\n"
        "                                    updated_metadata[\"agent_id\"] = meta[\"agent_id\"]\n"
        "                                if meta.get(\"run_id\"):\n"
        "                                    updated_metadata[\"run_id\"] = meta[\"run_id\"]\n"
        "                                updated_metadata[\"updated_at\"] = datetime.now(pytz.timezone(\"US/Pacific\")).isoformat()\n"
        "\n"
        "                                await asyncio.to_thread(\n"
        "                                    self.vector_store.update,\n"
        "                                    vector_id=mem_id,\n"
        "                                    vector=None,\n"
        "                                    payload=updated_metadata,\n"
        "                                )\n"
        "                                logger.info(f\"Updated session IDs for memory {mem_id} (async)\")\n"
        "\n"
        "                            await update_session_ids(memory_id, metadata)\n"
        "                        else:\n"
        "                            logger.info(\"NOOP for Memory (async).\")\n"
        "                except Exception as e:\n"
        "                    logger.error(f\"Error processing memory action (async): {resp}, Error: {e}\")\n"
        "        except Exception as e:\n"
        "            logger.error(f\"Error in memory processing loop (async): {e}\")\n"
    )

    updated = text[:start_idx] + replacement + text[end_idx:]
    return updated, True


def main() -> int:
    try:
        import mem0  # type: ignore
    except Exception as exc:
        print(f"❌ mem0 import error: {exc}")
        return 1

    mem0_path = Path(mem0.__file__).resolve()
    target_file = mem0_path.parent / "memory" / "main.py"

    if not target_file.exists():
        print(f"❌ mem0 main.py not found: {target_file}")
        return 1

    text = target_file.read_text(encoding="utf-8")
    updated, changed = _apply_patch(text)
    if not changed:
        print("ℹ️ Patch not applied (maybe already patched or pattern changed).")
        return 0

    target_file.write_text(updated, encoding="utf-8")
    print(f"✅ Patched mem0 async memory actions in {target_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
