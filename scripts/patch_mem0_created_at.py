from __future__ import annotations

import sys
from pathlib import Path


def _patch_created_at(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    updated = []
    changed = False

    target = 'metadata["created_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()'

    for line in lines:
        if target in line:
            indent = line[: len(line) - len(line.lstrip())]
            updated.append(f'{indent}if not metadata.get("created_at"):')
            updated.append(
                f'{indent}    metadata["created_at"] = datetime.now(pytz.timezone("US/Pacific")).isoformat()'
            )
            changed = True
            continue
        updated.append(line)

    if not changed:
        return False

    path.write_text("\n".join(updated) + "\n", encoding="utf-8")
    return True


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

    changed = _patch_created_at(target_file)
    if changed:
        print(f"✅ Patched mem0 created_at handling in {target_file}")
        return 0

    print("ℹ️ Patch not applied (maybe already patched).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
