"""Episode ID Mapper - Long ID <-> Short ID.

This helps reduce hallucination and token consumption of long IDs by the language model.

"""

from typing import Dict, Any
import copy


class EpisodeIdMapper:
    """Episode ID Mapper: Long ID <-> Short ID."""

    def __init__(self):
        self._long_to_short: Dict[str, str] = {}
        self._short_to_long: Dict[str, str] = {}
        self._counter = 0

    def get_short(self, long_id: str) -> str:
        """Get short ID, create if not exists."""
        if long_id not in self._long_to_short:
            self._counter += 1
            short_id = f"ep{self._counter}"
            self._long_to_short[long_id] = short_id
            self._short_to_long[short_id] = long_id
        return self._long_to_short[long_id]

    def to_long(self, short_id: str) -> str:
        """Convert short ID to long ID."""
        return self._short_to_long.get(short_id, short_id)

    def replace_sources_to_short(self, profile_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert all source IDs in profile dict to short IDs."""
        result = copy.deepcopy(profile_dict)

        def _map_source(source: Any, to_short: bool) -> Any:
            if not isinstance(source, str) or not source:
                return source
            if "|" in source:
                prefix, sid = source.rsplit("|", 1)
                sid = sid.strip()
                mapped = self.get_short(sid) if to_short else self.to_long(sid)
                return f"{prefix}|{mapped}"
            return self.get_short(source) if to_short else self.to_long(source)

        for item in result.get("explicit_info", []):
            item["sources"] = [_map_source(s, True) for s in item.get("sources", [])]
        for item in result.get("implicit_traits", []):
            item["sources"] = [_map_source(s, True) for s in item.get("sources", [])]

        return result

    def replace_sources_to_long(self, profile_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Convert all source IDs in profile dict back to long IDs."""
        result = copy.deepcopy(profile_dict)

        def _map_source(source: Any) -> Any:
            if not isinstance(source, str) or not source:
                return source
            if "|" in source:
                prefix, sid = source.rsplit("|", 1)
                sid = sid.strip()
                return f"{prefix}|{self.to_long(sid)}"
            return self.to_long(source)

        for item in result.get("explicit_info", []):
            item["sources"] = [_map_source(s) for s in item.get("sources", [])]
        for item in result.get("implicit_traits", []):
            item["sources"] = [_map_source(s) for s in item.get("sources", [])]

        return result

    def reset(self):
        """Reset the mapper."""
        self._long_to_short.clear()
        self._short_to_long.clear()
        self._counter = 0
