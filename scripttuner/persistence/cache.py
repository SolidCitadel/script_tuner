"""Disk-backed KV cache — sha256-keyed JSON files.

LLM 응답 등 비싼 호출의 결과를 디스크에 캐싱한다. 캐시 키 생성은 caller 책임이며,
`make_cache_key`로 여러 문자열 부분을 sha256 hex digest로 결합할 수 있다.

캐시는 단순 KV 스토어이므로 값은 JSON-serializable한 dict여야 한다.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast


class DiskCache:
    """sha256 키 기반 JSON KV 디스크 캐시."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def get(self, key: str) -> dict[str, Any] | None:
        """Return cached value for key, or None if missing."""
        path = self._path_for(key)
        if not path.exists():
            return None
        return cast(dict[str, Any], json.loads(path.read_text(encoding="utf-8")))

    def set(self, key: str, value: dict[str, Any]) -> None:
        """Store JSON-serializable value under key (overwrites existing)."""
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")

    def _path_for(self, key: str) -> Path:
        return self._root / f"{key}.json"


def make_cache_key(*parts: str) -> str:
    """Compose a sha256 hex digest from string parts.

    Parts are NUL-separated to prevent collision between e.g. ("ab", "c") and ("a", "bc").
    """
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()
