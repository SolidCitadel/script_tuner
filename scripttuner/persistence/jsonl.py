"""JSONL I/O — frozen dataclass <-> dict 변환 포함.

파이프라인 단계 간 디스크 적재 표준 형식. 정책은 ADR-0001 참조.

직렬화는 `dataclasses.asdict()`를 사용하며 tuple 필드는 JSON에서 list로 표현된다.
역직렬화 시 dataclass의 타입 힌트(`tuple[...]`)를 보고 list → tuple로 복원한다.
"""

from __future__ import annotations

import json
import typing
from collections.abc import Iterable
from dataclasses import asdict, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin

T = TypeVar("T")


def write_jsonl(path: Path, items: Iterable[Any]) -> int:
    """dataclass instance들을 JSONL로 저장한다.

    부모 디렉토리는 자동 생성. 반환값은 기록된 레코드 수.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(_to_dict(item), ensure_ascii=False))
            f.write("\n")
            n += 1
    return n


def read_jsonl(path: Path, item_type: type[T]) -> list[T]:
    """JSONL을 읽어 item_type dataclass 인스턴스 리스트로 반환한다.

    빈 라인은 무시. 깨진 JSON 또는 비-object 라인은 ValueError.
    """
    if not path.exists():
        raise FileNotFoundError(str(path))
    out: list[T] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                d = json.loads(stripped)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{line_no}: {e.msg}") from e
            if not isinstance(d, dict):
                raise ValueError(
                    f"Expected JSON object at {path}:{line_no}, got {type(d).__name__}"
                )
            out.append(_from_dict(d, item_type))
    return out


def _to_dict(obj: Any) -> dict[str, Any]:
    if not is_dataclass(obj) or isinstance(obj, type):
        raise TypeError(
            f"write_jsonl item must be a dataclass instance, got {type(obj).__name__}"
        )
    return asdict(obj)


def _from_dict(d: dict[str, Any], cls: type[T]) -> T:
    if not (is_dataclass(cls) and isinstance(cls, type)):
        raise TypeError(f"{cls!r} is not a dataclass type")
    hints = typing.get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for fld in fields(cls):
        if fld.name not in d:
            continue
        kwargs[fld.name] = _coerce(d[fld.name], hints.get(fld.name))
    return typing.cast(T, cls(**kwargs))


def _coerce(value: Any, hint: Any) -> Any:
    if value is None or hint is None:
        return value
    origin = get_origin(hint)
    if origin is tuple:
        args = get_args(hint)
        if not isinstance(value, list | tuple):
            return value
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(v, args[0]) for v in value)
        return tuple(_coerce(v, a) for v, a in zip(value, args, strict=False))
    return value
