from __future__ import annotations

from pathlib import Path

from scripttuner.persistence.cache import DiskCache, make_cache_key


def test_set_and_get_round_trip(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    cache.set("abc123", {"foo": "bar", "n": 42})
    assert cache.get("abc123") == {"foo": "bar", "n": 42}


def test_get_missing_returns_none(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    assert cache.get("missing") is None


def test_set_creates_parent_dirs(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c"
    cache = DiskCache(nested)
    cache.set("k", {"v": 1})
    assert (nested / "k.json").exists()


def test_overwrite(tmp_path: Path) -> None:
    cache = DiskCache(tmp_path)
    cache.set("k", {"v": 1})
    cache.set("k", {"v": 2})
    assert cache.get("k") == {"v": 2}


def test_make_cache_key_deterministic() -> None:
    a = make_cache_key("model", "prompt", "input")
    b = make_cache_key("model", "prompt", "input")
    assert a == b
    assert len(a) == 64  # sha256 hex digest length


def test_make_cache_key_distinguishes_parts() -> None:
    # NUL separator prevents collision between ("ab","c") and ("a","bc")
    assert make_cache_key("ab", "c") != make_cache_key("a", "bc")


def test_make_cache_key_distinguishes_input() -> None:
    assert make_cache_key("v1", "model", "input-a") != make_cache_key(
        "v1", "model", "input-b"
    )
