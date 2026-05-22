from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripttuner.preprocessing.ir import Monologue
from scripttuner.preprocessing.pairs import (
    DEFAULT_PROMPT_VERSION,
    _normalize_typography,
    _strip_special_tokens,
    convert_to_formal,
)


class FakeLLMClient:
    """Deterministic LLMClient for tests.

    - Echoes user input prefixed with "FORMAL: " by default
    - Tracks call count
    - Can be configured to raise on the nth call
    """

    def __init__(
        self,
        *,
        raise_on: list[int] | None = None,
        fixed_response: str | None = None,
    ) -> None:
        self.calls: list[tuple[str, str]] = []
        self._raise_on = set(raise_on or [])
        self._fixed = fixed_response

    def complete(self, system: str, user: str) -> tuple[str, dict[str, Any]]:
        self.calls.append((system, user))
        idx = len(self.calls)
        if idx in self._raise_on:
            raise RuntimeError(f"injected failure on call {idx}")
        text = self._fixed if self._fixed is not None else f"FORMAL: {user}"
        return text, {"prompt_tokens": 10, "completion_tokens": 5}


def _mono(mono_id: str, text: str, n_tokens: int = 30) -> Monologue:
    return Monologue(
        source="SBCSAE",
        monologue_id=mono_id,
        speaker="TAMM",
        text=text,
        utterance_ids=("u1", "u2"),
        n_tokens=n_tokens,
    )


# ----- _strip_special_tokens -----


def test_strip_removes_pause_short() -> None:
    assert _strip_special_tokens("hello <pause:short> world") == "hello world"


def test_strip_removes_pause_long() -> None:
    assert _strip_special_tokens("ok <pause:long> next") == "ok next"


def test_strip_handles_multiple_tokens() -> None:
    assert (
        _strip_special_tokens("<pause:long> well <pause:short> um <pause:short> ok")
        == "well um ok"
    )


def test_strip_collapses_whitespace() -> None:
    assert _strip_special_tokens("  a    b\tc\n d") == "a b c d"


def test_strip_leaves_text_without_tokens_alone() -> None:
    assert _strip_special_tokens("Just plain text.") == "Just plain text."


# ----- _normalize_typography -----


def test_normalize_smart_apostrophe() -> None:
    assert _normalize_typography("you’ll see it’s fine") == "you'll see it's fine"


def test_normalize_smart_double_quotes() -> None:
    assert _normalize_typography("“hello” he said") == '"hello" he said'


def test_normalize_em_dash_and_en_dash() -> None:
    assert _normalize_typography("from A—to B and A–B") == "from A-to B and A-B"


def test_normalize_ellipsis_char() -> None:
    assert _normalize_typography("wait… maybe") == "wait... maybe"


def test_normalize_leaves_ascii_alone() -> None:
    assert _normalize_typography("plain 'ascii' \"text\" - ok") == "plain 'ascii' \"text\" - ok"


def test_convert_normalizes_typography_in_formal_text() -> None:
    monos = [_mono("m1", "hello")]

    class _SmartClient:
        def complete(self, system: str, user: str) -> tuple[str, dict[str, Any]]:
            return "you’ll see—really", {}

    pairs = convert_to_formal(monos, client=_SmartClient(), model="m", progress=False)
    assert pairs[0].formal_text == "you'll see-really"


# ----- convert_to_formal: basic -----


def test_convert_returns_pairs(tmp_path: Path) -> None:
    monos = [
        _mono("SBC016#mono_0001", "hello world"),
        _mono("SBC016#mono_0002", "second monologue"),
    ]
    client = FakeLLMClient()
    pairs = convert_to_formal(monos, client=client, model="test-model", progress=False)
    assert len(pairs) == 2
    assert pairs[0].formal_text == "FORMAL: hello world"
    assert pairs[0].spoken_text == "hello world"
    assert pairs[0].source == "SBCSAE"
    assert pairs[0].style == "casual"
    assert pairs[0].monologue_id == "SBC016#mono_0001"
    assert pairs[0].pair_id == f"SBC016#mono_0001#casual#{DEFAULT_PROMPT_VERSION}"


def test_convert_passes_stripped_text_to_llm() -> None:
    monos = [_mono("m1", "<pause:long> hello <pause:short> world")]
    client = FakeLLMClient()
    convert_to_formal(monos, client=client, model="test-model", progress=False)
    assert client.calls[0][1] == "hello world"


def test_convert_preserves_pause_in_spoken_text() -> None:
    monos = [_mono("m1", "<pause:long> hello <pause:short> world")]
    client = FakeLLMClient()
    pairs = convert_to_formal(monos, client=client, model="test-model", progress=False)
    # spoken_text must preserve the original pause tokens
    assert pairs[0].spoken_text == "<pause:long> hello <pause:short> world"


def test_convert_metadata_includes_model_and_prompt_version() -> None:
    monos = [_mono("m1", "hi")]
    client = FakeLLMClient()
    pairs = convert_to_formal(
        monos,
        client=client,
        model="some/model-slug",
        prompt_version="vX",
        progress=False,
    )
    assert pairs[0].metadata["model"] == "some/model-slug"
    assert pairs[0].metadata["prompt_version"] == "vX"
    assert pairs[0].metadata["from_cache"] is False
    assert pairs[0].metadata["prompt_tokens"] == 10


# ----- convert_to_formal: cache -----


def test_cache_hit_skips_client_call(tmp_path: Path) -> None:
    monos = [_mono("m1", "hello")]
    client = FakeLLMClient()
    cache_dir = tmp_path / "cache"

    convert_to_formal(
        monos, client=client, model="m", cache_dir=cache_dir, progress=False
    )
    assert len(client.calls) == 1

    # second run: cache hit, no new call
    pairs = convert_to_formal(
        monos, client=client, model="m", cache_dir=cache_dir, progress=False
    )
    assert len(client.calls) == 1
    assert pairs[0].formal_text == "FORMAL: hello"
    assert pairs[0].metadata["from_cache"] is True


def test_cache_miss_on_different_model(tmp_path: Path) -> None:
    monos = [_mono("m1", "hello")]
    client = FakeLLMClient()
    cache_dir = tmp_path / "cache"

    convert_to_formal(monos, client=client, model="modelA", cache_dir=cache_dir, progress=False)
    convert_to_formal(monos, client=client, model="modelB", cache_dir=cache_dir, progress=False)
    assert len(client.calls) == 2


def test_cache_miss_on_different_prompt_version(tmp_path: Path) -> None:
    monos = [_mono("m1", "hello")]
    client = FakeLLMClient()
    cache_dir = tmp_path / "cache"

    convert_to_formal(
        monos,
        client=client,
        model="m",
        cache_dir=cache_dir,
        prompt_version="v1",
        progress=False,
    )
    convert_to_formal(
        monos,
        client=client,
        model="m",
        cache_dir=cache_dir,
        prompt_version="v2",
        progress=False,
    )
    assert len(client.calls) == 2


# ----- convert_to_formal: failure handling -----


def test_failure_is_skipped_not_raised(capsys: pytest.CaptureFixture[str]) -> None:
    monos = [_mono("m1", "first"), _mono("m2", "second"), _mono("m3", "third")]
    client = FakeLLMClient(raise_on=[2])
    pairs = convert_to_formal(monos, client=client, model="m", progress=False)
    # m2 skipped → 2 pairs returned
    assert len(pairs) == 2
    ids = [p.monologue_id for p in pairs]
    assert ids == ["m1", "m3"]
    captured = capsys.readouterr()
    assert "skip m2" in captured.err
    assert "RuntimeError" in captured.err


def test_failed_call_not_cached(tmp_path: Path) -> None:
    monos = [_mono("m1", "boom")]
    cache_dir = tmp_path / "cache"
    failing = FakeLLMClient(raise_on=[1])
    convert_to_formal(monos, client=failing, model="m", cache_dir=cache_dir, progress=False)

    # second run with succeeding client: still calls (no cache from failed call)
    succeeding = FakeLLMClient()
    pairs = convert_to_formal(
        monos, client=succeeding, model="m", cache_dir=cache_dir, progress=False
    )
    assert len(succeeding.calls) == 1
    assert len(pairs) == 1
