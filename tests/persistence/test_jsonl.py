from __future__ import annotations

from pathlib import Path

import pytest

from scripttuner.persistence.jsonl import read_jsonl, write_jsonl
from scripttuner.preprocessing.ir import Monologue, Utterance


def test_utterance_round_trip(tmp_path: Path) -> None:
    items = [
        Utterance(
            source="SBCSAE",
            utterance_id="SBC016#0001",
            speaker="TAMM",
            text="hello world",
            t_start_ms=100,
            t_end_ms=500,
            metadata={"line_no": 9},
        ),
        Utterance(
            source="SBCSAE",
            utterance_id="SBC016#0002",
            speaker="BRAD",
            text="good",
            t_start_ms=None,
            t_end_ms=None,
        ),
    ]
    out = tmp_path / "out.jsonl"
    n = write_jsonl(out, items)
    assert n == 2

    loaded = read_jsonl(out, Utterance)
    assert loaded == items


def test_monologue_round_trip_restores_tuple(tmp_path: Path) -> None:
    items = [
        Monologue(
            source="SBCSAE",
            monologue_id="SBC016#mono_0001",
            speaker="TAMM",
            text="hello world. this is a longer monologue.",
            utterance_ids=("SBC016#0001", "SBC016#0002", "SBC016#0003"),
            n_tokens=8,
            metadata={"note": "first"},
        ),
    ]
    out = tmp_path / "mono.jsonl"
    write_jsonl(out, items)

    loaded = read_jsonl(out, Monologue)
    assert loaded == items
    # tuple field must be restored as tuple (not list)
    assert isinstance(loaded[0].utterance_ids, tuple)


def test_write_creates_parent_dirs(tmp_path: Path) -> None:
    out = tmp_path / "a" / "b" / "c" / "out.jsonl"
    items = [
        Utterance(
            source="X",
            utterance_id="X#0001",
            speaker="A",
            text="t",
        )
    ]
    n = write_jsonl(out, items)
    assert n == 1
    assert out.exists()


def test_empty_file_returns_empty_list(tmp_path: Path) -> None:
    out = tmp_path / "empty.jsonl"
    write_jsonl(out, [])
    loaded = read_jsonl(out, Utterance)
    assert loaded == []


def test_blank_lines_skipped(tmp_path: Path) -> None:
    out = tmp_path / "blank.jsonl"
    out.write_text(
        '{"source":"X","utterance_id":"X#1","speaker":"A","text":"hi","t_start_ms":null,'
        '"t_end_ms":null,"metadata":{}}\n'
        "\n"
        "   \n"
        '{"source":"X","utterance_id":"X#2","speaker":"B","text":"yo","t_start_ms":null,'
        '"t_end_ms":null,"metadata":{}}\n',
        encoding="utf-8",
    )
    loaded = read_jsonl(out, Utterance)
    assert len(loaded) == 2
    assert loaded[0].utterance_id == "X#1"
    assert loaded[1].utterance_id == "X#2"


def test_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_jsonl(tmp_path / "nope.jsonl", Utterance)


def test_invalid_json_raises(tmp_path: Path) -> None:
    out = tmp_path / "bad.jsonl"
    out.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        read_jsonl(out, Utterance)


def test_non_object_line_raises(tmp_path: Path) -> None:
    out = tmp_path / "bad.jsonl"
    out.write_text('["array", "instead", "of", "object"]\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Expected JSON object"):
        read_jsonl(out, Utterance)


def test_write_non_dataclass_raises(tmp_path: Path) -> None:
    out = tmp_path / "out.jsonl"
    with pytest.raises(TypeError, match="must be a dataclass"):
        write_jsonl(out, [{"foo": "bar"}])  # plain dict not accepted


def test_extra_fields_in_json_ignored(tmp_path: Path) -> None:
    out = tmp_path / "extra.jsonl"
    out.write_text(
        '{"source":"X","utterance_id":"X#1","speaker":"A","text":"hi","t_start_ms":null,'
        '"t_end_ms":null,"metadata":{},"unknown_field":"ignored"}\n',
        encoding="utf-8",
    )
    loaded = read_jsonl(out, Utterance)
    assert len(loaded) == 1
    assert loaded[0].utterance_id == "X#1"
