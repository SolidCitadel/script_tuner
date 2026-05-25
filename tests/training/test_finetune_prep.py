from __future__ import annotations

import json
from pathlib import Path

from scripttuner import cli
from scripttuner.persistence.jsonl import read_jsonl, write_jsonl
from scripttuner.preprocessing.ir import Pair


def _pair(pair_id: str, speaker: str, style: str = "casual") -> Pair:
    return Pair(
        pair_id=pair_id,
        source="SBCSAE",
        style=style,
        speaker=speaker,
        spoken_text=f"Well, this is spoken text from {speaker}.",
        formal_text=f"This is formal text from {speaker}.",
        monologue_id=f"{pair_id}#mono",
    )


def test_split_subcommand_writes_speaker_disjoint_splits(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    input_path = data_dir / "pairs" / "SBCSAE" / "_all.jsonl"
    write_jsonl(
        input_path,
        [
            _pair("p1", "A"),
            _pair("p2", "A"),
            _pair("p3", "B"),
            _pair("p4", "C"),
            _pair("p5", "D"),
            _pair("p6", "E"),
        ],
    )

    rc = cli.main(["split", "sbcsae", "--data-dir", str(data_dir), "--seed", "7"])
    assert rc == 0

    splits_dir = data_dir / "finetune" / "SBCSAE" / "splits"
    assert (splits_dir / "train.jsonl").exists()
    assert (splits_dir / "validation.jsonl").exists()
    assert (splits_dir / "test.jsonl").exists()

    speaker_to_split: dict[str, str] = {}
    for split_name in ("train", "validation", "test"):
        for pair in read_jsonl(splits_dir / f"{split_name}.jsonl", Pair):
            previous = speaker_to_split.setdefault(pair.speaker, split_name)
            assert previous == split_name

    manifest = json.loads((splits_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    assert manifest["split_strategy"] == "speaker_aware_greedy"
    assert sum(manifest["counts"].values()) == 6


def test_format_subcommand_writes_chat_data(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    splits_dir = data_dir / "finetune" / "SBCSAE" / "splits"
    for split_name in ("train", "validation", "test"):
        write_jsonl(splits_dir / f"{split_name}.jsonl", [_pair(f"{split_name}-1", "A")])

    rc = cli.main(["format", "gemma4-e4b", "sbcsae", "--data-dir", str(data_dir)])
    assert rc == 0

    out_path = data_dir / "finetune" / "SBCSAE" / "formatted" / "gemma4-e4b" / "train.jsonl"
    row = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["messages"][0]["role"] == "user"
    assert "<STYLE=casual>" in row["messages"][0]["content"]
    assert row["messages"][1]["role"] == "assistant"
    assert row["style_token"] == "<STYLE=casual>"


def test_format_subcommand_writes_seq2seq_data(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    splits_dir = data_dir / "finetune" / "SBCSAE" / "splits"
    for split_name in ("train", "validation", "test"):
        write_jsonl(splits_dir / f"{split_name}.jsonl", [_pair(f"{split_name}-1", "A")])

    rc = cli.main(["format", "t5gemma2", "sbcsae", "--data-dir", str(data_dir)])
    assert rc == 0

    out_path = data_dir / "finetune" / "SBCSAE" / "formatted" / "t5gemma2" / "train.jsonl"
    row = json.loads(out_path.read_text(encoding="utf-8").splitlines()[0])
    assert row["input"].startswith("<STYLE=casual>")
    assert row["target"].startswith("Well,")
