from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from scripttuner import cli
from scripttuner.persistence.jsonl import read_jsonl, write_jsonl
from scripttuner.preprocessing.ir import Monologue, Pair, Utterance


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Prevent the developer's local .env from leaking into CLI tests.

    load_dotenv runs inside cli.main; if a real .env defines LLM_MODEL or
    OPENAI_* vars, it would override monkeypatch.delenv done before the call.
    Patching load_dotenv to a no-op + scrubbing LLM_MODEL keeps tests hermetic.
    """
    monkeypatch.setattr(cli, "load_dotenv", lambda: None)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_MODEL_ALIAS", raising=False)

SAMPLE_CHA = """@UTF8
@Begin
@Languages:\teng
@Participants:\tTAMM TAMMY Speaker, BRAD BRAD Speaker
*TAMM:\t(..) Well (.) u:m I'm gonna tell you a long story today. 760_2000
\tIt's all about how I went to the store yesterday and bought milk. 2000_4000
\tAnd then I came home and made some coffee for breakfast. 4000_6000
*BRAD:\tYeah . 6000_6500
*TAMM:\tAnd it was really good coffee, like the best I've had in a while. 6500_8500
\tI'm thinking I might go back to that store tomorrow as well. 8500_10000
@End
"""


def test_parse_clean_monologue_pipeline(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    sbcsae_dir = datasets_dir / "sbcsae"
    sbcsae_dir.mkdir(parents=True)
    (sbcsae_dir / "SBC016.cha").write_text(SAMPLE_CHA, encoding="utf-8")
    data_dir = tmp_path / "data"

    rc = cli.main(
        [
            "parse",
            "sbcsae",
            "SBC016",
            "--datasets-dir",
            str(datasets_dir),
            "--data-dir",
            str(data_dir),
        ]
    )
    assert rc == 0
    parsed_path = data_dir / "parsed" / "SBCSAE" / "SBC016.jsonl"
    assert parsed_path.exists()
    parsed = read_jsonl(parsed_path, Utterance)
    assert len(parsed) == 3
    assert parsed[0].speaker == "TAMM"
    # markers must be preserved at parse stage
    assert "(.)" in parsed[0].text or "(..)" in parsed[0].text

    rc = cli.main(["clean", "sbcsae", "SBC016", "--data-dir", str(data_dir)])
    assert rc == 0
    cleaned_path = data_dir / "cleaned" / "SBCSAE" / "SBC016.jsonl"
    assert cleaned_path.exists()
    cleaned = read_jsonl(cleaned_path, Utterance)
    assert len(cleaned) == 3
    # cleaner replaces (.) and (..) with pause tokens
    assert "<pause:short>" in cleaned[0].text or "<pause:long>" in cleaned[0].text
    assert "(.)" not in cleaned[0].text
    assert "(..)" not in cleaned[0].text

    rc = cli.main(
        [
            "monologue",
            "sbcsae",
            "SBC016",
            "--data-dir",
            str(data_dir),
            "--min-tokens",
            "5",
        ]
    )
    assert rc == 0
    mono_path = data_dir / "monologues" / "SBCSAE" / "SBC016.jsonl"
    assert mono_path.exists()
    monos = read_jsonl(mono_path, Monologue)
    # BRAD's "Yeah" is a backchannel → TAMM's utterances merge into one monologue
    assert len(monos) >= 1
    assert monos[0].speaker == "TAMM"
    assert isinstance(monos[0].utterance_ids, tuple)


def test_parse_writes_correct_count(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    sbcsae_dir = datasets_dir / "sbcsae"
    sbcsae_dir.mkdir(parents=True)
    (sbcsae_dir / "tiny.cha").write_text(
        "@UTF8\n@Begin\n*A:\thello.\n*B:\tworld.\n@End\n",
        encoding="utf-8",
    )
    data_dir = tmp_path / "data"
    rc = cli.main(
        [
            "parse",
            "sbcsae",
            "tiny",
            "--datasets-dir",
            str(datasets_dir),
            "--data-dir",
            str(data_dir),
        ]
    )
    assert rc == 0
    parsed = read_jsonl(data_dir / "parsed" / "SBCSAE" / "tiny.jsonl", Utterance)
    assert len(parsed) == 2
    assert parsed[0].source == "SBCSAE"


class _FakeOpenAIClient:
    """Stand-in for OpenAICompatibleClient — no network, deterministic."""

    def __init__(self, *, model: str, max_retries: int = 3) -> None:
        self._model = model

    def complete(self, system: str, user: str) -> tuple[str, dict[str, Any]]:
        return f"FORMAL[{self._model}]: {user}", {
            "prompt_tokens": 1,
            "completion_tokens": 1,
        }


def _write_monologues(path: Path, monos: list[Monologue]) -> None:
    write_jsonl(path, monos)


def test_pairs_subcommand_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "OpenAICompatibleClient", _FakeOpenAIClient)

    data_dir = tmp_path / "data"
    mono_path = data_dir / "monologues" / "SBCSAE" / "SBC016.jsonl"
    _write_monologues(
        mono_path,
        [
            Monologue(
                source="SBCSAE",
                monologue_id="SBC016#mono_0001",
                speaker="TAMM",
                text="<pause:long> hello <pause:short> there",
                utterance_ids=("u1", "u2"),
                n_tokens=2,
            ),
            Monologue(
                source="SBCSAE",
                monologue_id="SBC016#mono_0002",
                speaker="TAMM",
                text="another monologue",
                utterance_ids=("u3",),
                n_tokens=2,
            ),
        ],
    )

    rc = cli.main(
        [
            "pairs",
            "sbcsae",
            "SBC016",
            "--model",
            "fake/model",
            "--data-dir",
            str(data_dir),
            "--no-progress",
        ]
    )
    assert rc == 0

    out_path = data_dir / "pairs" / "SBCSAE" / "SBC016.jsonl"
    pairs = read_jsonl(out_path, Pair)
    assert len(pairs) == 2
    # spoken_text preserves pause tokens; formal_text is stripped (FakeClient echoes user)
    assert pairs[0].spoken_text == "<pause:long> hello <pause:short> there"
    assert pairs[0].formal_text == "FORMAL[fake/model]: hello there"
    assert pairs[0].metadata["model"] == "fake/model"


def test_pairs_subcommand_accepts_model_alias_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("LLM_MODEL_ALIAS", "shared-alias")
    monkeypatch.setattr(cli, "OpenAICompatibleClient", _FakeOpenAIClient)

    data_dir = tmp_path / "data"
    mono_path = data_dir / "monologues" / "SBCSAE" / "SBC016.jsonl"
    _write_monologues(
        mono_path,
        [
            Monologue(
                source="SBCSAE",
                monologue_id="SBC016#mono_0001",
                speaker="TAMM",
                text="hello",
                utterance_ids=("u1",),
                n_tokens=1,
            )
        ],
    )

    rc = cli.main(
        [
            "pairs",
            "sbcsae",
            "SBC016",
            "--model",
            "raw/slug:free",
            "--data-dir",
            str(data_dir),
            "--no-progress",
        ]
    )
    assert rc == 0
    pairs = read_jsonl(data_dir / "pairs" / "SBCSAE" / "SBC016.jsonl", Pair)
    assert pairs[0].metadata["model"] == "raw/slug:free"
    assert pairs[0].metadata["model_alias"] == "shared-alias"


def test_pairs_subcommand_requires_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LLM_MODEL", raising=False)
    data_dir = tmp_path / "data"
    mono_path = data_dir / "monologues" / "SBCSAE" / "SBC016.jsonl"
    _write_monologues(
        mono_path,
        [
            Monologue(
                source="SBCSAE",
                monologue_id="SBC016#mono_0001",
                speaker="TAMM",
                text="hello",
                utterance_ids=("u1",),
                n_tokens=1,
            )
        ],
    )

    rc = cli.main(
        [
            "pairs",
            "sbcsae",
            "SBC016",
            "--data-dir",
            str(data_dir),
            "--no-progress",
        ]
    )
    assert rc == 2


def test_run_subcommand_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "OpenAICompatibleClient", _FakeOpenAIClient)

    datasets_dir = tmp_path / "datasets"
    sbcsae_dir = datasets_dir / "sbcsae"
    sbcsae_dir.mkdir(parents=True)
    cha_path = sbcsae_dir / "SBC016.cha"
    cha_path.write_text(SAMPLE_CHA, encoding="utf-8")
    data_dir = tmp_path / "data"

    rc = cli.main(
        [
            "run",
            "sbcsae",
            "SBC016",
            "--datasets-dir",
            str(datasets_dir),
            "--data-dir",
            str(data_dir),
            "--model",
            "fake/model",
            "--min-tokens",
            "5",
            "--no-progress",
            "--no-pos",
        ]
    )
    assert rc == 0

    # all stage outputs present
    assert (data_dir / "parsed" / "SBCSAE" / "SBC016.jsonl").exists()
    assert (data_dir / "cleaned" / "SBCSAE" / "SBC016.jsonl").exists()
    assert (data_dir / "monologues" / "SBCSAE" / "SBC016.jsonl").exists()
    assert (data_dir / "pairs" / "SBCSAE" / "SBC016.jsonl").exists()
    stats_path = data_dir / "stats" / "SBCSAE" / "SBC016.json"
    assert stats_path.exists()
    result = json.loads(stats_path.read_text(encoding="utf-8"))
    assert result["source"] == "SBCSAE"
    assert result["n_pairs"] >= 1


def test_run_subcommand_missing_input(tmp_path: Path) -> None:
    rc = cli.main(
        [
            "run",
            "sbcsae",
            "NOTHERE",
            "--datasets-dir",
            str(tmp_path / "datasets"),
            "--data-dir",
            str(tmp_path / "data"),
            "--model",
            "fake/model",
            "--no-progress",
            "--no-pos",
        ]
    )
    assert rc == 1


def _setup_run_dirs(tmp_path: Path, stems: list[str]) -> tuple[Path, Path]:
    datasets_dir = tmp_path / "datasets"
    sbcsae_dir = datasets_dir / "sbcsae"
    sbcsae_dir.mkdir(parents=True)
    for stem in stems:
        (sbcsae_dir / f"{stem}.cha").write_text(SAMPLE_CHA, encoding="utf-8")
    return datasets_dir, tmp_path / "data"


def test_run_subcommand_multiple_stems(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "OpenAICompatibleClient", _FakeOpenAIClient)
    datasets_dir, data_dir = _setup_run_dirs(tmp_path, ["SBC016", "SBC017"])

    rc = cli.main(
        [
            "run",
            "sbcsae",
            "SBC016",
            "SBC017",
            "--datasets-dir",
            str(datasets_dir),
            "--data-dir",
            str(data_dir),
            "--model",
            "fake/model",
            "--min-tokens",
            "5",
            "--no-progress",
            "--no-pos",
        ]
    )
    assert rc == 0
    assert (data_dir / "stats" / "SBCSAE" / "SBC016.json").exists()
    assert (data_dir / "stats" / "SBCSAE" / "SBC017.json").exists()
    captured = capsys.readouterr()
    assert "summary: 2/2 succeeded" in captured.err


def test_run_subcommand_all_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "OpenAICompatibleClient", _FakeOpenAIClient)
    datasets_dir, data_dir = _setup_run_dirs(tmp_path, ["SBC001", "SBC002", "SBC003"])

    rc = cli.main(
        [
            "run",
            "sbcsae",
            "--all",
            "--datasets-dir",
            str(datasets_dir),
            "--data-dir",
            str(data_dir),
            "--model",
            "fake/model",
            "--min-tokens",
            "5",
            "--no-progress",
            "--no-pos",
        ]
    )
    assert rc == 0
    for stem in ("SBC001", "SBC002", "SBC003"):
        assert (data_dir / "stats" / "SBCSAE" / f"{stem}.json").exists()
    captured = capsys.readouterr()
    assert "summary: 3/3 succeeded" in captured.err


def test_run_subcommand_partial_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(cli, "OpenAICompatibleClient", _FakeOpenAIClient)
    # Only SBC016 is on disk; SBC999 is missing → that stem fails, the other succeeds.
    datasets_dir, data_dir = _setup_run_dirs(tmp_path, ["SBC016"])

    rc = cli.main(
        [
            "run",
            "sbcsae",
            "SBC016",
            "SBC999",
            "--datasets-dir",
            str(datasets_dir),
            "--data-dir",
            str(data_dir),
            "--model",
            "fake/model",
            "--min-tokens",
            "5",
            "--no-progress",
            "--no-pos",
        ]
    )
    assert rc == 1
    assert (data_dir / "stats" / "SBCSAE" / "SBC016.json").exists()
    assert not (data_dir / "stats" / "SBCSAE" / "SBC999.json").exists()
    captured = capsys.readouterr()
    assert "summary: 1/2 succeeded" in captured.err
    assert "SBC999" in captured.err


def test_run_subcommand_requires_stems_or_all(tmp_path: Path) -> None:
    rc = cli.main(
        [
            "run",
            "sbcsae",
            "--datasets-dir",
            str(tmp_path / "datasets"),
            "--data-dir",
            str(tmp_path / "data"),
            "--model",
            "fake/model",
            "--no-progress",
            "--no-pos",
        ]
    )
    assert rc == 2


def test_run_subcommand_rejects_stems_and_all_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cli, "OpenAICompatibleClient", _FakeOpenAIClient)
    datasets_dir, data_dir = _setup_run_dirs(tmp_path, ["SBC016"])
    rc = cli.main(
        [
            "run",
            "sbcsae",
            "SBC016",
            "--all",
            "--datasets-dir",
            str(datasets_dir),
            "--data-dir",
            str(data_dir),
            "--model",
            "fake/model",
            "--no-progress",
            "--no-pos",
        ]
    )
    assert rc == 2


def test_run_subcommand_all_flag_empty_dir(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets" / "sbcsae"
    datasets_dir.mkdir(parents=True)  # no .cha files
    rc = cli.main(
        [
            "run",
            "sbcsae",
            "--all",
            "--datasets-dir",
            str(tmp_path / "datasets"),
            "--data-dir",
            str(tmp_path / "data"),
            "--model",
            "fake/model",
            "--no-progress",
            "--no-pos",
        ]
    )
    assert rc == 1


def _write_switchboard_conv(corpus_dir: Path) -> None:
    """One conversation: A holds the floor (split by silence), B only backchannels."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    (corpus_dir / "sw2005A-ms98-a-trans.text").write_text(
        "sw2005A-ms98-a-0001 0.0 4.0 well i really think nursing homes are a hard "
        "decision for most families\n"
        "sw2005A-ms98-a-0002 4.0 6.0 [silence]\n"
        "sw2005A-ms98-a-0003 6.0 10.0 and you know my grandmother had to go into one "
        "just last year\n",
        encoding="utf-8",
    )
    (corpus_dir / "sw2005B-ms98-a-trans.text").write_text(
        "sw2005B-ms98-a-0001 0.0 4.0 [silence]\n"
        "sw2005B-ms98-a-0002 4.0 5.0 um-hum\n"
        "sw2005B-ms98-a-0003 5.0 10.0 [silence]\n",
        encoding="utf-8",
    )


def test_run_switchboard_through_monologue_stops_before_llm(tmp_path: Path) -> None:
    datasets_dir = tmp_path / "datasets"
    _write_switchboard_conv(datasets_dir / "switchboard")
    data_dir = tmp_path / "data"

    rc = cli.main(
        [
            "run",
            "switchboard",
            "--all",
            "--through",
            "monologue",
            "--datasets-dir",
            str(datasets_dir),
            "--data-dir",
            str(data_dir),
            "--min-tokens",
            "5",
        ]
    )
    assert rc == 0

    # stages up to monologue ran...
    assert (data_dir / "parsed" / "Switchboard" / "sw2005.jsonl").exists()
    assert (data_dir / "cleaned" / "Switchboard" / "sw2005.jsonl").exists()
    mono_path = data_dir / "monologues" / "Switchboard" / "sw2005.jsonl"
    assert mono_path.exists()
    # ...and stopped before the LLM pairs stage + stats
    assert not (data_dir / "pairs" / "Switchboard" / "sw2005.jsonl").exists()
    assert not (data_dir / "stats" / "Switchboard" / "sw2005.json").exists()

    monos = read_jsonl(mono_path, Monologue)
    # A's two speech segments merge across B's "um-hum" backchannel into one monologue
    assert len(monos) == 1
    assert monos[0].speaker == "A"
    assert monos[0].monologue_id.startswith("sw2005#mono_")
    assert "nursing homes" in monos[0].text
    assert "grandmother" in monos[0].text


def test_aggregate_subcommand_concats_and_stats(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    pairs_dir = data_dir / "pairs" / "SBCSAE"

    def _pair(stem: str, idx: int, speaker: str) -> Pair:
        return Pair(
            pair_id=f"{stem}#mono_{idx:04d}#casual#v1",
            source="SBCSAE",
            style="casual",
            speaker=speaker,
            spoken_text=f"um well hello {idx}",
            formal_text=f"Hello {idx}.",
            monologue_id=f"{stem}#mono_{idx:04d}",
        )

    write_jsonl(pairs_dir / "SBC001.jsonl", [_pair("SBC001", 1, "A"), _pair("SBC001", 2, "B")])
    write_jsonl(pairs_dir / "SBC002.jsonl", [_pair("SBC002", 1, "C")])
    # underscore-prefixed file must be excluded from input
    write_jsonl(pairs_dir / "_all.jsonl", [_pair("STALE", 99, "X")])

    rc = cli.main(
        ["aggregate", "sbcsae", "--data-dir", str(data_dir), "--no-pos"]
    )
    assert rc == 0

    all_path = pairs_dir / "_all.jsonl"
    assert all_path.exists()
    rewritten = read_jsonl(all_path, Pair)
    assert len(rewritten) == 3  # 2 + 1, stale entry overwritten
    assert {p.monologue_id for p in rewritten} == {
        "SBC001#mono_0001",
        "SBC001#mono_0002",
        "SBC002#mono_0001",
    }

    agg_path = data_dir / "stats" / "SBCSAE" / "_aggregate.json"
    assert agg_path.exists()
    agg = json.loads(agg_path.read_text(encoding="utf-8"))
    assert agg["n_pairs"] == 3
    assert agg["n_unique_speakers"] == 3
    assert set(agg["speakers"]) == {"A", "B", "C"}


def test_aggregate_subcommand_errors_when_no_pairs(tmp_path: Path) -> None:
    rc = cli.main(
        ["aggregate", "sbcsae", "--data-dir", str(tmp_path / "data"), "--no-pos"]
    )
    assert rc == 1


def test_stats_subcommand_writes_json(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    pairs_path = data_dir / "pairs" / "SBCSAE" / "SBC016.jsonl"
    write_jsonl(
        pairs_path,
        [
            Pair(
                pair_id="SBC016#mono_0001#casual#v1",
                source="SBCSAE",
                style="casual",
                speaker="TAMM",
                spoken_text="<pause:short> um well hello <pause:long> world",
                formal_text="Hello, world.",
                monologue_id="SBC016#mono_0001",
            ),
            Pair(
                pair_id="SBC016#mono_0002#casual#v1",
                source="SBCSAE",
                style="casual",
                speaker="BRAD",
                spoken_text="another monologue here",
                formal_text="another monologue here",
                monologue_id="SBC016#mono_0002",
            ),
        ],
    )

    rc = cli.main(
        ["stats", "sbcsae", "SBC016", "--data-dir", str(data_dir), "--no-pos"]
    )
    assert rc == 0
    out_path = data_dir / "stats" / "SBCSAE" / "SBC016.json"
    assert out_path.exists()

    result = json.loads(out_path.read_text(encoding="utf-8"))
    assert result["n_pairs"] == 2
    assert result["n_unique_speakers"] == 2
    assert set(result["speakers"]) == {"TAMM", "BRAD"}
    # pause counts
    assert result["spoken"]["pause_short_per_pair"]["max"] == 1.0
    assert result["spoken"]["pause_long_per_pair"]["max"] == 1.0
    # filler "um well" = 2 in pair 1; 0 in pair 2 → max 2
    assert result["spoken"]["fillers_per_pair"]["max"] == 2.0
    # --no-pos suppresses lexical_density
    assert "lexical_density" not in result["spoken"]
