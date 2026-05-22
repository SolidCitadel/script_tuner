from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scripttuner import cli
from scripttuner.persistence.jsonl import read_jsonl
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
    cha_path = tmp_path / "SBC016.cha"
    cha_path.write_text(SAMPLE_CHA, encoding="utf-8")
    data_dir = tmp_path / "data"

    rc = cli.main(["parse", "sbcsae", str(cha_path), "--data-dir", str(data_dir)])
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
    cha_path = tmp_path / "tiny.cha"
    cha_path.write_text(
        "@UTF8\n@Begin\n*A:\thello.\n*B:\tworld.\n@End\n",
        encoding="utf-8",
    )
    data_dir = tmp_path / "data"
    rc = cli.main(["parse", "sbcsae", str(cha_path), "--data-dir", str(data_dir)])
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
    from scripttuner.persistence.jsonl import write_jsonl

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
