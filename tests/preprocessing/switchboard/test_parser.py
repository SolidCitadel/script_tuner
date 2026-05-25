from __future__ import annotations

from pathlib import Path

import pytest

from scripttuner.preprocessing.switchboard import parser


def _write_side(corpus_dir: Path, stem: str, side: str, lines: list[str]) -> None:
    path = corpus_dir / f"{stem}{side}-ms98-a-trans.text"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_interleaves_two_sides_by_start_time(tmp_path: Path) -> None:
    # A speaks at 0 and 10; B speaks at 5 -> order must be A,B,A by t_start
    _write_side(
        tmp_path,
        "sw2005",
        "A",
        [
            "sw2005A-ms98-a-0001 0.000000 4.000000 okay first thing i want to say",
            "sw2005A-ms98-a-0002 4.000000 5.000000 [silence]",
            "sw2005A-ms98-a-0003 10.000000 14.000000 and then i continue my point here",
        ],
    )
    _write_side(
        tmp_path,
        "sw2005",
        "B",
        [
            "sw2005B-ms98-a-0001 0.000000 5.000000 [silence]",
            "sw2005B-ms98-a-0002 5.000000 6.000000 um-hum",
            "sw2005B-ms98-a-0003 6.000000 10.000000 [silence]",
        ],
    )

    utts = parser.parse_conversation(tmp_path, "sw2005")

    # 6 lines total, time-ordered
    assert [u.t_start_ms for u in utts] == [0, 0, 4000, 5000, 6000, 10000]
    # the B backchannel "um-hum" (start 5000) lands between A's 4000 and 10000 lines
    texts_by_start = {(u.t_start_ms, u.speaker): u.text for u in utts}
    assert texts_by_start[(5000, "B")] == "um-hum"


def test_utterance_id_and_speaker_format(tmp_path: Path) -> None:
    _write_side(
        tmp_path,
        "sw2005",
        "A",
        ["sw2005A-ms98-a-0013 1.500000 2.250000 hello there"],
    )
    utts = parser.parse_conversation(tmp_path, "sw2005")
    u = utts[0]
    # id format {conv}#{side}_{nnnn} so monologue._file_stem_from_ids -> "sw2005"
    assert u.utterance_id == "sw2005#A_0013"
    assert u.utterance_id.split("#", 1)[0] == "sw2005"
    assert u.speaker == "A"
    assert u.source == "Switchboard"
    assert u.t_start_ms == 1500
    assert u.t_end_ms == 2250
    assert u.metadata["orig_id"] == "sw2005A-ms98-a-0013"
    assert u.metadata["side"] == "A"


def test_markers_preserved_at_parse_stage(tmp_path: Path) -> None:
    _write_side(
        tmp_path,
        "sw2005",
        "A",
        [
            "sw2005A-ms98-a-0001 0.0 1.0 [silence]",
            "sw2005A-ms98-a-0002 1.0 3.0 well i [laughter] think so h[ow]- how about you",
        ],
    )
    utts = parser.parse_conversation(tmp_path, "sw2005")
    assert utts[0].text == "[silence]"
    assert "[laughter]" in utts[1].text
    assert "h[ow]-" in utts[1].text


def test_single_side_present(tmp_path: Path) -> None:
    _write_side(tmp_path, "sw2005", "A", ["sw2005A-ms98-a-0001 0.0 1.0 hello"])
    utts = parser.parse_conversation(tmp_path, "sw2005")
    assert len(utts) == 1
    assert utts[0].speaker == "A"


def test_missing_conversation_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parser.parse_conversation(tmp_path, "sw9999")


def test_enumerate_stems(tmp_path: Path) -> None:
    _write_side(tmp_path, "sw2005", "A", ["sw2005A-ms98-a-0001 0.0 1.0 hi"])
    _write_side(tmp_path, "sw2005", "B", ["sw2005B-ms98-a-0001 0.0 1.0 hi"])
    _write_side(tmp_path, "sw2006", "A", ["sw2006A-ms98-a-0001 0.0 1.0 hi"])
    _write_side(tmp_path, "sw2006", "B", ["sw2006B-ms98-a-0001 0.0 1.0 hi"])
    assert parser.enumerate_stems(tmp_path) == ["sw2005", "sw2006"]
