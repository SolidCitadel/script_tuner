from __future__ import annotations

from pathlib import Path

import pytest

from scripttuner.preprocessing.chat import parser

# 합성 CHAT 파일: 헤더(멀티라인 포함), 단일 발화, 멀티라인 발화
SAMPLE_CHA = """@UTF8
@Begin
@Languages:\teng
@Participants:\tTAMM TAMMY Speaker, BRAD BRAD Speaker
@ID:\teng|SBCSAE|TAMM|||||Speaker|||
@Comment:\tFirst line of comment
\tcontinuation of comment
\tand a third line
*TAMM:\t(..) Well (.) u:m 760_1735
\t(..) &=tsk I got now 2160_6130
\t(..) ⌈ &=laugh ⌉ . 6160_6785
*BRAD:\t⌊ Good ⌋ . 6165_6785
*TAMM:\tI mean I h- +... 25670_26200
@End
"""


@pytest.fixture
def sample_path(tmp_path: Path) -> Path:
    path = tmp_path / "sample.cha"
    path.write_text(SAMPLE_CHA, encoding="utf-8")
    return path


def test_header_metadata_parsed(sample_path: Path) -> None:
    header, _ = parser.parse(sample_path)
    assert header["Languages"] == ["eng"]
    assert header["Participants"] == ["TAMM TAMMY Speaker, BRAD BRAD Speaker"]
    assert header["ID"] == ["eng|SBCSAE|TAMM|||||Speaker|||"]


def test_header_multiline_joined(sample_path: Path) -> None:
    header, _ = parser.parse(sample_path)
    assert header["Comment"] == [
        "First line of comment continuation of comment and a third line"
    ]


def test_utterance_count(sample_path: Path) -> None:
    _, utterances = parser.parse(sample_path)
    assert len(utterances) == 3


def test_utterance_speakers(sample_path: Path) -> None:
    _, utterances = parser.parse(sample_path)
    assert [u.speaker for u in utterances] == ["TAMM", "BRAD", "TAMM"]


def test_multiline_utterance_combined(sample_path: Path) -> None:
    _, utterances = parser.parse(sample_path)
    # 첫 번째 발화는 3줄 합쳐서 만들어져야 함
    first = utterances[0]
    assert "Well" in first.text
    assert "I got now" in first.text
    assert "&=laugh" in first.text


def test_nak_delimiters_removed(tmp_path: Path) -> None:
    """CHAT의 sound-bullet delimiter \\x15가 spoken text에 안 남아야 한다."""
    cha = "@UTF8\n@Begin\n*TAMM:\thello \x15760_1735\x15 world.\n@End\n"
    p = tmp_path / "nak.cha"
    p.write_text(cha, encoding="utf-8")
    _, utts = parser.parse(p)
    assert "\x15" not in utts[0].text


def test_timestamps_extracted_and_stripped(sample_path: Path) -> None:
    _, utterances = parser.parse(sample_path)
    first = utterances[0]
    # 첫 발화의 타임스탬프: 760, 1735, 2160, 6130, 6160, 6785
    assert first.t_start_ms == 760
    assert first.t_end_ms == 6785
    # 타임스탬프 토큰은 text에서 제거됐어야 함
    assert "760_1735" not in first.text
    assert "6160_6785" not in first.text


def test_markers_preserved_in_text(sample_path: Path) -> None:
    _, utterances = parser.parse(sample_path)
    first = utterances[0]
    # 마커는 보존
    assert "(.)" in first.text
    assert "(..)" in first.text
    assert "&=tsk" in first.text
    assert "&=laugh" in first.text
    assert "⌈" in first.text and "⌉" in first.text


def test_utterance_id_format(sample_path: Path) -> None:
    _, utterances = parser.parse(sample_path)
    ids = [u.utterance_id for u in utterances]
    assert ids == ["sample#0001", "sample#0002", "sample#0003"]


def test_metadata_line_no(sample_path: Path) -> None:
    _, utterances = parser.parse(sample_path)
    # 첫 발화는 9번째 라인에서 시작 (1-indexed)
    assert utterances[0].metadata["line_no"] == 9


def test_source_field_default_and_override(sample_path: Path) -> None:
    _, default = parser.parse(sample_path)
    assert default[0].source == "SBCSAE"
    _, custom = parser.parse(sample_path, source="CustomCorpus")
    assert custom[0].source == "CustomCorpus"


def test_short_utterance_no_continuation(sample_path: Path) -> None:
    _, utterances = parser.parse(sample_path)
    # 두 번째 발화 (BRAD)는 단일 라인
    brad = utterances[1]
    assert brad.speaker == "BRAD"
    assert "Good" in brad.text
    assert brad.t_start_ms == 6165
    assert brad.t_end_ms == 6785


# ----- 통합 검증: 실제 SBC016.cha -----

_REAL_FILE = Path("datasets/sbcsae/SBC016.cha")


@pytest.mark.skipif(not _REAL_FILE.exists(), reason="SBC016.cha not downloaded")
def test_integration_sbc016_parses() -> None:
    header, utterances = parser.parse(_REAL_FILE)
    # 헤더 정보
    assert "Languages" in header
    assert header["Languages"] == ["eng"]
    assert "Participants" in header
    # 화자 4명 (TAMM/BRAD/TODD/JONA가 등록됨)
    speakers = {u.speaker for u in utterances}
    assert speakers >= {"TAMM", "BRAD"}
    # 발화가 충분히 추출됨 (대화 22분, 수백 발화 예상)
    assert len(utterances) > 500
    # 첫 발화 검증
    first = utterances[0]
    assert first.speaker == "TAMM"
    assert first.t_start_ms == 760
