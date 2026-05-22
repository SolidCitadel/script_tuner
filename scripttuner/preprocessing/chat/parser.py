"""CHAT (CHILDES) 포맷 파서.

SBCSAE 등 TalkBank/CHILDES 호환 .cha 파일을 공통 IR(`Utterance` 리스트)로 변환한다.
마커는 손대지 않고 보존하되, 타임스탬프 토큰만 분리해 메타데이터로 추출한다.
"""

from __future__ import annotations

import re
from pathlib import Path

from scripttuner.preprocessing.ir import Utterance

# 발화 라인: "*XXX:\t..." 형식 (화자 코드는 알파벳/숫자)
_UTTERANCE_LINE_RE = re.compile(r"^\*([A-Z0-9]+):\s*(.*)$")
# 헤더 라인: "@Key:\tValue" 또는 "@Marker"
_HEADER_LINE_RE = re.compile(r"^@(\w+)(?::\s*(.*))?$")
# 이어지는 라인: 공백/탭으로 시작
_CONTINUATION_RE = re.compile(r"^\s+(.*)$")
# 타임스탬프 토큰: 발화 라인 끝의 "1234_5678" 형식.
# CHAT의 sound-bullet delimiter `\x15` (NAK)이 양옆에 붙는 경우도 함께 제거.
_TIMESTAMP_RE = re.compile(r"\x15?(?<!\d)(\d+)_(\d+)(?!\d)\x15?")
# 잔존 NAK 정리용 (timestamp 외 위치에 떠도는 NAK)
_NAK_RE = re.compile(r"\x15")


def parse(path: Path, source: str = "SBCSAE") -> tuple[dict[str, list[str]], list[Utterance]]:
    """Parse a CHAT (.cha) file.

    Returns (header_metadata, utterances).

    - header_metadata: dict mapping @Key to list of values (multiple @Key entries supported,
      multi-line values joined with a single space).
    - utterances: list of Utterance with text preserving CHAT markers
      (timestamps are removed and extracted into t_start_ms / t_end_ms).
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    file_stem = path.stem

    header_meta: dict[str, list[str]] = {}
    utterances: list[Utterance] = []

    cur_kind: str = "none"
    cur_header_key: str | None = None
    cur_speaker: str | None = None
    cur_text_lines: list[str] = []
    cur_start_line: int = 0
    seq: int = 0

    def flush() -> None:
        nonlocal cur_kind, cur_header_key, cur_speaker, cur_text_lines, seq
        if cur_kind == "header" and cur_header_key is not None:
            value = " ".join(s.strip() for s in cur_text_lines).strip()
            header_meta.setdefault(cur_header_key, []).append(value)
        elif cur_kind == "utterance" and cur_speaker is not None:
            seq += 1
            full_text = " ".join(s.strip() for s in cur_text_lines).strip()
            t_start, t_end, stripped = _extract_timestamps(full_text)
            utterances.append(
                Utterance(
                    source=source,
                    utterance_id=f"{file_stem}#{seq:04d}",
                    speaker=cur_speaker,
                    text=stripped,
                    t_start_ms=t_start,
                    t_end_ms=t_end,
                    metadata={"line_no": cur_start_line},
                )
            )
        cur_kind = "none"
        cur_header_key = None
        cur_speaker = None
        cur_text_lines = []

    for i, raw_line in enumerate(lines, start=1):
        if not raw_line:
            continue

        m_utt = _UTTERANCE_LINE_RE.match(raw_line)
        if m_utt:
            flush()
            cur_kind = "utterance"
            cur_speaker = m_utt.group(1)
            cur_text_lines = [m_utt.group(2)]
            cur_start_line = i
            continue

        m_hdr = _HEADER_LINE_RE.match(raw_line)
        if m_hdr:
            flush()
            cur_kind = "header"
            cur_header_key = m_hdr.group(1)
            cur_text_lines = [m_hdr.group(2) or ""]
            cur_start_line = i
            continue

        m_cont = _CONTINUATION_RE.match(raw_line)
        if m_cont and cur_kind != "none":
            cur_text_lines.append(m_cont.group(1))
            continue

        # 알 수 없는 라인은 무시 (실제 CHAT 파일에는 거의 없음)

    flush()
    return header_meta, utterances


def _extract_timestamps(text: str) -> tuple[int | None, int | None, str]:
    """Extract timestamp tokens from text. Returns (t_start, t_end, stripped_text).

    NAK characters (CHAT sound-bullet delimiter) are removed even when no
    timestamp pattern matches, since they sometimes drift apart from the
    timestamp pair in malformed lines.
    """
    matches = list(_TIMESTAMP_RE.finditer(text))
    if not matches:
        stripped = _NAK_RE.sub("", text)
        if stripped != text:
            stripped = re.sub(r"\s+", " ", stripped).strip()
        return None, None, stripped

    starts = [int(m.group(1)) for m in matches]
    ends = [int(m.group(2)) for m in matches]
    t_start = min(starts)
    t_end = max(ends)

    stripped = _TIMESTAMP_RE.sub("", text)
    stripped = _NAK_RE.sub("", stripped)
    stripped = re.sub(r"\s+", " ", stripped).strip()
    return t_start, t_end, stripped
