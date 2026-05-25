"""Switchboard (MSU/ISIP) turn-level transcript parser.

대화 1건의 두 화자 면(A/B)을 별도 `*-trans.text` 파일에서 읽어 `t_start_ms`
기준 단일 시간순 `Utterance` 스트림으로 인터리브한다 (cf. ADR-0009). 이렇게
SBCSAE와 동일한 "화자 교대 스트림"이 복원되어 ③ monologue.py가 변경 없이
재사용된다.

마커는 손대지 않고 보존한다(`[silence]` 라인 포함). 마커 처리는 cleaner 담당.

라인 포맷:
    sw2005A-ms98-a-0013 97.971250 112.239500 um-hum yeah probably the hardest ...
    <utt_id> <start_sec> <end_sec> <text>
"""

from __future__ import annotations

import re
from pathlib import Path

from scripttuner.preprocessing.ir import Utterance

SOURCE_NAME = "Switchboard"
_SIDES = ("A", "B")
_TRANS_TEMPLATE = "{stem}{side}-ms98-a-trans.text"
_A_FILE_SUFFIX = "A-ms98-a-trans.text"

# <utt_id> <start_sec> <end_sec> <text...>
_LINE_RE = re.compile(r"^(\S+)\s+([\d.]+)\s+([\d.]+)\s+(.*)$")
# trailing 4-digit sequence in utt_id, e.g. "...-0013" -> "0013"
_SEQ_RE = re.compile(r"-(\d+)$")


def _side_path(corpus_dir: Path, stem: str, side: str) -> Path:
    return corpus_dir / _TRANS_TEMPLATE.format(stem=stem, side=side)


def enumerate_stems(corpus_dir: Path) -> list[str]:
    """Return sorted conversation stems (e.g. ['sw2005', 'sw2006']).

    Derived from the A-side transcript filenames present in corpus_dir.
    """
    stems = [
        p.name[: -len(_A_FILE_SUFFIX)]
        for p in corpus_dir.glob(f"sw*{_A_FILE_SUFFIX}")
    ]
    return sorted(stems)


def _parse_side(path: Path, stem: str, side: str) -> list[Utterance]:
    utterances: list[Utterance] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line:
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        orig_id, start_s, end_s, text = m.groups()
        seq_m = _SEQ_RE.search(orig_id)
        seq = seq_m.group(1) if seq_m else f"{len(utterances):04d}"
        utterances.append(
            Utterance(
                source=SOURCE_NAME,
                utterance_id=f"{stem}#{side}_{seq}",
                speaker=side,
                text=text.strip(),
                t_start_ms=round(float(start_s) * 1000),
                t_end_ms=round(float(end_s) * 1000),
                metadata={"orig_id": orig_id, "side": side},
            )
        )
    return utterances


def parse_conversation(corpus_dir: Path, stem: str) -> list[Utterance]:
    """Parse both sides of a conversation and interleave by start time.

    Markers (incl. ``[silence]``) are preserved verbatim in ``text``; timestamps
    are extracted into ``t_start_ms`` / ``t_end_ms``. Sides are merged with a
    stable sort on ``t_start_ms`` (ties keep A before B).

    Raises FileNotFoundError if neither side file exists.
    """
    merged: list[Utterance] = []
    found = False
    for side in _SIDES:
        path = _side_path(corpus_dir, stem, side)
        if path.exists():
            found = True
            merged.extend(_parse_side(path, stem, side))
    if not found:
        raise FileNotFoundError(
            f"no Switchboard transcript for stem {stem!r} under {corpus_dir}"
        )
    # stable sort: equal t_start_ms preserves insertion order (A side first)
    merged.sort(key=lambda u: (u.t_start_ms if u.t_start_ms is not None else 0))
    return merged
