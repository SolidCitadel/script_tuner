"""Switchboard (MSU/ISIP) 발화 정규화.

파서가 추출한 `Utterance` 리스트를 받아 Switchboard-specific 마커를 정규화하고,
정규화 후 텍스트가 빈 발화(예: `[silence]` 전용 라인)는 **제거**한다. 빈 발화를
③ monologue 재조립에 넘기면 `is_backchannel("")==False`라 화자 버퍼를 조기
flush하므로, drop은 필수다 (cf. ADR-0009).

마커 정책은 PoC(`.work/switchboard-poc`) 실측 + `docs/design/preprocessing_pipeline.md`
"Switchboard 마커 처리 정책" 표 참조. 처리 순서는 마커 간 의존성을 고려한다.
"""

from __future__ import annotations

import re
from dataclasses import replace

from scripttuner.preprocessing.ir import Utterance

# 1. 단어 재시작/중단 stub: prefix[completion]- (예: h[ow]-, tr[aveled]-, re[new]-)
#    내부에 슬래시가 없고 끝에 '-'가 붙는다 → 오발음 [x/y]와 구분된다. 통째로 제거.
_PARTIAL_RE = re.compile(r"\w*\[\w+\]-")

# 2. 웃으며 발화한 단어: [laughter-WORD] → WORD (일반 비언어 제거보다 먼저)
_LAUGHTER_WORD_RE = re.compile(r"\[laughter-([^\]]+)\]")

# 3. 오발음 표기: [said/intended] → intended (슬래시 뒤) (예: [bidness/business])
_MISPRONOUNCE_RE = re.compile(r"\[[^\]/]*/([^\]]*)\]")

# 4. 신조어/비표준어: {word} → word (중괄호만 제거)
_COINAGE_RE = re.compile(r"\{([^}]*)\}")

# 5. aside 경계 마커: <b_aside>, <e_aside> → 제거 (내부 텍스트는 실발화라 보존)
_ASIDE_RE = re.compile(r"<[be]_aside>")

# 6. 잔여 비언어/이벤트 브래킷: [silence], [noise], [laughter], [vocalized-noise] 등
#    (위 2·3에서 처리되지 않은 모든 [...]) → 제거
_BRACKET_RE = re.compile(r"\[[^\]]*\]")

# 7. 단어 뒤 disambiguation 인덱스: word_1, them_1 등 _<digit> 접미 → 제거
_UNDERSCORE_IDX_RE = re.compile(r"(?<=\w)_\d+")

# 8. 다중 공백
_MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """단일 발화 텍스트를 정규화한다. 처리 순서는 마커 의존성을 고려한 안전 순서."""
    # 1. 단어 재시작 stub 제거
    text = _PARTIAL_RE.sub(" ", text)
    # 2. 웃음-단어는 단어만 남김 (일반 브래킷 제거 전에)
    text = _LAUGHTER_WORD_RE.sub(r"\1", text)
    # 3. 오발음 → 의도 단어
    text = _MISPRONOUNCE_RE.sub(r"\1", text)
    # 4. 신조어 → 중괄호 제거
    text = _COINAGE_RE.sub(r"\1", text)
    # 5. aside 경계 마커 제거 (내부 텍스트 보존)
    text = _ASIDE_RE.sub(" ", text)
    # 6. 잔여 비언어 브래킷 제거
    text = _BRACKET_RE.sub(" ", text)
    # 7. disambiguation 인덱스 제거
    text = _UNDERSCORE_IDX_RE.sub("", text)
    # 8. 공백 정리
    text = _MULTISPACE_RE.sub(" ", text).strip()
    return text


def clean(utterances: list[Utterance]) -> list[Utterance]:
    """text 필드를 정규화하고, 정규화 후 빈 발화는 제거한 새 리스트를 반환한다.

    Utterance는 frozen이므로 dataclasses.replace로 새 인스턴스를 생성한다.
    빈 발화 drop은 ③ 도달 전 필수 단계다 (cf. ADR-0009).
    """
    cleaned: list[Utterance] = []
    for u in utterances:
        text = clean_text(u.text)
        if text:
            cleaned.append(replace(u, text=text))
    return cleaned
