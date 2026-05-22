"""CHAT (CHILDES) 발화 정규화.

파서가 추출한 `Utterance` 리스트를 받아 CHA-specific 마커를 정규화한다.
처리 정책은 `docs/design/preprocessing_pipeline.md`의 "마커 처리 정책" 표 참조.
포즈 토큰화 근거는 ADR-0003.
"""

from __future__ import annotations

import re
from dataclasses import replace

from scripttuner.preprocessing.ir import Utterance

# 1. 오버랩 마커: ⌈...⌉ 또는 ⌊2...⌋2 형태 (숫자 인덱스 옵셔널)
_OVERLAP_OPEN_RE = re.compile(r"⌈\d*")
_OVERLAP_CLOSE_RE = re.compile(r"⌉\d*")
_OVERLAP_OPEN_LOW_RE = re.compile(r"⌊\d*")
_OVERLAP_CLOSE_LOW_RE = re.compile(r"⌋\d*")

# 2. 코드 스위칭 / L2 표기: &{l=X ... &}l=X — 외곽 마커만 제거, 내부 보존
_LANG_OPEN_RE = re.compile(r"&\{l=\S+\s*")
_LANG_CLOSE_RE = re.compile(r"\s*&\}l=\S+")

# 3. 비언어 어노테이션: &=tsk, &=laugh, &=in, &=ex 등 (&=word 패턴)
_NONVERBAL_RE = re.compile(r"&=\S+")

# 4. 성문 폐쇄음 표기: 단어 시작의 ʔ (예: ʔuh → uh, youʔ → you)
_GLOTTAL_RE = re.compile(r"ʔ")

# 4b. Vowel lengthening: 알파벳 직후의 ":" (CHAT에서 모음 늘림 표기)
# 예: "I:" → "I", "u:m" → "um", "perc:e:nt" → "percent"
# overlap marker가 vowel lengthening 사이에 끼면 1단계 제거 후 "Yeah::" 같은
# 연속 colon이 생기므로 `:+`로 연속 colon을 한 번에 잡는다.
_VOWEL_LENGTH_RE = re.compile(r"(?<=[a-zA-Z]):+")

# 5. 발화 중단: +/.  (다른 +/ 형식 변형이 있으면 추가)
_TRAILOFF_INTERRUPT_RE = re.compile(r"\+/\.")

# 6. 말끝 흐림: +...
_TRAILOFF_RE = re.compile(r"\+\.\.\.")

# 7. 포즈 마커: (.) 와 (..) — 정확히 매치 (다른 괄호 표기와 혼동 방지)
#    먼저 (..) 매치한 후 (.) 매치 (긴 것 먼저)
_PAUSE_LONG_RE = re.compile(r"\(\.\.\)")
_PAUSE_SHORT_RE = re.compile(r"\(\.\)")

# 8. 다중 공백
_MULTISPACE_RE = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """단일 발화 텍스트를 정규화한다.

    처리 순서는 마커 간 의존성을 고려한 안전 순서이다 (모듈 docstring 참조).
    """
    # 1. 오버랩 마커 제거
    text = _OVERLAP_OPEN_RE.sub("", text)
    text = _OVERLAP_CLOSE_RE.sub("", text)
    text = _OVERLAP_OPEN_LOW_RE.sub("", text)
    text = _OVERLAP_CLOSE_LOW_RE.sub("", text)

    # 2. 코드 스위칭 / L2 외곽 마커 제거 (내부 보존)
    text = _LANG_OPEN_RE.sub("", text)
    text = _LANG_CLOSE_RE.sub("", text)

    # 3. 비언어 어노테이션 제거
    text = _NONVERBAL_RE.sub("", text)

    # 4. 성문 폐쇄음 표기 정규화
    text = _GLOTTAL_RE.sub("", text)

    # 4b. Vowel lengthening colon 제거
    text = _VOWEL_LENGTH_RE.sub("", text)

    # 5. 발화 중단 → 자연 종결
    text = _TRAILOFF_INTERRUPT_RE.sub(".", text)

    # 6. 말끝 흐림 → ...
    text = _TRAILOFF_RE.sub("...", text)

    # 7. 포즈 마커 토큰화 (긴 것부터)
    text = _PAUSE_LONG_RE.sub("<pause:long>", text)
    text = _PAUSE_SHORT_RE.sub("<pause:short>", text)

    # 8. 다중 공백 정리
    text = _MULTISPACE_RE.sub(" ", text).strip()

    return text


def clean(utterances: list[Utterance]) -> list[Utterance]:
    """Utterance 리스트의 text 필드를 정규화한 새 리스트를 반환한다.

    Utterance는 frozen이므로 dataclasses.replace로 새 인스턴스를 생성한다.
    """
    return [replace(u, text=clean_text(u.text)) for u in utterances]
