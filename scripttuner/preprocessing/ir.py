"""Common intermediate representation shared across corpus adapters.

Every corpus adapter (CHAT, Switchboard, BNC, ...) produces a list of `Utterance`
so that downstream modules (③ Monologue, ④ Pairs, ⑤ Stats) are corpus-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Utterance:
    """단일 발화. 모든 코퍼스 어댑터의 공통 출력 형식.

    text 필드의 의미는 파이프라인 단계에 따라 다르다:
      - parser 출력: 코퍼스 마커가 보존된 원본 텍스트 (타임스탬프 토큰만 분리됨)
      - cleaner 출력: 정규화된 텍스트 (마커 제거/변환됨)
    """

    source: str
    """Corpus identifier (e.g. 'SBCSAE')."""

    utterance_id: str
    """Corpus-unique identifier (e.g. 'SBC016#0042')."""

    speaker: str
    """Speaker identifier within the corpus."""

    text: str
    """Utterance text. Meaning depends on pipeline stage."""

    t_start_ms: int | None = None
    """Start time in milliseconds, if the corpus provides timing info."""

    t_end_ms: int | None = None
    """End time in milliseconds, if the corpus provides timing info."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Corpus-specific metadata (header fields, line numbers, etc.)."""


@dataclass(frozen=True)
class Monologue:
    """동일 화자의 연속 발화를 결합한 단위. 모듈 ③ Monologue 재조립의 출력 형식.

    백채널(짧은 응답)은 결합 과정에서 skip되며, 결합 후 토큰 수가 임계값 미만인
    monologue는 필터링된다.
    """

    source: str
    """Corpus identifier (e.g. 'SBCSAE')."""

    monologue_id: str
    """Monologue identifier (e.g. 'SBC016#mono_0001')."""

    speaker: str
    """Speaker identifier."""

    text: str
    """결합된 monologue 텍스트 (cleaned utterance text들을 공백으로 join)."""

    utterance_ids: tuple[str, ...]
    """이 monologue를 구성한 원본 utterance ID들 (디버깅·트레이스용)."""

    n_tokens: int
    """단어 토큰 수 (pause 등 특수 토큰 제외)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """추가 메타데이터."""


@dataclass(frozen=True)
class Pair:
    """(formal_text, spoken_text) 학습 페어. 모듈 ④ LLM 역변환의 출력 형식.

    파인튜닝 시 모델은 formal_text → spoken_text 방향으로 학습한다.
    spoken_text는 monologue.text 원본을 그대로 보존하며(pause 토큰 포함),
    formal_text는 LLM이 생성한 정제된 문어체 paraphrase이다.
    """

    pair_id: str
    """Pair identifier (e.g. 'SBC016#mono_0001#casual#v1-zero-shot')."""

    source: str
    """Corpus identifier (e.g. 'SBCSAE')."""

    style: str
    """Style label (e.g. 'casual', 'semi-formal'). cf. ADR-0005."""

    speaker: str
    """Speaker identifier."""

    spoken_text: str
    """원본 monologue 텍스트 (pause 토큰 보존)."""

    formal_text: str
    """LLM이 생성한 정제된 문어체 paraphrase."""

    monologue_id: str
    """이 페어를 생성한 원본 monologue ID (트레이스용)."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """LLM 모델, prompt_version, token usage 등."""
