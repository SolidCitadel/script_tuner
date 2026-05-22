# ADR-0008: LLM 입력 전 pause 토큰 strip (spoken은 보존)

- **Status**: Accepted
- **Date**: 2026-05-22

## Context

[ADR-0003](0003-pause-marker-tokenization.md)에서 CHA 포즈 마커를 `<pause:short>`, `<pause:long>` 특수 토큰으로 보존하기로 했다. `Monologue.text`는 이 토큰을 포함한다.

모듈 ④ (pairs)는 monologue를 LLM에 보내 문어체 paraphrase를 받는다. 여기서 두 가지 질문이 생긴다:

1. LLM은 우리가 만든 `<pause:*>` 토큰을 모른다. 그대로 보내면 LLM이 토큰을 그대로 echo하거나, 일반 텍스트로 오해해 출력에 섞을 수 있다.
2. 그러나 학습 시점 모델은 (formal → spoken) 방향으로 파인튜닝되고, 출력 쪽 spoken에는 pause 패턴이 반드시 있어야 한다 — 학습 모델이 자연 발화 중 끊김 위치를 학습할 신호이기 때문이다 (cf. ADR-0003).

따라서 "LLM에 어떻게 줄 것인가"와 "학습 데이터에 어떻게 보존할 것인가"를 분리해서 결정해야 한다.

## Decision

**LLM 입력에서만 pause 토큰을 strip하고, `Pair.spoken_text`에는 원본 `Monologue.text`를 그대로 보존한다.**

- `scripttuner/preprocessing/pairs.py`의 `_strip_special_tokens()`가 정규식 `<pause:\w+>`을 매치해 제거 + multispace 정리.
- LLM의 user 메시지에 들어가는 텍스트만 strip 적용.
- `Pair.spoken_text`는 `Monologue.text` 원본을 그대로 저장 (pause 토큰 포함).
- 캐시 키도 strip된 텍스트의 해시를 사용한다 (cf. ADR-0007) — 동일 strip 결과는 동일 LLM 호출이므로 캐시 적중이 자연스럽다.

## Consequences

### 긍정적

- LLM이 알 수 없는 토큰을 입력에 보지 않으므로 출력 품질 안정.
- 학습 데이터 spoken 쪽은 pause 패턴이 그대로 살아있어 변환 모델 학습 신호 보존 (cf. ADR-0003).
- 입력 strip은 후처리 한 줄이라 ADR-0003 결정과 직교 (서로 영향 없음).

### 부정적

- formal_text에는 pause 정보가 반영되지 않음 — 모델은 formal 입력만 보고 pause 위치를 추론해야 한다. 다만 본 프로젝트의 정의상 formal은 "정돈된 written script"이고 pause는 자연 발화 단계에서 부여되는 신호이므로, formal에 pause가 없는 것이 오히려 데이터 의미와 일치한다.

## Alternatives Considered

| 안 | 채택 안 한 이유 |
|---|---|
| 토큰 그대로 LLM에 전달 | LLM이 토큰을 모르고, 출력에 echo하거나 무시. spoken_text와 일관성 없는 결과 위험 |
| 자연어로 풀어서 전달 (e.g. `(pause)`) | LLM 입력 토큰 ↑. 무엇보다 결과가 다시 spoken 형태에 섞일 위험 |
| 시스템 프롬프트에 토큰 의미 설명 | 토큰 ↑, LLM 준수 의존. 단순 strip이 더 안전하고 결정적 |
| spoken_text에서도 pause 제거 | ADR-0003의 학습 신호를 잃음 |

## References

- [ADR-0003](0003-pause-marker-tokenization.md) — pause 토큰화 결정
- [ADR-0007](0007-llm-client-provider-agnostic-and-caching.md) — LLM 클라이언트 + 캐싱
- `scripttuner/preprocessing/pairs.py` — `_strip_special_tokens`, `convert_to_formal`