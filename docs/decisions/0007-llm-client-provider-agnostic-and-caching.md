# ADR-0007: LLM 클라이언트 provider-agnostic + 디스크 캐싱

- **Status**: Accepted
- **Date**: 2026-05-22

## Context

모듈 ④ (pairs)에서 (구어체 → 문어체) paraphrase를 생성하기 위해 외부 LLM API를 호출한다. 두 가지 결정이 필요했다:

1. **어떤 SDK / provider를 어떻게 통합할 것인가** — 운영 default 모델이 바뀌거나 provider를 옮길 때 코드 변경 비용이 들지 않아야 한다.
2. **반복 호출의 비용/시간을 어떻게 줄일 것인가** — PoC 단계에서 프롬프트 튜닝, 재실행, 부분 확장이 잦다. 동일 입력은 결정성이 충분히 높으므로 캐시 가치가 크다.

## Decision

### 1. provider-agnostic OpenAI-호환 SDK 채택

- 클라이언트는 `openai` Python SDK 한 가지만 사용한다.
- SDK가 자동 인식하는 표준 환경변수 두 개에 모든 provider 설정을 위임한다:
  - `OPENAI_API_KEY` — provider의 API 키
  - `OPENAI_BASE_URL` — provider endpoint (없으면 OpenAI 기본)
- 모델 슬러그는 `LLM_MODEL` 환경변수 또는 CLI `--model` 옵션으로 받는다.
- 코드에는 어떤 provider URL이나 모델 슬러그도 default로 하드코딩하지 않는다.
- OpenAI 호환 endpoint면 어떤 backend(OpenAI, OpenRouter, Together, Groq, 로컬 vLLM 등)든 동일 코드로 동작한다.

`scripttuner/preprocessing/pairs.py`는 `LLMClient` Protocol에만 의존하고, 구현은 `scripttuner/llm/openai_compatible.py`에 둔다. 다른 backend 필요 시 동일 Protocol을 만족하는 모듈 추가만 하면 된다.

### 2. sha256-keyed JSON 디스크 캐싱

- LLM 응답을 `data/cache/pairs/<sha256>.json`에 저장한다.
- 캐시 키 = `sha256(prompt_version + model + stripped_user_text)`.
  - `prompt_version`이 들어가므로 프롬프트 변경 시 자동 cache miss.
  - `model`이 들어가므로 모델 교체 시 자동 cache miss.
  - `stripped_user_text`는 pause 토큰 제거 후 정규화된 LLM 입력 (cf. ADR-0008).
- **실패 응답은 캐시에 저장하지 않는다.**
- 캐시 모듈은 `scripttuner/persistence/cache.py`의 `DiskCache` + `make_cache_key`. JSONL과 같은 영역(`persistence/`)에 둔다.
- LLM 응답에 대한 typography 정규화(스마트 따옴표 → ASCII 등)는 **캐시 읽은 후 적용**한다. 정규화 규칙 변경 시 캐시 무효화 없이 새 결과를 얻을 수 있다.

## Consequences

### 긍정적

- provider 락인 회피 — 모델 비용/성능 변경 시 `.env`만 수정.
- 환경 변수 이름이 SDK 표준이라 다른 LLM 도구와 충돌 없음 (`OPENROUTER_API_KEY` 같은 provider-특정 이름 회피).
- 동일 입력 재호출 비용 0, 시간 즉시. PoC의 반복 실행 사이클을 짧게 유지.
- 프롬프트/모델/입력 어느 차원이 바뀌어도 캐시 키 충돌 없이 자동 분리.

### 부정적

- 비-OpenAI-호환 backend(예: Anthropic 네이티브 SDK)를 직접 쓰려면 별도 `LLMClient` 구현이 필요 — 단, OpenAI 호환 어댑터 layer가 워낙 흔해서 실제로는 거의 발생하지 않음.
- 캐시 디렉토리가 누적 증가 — `data/cache/` 자체가 gitignore이고 디스크 비용이 미미해 운영 부담 없음.

## Alternatives Considered

| 안 | 채택 안 한 이유 |
|---|---|
| Anthropic SDK 직접 | 단일 provider 락인. 모델 비용/가용성 변경 시 코드 수정 |
| LiteLLM 같은 추가 추상화 layer | OpenAI 호환 endpoint면 불필요. 의존성 추가 비용 |
| 직접 HTTP 호출 | 가독성·재시도·인증 처리 모두 수동. SDK가 무료로 처리해주는 것 재구현 |
| provider별 환경변수 (`OPENROUTER_API_KEY` 등) | provider 락인이 환경변수 이름에까지 새겨짐. SDK 자동 인식 표준명에서 벗어남 |
| 캐시 없음 | PoC의 잦은 재호출 비용/시간 낭비 |
| 인메모리 캐시 | 프로세스 재시작 시 사라짐 — PoC에서 가장 자주 발생하는 사이클 |
| sqlite 캐시 | 단일 KV에 과한 의존성. JSON 파일 한 개로 충분 |

## References

- [docs/design/preprocessing_pipeline.md](../design/preprocessing_pipeline.md) — 모듈 ④ pairs
- [ADR-0008](0008-pause-token-strip-on-llm-input.md) — pause 토큰 LLM 입력 strip 정책
- `scripttuner/preprocessing/pairs.py` — `LLMClient` Protocol + `convert_to_formal`
- `scripttuner/llm/openai_compatible.py` — OpenAI-호환 구현
- `scripttuner/persistence/cache.py` — 디스크 캐시
