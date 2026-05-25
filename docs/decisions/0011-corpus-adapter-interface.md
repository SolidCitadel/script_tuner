# ADR-0011: 코퍼스 어댑터 인터페이스 + stem-centric 파이프라인

- **Status**: Accepted
- **Date**: 2026-05-26

## Context

[ADR-0006](0006-adapter-structure-and-common-ir.md)이 "코퍼스별 ①② 어댑터 + 공통 ③~⑤" 구조를 정했지만, PoC(SBCSAE 단독) 단계의 CLI는 그 구조를 실체화하지 않았다:

- `parse`/`clean` 러너가 `if args.corpus != "sbcsae": raise`로 하드코딩.
- `parse` 서브커맨드가 **단일 파일 경로**(`input_path`)를 받음.

두 번째 코퍼스 Switchboard는 대화당 **2파일**(A/B면)을 인터리브하므로(cf. [ADR-0009](0009-switchboard-turn-reconstruction.md)) "파일 1개 = 처리 단위" 가정이 깨진다. 두 번째 코퍼스 추가가 어댑터 추상화를 실체화할 적기다.

## Decision

**`Adapter` 프로즌 dataclass + 레지스트리**를 도입한다 (`scripttuner/corpora.py`).

```python
@dataclass(frozen=True)
class Adapter:
    source_name: str
    download: Callable[..., list[Path]]
    enumerate_stems: Callable[[Path], list[str]]      # corpus_dir -> stems
    parse_stem: Callable[[Path, str], list[Utterance]]  # (corpus_dir, stem) -> utterances
    clean: Callable[[list[Utterance]], list[Utterance]]
    backchannel_words: frozenset[str] = DEFAULT_BACKCHANNEL_WORDS

REGISTRY: dict[str, Adapter] = {"sbcsae": ..., "switchboard": ...}
```

- **stem-centric 계약**: 모든 어댑터는 `(corpus_dir, stem)`만으로 입력을 해석한다. CHAT은 `{stem}.cha` 1파일, Switchboard는 `{stem}{A,B}-...trans.text` 2파일 인터리브.
- CLI(`download/parse/clean/monologue/run`)는 `REGISTRY`로 디스패치. `parse` 서브커맨드를 경로 기반 → **stem 기반**(`parse <corpus> <stem> --datasets-dir`)으로 통일.
- `run`에 **`--through {stage}`** 추가: 지정 단계까지만 실행(기본 `stats`). `--through monologue`로 **LLM(④ pairs) 전에 중단** 가능 — 결정적 전처리 산출물을 먼저 확보하고 학습 단계에서 데이터를 증분 투입하기 위함.
- 백채널 사전을 어댑터가 보유하고 ③ `build_monologues`가 주입받는다(코퍼스 격리, cf. ADR-0009).

## Consequences

### 긍정적

- 새 코퍼스 추가 = 어댑터 1건 등록. ③~⑤·IR 무변경 (ADR-0006 계약 유지).
- `parse`/`clean`의 corpus 하드코딩 제거, 입력 해석이 stem-centric으로 일관.
- `--through`로 비용 단계(LLM) 분리 — 전처리/학습 라이프사이클 디커플링.

### 부정적

- `parse` 서브커맨드 API 변경(경로 → stem). 기존 CLI 테스트 갱신 필요(완료). dev CLI라 영향 한정.

## Alternatives Considered

| 안 | 기각 이유 |
|---|---|
| 최소 디스패치(corpus→함수 dict만, SBCSAE 경로 보존, parse는 path 유지) | `if corpus==` 분기 누적, parse 입력 단위가 코퍼스마다 달라 장기 일관성 저하 (사용자 결정으로 어댑터 인터페이스 채택) |

## References

- [ADR-0006](0006-adapter-structure-and-common-ir.md), [ADR-0009](0009-switchboard-turn-reconstruction.md)
- `scripttuner/corpora.py`, `scripttuner/cli.py`
