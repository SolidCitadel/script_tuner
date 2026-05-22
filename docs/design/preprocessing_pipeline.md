# 전처리 파이프라인 설계

> 본 문서는 CHA 원본을 (문어체, 구어체) 병렬 코퍼스로 변환하는 전처리 파이프라인의 정적 설계서이다. 본 단계에서는 학습/추론은 다루지 않는다. 관련 결정 이력은 [`../decisions/`](../decisions/), 진행 상황은 [`../status.md`](../status.md)를 참고한다.

## 목표

CHA(CHILDES) 포맷의 원어민 대화 전사 데이터를 **OPIc 모놀로그 학습용 (문어체 ↔ 구어체) 병렬 코퍼스**로 변환한다.

## 입출력

- **입력**: 코퍼스별 원본 파일 (현재는 CHAT `.cha`, 향후 NXT/XML/plain 등)
- **출력**: 모듈별 JSONL 산출물 (cf. [ADR-0001](../decisions/0001-jsonl-output-format.md))

## 아키텍처

본 파이프라인은 **어댑터 구조 + 공통 IR** 패턴을 따른다 (cf. [ADR-0006](../decisions/0006-adapter-structure-and-common-ir.md)).

- ① 파서 + ② Cleaner는 **코퍼스/포맷별 어댑터**에 묶임
- 어댑터는 자기 포맷을 공통 IR(`Utterance`)로 변환
- ③ Monologue 재조립, ④ LLM 역변환, ⑤ 통계는 IR만 다루므로 **코퍼스 무관**

### 패키지 구조

```
scripttuner/
├── preprocessing/
│   ├── ir.py                # 공통 IR — Utterance, Monologue, Pair dataclass
│   ├── chat/                # CHAT (CHILDES) 어댑터 — SBCSAE 등
│   │   ├── parser.py        # ① CHAT 파서
│   │   └── cleaner.py       # ② CHAT 정규화
│   ├── switchboard/         # (미래) Switchboard 어댑터
│   ├── monologue.py         # ③ 공통
│   ├── pairs.py             # ④ 공통 — LLMClient Protocol + convert_to_formal
│   └── stats.py             # ⑤ 공통
├── persistence/             # 디스크 적재/직렬화 영역
│   ├── jsonl.py             #   JSONL I/O (dataclass ↔ dict)
│   └── cache.py             #   sha256-keyed JSON KV 캐시 (LLM 응답 등)
└── llm/                     # LLM 클라이언트 (provider-agnostic, cf. ADR-0007)
    └── openai_compatible.py #   OpenAI SDK 래퍼
```

## 파이프라인 구조

```
[입력] *.cha (or other corpus format)
   │
   ▼
① 파서 (e.g. chat/parser.py)
   - 코퍼스별 헤더/메타 파싱
   - 발화 라인 → Utterance(speaker, text=raw, t_start, t_end, metadata)
   - 멀티라인 발화 결합
   - 마커는 손대지 않음 (text에 그대로, 타임스탬프 토큰만 분리)
   │
   ▼ parsed/*.jsonl
   │
② Cleaner (e.g. chat/cleaner.py)
   - 코퍼스별 마커 처리 (CHAT의 경우 아래 "마커 처리 정책" 참조)
   - text를 정규화된 형태로 갱신
   │
   ▼ cleaned/*.jsonl
   │
③ Monologue 재조립 (monologue.py) — 코퍼스 무관
   - 동일 화자 연속 발화 병합
   - 백채널(Okay/Yeah/Mhm 등 짧은 응답) 식별 후 skip (cf. ADR-0004)
   - 최소 길이 필터링 (예: ≥30 tokens)
   │
   ▼ monologues/*.jsonl
   │
④ (구어체 → 문어체) 역변환 (pairs.py) — 코퍼스 무관
   - LLM 호출로 정제된 문어체 페어 생성 (provider-agnostic, cf. ADR-0007)
   - 입력 전처리: <pause:*> 토큰 strip (LLM은 토큰을 모름, cf. ADR-0008)
   - 출력 후처리: typography ASCII 정규화 (스마트 따옴표·em-dash → ASCII)
   - 디스크 캐싱: sha256(prompt_version+model+stripped_input) 키 (cf. ADR-0007)
   - 메타: style="casual" (cf. ADR-0005)
   │
   ▼ pairs/*.jsonl
   │
⑤ 통계 / 검증 (stats.py) — 코퍼스 무관
   - 페어 수, 길이 분포, 어휘 밀도, 구동사 비율, 필러 빈도, 포즈 빈도
   - 진단 모듈의 ground truth로 활용
   │
   ▼ stats/*.json
```

## 공통 IR (`Utterance`)

모든 어댑터의 출력 표준. 코퍼스 무관 최소 공통 필드만 둔다.

```python
@dataclass(frozen=True)
class Utterance:
    source: str                       # "SBCSAE", "Switchboard" 등
    utterance_id: str                 # 코퍼스 내 고유 ID
    speaker: str                      # 필수 — monologue 재조립의 키
    text: str                         # stage에 따라 raw or cleaned
    t_start_ms: int | None = None
    t_end_ms: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
```

자세한 설계 근거는 [ADR-0006](../decisions/0006-adapter-structure-and-common-ir.md) 참조.

## 모듈별 책임 분리 원칙

- ① 파서: 코퍼스별 — **데이터 추출만**, 마커는 text에 그대로 보존, 타임스탬프 토큰만 분리
- ② Cleaner: 코퍼스별 — **마커 처리 전담**, parsing·monologue 로직 없음
- ③ Monologue: 공통 — **턴 구조 처리 전담**, 어휘적 변경 없음
- ④ Pairs: 공통 — **외부 LLM 호출 전담**
- ⑤ Stats: 공통 — **read-only**, 데이터 변경 없음

각 모듈은 JSONL 입출력으로 결합되므로 단위 실행·재실행이 격리된다.

## 마커 처리 정책 (모듈 ② Cleaner)

| 마커 | 의미 | 처리 |
|---|---|---|
| `(.)`, `(..)` | 짧은/긴 포즈 | `<pause:short>` / `<pause:long>` |
| `um`, `uh`, `you know`, `I mean`, `well` | 단어형 필러 | 보존 |
| `+/.` | 발화 중단 | 자연 종결 처리 |
| `+...` | 말끝 흐림 | `...` |
| `ʔuh` | 성문 폐쇄음 표기 | `uh` |
| `I:`, `u:m`, `perc:e:nt` | 모음 늘림 (vowel lengthening) | 알파벳 직후 `:` 제거 |
| `&=tsk`, `&=laugh`, `&=in`, `&=ex` | 비언어 어노테이션 | 제거 |
| `&{l=X ... &}l=X` | 불명확/L2 표기 | 제거 |
| `⌈ ⌉`, `⌊n ⌋n` | 오버랩 마커 | 제거 |
| `760_1735` | 타임스탬프 | 제거 |

## 출력 디렉토리 구조

```
data/                                  # gitignore
├── parsed/<SOURCE>/<stem>.jsonl       # ① 파서 출력 (Utterance)
├── cleaned/<SOURCE>/<stem>.jsonl      # ② 정규화 출력 (Utterance)
├── monologues/<SOURCE>/<stem>.jsonl   # ③ 재조립 출력 (Monologue)
├── pairs/<SOURCE>/<stem>.jsonl        # ④ 역변환 출력 (Pair)
├── stats/<SOURCE>/<stem>.json         # ⑤ 통계
└── cache/pairs/<sha256>.json          # ④ LLM 응답 디스크 캐시 (cf. ADR-0007)
```

`<SOURCE>`는 corpus의 `SOURCE_NAME` (예: `SBCSAE`), `<stem>`은 파일 식별자(예: `SBC016`).

원본 CHA는 `datasets/`에 위치하며 다운로드 스크립트로 확보한다 (cf. [ADR-0002](../decisions/0002-sbcsae-license-policy.md)).

## 출력 페어 스키마 (모듈 ④ 산출물)

`Pair` dataclass (`scripttuner/preprocessing/ir.py`) 를 JSONL로 직렬화.

```python
@dataclass(frozen=True)
class Pair:
    pair_id: str          # e.g. "SBC016#mono_0001#casual#v2-zero-shot"
    source: str           # "SBCSAE"
    style: str            # "casual" (cf. ADR-0005)
    speaker: str
    spoken_text: str      # 원본 monologue.text (pause 토큰 포함, cf. ADR-0008)
    formal_text: str      # LLM 출력 (typography ASCII 정규화 적용)
    monologue_id: str     # 트레이스용
    metadata: dict        # model, prompt_version, prompt/completion_tokens, from_cache 등
```

JSON 예시:

```json
{
  "pair_id": "SBC016#mono_0002#casual#v2-zero-shot",
  "source": "SBCSAE",
  "style": "casual",
  "speaker": "TAMM",
  "spoken_text": "Um  <pause:long> because on a  <pause:short> ... you're not gonna be able ...",
  "formal_text": "Because on a tape deck like that you realize that you are not going to be able ...",
  "monologue_id": "SBC016#mono_0002",
  "metadata": {
    "model": "openai/gpt-oss-120b:free",
    "prompt_version": "v2-zero-shot",
    "prompt_tokens": 255,
    "completion_tokens": 67,
    "from_cache": false
  }
}
```

단순 페어 단위 메트릭(spoken/formal 토큰 수, 필러 빈도, pause 빈도 등)은 모듈 ⑤ Stats에서 별도 산출한다.

## 향후 확장

- **Semi-formal 스타일 데이터 통합**: 동일 파이프라인 재사용, `style` 메타속성만 변경 (cf. [ADR-0005](../decisions/0005-style-as-dataset-metadata.md))
- **다른 코퍼스(Switchboard, BNC, TED 등) 통합**: `scripttuner/preprocessing/<format>/` 하위에 새 어댑터(parser + cleaner) 추가. ③~⑤ 재사용. IR 확장이 필요한 경우 `ir.py`의 `metadata` dict로 흡수 가능.