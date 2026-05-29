# ScriptTuner

> 구어체 특화형 영어 스크립트 변환 모델 — OPIc 등 영어 말하기 평가 대비용 비원어민 스크립트를 파인튜닝된 어댑터로 자연스러운 구어체로 변환한다.

**팀**: QAIP (임준혁, 전현준, 최인재)

자세한 프로젝트 배경은 [`background/QAIP_proposal_report.md`](background/QAIP_proposal_report.md).

## 디렉토리 구조

```
aip/
├── scripttuner/                    # 메인 Python 패키지
│   ├── cli.py                      #   CLI 진입점 (subcommands)
│   ├── persistence/                #   디스크 적재/직렬화 영역
│   │   ├── jsonl.py                #     JSONL I/O (dataclass ↔ dict)
│   │   └── cache.py                #     sha256-keyed JSON KV 캐시
│   ├── llm/                        #   LLM 클라이언트 (provider-agnostic)
│   │   └── openai_compatible.py    #     OpenAI SDK 래퍼
│   ├── preprocessing/
│   │   ├── ir.py                   #   공통 IR (Utterance, Monologue, Pair)
│   │   ├── monologue.py            #   ③ Monologue 재조립
│   │   ├── pairs.py                #   ④ LLM 역변환 (구어체→문어체)
│   │   ├── stats.py                #   ⑤ Pair 집계 통계
│   │   └── chat/                   #   CHAT (CHILDES) 어댑터
│   │       ├── parser.py           #     ① 파서
│   │       └── cleaner.py          #     ② 정규화
│   └── data_sources/
│       └── sbcsae.py               #   SBCSAE 다운로더
├── tests/                          # 단위 테스트
├── datasets/                       # 원본 데이터 (gitignore, 다운로드 스크립트로 확보)
├── data/                           # 파이프라인 산출물 (gitignore)
├── docs/                           # 프로젝트 문서
│   ├── status.md                   #   진행 현황 (자주 업데이트)
│   ├── design/                     #   정적 설계 문서
│   └── decisions/                  #   ADR — 결정 이력
├── background/                     # 외부 자료 (제안서 등)
├── .work/                          # 개인 작업 메모 (gitignore)
└── pyproject.toml
```

**아직 미생성 (계획)**: 진단 모듈(`diagnosis/`), 변환 모델(`transform/`), 백엔드(`api/`). 진행도는 [`docs/status.md`](docs/status.md) 참조.

## 빠른 진입점

| 알고 싶은 것 | 보세요 |
|---|---|
| 지금 어디까지 진행됐는지 | [`docs/status.md`](docs/status.md) |
| 전처리 파이프라인 설계 | [`docs/design/preprocessing_pipeline.md`](docs/design/preprocessing_pipeline.md) |
| 데이터셋 분석 | [`docs/design/dataset_review.md`](docs/design/dataset_review.md) |
| "왜 이렇게 결정했는가" | [`docs/decisions/`](docs/decisions/) — ADR 목록 |
| 프로젝트 제안서 원본 | [`background/QAIP_proposal_report.md`](background/QAIP_proposal_report.md) |

## 데이터 라이선스

본 프로젝트가 사용하는 1차 데이터셋은 **SBCSAE (Santa Barbara Corpus of Spoken American English)**, CC BY-ND 3.0 US 라이선스이다. 정책은 [ADR-0002](docs/decisions/0002-sbcsae-license-policy.md) 참조.

- 원본 데이터는 git에 포함되지 않음 → `uv run scripttuner download sbcsae`로 확보
- 전처리 산출물은 ND 조항에 따라 공개 배포하지 않음

## 환경 셋업

전제: Python 3.11+, [uv](https://docs.astral.sh/uv/) 설치 필요.

```bash
# 1) 의존성 설치 (.venv 생성)
uv sync

# 2) SBCSAE 데이터셋 다운로드 (60개 .cha → datasets/sbcsae/)
uv run scripttuner download sbcsae

# 3) end-to-end (parse → clean → monologue → pairs → stats)
uv run scripttuner run sbcsae SBC016                  # 단일 파일
uv run scripttuner run sbcsae SBC016 SBC017 SBC018    # 여러 파일
uv run scripttuner run sbcsae --all                   # corpus 전체
# → data/{parsed,cleaned,monologues,pairs,stats}/SBCSAE/<stem>.*

# 또는 단계별로 실행
uv run scripttuner parse sbcsae datasets/sbcsae/SBC016.cha
uv run scripttuner clean sbcsae SBC016
uv run scripttuner monologue sbcsae SBC016
uv run scripttuner pairs sbcsae SBC016 --model <provider/model-slug>   # .env 필요
uv run scripttuner stats sbcsae SBC016                                  # spacy 모델 필요

# 4) 테스트 / 린트 / 타입체크
uv run pytest
uv run ruff check .
uv run mypy scripttuner tests
```

### spaCy 영어 모델 (stats 모듈의 POS 지표용)

`en_core_web_sm`은 의존성으로 선언돼 있어 `uv sync` 시 자동 설치된다 (별도 다운로드 불필요).

### 파인튜닝 학습 의존성 (optional `train` 그룹)

QLoRA 파인튜닝 스택(Unsloth + torch)은 무겁고 GPU 종속이라 core가 아닌 optional 그룹이다.

```bash
uv sync --group train
```

> ⚠️ 이후 동기화는 `uv sync --group train`으로 할 것. 그냥 `uv sync`를 돌리면 학습 의존성(torch/unsloth 등)이 **제거된다**.

### LLM provider 설정

`.env.example`을 `.env`로 복사 후 값 채움 (`.env`는 gitignore됨):

```bash
OPENAI_API_KEY=<your-key>
OPENAI_BASE_URL=<provider-endpoint>
# LLM_MODEL=<model-slug>   # 선택; CLI --model로도 가능
```

## 팀원 Quick Start (학습된 모델 사용/재현)

학습된 어댑터·데이터는 HF org **`aip-scripttuner-team`**(private)에 있다.
SBCSAE가 CC BY-ND라 전부 private이며, 접근하려면 org 멤버 초대가 필요하다.

**사전 준비**
1. [HF 계정](https://huggingface.co) → org `aip-scripttuner-team` 멤버 초대 수락 (유지보수자에게 요청)
2. base 모델 [라이선스 수락](https://huggingface.co/google/t5gemma-2-1b-1b) (gated)
3. HF Settings → Access Tokens에서 read 토큰 발급 → `.env`에 `HF_TOKEN=<토큰>`
4. `git clone https://github.com/SolidCitadel/script-tuner.git && cd script-tuner && uv sync --group train`

**A. 학습된 모델만 써보기 (재학습 X, GPU 권장)**
```bash
# 어댑터는 HF에서 자동 다운로드/캐시된다 (--adapter-dir에 repo id 지정)
uv run scripttuner generate t5gemma2-1b sbcsae --split test \
    --adapter-dir aip-scripttuner-team/scripttuner-t5gemma2-1b-sbcsae-casual
uv run scripttuner evaluate --predictions runs/eval/t5gemma2-1b-SBCSAE-lora/predictions.jsonl
```

**B. 직접 학습 (전처리·LLM 비용 스킵)**
```bash
# formatted splits를 HF dataset에서 받아 학습 입력 위치에 둠
huggingface-cli download aip-scripttuner-team/scripttuner-sbcsae-formatted-t5gemma2-1b \
    --repo-type dataset --local-dir data/finetune/SBCSAE/formatted/t5gemma2-1b
uv run scripttuner train t5gemma2-1b sbcsae \
    --batch-size 1 --grad-accum 16 --max-seq-length 1024 --epochs 8   # ~2h on 8GB GPU
uv run scripttuner generate t5gemma2-1b sbcsae --split test
uv run scripttuner evaluate --predictions runs/eval/t5gemma2-1b-SBCSAE-lora/predictions.jsonl
uv run scripttuner plot t5gemma2-1b sbcsae
```

**C. 전처리부터 전부 재현** — 위 [환경 셋업](#환경-셋업)의 `download → run → aggregate → split → format`을 거친 뒤 B의 `train` 이후를 실행. `pairs` 단계는 LLM API 키가 필요하다.

> 어댑터 재업로드(재학습 후)는 `scripts/push_to_hf.py` 참조. 보고서용 집계 결과는 [`report/`](report/).