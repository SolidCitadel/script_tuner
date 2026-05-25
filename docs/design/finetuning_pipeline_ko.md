# 파인튜닝 파이프라인 상세 설계

이 문서는 ScriptTuner 프로젝트의 파인튜닝 사전 준비 과정을 팀원들과 공유하기 위한 한국어 설명 문서다. 기존 전처리 파이프라인에서 만들어진 `Pair` 데이터를 어떤 방식으로 나누고, 모델별 학습 포맷으로 변환하며, 이후 Gemma 4 E4B를 중심으로 어떻게 QLoRA/SFT 학습으로 이어갈지 정리한다.

## 1. 전체 목표

ScriptTuner의 목표는 사용자가 작성한 영어 스크립트를 OPIc 말하기 시험에 더 적합한 자연스러운 구어체 영어 답변으로 바꾸는 것이다.

모델 관점에서 학습 문제는 다음과 같이 정의한다.

```text
입력: formal_text
출력: spoken_text
```

즉, 정제된 문어체 영어 또는 다소 딱딱한 스크립트를 입력하면, 모델이 자연스럽고 말하듯 들리는 영어 답변을 생성하도록 학습한다.

## 2. 기존 전처리 파이프라인과의 연결

기존 프로젝트는 SBCSAE 데이터를 다음 단계로 처리한다.

```text
CHA 원본
-> parse
-> clean
-> monologue
-> pairs
-> stats
```

각 단계의 역할은 다음과 같다.

| 단계 | 입력 | 출력 | 설명 |
|---|---|---|---|
| parse | `.cha` 원본 파일 | `Utterance` JSONL | CHAT 형식 발화 라인을 공통 IR로 변환 |
| clean | parsed utterances | cleaned utterances | pause, filler, 전사 마커 등을 정리 |
| monologue | cleaned utterances | `Monologue` JSONL | 같은 화자의 연속 발화를 하나의 monologue로 결합 |
| pairs | monologues | `Pair` JSONL | LLM을 사용해 spoken text의 formal paraphrase 생성 |
| stats | pairs | 통계 JSON | token 길이, filler, pause, lexical density 등 분석 |

이번에 추가한 파인튜닝 준비 파이프라인은 `pairs` 이후에 붙는다.

```text
pairs
-> split
-> format
-> train
-> generate
-> evaluate
```

현재 구현된 것은 `split`과 `format`까지다. `train`, `generate`, `evaluate`는 다음 단계에서 추가할 예정이다.

## 3. 원천 학습 데이터

현재 파인튜닝 준비의 입력 파일은 다음이다.

```text
data/pairs/SBCSAE/_all.jsonl
```

이 파일은 전체 SBCSAE 60개 파일에 대한 pair aggregate다.

현재 데이터 요약:

| 항목 | 값 |
|---|---:|
| 전체 pair 수 | 1,757 |
| 고유 speaker 수 | 131 |
| 현재 style | casual |
| 학습 입력 | `formal_text` |
| 학습 target | `spoken_text` |

`Pair` 데이터의 주요 필드는 다음과 같다.

```json
{
  "pair_id": "...",
  "source": "SBCSAE",
  "style": "casual",
  "speaker": "...",
  "spoken_text": "...",
  "formal_text": "...",
  "monologue_id": "...",
  "metadata": {}
}
```

여기서 실제 파인튜닝에 가장 중요한 필드는 `formal_text`, `spoken_text`, `style`, `speaker`다.

## 4. Split 단계

### 4.1 목적

파인튜닝에서는 데이터를 train, validation, test로 나눠야 한다.

단순히 pair 단위로 랜덤 분리하면 같은 speaker의 말투가 train과 test에 동시에 들어갈 수 있다. 그러면 모델이 실제로 일반화한 것이 아니라, 같은 화자의 반복적인 표현이나 전사 특성을 기억한 것처럼 보일 수 있다.

따라서 이번 파이프라인에서는 speaker-aware split을 사용한다.

### 4.2 방식

같은 speaker에 속한 pair는 반드시 하나의 split에만 들어간다.

```text
speaker A -> train
speaker B -> validation
speaker C -> test
```

이렇게 하면 speaker 단위 누수를 줄일 수 있다.

### 4.3 실행 명령

프로젝트 루트가 `script-tuner-main`일 때 다음 명령을 실행한다.

```bash
python -m scripttuner.cli split sbcsae --data-dir data --seed 42
```

### 4.4 출력 위치

```text
data/finetune/SBCSAE/splits/
  train.jsonl
  validation.jsonl
  test.jsonl
  MANIFEST.json
```

### 4.5 현재 생성 결과

| split | pair 수 | speaker 수 |
|---|---:|---:|
| train | 1,405 | 107 |
| validation | 176 | 12 |
| test | 176 | 12 |

`MANIFEST.json`에는 split 비율, seed, speaker 목록, split별 pair 수가 저장된다.

## 5. Format 단계

### 5.1 목적

모델마다 학습 데이터 형식이 다르다.

Gemma 4와 Qwen 계열은 instruction 또는 chat 형식으로 학습하는 것이 자연스럽다. 반면 T5Gemma 2는 encoder-decoder 구조이므로 `input` / `target` text-to-text 형식이 더 잘 맞는다.

따라서 `format` 단계는 같은 split 데이터를 모델별 학습 포맷으로 변환한다.

### 5.2 지원 모델 키

현재 formatter가 지원하는 모델 키는 다음과 같다.

| model_key | 형식 | 용도 |
|---|---|---|
| `gemma4-e4b` | chat | 1차 주력 모델 |
| `gemma4-e2b` | chat | 더 가벼운 Gemma 4 비교 모델 |
| `qwen3-4b` | chat | Gemma 4 E4B와 비교할 대안 |
| `qwen3-1.7b` | chat | 경량 비교 모델 |
| `t5gemma2` | seq2seq | encoder-decoder 비교 모델 |

### 5.3 실행 명령

```bash
python -m scripttuner.cli format gemma4-e4b sbcsae --data-dir data
python -m scripttuner.cli format gemma4-e2b sbcsae --data-dir data
python -m scripttuner.cli format qwen3-4b sbcsae --data-dir data
python -m scripttuner.cli format qwen3-1.7b sbcsae --data-dir data
python -m scripttuner.cli format t5gemma2 sbcsae --data-dir data
```

### 5.4 출력 위치

```text
data/finetune/SBCSAE/formatted/
  gemma4-e4b/
    train.jsonl
    validation.jsonl
    test.jsonl
    MANIFEST.json
  gemma4-e2b/
  qwen3-4b/
  qwen3-1.7b/
  t5gemma2/
```

각 모델 폴더에는 동일한 split 구조가 유지된다. 따라서 나중에 모델별 성능을 비교할 때 데이터 분리 차이 때문에 생기는 변수를 줄일 수 있다.

## 6. Chat 모델용 포맷

Gemma 4 E4B, Gemma 4 E2B, Qwen3-4B, Qwen3-1.7B는 chat-style JSONL로 변환한다.

예시는 다음과 같다.

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<STYLE=casual>\nRewrite the input into natural casual spoken English...\n\nInput:\n...\n\nOutput:"
    },
    {
      "role": "assistant",
      "content": "Well, I think ..."
    }
  ],
  "pair_id": "...",
  "source": "SBCSAE",
  "style": "casual",
  "speaker": "...",
  "monologue_id": "...",
  "style_token": "<STYLE=casual>",
  "source_metadata": {}
}
```

`messages[0]`은 모델 입력이고, `messages[1]`은 정답 target이다.

## 7. T5Gemma 2용 포맷

T5Gemma 2는 encoder-decoder 모델이므로 seq2seq 구조로 변환한다.

예시는 다음과 같다.

```json
{
  "input": "<STYLE=casual>\nRewrite the input into natural casual spoken English...\n\nInput:\n...\n\nOutput:",
  "target": "Well, I think ...",
  "pair_id": "...",
  "source": "SBCSAE",
  "style": "casual",
  "speaker": "...",
  "monologue_id": "...",
  "style_token": "<STYLE=casual>",
  "source_metadata": {}
}
```

## 8. 스타일 제어 토큰

제안서에는 사용자가 변환 스타일을 Casual 또는 Semi-formal spoken 중 하나로 선택할 수 있다고 되어 있다. 이를 구현하기 위해 학습 입력 앞에 스타일 제어 토큰을 붙인다.

현재 정의된 토큰은 다음과 같다.

```text
<STYLE=casual>
<STYLE=semi_formal>
```

### 8.1 Casual

`casual`은 일상 대화체에 가까운 스타일이다.

특징:

- 자연스러운 담화 표지 사용
- 예: `well`, `you know`, `I mean`, `actually`
- 짧은 문장 단위
- contraction 사용 가능
- 너무 딱딱한 essay 느낌을 피함

### 8.2 Semi-formal spoken

`semi_formal`은 말하기 시험이나 영어 면접에 적합한 구어체다.

특징:

- spoken style이지만 지나치게 casual하지 않음
- filler를 과도하게 쓰지 않음
- 답변 구조가 비교적 명확함
- OPIc나 영어 인터뷰 답변에 적합

### 8.3 현재 상태

현재 SBCSAE 기반 데이터는 사실상 casual spoken 데이터다. 따라서 지금 생성된 formatted 데이터에는 `casual` 예시만 들어 있다.

`semi_formal`은 아직 실제 학습 예시가 없으므로, 다음 단계에서 데이터를 추가해야 한다.

가능한 방법:

1. 외부 semi-formal corpus 추가
2. teacher LLM으로 semi-formal target 생성
3. 초기 버전에서는 semi-formal을 prompt-only로 처리한 뒤, 추후 데이터가 생기면 학습에 포함

팀 프로젝트 일정상 가장 현실적인 순서는 다음과 같다.

```text
1차: casual 데이터로 Gemma 4 E4B QLoRA 성공
2차: teacher LLM으로 semi_formal 데이터 생성
3차: casual + semi_formal style-control 학습
4차: 모델별 성능 비교
```

## 9. Gemma 4 E4B를 1차 모델로 두는 이유

현재 1차 파인튜닝 모델은 Gemma 4 E4B로 잡는다.

이유:

- 경량 모델이라 실습 환경에서 QLoRA를 시도하기 좋음
- Gemma 4 계열은 최신 open-weight 모델 후보로 프로젝트 제안서 방향과 맞음
- E2B보다 품질이 좋을 가능성이 높고, 26B/31B보다 훨씬 가벼움
- 이후 Gemma 4 E2B, Qwen3, T5Gemma 2와 비교하기 좋은 기준점이 됨

## 10. 이후 학습 단계 설계

다음에 추가할 학습 단계는 다음 구조가 적절하다.

```text
scripttuner/
  training/
    train_lora.py
    generate.py
    evaluate.py
    registry.py
```

예상 명령:

```bash
python -m scripttuner.cli train gemma4-e4b --data-dir data --run-name gemma4-e4b-sbcsae-lora-001
python -m scripttuner.cli generate gemma4-e4b --checkpoint runs/finetune/gemma4-e4b-sbcsae-lora-001
python -m scripttuner.cli evaluate --predictions runs/eval/gemma4-e4b-sbcsae-lora-001/predictions.jsonl
```

아직 이 명령들은 구현 전이다. 이번 작업에서는 학습 전 데이터 준비까지만 완료했다.

## 11. 권장 run 산출물 구조

학습 결과는 다음처럼 저장하는 것을 권장한다.

```text
runs/
  finetune/
    gemma4-e4b-sbcsae-lora-001/
      adapter/
      tokenizer/
      training_args.json
      metrics.json
      MANIFEST.json
  eval/
    gemma4-e4b-sbcsae-lora-001/
      predictions.jsonl
      metrics.json
      samples.md
```

`MANIFEST.json`에는 다음 정보를 넣는 것이 좋다.

- base model name
- dataset path
- split manifest path
- formatter manifest path
- LoRA 설정
- 학습 epoch
- learning rate
- batch size
- seed
- 실행 날짜

## 12. 평가 계획

모델 평가에는 자동 평가와 사람 평가를 함께 사용해야 한다.

### 12.1 자동 평가 후보

| 평가 항목 | 목적 |
|---|---|
| validation loss | 학습 안정성 확인 |
| token length ratio | 출력이 너무 길거나 짧아지는지 확인 |
| embedding similarity | 의미 보존 확인 |
| BLEU/ROUGE | reference와의 표면 유사도 참고 |
| filler count | 구어체 표지 사용량 확인 |
| pause token count | pause token 생성 경향 확인 |
| lexical density | spoken/formal 스타일 차이 확인 |

BLEU나 ROUGE는 자연스러운 문장 변환 문제에서는 한계가 있으므로 참고용으로만 쓰는 것이 좋다.

### 12.2 사람 평가 후보

| 평가 항목 | 질문 |
|---|---|
| 의미 보존 | 원래 내용이 유지되었는가? |
| 자연스러움 | 실제 사람이 말하는 것처럼 들리는가? |
| OPIc 적합성 | 말하기 시험 답변으로 적절한가? |
| 과도한 filler 여부 | `um`, `uh`, `you know`가 너무 많지 않은가? |
| 문법/유창성 | 답변이 이해하기 쉽고 유창한가? |

## 13. 데이터 품질 관련 주의점

현재 `spoken_text`는 원본 구어체 전사 특성을 최대한 보존한다. 따라서 다음과 같은 요소가 target에 포함될 수 있다.

- `<pause:short>`
- `<pause:long>`
- filler
- 반복 표현
- 일부 전사 잔여 표기
- 웃음, 불명확 발화 등 corpus marker

이 중 pause token은 프로젝트에서 의도적으로 보존한 요소다. 다만 실제 사용자에게 보여줄 최종 출력에 pause token이나 전사 marker가 필요한지는 별도 정책이 필요하다.

가능한 정책:

1. 학습 target에는 pause token을 보존하고, 서비스 출력에서 후처리로 제거
2. 학습 target에서 pause token과 전사 marker를 제거
3. `<pause:short>` 정도만 남기고 나머지 marker 제거
4. pause token을 모델 학습에는 사용하되 UI에서는 자연스러운 쉼표나 문장 분리로 변환

현재는 기존 파이프라인 결정에 맞춰 보존하는 상태다. 본격 학습 전에는 이 정책을 팀에서 다시 결정하는 것이 좋다.

## 14. 현재 완료된 것과 남은 것

### 완료

- `formal_text -> spoken_text` 학습 방향 확정
- speaker-aware split 구현
- Gemma/Qwen/T5Gemma용 formatter 구현
- style control token 구조 도입
- casual 데이터 포맷 생성
- semi-formal 확장을 위한 구조 예약
- 한국어/영어 설계 문서 추가

### 남은 작업

- Gemma 4 E4B QLoRA 학습 코드 작성
- Hugging Face/Transformers/PEFT/TRL 의존성 추가 여부 결정
- GPU 환경 확인
- target cleaning 정책 결정
- semi-formal 데이터 확보
- evaluation script 작성
- 모델별 비교 실험 수행

## 15. 팀 공유용 요약

이번 작업의 핵심은 기존 전처리 산출물 `_all.jsonl`을 실제 모델 학습에 넣을 수 있는 형태로 정리한 것이다. 데이터는 speaker 기준으로 train/validation/test로 나누었고, Gemma 4 E4B를 중심으로 여러 모델이 같은 split에서 비교될 수 있도록 모델별 formatted 데이터를 생성했다. 또한 제안서의 Casual / Semi-formal spoken 선택 기능을 구현하기 위해 `<STYLE=casual>`, `<STYLE=semi_formal>` 제어 토큰 구조를 미리 넣었다.

현재는 casual 데이터만 준비되어 있으므로, 다음 단계에서는 Gemma 4 E4B QLoRA 학습을 먼저 성공시키고, 이후 semi-formal 데이터를 외부 corpus나 teacher LLM으로 추가해 style-control 학습으로 확장하는 것이 가장 현실적이다.
