# 파인튜닝 파이프라인 상세 설계

이 문서는 ScriptTuner 프로젝트의 파인튜닝 파이프라인을 팀원들과 공유하기 위한 한국어 설명 문서다. 기존 전처리 파이프라인에서 만들어진 `Pair` 데이터를 split → format → train → generate → evaluate → plot 순으로 처리하는 전체 흐름과, 1차 base model로 **T5Gemma 2-1B**를 선택한 이유, 그리고 실제 구현 시 부딪힌 주요 이슈(라벨 EOS, early stopping)를 정리한다.

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

현재 `split`부터 `evaluate`까지 모두 구현됐고, 학습 곡선 시각화를 위한 `plot` 단계가 추가됐다.

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

T5Gemma 2는 encoder-decoder(seq2seq) 구조라 `input` / `target` text-to-text 형식이 자연스럽다. Gemma 4·Qwen 계열은 decoder-only이므로 instruction/chat 형식이 더 적합하다.

따라서 `format` 단계는 같은 split 데이터를 모델별 학습 포맷으로 변환한다. dispatch는 `scripttuner/training/registry.py`의 `ModelSpec.format_kind`(`"chat"` 또는 `"seq2seq"`)에 기록된다.

### 5.2 지원 모델 키

현재 formatter가 지원하는 모델 키는 다음과 같다.

| model_key | 형식 | 용도 |
|---|---|---|
| `t5gemma2-1b` | seq2seq | **1차 주력 모델** (8GB GPU에 LoRA + grad checkpoint로 적합) |
| `t5gemma2-270m` | seq2seq | smoke test용 경량 |
| `t5gemma2-4b` | seq2seq | escalation 후보 (12GB+ GPU 필요) |
| `gemma4-e2b` | chat | escalation 후보 (chat backend, 학습은 deferred) |
| `gemma4-e4b` | chat | escalation 후보 (≥12GB 권장, deferred) |
| `qwen3-1.7b` | chat | formatting only (학습 backend 미구현) |
| `qwen3-4b` | chat | formatting only (학습 backend 미구현) |

### 5.3 실행 명령

```bash
python -m scripttuner.cli format t5gemma2-1b sbcsae --data-dir data
python -m scripttuner.cli format t5gemma2-270m sbcsae --data-dir data    # smoke
# 아래는 escalation/비교 시 추가 생성
python -m scripttuner.cli format gemma4-e2b sbcsae --data-dir data
python -m scripttuner.cli format gemma4-e4b sbcsae --data-dir data
```

### 5.4 출력 위치

```text
data/finetune/SBCSAE/formatted/
  t5gemma2-1b/
    train.jsonl
    validation.jsonl
    test.jsonl
    MANIFEST.json
  t5gemma2-270m/
  gemma4-e2b/        # 생성한 경우만
  gemma4-e4b/        # 생성한 경우만
```

각 모델 폴더에는 동일한 split 구조가 유지된다. 따라서 나중에 모델별 성능을 비교할 때 데이터 분리 차이 때문에 생기는 변수를 줄일 수 있다.

## 6. Chat 모델용 포맷 (escalation/비교용)

Gemma 4 E4B, Gemma 4 E2B, Qwen3-4B, Qwen3-1.7B는 chat-style JSONL로 변환한다. 학습 backend(`_train_chat`, Unsloth/TRL)는 구현되어 있으나 8GB GPU 환경에선 운용이 어려워 현재 1차 학습에서는 사용하지 않는다.

예시는 다음과 같다.

```json
{
  "messages": [
    {
      "role": "user",
      "content": "<style:casual>\nRewrite the input into natural casual spoken English...\n\nInput:\n...\n\nOutput:"
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
  "style_token": "<style:casual>",
  "source_metadata": {}
}
```

`messages[0]`은 모델 입력이고, `messages[1]`은 정답 target이다.

## 7. T5Gemma 2용 포맷 (1차 주력)

T5Gemma 2는 encoder-decoder 모델이므로 seq2seq 구조로 변환한다.

예시는 다음과 같다.

```json
{
  "input": "<style:casual>\nRewrite the input into natural casual spoken English...\n\nInput:\n...\n\nOutput:",
  "target": "Well, I think ...",
  "pair_id": "...",
  "source": "SBCSAE",
  "style": "casual",
  "speaker": "...",
  "monologue_id": "...",
  "style_token": "<style:casual>",
  "source_metadata": {}
}
```

## 8. 스타일 제어 토큰

제안서에는 사용자가 변환 스타일을 Casual 또는 Semi-formal spoken 중 하나로 선택할 수 있다고 되어 있다. 이를 구현하기 위해 학습 입력 앞에 스타일 제어 토큰을 붙인다.

현재 정의된 토큰은 다음과 같다.

```text
<style:casual>
<style:semi_formal>
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
1차: casual 데이터로 T5Gemma 2-1B LoRA 성공 (✅ 완료)
2차: teacher LLM 또는 외부 corpus로 semi_formal 데이터 확보
3차: casual + semi_formal style-control 학습
4차: 모델별 성능 비교 (필요 시 T5Gemma 2-4B 또는 Gemma 4 escalation)
```

## 9. T5Gemma 2-1B를 1차 모델로 두는 이유

현재 1차 파인튜닝 모델은 **T5Gemma 2-1B**로 잡는다 (제안서의 1순위 후보와도 일치).

이유:

- **task 적합성**: 우리 학습 문제는 `formal_text → spoken_text` 1대1 rewrite다. 채팅 turn-taking이 아니라 단방향 변환이므로 encoder-decoder(seq2seq) 구조가 자연스럽다. T5Gemma 2는 Gemma 3 기반 UL2 adaptation으로 nominally seq2seq를 제공.
- **메모리 적합성**: 1B 사이즈는 bf16 + LoRA + gradient checkpointing 조합으로 8GB GPU(RTX 4060)에 적재 가능. 4B 또는 Gemma 4 E4B는 12GB+를 권장하므로 escalation 후보.
- **escalation 경로 명확**: 품질이 부족하면 T5Gemma 2-4B → Gemma 4 E4B 순서로 키울 수 있고, 같은 base 패밀리(Gemma 3 계열)라서 비교 변수가 적다.

decoder-only Gemma 4 backend(`_train_chat`, Unsloth/TRL)도 코드 상으로 구현되어 있으나, 8GB 환경 제약 + task 구조 fit 이유로 현재 1차에서는 사용하지 않는다.

## 10. 학습 단계 구현

### 10.1 패키지 구조

```text
scripttuner/training/
  registry.py    # ModelSpec(format_kind, hf_id) 단일 source of truth
  formatters.py  # split → formatted JSONL (Stage 5)
  train.py       # _train_seq2seq / _train_chat dispatch (Stage 10)
  generate.py    # 추론 (Stage 11)
  evaluate.py    # spoken-ness 메트릭 (Stage 12)
  plot.py        # 학습 곡선 시각화 (Stage 13)
```

### 10.2 실행 명령

```bash
# 학습
python -m scripttuner.cli train t5gemma2-1b sbcsae \
    --batch-size 1 --grad-accum 16 --max-seq-length 1024 \
    --epochs 8

# 추론 (test split에 대해)
python -m scripttuner.cli generate t5gemma2-1b sbcsae \
    --split test --run-name t5gemma2-1b-SBCSAE-lora-es

# 평가 (spoken-ness 메트릭)
python -m scripttuner.cli evaluate \
    --predictions runs/eval/t5gemma2-1b-SBCSAE-lora-es/predictions.jsonl

# 학습 곡선 PNG
python -m scripttuner.cli plot t5gemma2-1b sbcsae \
    --run-name t5gemma2-1b-SBCSAE-lora-es
```

### 10.3 `_train_seq2seq` 핵심 동작

- PEFT LoRA (`task_type="SEQ_2_SEQ_LM"`, `target_modules="all-linear"`, r=16)
- base는 bf16, gradient checkpointing on
- `model.enable_input_require_grads()` — PEFT(동결 base) + grad_checkpoint 조합에서 입력이 grad를 요구해야 backward가 동작
- `DataCollatorForSeq2Seq(tokenizer)` — `model=` 인자 생략. T5Gemma 2의 `prepare_decoder_input_ids_from_labels` 시그니처와 collator가 호환되지 않으므로, `model.forward`가 라벨을 내부적으로 shift하도록 둔다.
- validation 데이터가 있을 때 `eval_strategy="epoch"`, `save_strategy="epoch"`, `save_total_limit=2`, `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`, `EarlyStoppingCallback(early_stopping_patience=2)` 활성화.
- `per_device_eval_batch_size=batch_size` — HF 기본값 8은 train 1과 무관하게 적용되므로 8GB GPU에서 즉시 OOM. train과 동일 batch로 고정.

### 10.4 산출물

```text
runs/finetune/<run-name>/
  adapter/                # LoRA 가중치 (load_best_model_at_end=True면 best epoch 시점)
  trainer/                # HF Trainer checkpoints (best + 최근, pruning됨)
  log_history.json        # step별 train loss + epoch별 eval loss
  training_curves.png     # plot 단계 생성
  MANIFEST.json           # 하이퍼파라미터, train_loss, best_eval_loss, stopped_epoch

runs/eval/<run-name>/
  predictions.jsonl       # generate 단계 출력 (pair_id/style/input/reference/prediction)
  metrics.json            # evaluate 단계 출력 (filler/pause/length/lexical_density)
```

### 10.5 구현 주의 — seq2seq 라벨 EOS

T5Gemma 2 tokenizer는 `text_target=` 호출 시 **BOS는 자동 추가하지만 EOS는 추가하지 않는다**. 라벨이 `[BOS, t1, ..., tN]`로 끝나면 모델이 종료 토큰을 학습하지 못해, 추론 시 `max_new_tokens`에 도달할 때까지 filler·pause로 padding되는 degenerate tail이 발생한다.

`_tokenize` 내부에서 라벨을 `max_seq_length - 1`로 truncate한 뒤 `tokenizer.eos_token_id`를 명시적으로 append하는 식으로 우회한다. 다른 seq2seq tokenizer를 추가할 때는 이 동작을 다시 확인해야 한다.

## 11. 권장 run 산출물 구조

실제 구현된 산출물 레이아웃은 §10.4 참조. `MANIFEST.json` 필드:

- `model_key` / `base_model` (HF id)
- `format` (chat/seq2seq), `backend` (transformers-seq2seq-lora-bf16 등)
- `formatted_dir`, `adapter_dir`
- `n_train`, `n_validation`
- `lora` (r/alpha/dropout)
- `training` (max_seq_length, epochs, max_steps, batch_size, grad_accum, learning_rate, seed)
- `train_loss`, `train_runtime_sec`
- `early_stopping` (bool), `best_eval_loss`, `stopped_epoch`
- `log_history_path`
- `date`

run-name 컨벤션: `<model_key>-<CORPUS>-lora[-<suffix>]` (예: `t5gemma2-1b-SBCSAE-lora-es`, `-es`는 early stopping 적용 run을 의미하는 임의 접미사).

## 12. 평가 계획

모델 평가에는 자동 평가와 사람 평가를 함께 사용해야 한다.

### 12.1 자동 평가 후보

| 평가 항목 | 목적 | 구현 상태 |
|---|---|---|
| train / val loss | 학습 안정성 확인 | ✅ (log_history + plot) |
| token length ratio | 출력이 너무 길거나 짧아지는지 확인 | ✅ (evaluate) |
| filler count | 구어체 표지 사용량 확인 | ✅ |
| pause token count | pause token 생성 경향 확인 | ✅ |
| lexical density | spoken/formal 스타일 차이 확인 | ✅ (spaCy POS) |
| embedding similarity | 의미 보존 확인 | 미구현 (future) |
| BLEU/ROUGE | reference와의 표면 유사도 참고 | 미구현 (future, 우선순위 낮음) |

현재 구현된 메트릭은 `scripttuner/training/evaluate.py`에서 산출하며 module-5 stats helper를 재사용한다.

### 12.2 val loss와 spoken-ness 메트릭의 신호 차이

1B SBCSAE 학습(`epochs=8 + patience=2`)에서 early stopping은 epoch 4에서 발동, best는 epoch 2(`eval_loss=0.702`)였다. 그러나 같은 어댑터를 test에 적용한 spoken-ness 메트릭은 ref보다 약간 보수적인 분포(filler 1.97 vs ref 2.16, pause:long 2.80 vs ref 4.63)였다. 단일 epoch만 돌린 어댑터가 오히려 ref 분포에 더 가까웠다(filler 2.19, pause:long 3.49).

→ **eval_loss와 spoken-ness 메트릭은 다른 신호를 본다.** eval_loss는 teacher-forced 예측 confidence이고, spoken-ness는 autoregressive 생성 결과의 분포 정렬이다. 둘 다 valid한 신호이므로 사용 목적에 맞춰 선택해야 한다.

### 12.3 사람 평가 후보

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

현재는 기존 파이프라인 결정에 맞춰 보존하는 상태다. 1B 학습 결과의 spoken-ness 메트릭이 ref 분포와 잘 정렬되어 모델이 pause/filler 마커를 자연스러운 빈도로 재현하고 있으므로 target cleaning은 현재 blocking 사항이 아니다. UI/서빙 단계에서 marker-free 출력이 필요해지면 그 시점에 재검토.

## 14. 현재 완료된 것과 남은 것

### 완료

- `formal_text → spoken_text` 학습 방향 확정
- speaker-aware split 구현 (M-?)
- T5Gemma 2 / Gemma 4 / Qwen3용 formatter 구현 + style control token 구조
- casual 데이터 포맷 생성
- `_train_seq2seq` (PEFT LoRA bf16 + grad checkpointing) 구현 — T5Gemma 2-1B에서 검증
- `_train_chat` (Unsloth/TRL) 구현 — backend 코드만, 8GB 환경 미운용
- generate / evaluate / plot CLI 구현
- validation + EarlyStopping(patience=2) + log_history 영구화
- 8GB GPU(RTX 4060) bs=1 + grad_accum=16 + seq=1024 운용 검증
- Kaggle T4 16GB 환경 portability 확인 (같은 config면 4060과 동등 속도)
- 한국어/영어 설계 문서 동기화

### 남은 작업

- semi_formal 데이터 확보 (teacher LLM 또는 외부 corpus)
- semi_formal + casual 결합 학습 (style control token 실효성 검증)
- (선택) 모델 escalation — T5Gemma 2-4B 또는 Gemma 4 E4B (12GB+ GPU 필요)
- (선택) embedding similarity / BLEU 등 추가 메트릭
- (선택) 사람 평가 라운드 설계
- target cleaning 정책 (UI/서빙 요구사항 결정 후)

## 15. 팀 공유용 요약

전처리 산출물 `_all.jsonl`에서 출발해 speaker-aware split → 모델별 formatted JSONL → LoRA 학습 → 추론 → spoken-ness 평가 → 학습 곡선 시각화까지 전 단계를 구현했다. 1차 base는 **T5Gemma 2-1B**로, seq2seq task 적합성과 8GB GPU 적재 가능성을 모두 만족한다. casual 단일 스타일 학습 결과 spoken-ness 메트릭이 reference 분포와 잘 정렬되었다(length ratio ~0.95, filler/pause/lexical density 모두 ref 근사).

학습 구현에서 발견한 주요 이슈는 두 가지였다.

1. **T5Gemma 2 tokenizer가 label에 EOS를 자동 추가하지 않아** 모델이 종료를 학습하지 못하고 추론 시 degenerate tail이 생기는 문제 — `_tokenize`에서 EOS를 명시 append해 해결.
2. **val loss와 spoken-ness 메트릭이 다른 신호를 본다는 점** — early stopping이 val loss 기준 best epoch을 골라도, 그 어댑터의 출력 분포는 ref와 약간 어긋날 수 있음. 본격 운용 시에는 두 신호를 분리해 평가.

다음 단계는 `<style:casual>` / `<style:semi_formal>` 제어 토큰의 실효성을 검증하기 위한 semi_formal 데이터 확보다. teacher LLM 합성 또는 외부 corpus(인터뷰/TED 등) 중 어느 쪽을 선택할지가 가장 큰 결정 사항.
