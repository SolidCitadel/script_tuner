# Fine-tuning pipeline design

This document defines the fine-tuning preparation steps that start after the
existing preprocessing pipeline has produced `Pair` JSONL data.

## Input

- Source file: `data/pairs/SBCSAE/_all.jsonl`
- Schema: `Pair`
- Training direction: `formal_text -> spoken_text`
- Current style coverage: `casual`
- Reserved future style: `semi_formal`

The current SBCSAE data is casual spoken English. Semi-formal spoken data should
be added later from an external corpus or teacher-LLM generated targets.

## Stages

### 1. Speaker-aware split

Command:

```bash
python -m scripttuner.cli split sbcsae --data-dir data --seed 42
```

Output:

```text
data/finetune/SBCSAE/splits/
  train.jsonl
  validation.jsonl
  test.jsonl
  MANIFEST.json
```

The split is speaker-aware: one speaker appears in only one split. This reduces
leakage from speaker-specific phrasing.

### 2. Model-specific formatting

Commands:

```bash
python -m scripttuner.cli format gemma4-e4b sbcsae --data-dir data
python -m scripttuner.cli format gemma4-e2b sbcsae --data-dir data
python -m scripttuner.cli format qwen3-4b sbcsae --data-dir data
python -m scripttuner.cli format qwen3-1.7b sbcsae --data-dir data
python -m scripttuner.cli format t5gemma2 sbcsae --data-dir data
```

Chat-style models use a `messages` format. T5Gemma 2 uses an `input` / `target`
seq2seq format.

### 3. Style control

Style control is represented with explicit control tokens:

```text
<STYLE=casual>
<STYLE=semi_formal>
```

The token is included at the beginning of every training input. The formatter
also stores `style`, `style_token`, `pair_id`, `speaker`, and `monologue_id` so
future evaluation can be grouped by style and source.

### 4. Future training stage

The first training target is Gemma 4 E4B with QLoRA/SFT. The same split and
evaluation data should then be reused for Gemma 4 E2B, Qwen-family models, and
T5Gemma 2.

Recommended future output layout:

```text
runs/finetune/<model-run-id>/
runs/eval/<model-run-id>/
```

## Data-quality note

Current targets preserve `spoken_text` as produced by the existing pipeline,
including pause tokens. Some SBCSAE transcription artifacts may remain in the
target text. Before final training, add either a target-cleaning policy or a
quality-filtering step if the model should not emit transcription markers.

