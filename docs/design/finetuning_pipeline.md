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
python -m scripttuner.cli format t5gemma2-1b sbcsae --data-dir data
python -m scripttuner.cli format gemma4-e2b sbcsae --data-dir data   # chat backend (deferred)
python -m scripttuner.cli format gemma4-e4b sbcsae --data-dir data   # chat backend (deferred)
python -m scripttuner.cli format qwen3-1.7b sbcsae --data-dir data   # formatting only
python -m scripttuner.cli format qwen3-4b sbcsae --data-dir data     # formatting only
```

Chat-style models use a `messages` format. T5Gemma 2 uses an `input` / `target`
seq2seq format. The dispatch is recorded in `scripttuner/training/registry.py`
via `ModelSpec.format_kind` (`"chat"` or `"seq2seq"`).

### 3. Style control

Style control is represented with explicit control tokens:

```text
<style:casual>
<style:semi_formal>
```

The token is included at the beginning of every training input. The formatter
also stores `style`, `style_token`, `pair_id`, `speaker`, and `monologue_id` so
future evaluation can be grouped by style and source.

### 4. Training

Base model selection — **T5Gemma 2** is primary. It is encoder-decoder
(`seq2seq`), matches the task structurally (formal → spoken rewrite, not chat
turn-taking), and the 1B size fits an 8GB GPU with LoRA + gradient
checkpointing. Decoder-only Gemma 4 is supported by `format_kind="chat"` and
`_train_chat` (Unsloth/TRL) but is deferred until a larger GPU is available;
Unsloth recommends ≥12GB for E4B QLoRA.

Command:

```bash
python -m scripttuner.cli train t5gemma2-1b sbcsae \
    --batch-size 1 --grad-accum 16 --max-seq-length 1024 \
    --epochs 8
```

Implementation (`scripttuner/training/train.py:_train_seq2seq`):

- PEFT LoRA (`task_type="SEQ_2_SEQ_LM"`, `target_modules="all-linear"`)
- bf16 on base model; gradient checkpointing on
- `model.enable_input_require_grads()` required for PEFT+grad_checkpoint
- `DataCollatorForSeq2Seq(tokenizer)` without `model=` (T5Gemma 2's
  `prepare_decoder_input_ids_from_labels` signature is incompatible; the
  model's `forward` shifts labels internally)
- When `validation.jsonl` exists: `eval_strategy="epoch"`,
  `save_strategy="epoch"`, `save_total_limit=2`,
  `load_best_model_at_end=True`, `metric_for_best_model="eval_loss"`,
  `EarlyStoppingCallback(early_stopping_patience=2)`
- `per_device_eval_batch_size=batch_size` — HF default 8 OOMs on 8GB when
  train batch is 1

Output:

```text
runs/finetune/<run-name>/
  adapter/                # LoRA weights
  trainer/                # HF Trainer checkpoints (most recent + best, pruned)
  log_history.json        # per-step train loss + per-epoch eval loss
  training_curves.png     # produced by `plot` subcommand
  MANIFEST.json           # hyperparams, train_loss, best_eval_loss, stopped_epoch
```

#### Implementation note — seq2seq label EOS

The T5Gemma 2 tokenizer adds BOS but not EOS to `text_target=` output. Without
manual EOS append the model never learns where to stop, and inference produces
degenerate tails padded with fillers/pauses until `max_new_tokens`. Labels are
constructed by truncating to `max_seq_length - 1` and appending
`tokenizer.eos_token_id` (see `_tokenize` in `_train_seq2seq`). Adding another
seq2seq tokenizer requires re-verifying this behavior.

### 5. Inference / generation

```bash
python -m scripttuner.cli generate t5gemma2-1b sbcsae \
    --split test --run-name t5gemma2-1b-SBCSAE-lora-es
```

`scripttuner/training/generate.py` loads the base model + LoRA adapter, runs
greedy decoding on `<split>.jsonl`, and writes `predictions.jsonl` with
`pair_id`, `style`, `speaker`, `input`, `reference`, `prediction`. Optional
HuggingFace `generate()` knobs are exposed via
`--repetition-penalty` / `--no-repeat-ngram-size`; both default to no-op.
The chat (decoder-only) generation path is not implemented yet.

### 6. Evaluation

```bash
python -m scripttuner.cli evaluate \
    --predictions runs/eval/<run-name>/predictions.jsonl
```

`scripttuner/training/evaluate.py` reuses the module-5 stats helpers to compute
length / filler / pause / lexical-density distributions on predictions vs.
references. Output is `metrics.json` colocated with the predictions. The
spoken-ness metric is task-specific: a model that emits more fillers/pauses
than the reference is *over-applying* the spoken style, while fewer markers
than the reference is *under-applying* it.

### 7. Visualization

```bash
python -m scripttuner.cli plot t5gemma2-1b sbcsae --run-name <run-name>
```

`scripttuner/training/plot.py` reads `log_history.json` and emits
`training_curves.png` (train loss per step + val loss per epoch). matplotlib
is in the `train` dependency group; install with `uv sync --group train`.

#### Observation — val loss vs. spoken-ness metric

In the 1B SBCSAE run with `epochs=8 + patience=2`, early stopping fired at
epoch 4 with the best checkpoint at epoch 2 (`eval_loss=0.702`). The selected
checkpoint produced fewer fillers/pauses than the reference (filler mean 1.97
vs. ref 2.16, pause:long mean 2.80 vs. ref 4.63), while a single-epoch run on
the same data was closer to the reference distribution (filler 2.19,
pause:long 3.49). **`eval_loss` and the task-specific spoken-ness metric carry
different signals**: lower `eval_loss` means more confident teacher-forced
predictions, not necessarily closer alignment with the reference output
distribution. Both checkpoints are valid; selection depends on which signal is
prioritized for downstream use.

## Data-quality note

Current targets preserve `spoken_text` as produced by the existing pipeline,
including pause tokens. Some SBCSAE transcription artifacts may remain in the
target text. The current spoken-ness metrics indicate the model has learned to
reproduce these markers in a ref-aligned distribution, so a target-cleaning
step is not blocking; revisit only if downstream consumers (UI/serving) need
marker-free output.

